"""Distribution statistics. Pure, and dependency-free.

``psi`` lived in ``learning.drift_detection``, which meant the feature-discovery
agent had to import the learning package to ask whether a candidate's
distribution was stable. PSI is not a drift-detection concept - drift detection
is one thing that uses it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index. >0.10 watch, >0.25 investigate."""
    # astype(float) is not redundant: pd.to_numeric leaves a boolean Series boolean,
    # and np.quantile refuses to subtract booleans. A boolean feature has a
    # perfectly meaningful two-bucket PSI, so it is cast rather than excluded.
    expected = pd.to_numeric(pd.Series(expected), errors="coerce").dropna().astype(float).to_numpy()
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().astype(float).to_numpy()
    if len(expected) < 10 or len(actual) < 10:
        return float("nan")

    # A low-cardinality feature - a flag, a small count - has no meaningful
    # quantile buckets: np.unique collapses the edges and the quantile path
    # returns 0.0, silently reporting "no drift" for a binary feature whose rate
    # has genuinely moved. Those are compared over their observed categories.
    levels = np.unique(np.concatenate([expected, actual]))
    if len(levels) <= buckets:
        expected_share = np.array([(expected == level).mean() for level in levels])
        actual_share = np.array([(actual == level).mean() for level in levels])
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        expected_share = np.histogram(expected, bins=edges)[0] / len(expected)
        actual_share = np.histogram(actual, bins=edges)[0] / len(actual)

    floor = 1e-4
    expected_share = np.clip(expected_share, floor, None)
    actual_share = np.clip(actual_share, floor, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def ks(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value (NaN, NaN when too few values).

    At the sizes a loan book reaches, the p-value flags shifts nobody would act on;
    the statistic - the largest gap between the two cumulative distributions - is
    the part to threshold.
    """
    from scipy import stats

    expected = pd.to_numeric(pd.Series(expected), errors="coerce").dropna().astype(float).to_numpy()
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().astype(float).to_numpy()
    if len(expected) < 10 or len(actual) < 10:
        return float("nan"), float("nan")
    result = stats.ks_2samp(expected, actual)
    return float(result.statistic), float(result.pvalue)


def category_psi(expected, actual) -> tuple[float, float]:
    """PSI over category shares, and the chi-square p-value for a categorical column."""
    from scipy import stats

    expected, actual = pd.Series(expected).astype(str), pd.Series(actual).astype(str)
    if len(expected) < 10 or len(actual) < 10:
        return float("nan"), float("nan")
    levels = sorted(set(expected) | set(actual))
    e = expected.value_counts().reindex(levels, fill_value=0).to_numpy(float)
    a = actual.value_counts().reindex(levels, fill_value=0).to_numpy(float)
    floor = 1e-4
    es, as_ = np.clip(e / e.sum(), floor, None), np.clip(a / a.sum(), floor, None)
    value = float(np.sum((as_ - es) * np.log(as_ / es)))
    p = float(stats.chi2_contingency(np.vstack([e, a]) + 0.5)[1]) if len(levels) > 1 else 1.0
    return value, p


def page_hinkley(values: np.ndarray, delta: float, threshold: float, min_observations: int = 30) -> int | None:
    """Index of the first observation where the Page-Hinkley test signals an upward
    change in the mean of ``values`` (e.g. an error stream in date order), or None.

    ``delta`` is the drift tolerated per observation and ``threshold`` the alarm
    level, both in the units of ``values``.
    """
    values = np.asarray(values, dtype=float)
    mean, cumulative, minimum = 0.0, 0.0, 0.0
    for i, x in enumerate(values, start=1):
        mean += (x - mean) / i
        cumulative += x - mean - delta
        minimum = min(minimum, cumulative)
        if i >= min_observations and cumulative - minimum > threshold:
            return i - 1
    return None


def cusum(values: np.ndarray, target: float, slack: float, threshold: float,
          min_observations: int = 30, side: str = "upper") -> int | None:
    """Index where a CUSUM against a known in-control mean ``target`` first signals, or None.

    Use when the in-control level is known (a model card's out-of-sample error): it
    accumulates the excess over ``target + slack`` (``upper``), the shortfall under
    ``target - slack`` (``lower``), or both, and signals past ``threshold``.
    """
    values = np.asarray(values, dtype=float)
    high = low = 0.0
    for i, x in enumerate(values):
        high = max(0.0, high + (x - target - slack))
        low = max(0.0, low + (target - slack - x))
        hit = (side in ("upper", "both") and high > threshold) or (side in ("lower", "both") and low > threshold)
        if i + 1 >= min_observations and hit:
            return i
    return None
