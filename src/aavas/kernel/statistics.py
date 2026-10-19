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
