"""The data profiler and the statistical battery.

Two of the ten specialist roles, and the two that have to come first: nothing
downstream is worth running until the data has been understood and the evidence
computed. Everything here is deterministic - no model, no LLM, no network - which
is what makes it cheap enough to run over every column of every dataset every
time.

The battery is chosen for lending rather than for completeness. Information Value
and WoE are here because a credit reviewer reads them; VIF is here because
collinear features make a coefficient uninterpretable and an explanation
unstable; PSI is here because a feature that drifts invalidates the model quietly.
Tests that would produce a number nobody acts on are left out.

Every statistic reports *insufficient evidence* rather than a value when the slice
is too thin. A p-value from eleven rows is a number, not evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..kernel.statistics import psi
from ..logging_utils import get_logger

log = get_logger(__name__)

MIN_ROWS = 50
MIN_BIN = 30
HIGH_CARDINALITY = 50
# Reported instead of infinity when a column is an exact combination of others.
VIF_CAP = 10_000.0
RARE_CATEGORY_SHARE = 0.01
INSUFFICIENT = "insufficient evidence"


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _round(value: Any, places: int = 4) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), places)


# --------------------------------------------------------------- 1. profiling


def profile_numeric(series: pd.Series) -> dict[str, Any]:
    values = _finite(series)
    if values.empty:
        return {"kind": "numeric", "usable": False, "why": "no finite values"}

    q = values.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    iqr = float(q[0.75] - q[0.25])
    lower, upper = q[0.25] - 1.5 * iqr, q[0.75] + 1.5 * iqr
    outliers = float(((values < lower) | (values > upper)).mean()) if iqr > 0 else 0.0

    return {
        "kind": "numeric", "usable": True,
        "count": int(len(values)), "missing_pct": _round(100 * (1 - len(values) / len(series)), 2),
        "mean": _round(values.mean()), "std": _round(values.std()),
        "min": _round(values.min()), "max": _round(values.max()),
        "p01": _round(q[0.01]), "p25": _round(q[0.25]), "median": _round(q[0.5]),
        "p75": _round(q[0.75]), "p99": _round(q[0.99]),
        "skew": _round(values.skew()), "kurtosis": _round(values.kurtosis()),
        "zeros_pct": _round(100 * float((values == 0).mean()), 2),
        "distinct": int(values.nunique()),
        "outlier_pct": _round(100 * outliers, 2),
        "shape": _shape_label(values),
    }


def _shape_label(values: pd.Series) -> str:
    skew = float(values.skew())
    if not np.isfinite(skew):
        return "undefined"
    if values.nunique() <= 2:
        return "binary"
    if skew > 1:
        return "right-skewed"
    if skew < -1:
        return "left-skewed"
    return "roughly symmetric"


def profile_categorical(series: pd.Series) -> dict[str, Any]:
    values = series.dropna().astype(str)
    if values.empty:
        return {"kind": "categorical", "usable": False, "why": "no values"}
    counts = values.value_counts()
    share = counts / len(values)
    rare = share[share < RARE_CATEGORY_SHARE]
    return {
        "kind": "categorical", "usable": True,
        "count": int(len(values)),
        "missing_pct": _round(100 * (1 - len(values) / len(series)), 2),
        "cardinality": int(counts.size),
        "high_cardinality": bool(counts.size > HIGH_CARDINALITY),
        "top": [{"value": str(k), "share_pct": _round(100 * v, 2)} for k, v in share.head(5).items()],
        "rare_categories": int(rare.size),
        "rare_share_pct": _round(100 * float(rare.sum()), 2),
        # A category seen in 0.2% of training rows will arrive unseen at scoring
        # time often enough to matter, and an unseen category is a silent default.
        "unseen_category_risk": "high" if rare.size > 5 else "low" if rare.empty else "moderate",
    }


def profile(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    columns = columns or list(df.columns)
    out: dict[str, Any] = {}
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            out[column] = profile_numeric(series)
        elif pd.api.types.is_datetime64_any_dtype(series):
            out[column] = _profile_temporal(series)
        else:
            out[column] = profile_categorical(series)
    return out


def _profile_temporal(series: pd.Series) -> dict[str, Any]:
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return {"kind": "temporal", "usable": False, "why": "no parseable dates"}
    return {"kind": "temporal", "usable": True, "count": int(len(values)),
            "from": str(values.min().date()), "to": str(values.max().date()),
            "span_days": int((values.max() - values.min()).days),
            "note": ("A temporal column is a leakage surface, not a feature. It orders the "
                     "split; it does not enter the model.")}


# ------------------------------------------------- 2. the statistical battery


def information_value(values: pd.Series, target: pd.Series, bins: int = 10) -> dict[str, Any]:
    """IV and WoE by bin. The statistic a credit reviewer actually reads."""
    frame = pd.concat([pd.to_numeric(values, errors="coerce"),
                       pd.to_numeric(target, errors="coerce")], axis=1).dropna()
    frame.columns = ["x", "y"]
    if len(frame) < MIN_ROWS or frame["y"].nunique() != 2:
        return {"iv": None, "why": INSUFFICIENT if len(frame) < MIN_ROWS
                else "target is not binary"}

    try:
        frame["bin"] = pd.qcut(frame["x"], bins, duplicates="drop")
    except ValueError:
        frame["bin"] = frame["x"]

    good, bad = (frame["y"] == 0).sum(), (frame["y"] == 1).sum()
    if good == 0 or bad == 0:
        return {"iv": None, "why": "one class is absent"}

    rows, total_iv = [], 0.0
    for name, group in frame.groupby("bin", observed=True):
        if len(group) < MIN_BIN:
            continue
        g = max((group["y"] == 0).sum(), 0.5) / good
        b = max((group["y"] == 1).sum(), 0.5) / bad
        woe = float(np.log(g / b))
        contribution = (g - b) * woe
        total_iv += contribution
        rows.append({"bin": str(name), "n": int(len(group)),
                     "event_rate": _round(float(group["y"].mean())), "woe": _round(woe)})

    if not rows:
        return {"iv": None, "why": f"no bin reached {MIN_BIN} rows"}
    return {"iv": _round(total_iv), "strength": _iv_label(total_iv), "bins": rows,
            "monotonic_woe": _is_monotonic([r["woe"] for r in rows])}


def _iv_label(iv: float) -> str:
    # The conventional lending bands. >0.5 is usually a sign of leakage rather
    # than of an unusually good feature, and is labelled as such.
    if iv < 0.02:
        return "not predictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspiciously strong - check for leakage"


def _is_monotonic(values: list[float]) -> bool:
    if len(values) < 3:
        return False
    series = pd.Series(values)
    return bool(series.is_monotonic_increasing or series.is_monotonic_decreasing)


def mutual_information(values: pd.Series, target: pd.Series, bins: int = 10) -> float | None:
    """Captures a non-linear relationship that correlation misses."""
    frame = pd.concat([pd.to_numeric(values, errors="coerce"),
                       pd.to_numeric(target, errors="coerce")], axis=1).dropna()
    if len(frame) < MIN_ROWS:
        return None
    frame.columns = ["x", "y"]
    try:
        x = pd.qcut(frame["x"], bins, duplicates="drop", labels=False)
    except ValueError:
        return None
    y = frame["y"] if frame["y"].nunique() <= bins else pd.qcut(
        frame["y"], bins, duplicates="drop", labels=False)

    joint = pd.crosstab(x, y, normalize=True).to_numpy()
    px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = joint * np.log(joint / (px * py))
    return _round(float(np.nansum(term)))


def variance_inflation(frame: pd.DataFrame, column: str) -> float | None:
    """VIF by regressing one column on the rest. A collinear feature makes its own
    contribution uninterpretable, which matters more here than it does for
    accuracy: an unstable attribution becomes an unstable customer explanation."""
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if column not in numeric.columns or len(numeric) < MIN_ROWS or numeric.shape[1] < 2:
        return None
    y = numeric[column].to_numpy(float)
    X = numeric.drop(columns=[column]).to_numpy(float)
    if y.std() == 0:
        return None
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    residual = y - X @ beta
    ss_res, ss_tot = float((residual ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return None
    r2 = 1 - ss_res / ss_tot
    # VIF = 1/(1-R^2), which diverges as R^2 approaches 1. Returning None there
    # would report the *most severe* collinearity as "not computed", which is the
    # wrong way round: a column that is an exact combination of others is the
    # clearest finding this test can produce. It is capped instead, and the cap is
    # a true statement - "at least VIF_CAP" - rather than an unrepresentable
    # infinity that would not survive JSON.
    if r2 >= 1 - 1e-9:
        return VIF_CAP
    return _round(min(1.0 / (1.0 - r2), VIF_CAP), 3)


def group_separation(values: pd.Series, target: pd.Series) -> dict[str, Any]:
    """Does the feature separate the two outcome classes?

    Mann-Whitney U rather than a t-test, because almost nothing in a loan book is
    normally distributed and a t-test on a right-skewed amount reports confidence
    it has not earned. The KS statistic is reported alongside because a credit
    team reads it.
    """
    frame = pd.concat([pd.to_numeric(values, errors="coerce"),
                       pd.to_numeric(target, errors="coerce")], axis=1).dropna()
    frame.columns = ["x", "y"]
    if len(frame) < MIN_ROWS or frame["y"].nunique() != 2:
        return {"test": "mann_whitney", "p_value": None, "why": INSUFFICIENT}

    a = frame.loc[frame["y"] == 0, "x"].to_numpy()
    b = frame.loc[frame["y"] == 1, "x"].to_numpy()
    if len(a) < MIN_BIN or len(b) < MIN_BIN:
        return {"test": "mann_whitney", "p_value": None, "why": INSUFFICIENT}

    try:
        from scipy import stats

        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        ks = stats.ks_2samp(a, b)
        return {"test": "mann_whitney", "statistic": _round(u.statistic, 1),
                "p_value": _round(u.pvalue, 6), "ks_statistic": _round(ks.statistic),
                "ks_p_value": _round(ks.pvalue, 6),
                "significant": bool(u.pvalue < 0.01),
                "effect_size_auc": _round(u.statistic / (len(a) * len(b)))}
    except Exception:                                           # noqa: BLE001
        # scipy is not a hard dependency. The AUC-equivalent of the U statistic is
        # computable without it, and is the number that actually matters.
        order = pd.Series(np.concatenate([a, b])).rank().to_numpy()
        auc = (order[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        return {"test": "rank_auc_without_scipy", "p_value": None,
                "effect_size_auc": _round(auc),
                "why": "scipy unavailable; the rank AUC is reported without a p-value"}


def cramers_v(values: pd.Series, target: pd.Series) -> float | None:
    """Association between a categorical feature and a categorical target."""
    frame = pd.concat([values.astype(str), target], axis=1).dropna()
    if len(frame) < MIN_ROWS:
        return None
    table = pd.crosstab(frame.iloc[:, 0], frame.iloc[:, 1]).to_numpy()
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    n = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((table - expected) ** 2 / expected))
    denominator = n * (min(table.shape) - 1)
    return _round(float(np.sqrt(chi2 / denominator))) if denominator else None


def stability(values: pd.Series, halves: int = 2) -> dict[str, Any]:
    clean = _finite(values)
    if len(clean) < 2 * MIN_ROWS:
        return {"psi": None, "why": INSUFFICIENT}
    cut = len(clean) // halves
    value = psi(clean.iloc[:cut].to_numpy(), clean.iloc[cut:].to_numpy())
    return {"psi": _round(value), "stable": bool(np.isfinite(value) and value < 0.25),
            "note": "computed between the older and newer halves of the same frame"}


def examine(values: pd.Series, target: pd.Series, contract: pd.DataFrame | None = None,
            name: str = "") -> dict[str, Any]:
    """The full battery for one feature. Every number, no verdict."""
    numeric = pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
    report: dict[str, Any] = {
        "feature": name,
        "profile": profile_numeric(values) if numeric else profile_categorical(values),
    }
    if not numeric:
        report["association"] = {"cramers_v": cramers_v(values, target)}
        return report

    report["information_value"] = information_value(values, target)
    report["mutual_information"] = mutual_information(values, target)
    report["separation"] = group_separation(values, target)
    report["stability"] = stability(values)
    if contract is not None and name and name in contract.columns:
        report["vif"] = variance_inflation(contract, name)
    elif contract is not None and name:
        widened = contract.copy()
        widened[name] = pd.to_numeric(values, errors="coerce")
        report["vif"] = variance_inflation(widened, name)
    return report
