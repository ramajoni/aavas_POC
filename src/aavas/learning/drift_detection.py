"""Drift detection: is the world still the one the models were fitted on?

Two kinds of drift matter to a pricing engine and they fail differently:

* input drift - the mix of files changed (a new geography, a channel push).
  Detected with PSI against the training distribution.
* outcome drift - the same file now converts differently, usually because
  competitors moved. This is the dangerous one: the model keeps looking
  confident while its demand curve is stale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import model_config
# PSI moved to the kernel: the feature-discovery agent needed it and had to
# import this package for one pure function.
from ..kernel.statistics import psi
from ..logging_utils import get_logger

log = get_logger(__name__)


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  features: list[str] | None = None) -> pd.DataFrame:
    cfg = model_config()["drift"]
    features = features or [c for c in reference.columns
                            if pd.api.types.is_numeric_dtype(reference[c]) and c in current.columns]
    rows = []
    for feature in features:
        value = psi(reference[feature].to_numpy(), current[feature].to_numpy())
        status = "ok"
        if np.isfinite(value):
            if value >= cfg["psi_alert"]:
                status = "alert"
            elif value >= cfg["psi_warn"]:
                status = "warn"
        rows.append({"feature": feature, "psi": round(value, 4) if np.isfinite(value) else None,
                     "status": status})
    table = pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
    alerts = table[table["status"] == "alert"]
    if len(alerts):
        log.warning("input drift alert on: %s", ", ".join(alerts["feature"].tolist()))
    return table.reset_index(drop=True)


def conversion_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Has the price/conversion relationship itself moved?

    Compares realized S2D at comparable competitor gaps. A shift here means the
    demand curve is stale, which no amount of input-distribution monitoring
    would catch.
    """
    cfg = model_config()["drift"]
    if len(current) < cfg["min_samples_for_check"]:
        return {"status": "insufficient_data", "n_current": len(current)}

    bins = [-np.inf, -25, 0, 25, 50, 75, np.inf]
    labels = ["below -25", "-25..0", "0..25", "25..50", "50..75", "75+"]
    reference = reference.assign(_bucket=pd.cut(reference["competitor_gap_bps"], bins=bins, labels=labels))
    current = current.assign(_bucket=pd.cut(current["competitor_gap_bps"], bins=bins, labels=labels))

    ref_rate = reference.groupby("_bucket", observed=True)["disbursed"].mean()
    cur_rate = current.groupby("_bucket", observed=True)["disbursed"].mean()
    comparison = pd.DataFrame({"reference_s2d": ref_rate, "current_s2d": cur_rate}).dropna()
    comparison["change_pts"] = ((comparison["current_s2d"] - comparison["reference_s2d"]) * 100).round(2)

    worst = comparison["change_pts"].abs().max() if len(comparison) else 0.0
    status = "alert" if worst >= 8 else "warn" if worst >= 4 else "ok"
    if status != "ok":
        log.warning("conversion drift %s: largest bucket shift %.1f pts", status, worst)
    return {
        "status": status,
        "max_shift_pts": float(worst),
        "by_competitor_gap": comparison.reset_index().to_dict(orient="records"),
        "overall_s2d_reference": round(float(reference["disbursed"].mean()), 4),
        "overall_s2d_current": round(float(current["disbursed"].mean()), 4),
    }


def performance_drift(decisions: pd.DataFrame) -> dict:
    """Are recommendations still landing where the models said they would?"""
    if "expected_conversion" not in decisions.columns or decisions.empty:
        return {"status": "insufficient_data"}
    expected = pd.to_numeric(decisions["expected_conversion"], errors="coerce")
    actual = pd.to_numeric(decisions["converted"], errors="coerce")
    valid = expected.notna() & actual.notna()
    if valid.sum() < model_config()["drift"]["min_samples_for_check"]:
        return {"status": "insufficient_data", "n": int(valid.sum())}

    gap = float(actual[valid].mean() - expected[valid].mean())
    status = "alert" if abs(gap) > 0.10 else "warn" if abs(gap) > 0.05 else "ok"
    return {
        "status": status, "n": int(valid.sum()),
        "expected_conversion": round(float(expected[valid].mean()), 4),
        "actual_conversion": round(float(actual[valid].mean()), 4),
        "calibration_gap": round(gap, 4),
    }
