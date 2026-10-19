"""Group fairness metrics. Pure, and dependency-free.

Moved from ``automl`` to ``models`` when the training path needed it, and
here because it is a primitive: pure numpy and pandas, shared by the model
trainers and the AutoML search. Neither owns it.

Originally:

This lived in ``aavas.automl`` and gated candidates in a search that has never
promoted anything, while the models actually pricing decisions were fitted by
``scripts/train_all.py`` and never evaluated at all. It sits here now because
fairness is a property of a model, not of the way it was selected, and both paths
must reach it - ``models`` may not import ``automl``.

Outcome fairness, error fairness and - the one that matters most for lending -
calibration fairness.

Calibration-by-group is the primary criterion here, not demographic parity. The
two are mathematically incompatible whenever base rates differ across groups, and
for a lender the defensible statement is that a predicted PD of 10% means 10% in
every group. Parity is reported alongside and explained, never optimised.

Thin slices return `insufficient evidence` rather than a number, following the
same discipline as `learning/override_learning.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MIN_GROUP_SIZE = 50

# The axis the gate is computed over. Age is used because it is the protected
# characteristic this dataset actually carries; the banding is coarse on purpose,
# because a thin band produces a confident-looking number from ten rows.
AGE_BANDS = [(18, 30, "18-29"), (30, 40, "30-39"), (40, 50, "40-49"),
             (50, 60, "50-59"), (60, 200, "60+")]
DEFAULT_GROUP_COLUMN = "age_band"


def age_band(series) -> pd.Series:
    ages = pd.to_numeric(series, errors="coerce")
    out = pd.Series("unknown", index=series.index, dtype=object)
    for low, high, label in AGE_BANDS:
        out[(ages >= low) & (ages < high)] = label
    return out


def groups_for(frame: pd.DataFrame, column: str = DEFAULT_GROUP_COLUMN):
    """The group labels for a frame, or None when the axis is not available.

    Returning None rather than raising matters: a model fitted on a frame without
    the protected characteristic must record that fairness could not be measured,
    not that it passed.
    """
    if column == DEFAULT_GROUP_COLUMN:
        if "applicant_age" not in frame.columns:
            return None
        return age_band(frame["applicant_age"]).to_numpy()
    if column not in frame.columns:
        return None
    return frame[column].astype(str).to_numpy()


def _rates(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "selection_rate": float(np.mean(y_pred)) if len(y_pred) else float("nan"),
        "tpr": tp / (tp + fn) if (tp + fn) else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "fnr": fn / (tp + fn) if (tp + fn) else float("nan"),
    }


def by_group(y_true, p_pred, groups, *, threshold: float = 0.5,
             min_size: int = MIN_GROUP_SIZE) -> dict[str, Any]:
    """Per-group outcome, error and calibration metrics."""
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred, dtype=float)
    groups = np.asarray(groups)
    y_pred = (p_pred >= threshold).astype(int)

    out: dict[str, Any] = {}
    for name in np.unique(groups):
        mask = groups == name
        n = int(mask.sum())
        if n < min_size:
            out[str(name)] = {"n": n, "verdict": "insufficient evidence"}
            continue
        metrics = _rates(y_true[mask], y_pred[mask])
        metrics["n"] = n
        metrics["mean_predicted"] = float(p_pred[mask].mean())
        metrics["actual_rate"] = float(y_true[mask].mean())
        # Calibration gap: predicted minus realised, in probability points.
        metrics["calibration_gap"] = round(metrics["mean_predicted"] - metrics["actual_rate"], 4)
        out[str(name)] = {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in metrics.items()}
    return out


def gaps(per_group: dict[str, Any]) -> dict[str, float]:
    """Max-minus-min across groups with enough evidence to compare."""
    usable = {k: v for k, v in per_group.items() if v.get("verdict") != "insufficient evidence"}
    if len(usable) < 2:
        return {"comparable_groups": len(usable)}

    def spread(key: str) -> float:
        values = [v[key] for v in usable.values() if v.get(key) == v.get(key)]
        return round(max(values) - min(values), 4) if values else float("nan")

    return {
        "comparable_groups": len(usable),
        "selection_rate_gap": spread("selection_rate"),
        "tpr_gap": spread("tpr"),
        "fpr_gap": spread("fpr"),
        "fnr_gap": spread("fnr"),
        "calibration_gap": spread("calibration_gap"),
    }


def evaluate(y_true, p_pred, groups, *, threshold: float = 0.5,
             max_calibration_gap: float = 0.05,
             max_selection_rate_gap: float = 0.10) -> dict[str, Any]:
    """The gate the AutoML search calls. Calibration is the binding criterion."""
    per_group = by_group(y_true, p_pred, groups, threshold=threshold)
    spread = gaps(per_group)

    reasons: list[str] = []
    if spread.get("comparable_groups", 0) < 2:
        return {"passed": None, "per_group": per_group, "gaps": spread,
                "reasons": ["fewer than two groups have enough evidence to compare"]}

    calibration = spread.get("calibration_gap", float("nan"))
    if calibration == calibration and calibration > max_calibration_gap:
        reasons.append(f"calibration gap {calibration:.3f} exceeds {max_calibration_gap}")

    selection = spread.get("selection_rate_gap", float("nan"))
    parity_note = None
    if selection == selection and selection > max_selection_rate_gap:
        parity_note = (f"selection-rate gap is {selection:.3f}. Reported, not gated: parity and "
                       "calibration cannot both hold when base rates differ across groups.")

    return {"passed": not reasons, "per_group": per_group, "gaps": spread,
            "reasons": reasons, "parity_note": parity_note,
            "primary_criterion": "calibration_by_group"}
