"""Branch-level evaluation - because a global average hides a local failure.

One number for the whole book is the most comfortable and least useful way to
report a model. A model at AUC 0.87 across 14 branches can be at 0.77 in one of
them, and the branch that is being priced badly is not visible anywhere in the
headline.

This module answers three questions the global metrics cannot:

* **Does every branch clear the bar, or only the average?**  ``by_unit``
* **How much do branches differ from each other?**  ``variance``
* **Should this model be promoted at all?**  ``evaluate`` - which fails a
  candidate when the global result passes but a branch does not.

The same machinery works for any operating unit: ``branch_id``, ``city``,
``state``, or a partner institution in a federated setting. Branch is the default
because it is the level at which a sales officer actually meets the customer.

Thin units return ``insufficient evidence`` rather than a number, following
``learning/override_learning.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..logging_utils import get_logger
from .calibration import expected_calibration_error
from .selection import ks_statistic

log = get_logger(__name__)

MIN_UNIT_SIZE = 150


def _unit_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    y_pred = (p >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    try:
        auc = float(roc_auc_score(y_true, p))
    except ValueError:                      # a branch with one outcome class only
        auc = float("nan")

    return {
        "auc": auc,
        "ks": ks_statistic(y_true, p),
        "ece": expected_calibration_error(p, y_true),
        "tpr": tp / (tp + fn) if (tp + fn) else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "selection_rate": float(np.mean(y_pred)),
        "mean_predicted": float(np.mean(p)),
        "actual_rate": float(np.mean(y_true)),
    }


def by_unit(y_true, p_pred, units, *, threshold: float = 0.5,
            min_size: int = MIN_UNIT_SIZE) -> dict[str, Any]:
    """Per-branch metrics, in the shape of the federated comparison table."""
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred, dtype=float)
    units = np.asarray(units)

    out: dict[str, Any] = {}
    for name in np.unique(units):
        mask = units == name
        n = int(mask.sum())
        if n < min_size:
            out[str(name)] = {"n": n, "verdict": "insufficient evidence"}
            continue
        metrics = _unit_metrics(y_true[mask], p_pred[mask], threshold)
        metrics["calibration_gap"] = metrics["mean_predicted"] - metrics["actual_rate"]
        metrics["n"] = n
        out[str(name)] = {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in metrics.items()}
    return out


def variance(per_unit: dict[str, Any]) -> dict[str, Any]:
    """How far apart the branches are - the number a single average conceals."""
    usable = {k: v for k, v in per_unit.items() if v.get("verdict") != "insufficient evidence"}
    if len(usable) < 2:
        return {"comparable_units": len(usable)}

    def spread(key: str) -> dict[str, Any]:
        pairs = [(k, v[key]) for k, v in usable.items()
                 if isinstance(v.get(key), float) and v[key] == v[key]]
        if not pairs:
            return {}
        values = np.array([v for _, v in pairs], dtype=float)
        worst = min(pairs, key=lambda kv: kv[1]) if key in {"auc", "ks"} \
            else max(pairs, key=lambda kv: abs(kv[1]))
        return {"min": round(float(values.min()), 4), "max": round(float(values.max()), 4),
                "spread": round(float(values.max() - values.min()), 4),
                "std": round(float(values.std()), 4),
                "worst_unit": worst[0], "worst_value": round(float(worst[1]), 4)}

    return {"comparable_units": len(usable), "auc": spread("auc"), "ece": spread("ece"),
            "fpr": spread("fpr"), "calibration_gap": spread("calibration_gap")}


def evaluate(y_true, p_pred, units, *, threshold: float = 0.5,
             min_unit_size: int = MIN_UNIT_SIZE,
             max_auc_drop_vs_global: float = 0.07,
             min_unit_auc: float = 0.55,
             max_unit_ece: float = 0.08,
             max_auc_spread: float = 0.15) -> dict[str, Any]:
    """The gate. A global pass with a failing branch is a fail.

    ``max_auc_drop_vs_global`` is the one that catches the case the headline
    hides: a branch materially worse than the book it is part of, even when its
    absolute AUC clears the floor.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred, dtype=float)

    try:
        global_auc = float(roc_auc_score(y_true, p_pred))
    except ValueError:
        global_auc = float("nan")
    global_metrics = {"auc": round(global_auc, 4),
                      "ece": round(expected_calibration_error(p_pred, y_true), 4),
                      "n": int(len(y_true))}

    per_unit = by_unit(y_true, p_pred, units, threshold=threshold, min_size=min_unit_size)
    spread = variance(per_unit)

    failing: list[dict[str, Any]] = []
    for name, metrics in per_unit.items():
        if metrics.get("verdict") == "insufficient evidence":
            continue
        reasons: list[str] = []
        auc = metrics.get("auc")
        if auc is not None and auc == auc:
            if auc < min_unit_auc:
                reasons.append(f"AUC {auc:.3f} below the {min_unit_auc} local floor")
            if global_auc == global_auc and (global_auc - auc) > max_auc_drop_vs_global:
                reasons.append(f"AUC {auc:.3f} is {global_auc - auc:.3f} below the "
                               f"global {global_auc:.3f}")
        ece = metrics.get("ece")
        if ece is not None and ece == ece and ece > max_unit_ece:
            reasons.append(f"calibration error {ece:.3f} exceeds {max_unit_ece}")
        if reasons:
            failing.append({"unit": name, "n": metrics.get("n"), "reasons": reasons})

    auc_spread = (spread.get("auc") or {}).get("spread")
    if auc_spread is not None and auc_spread > max_auc_spread:
        failing.append({"unit": "<across units>", "n": None,
                        "reasons": [f"AUC spread {auc_spread:.3f} across branches exceeds "
                                    f"{max_auc_spread}"]})

    thin = [k for k, v in per_unit.items() if v.get("verdict") == "insufficient evidence"]
    passed = not failing
    verdict = _verdict(global_metrics, passed, failing)

    return {"passed": passed, "verdict": verdict, "global": global_metrics,
            "per_unit": per_unit, "variance": spread, "failing_units": failing,
            "units_with_insufficient_evidence": thin}


def _verdict(global_metrics: dict[str, Any], passed: bool,
             failing: list[dict[str, Any]]) -> str:
    if passed:
        return f"global AUC {global_metrics['auc']} and every branch clears its floor"
    names = ", ".join(f["unit"] for f in failing[:3])
    more = f" (+{len(failing) - 3} more)" if len(failing) > 3 else ""
    return (f"global AUC {global_metrics['auc']} passes, but {len(failing)} unit(s) fail: "
            f"{names}{more}. Do not promote on the global number alone.")


def table(per_unit: dict[str, Any], sort_by: str = "auc") -> list[dict[str, Any]]:
    """Flat rows for a report or the operations screen, worst first."""
    rows = []
    for name, metrics in per_unit.items():
        if metrics.get("verdict") == "insufficient evidence":
            rows.append({"unit": name, "n": metrics["n"], "verdict": "insufficient evidence"})
            continue
        rows.append({"unit": name, "n": metrics["n"], "auc": metrics.get("auc"),
                     "ks": metrics.get("ks"), "ece": metrics.get("ece"),
                     "fpr": metrics.get("fpr"),
                     "calibration_gap": metrics.get("calibration_gap")})
    rows.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by, 0)))
    return rows
