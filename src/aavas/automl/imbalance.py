"""Imbalance handling and cost-sensitive learning.

Every method in the catalogue is here. So is the warning that goes with them:
**resampling moves the base rate, and this repository multiplies probabilities by
rupees.** A model trained on a rebalanced sample predicts the frequency of the
rebalanced world, not the real one, so `P(conversion) x lifetime value` stops
meaning anything until the model is recalibrated on an unweighted holdout.

``apply`` therefore always reports ``recalibration_required``. The search flow
reads that flag and forces a calibration step; it is not advisory.

Base rates in this book (from the shipped model cards) are 9.4% to 65.9%, none of
which is severe imbalance - so these strategies are available rather than needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

STRATEGIES = ["none", "smote", "adasyn", "random_oversample", "random_undersample",
              "class_weight", "scale_pos_weight"]


@dataclass
class Resampled:
    X: Any
    y: Any
    strategy: str
    applied: bool
    recalibration_required: bool
    fit_params: dict[str, Any]
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "applied": self.applied,
                "recalibration_required": self.recalibration_required,
                "n_rows": int(len(self.y)), "positive_rate": round(float(np.mean(self.y)), 4),
                "note": self.note}


def _imblearn(name: str):
    try:
        module = __import__("imblearn.over_sampling", fromlist=[name])
        return getattr(module, name)
    except Exception:  # noqa: BLE001
        return None


def _native_resample(X, y, *, over: bool, seed: int = 42):
    """Random over/under sampling with no dependency.

    Included so the imbalance step is never silently skipped just because
    imbalanced-learn is not installed.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max() if over else counts.min()
    idx: list[np.ndarray] = []
    for cls, count in zip(classes, counts):
        pool = np.flatnonzero(y == cls)
        if count == target:
            idx.append(pool)
        elif over:
            idx.append(rng.choice(pool, size=target, replace=True))
        else:
            idx.append(rng.choice(pool, size=target, replace=False))
    order = np.sort(np.concatenate(idx))
    X_out = X.iloc[order] if hasattr(X, "iloc") else np.asarray(X)[order]
    return X_out, y[order]


def apply(X, y, strategy: str = "none", *, seed: int = 42) -> Resampled:
    """Apply one strategy. Returns the data plus whatever fit params it implies."""
    y_arr = np.asarray(y)

    if strategy in {"none", None}:
        return Resampled(X, y, "none", False, False, {})

    if strategy == "class_weight":
        return Resampled(X, y, strategy, True, True, {"class_weight": "balanced"},
                         "Class weights distort predicted probabilities.")

    if strategy == "scale_pos_weight":
        positives = max(int(y_arr.sum()), 1)
        ratio = float((len(y_arr) - positives) / positives)
        return Resampled(X, y, strategy, True, True, {"scale_pos_weight": round(ratio, 4)},
                         "scale_pos_weight distorts predicted probabilities.")

    if strategy in {"smote", "adasyn"}:
        cls = _imblearn("SMOTE" if strategy == "smote" else "ADASYN")
        if cls is None:
            log.warning("automl: %s needs imbalanced-learn; using random oversampling", strategy)
            X_out, y_out = _native_resample(X, y_arr, over=True, seed=seed)
            return Resampled(X_out, y_out, f"{strategy}->random_oversample", True, True, {},
                             "imbalanced-learn unavailable; fell back to random oversampling.")
        sampler = cls(random_state=seed)
        X_out, y_out = sampler.fit_resample(X, y_arr)
        return Resampled(X_out, y_out, strategy, True, True, {},
                         "Synthetic minority rows change the base rate.")

    if strategy in {"random_oversample", "random_undersample"}:
        X_out, y_out = _native_resample(X, y_arr, over=strategy == "random_oversample", seed=seed)
        return Resampled(X_out, y_out, strategy, True, True, {}, "Base rate changed by resampling.")

    raise ValueError(f"unknown imbalance strategy {strategy!r}; expected one of {STRATEGIES}")


def diagnose(y) -> dict[str, Any]:
    """Is intervention even warranted? Usually, here, it is not."""
    y_arr = np.asarray(y)
    rate = float(np.mean(y_arr))
    minority = min(rate, 1 - rate)
    if minority >= 0.20:
        verdict, recommendation = "balanced", "none"
    elif minority >= 0.05:
        verdict, recommendation = "mild", "none"
    elif minority >= 0.01:
        verdict, recommendation = "moderate", "class_weight"
    else:
        verdict, recommendation = "severe", "smote"
    return {"positive_rate": round(rate, 4), "minority_share": round(minority, 4),
            "verdict": verdict, "recommended_strategy": recommendation,
            "note": ("Resampling is not indicated; calibrated probabilities are worth more "
                     "than a balanced training set." if recommendation == "none" else
                     "Recalibrate on an unweighted holdout after applying this.")}
