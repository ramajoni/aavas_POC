"""Calibration - Platt, isotonic and beta.

Probabilities here get multiplied by rupees, so calibration is not a finishing
touch. Two properties matter:

* **Monotonicity must survive.** Isotonic and beta are both monotone, so a model
  constrained non-increasing in price stays non-increasing after calibration.
  Platt is monotone too. Nothing in this module can bend a demand curve.
* **Thin positive classes.** Isotonic is non-parametric and produces a step
  function; on a small validation fold it overfits. Beta calibration is smooth,
  parametric and three-parameter - the better choice for `prepayment`, whose
  positive class is 15.3%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

METHODS = ["none", "platt", "isotonic", "beta"]


class BetaCalibrator:
    """Kull, Silva Filho & Flach beta calibration.

    Fits a logistic regression on [log p, -log(1-p)], which gives a smooth
    monotone map with more shape freedom than Platt and far less variance than
    isotonic on small samples.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def _design(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), self.eps, 1 - self.eps)
        return np.column_stack([np.log(p), -np.log(1 - p)])

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression

        self.model_ = LogisticRegression(max_iter=1000)
        self.model_.fit(self._design(p), np.asarray(y).astype(int))
        return self

    def predict(self, p) -> np.ndarray:
        return self.model_.predict_proba(self._design(p))[:, 1]


@dataclass
class Calibrated:
    method: str
    predict: Any                 # callable: raw probability -> calibrated probability
    metrics: dict[str, float]

    def __call__(self, p):
        return self.predict(p)


def expected_calibration_error(p, y, bins: int = 10) -> float:
    """Mean |confidence - accuracy| across equal-width bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        error += (mask.sum() / len(p)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(error)


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y).astype(int)) ** 2))


def fit(p_raw, y, method: str = "isotonic") -> Calibrated:
    """Fit one calibrator on held-out predictions."""
    p_raw = np.asarray(p_raw, dtype=float)
    y = np.asarray(y).astype(int)

    if method in {"none", None}:
        mapper = lambda p: np.asarray(p, dtype=float)  # noqa: E731
    elif method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw, y)
        mapper = iso.predict
    elif method == "platt":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=1000).fit(p_raw.reshape(-1, 1), y)
        mapper = lambda p: lr.predict_proba(np.asarray(p, dtype=float).reshape(-1, 1))[:, 1]  # noqa: E731
    elif method == "beta":
        beta = BetaCalibrator().fit(p_raw, y)
        mapper = beta.predict
    else:
        raise ValueError(f"unknown calibration method {method!r}; expected one of {METHODS}")

    after = mapper(p_raw)
    return Calibrated(method=method, predict=mapper, metrics={
        "ece_before": round(expected_calibration_error(p_raw, y), 5),
        "ece_after": round(expected_calibration_error(after, y), 5),
        "brier_before": round(brier(p_raw, y), 5),
        "brier_after": round(brier(after, y), 5),
    })


def is_monotone(calibrated: Calibrated, points: int = 200) -> bool:
    """A calibrator that is not monotone would break the price constraint."""
    grid = np.linspace(0.001, 0.999, points)
    mapped = np.asarray(calibrated.predict(grid), dtype=float)
    return bool(np.all(np.diff(mapped) >= -1e-9))


def best(p_raw, y, methods: list[str] | None = None) -> Calibrated:
    """Pick the calibrator with the lowest ECE, monotonicity enforced."""
    out: list[Calibrated] = []
    for method in (methods or ["platt", "isotonic", "beta"]):
        try:
            fitted = fit(p_raw, y, method)
        except Exception as exc:  # noqa: BLE001
            log.warning("automl: calibration %r failed (%s)", method, exc)
            continue
        if not is_monotone(fitted):
            log.warning("automl: calibration %r is not monotone; rejected", method)
            continue
        out.append(fitted)
    if not out:
        return fit(p_raw, y, "none")
    return min(out, key=lambda c: c.metrics["ece_after"])
