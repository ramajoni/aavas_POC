"""Shared model plumbing: learner construction, persistence and model cards.

Every trained artefact is saved with the feature contract it was fitted on and a
model card recording data volumes, metrics and the config versions in force.
Nothing is loadable without that provenance, because a pricing recommendation has
to be explainable months after the fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import model_config, seed, versions
from ..features.feature_pipeline import FeatureSpec, FeatureTransformer
from ..logging_utils import get_logger
from ..paths import MODEL_DIR, resolve

log = get_logger(__name__)


# --------------------------------------------------------------------- model card


@dataclass
class ModelCard:
    name: str
    version: str
    learner: str
    target: str
    trained_at: str = ""
    n_train: int = 0
    n_test: int = 0
    features: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    config_versions: dict[str, str] = field(default_factory=dict)
    training_window: dict[str, str] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    # Identity, not just provenance: a content hash over the frame this model was
    # fitted on. See models/provenance.py.
    dataset: dict[str, Any] = field(default_factory=dict)
    # Group fairness as measured on the held-out split. None means it could not be
    # measured - which is not the same as passing, and is recorded as such.
    fairness: dict[str, Any] | None = None

    def finalize(self) -> "ModelCard":
        self.trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.config_versions = versions()
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ persistence


def fairness_report(frame: pd.DataFrame, y_true, p_pred, *,
                    group_column: str | None = None) -> dict[str, Any] | None:
    """Group fairness for a fitted classifier, for the model card.

    Returns None when the protected axis is not on the frame. A card that recorded
    "passed" in that case would be claiming a measurement nobody made.
    """
    from . import fairness as fairness_module

    column = group_column or fairness_module.DEFAULT_GROUP_COLUMN
    groups = fairness_module.groups_for(frame, column)
    if groups is None or len(groups) != len(y_true):
        log.warning("fairness not measurable on %r: group axis unavailable", column)
        return None
    report = fairness_module.evaluate(y_true, p_pred, groups)
    return {"group_column": column, **report}


def artifact_path(name: str, suffix: str = ".joblib") -> Path:
    version = model_config()["registry"]["version"]
    directory = resolve(model_config()["registry"]["dir"])
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}__{version}{suffix}"


def save(name: str, payload: dict[str, Any], card: ModelCard) -> Path:
    path = artifact_path(name)
    joblib.dump({"payload": payload, "card": card.as_dict()}, path)
    with open(artifact_path(name, ".card.json"), "w", encoding="utf-8") as fh:
        json.dump(card.as_dict(), fh, indent=2)
    log.info("saved %s -> %s", name, path.name)
    return path


def load(name: str) -> tuple[dict[str, Any], ModelCard]:
    path = artifact_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train the models first: python scripts/train_all.py")
    blob = joblib.load(path)
    return blob["payload"], ModelCard(**blob["card"])


def exists(name: str) -> bool:
    return artifact_path(name).exists()


# --------------------------------------------------------------------- learners


def hgb_classifier(params: dict, spec: FeatureSpec | None = None):
    from sklearn.ensemble import HistGradientBoostingClassifier

    kwargs = dict(params)
    monotone = spec.monotone_vector() if spec else None
    if monotone is not None:
        kwargs["monotonic_cst"] = monotone
    return HistGradientBoostingClassifier(random_state=seed(), **kwargs)


def hgb_regressor(params: dict, spec: FeatureSpec | None = None):
    from sklearn.ensemble import HistGradientBoostingRegressor

    kwargs = dict(params)
    monotone = spec.monotone_vector() if spec else None
    if monotone is not None:
        kwargs["monotonic_cst"] = monotone
    return HistGradientBoostingRegressor(random_state=seed(), **kwargs)


def quantile_regressor(alpha: float, params: dict):
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=seed(), **params)


def stage2_learner():
    """XGBoost when installed, sklearn otherwise. The POC must run either way."""
    cfg = model_config()["stage2"]
    choice = cfg.get("learner", "auto")
    if choice in ("auto", "xgboost"):
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(random_state=seed(), n_jobs=-1, **cfg["params"]), "xgboost"
        except ImportError:
            if choice == "xgboost":
                raise
            log.info("xgboost unavailable, falling back to HistGradientBoostingRegressor")
    return hgb_regressor(cfg["hist_params"]), "hist_gradient_boosting"


def calibrate(estimator, X: pd.DataFrame, y: np.ndarray, method: str | None):
    """Probabilities feed money arithmetic, so they are calibrated before use.

    Isotonic calibration is monotone, so a monotone constraint on price survives
    calibration intact.
    """
    if not method or method == "none":
        estimator.fit(X, y)
        return estimator
    from sklearn.calibration import CalibratedClassifierCV

    calibrated = CalibratedClassifierCV(estimator, method=method, cv=3)
    calibrated.fit(X, y)
    return calibrated


def base_estimator(model):
    """Reach through a calibration wrapper to the tree model, for explainers."""
    inner = getattr(model, "calibrated_classifiers_", None)
    if inner:
        est = getattr(inner[0], "estimator", None)
        if est is not None:
            return est
    return model


# ------------------------------------------------------------------ rate grids


def expand_over_rates(application: pd.DataFrame, rates: np.ndarray) -> pd.DataFrame:
    """One application replicated across N candidate rates, as a single frame.

    Scoring rate-by-rate is what made a Stage-2 decision take seconds: the tree
    inference is trivial, but each one-row call pays a full DataFrame transform
    plus sklearn input validation. Expanding first turns N scoring calls into one.
    """
    rates = np.asarray(rates, dtype=float)
    row = application.iloc[[0]]
    expanded = row.iloc[np.zeros(len(rates), dtype=int)].reset_index(drop=True)
    expanded["offered_rate"] = rates
    if "competitor_rate" in expanded.columns:
        competitor = pd.to_numeric(expanded["competitor_rate"], errors="coerce")
        expanded["competitor_gap_bps"] = ((expanded["offered_rate"] - competitor) * 100).fillna(0.0)
    return expanded


# ----------------------------------------------------------------------- metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    err_bps = np.abs(np.asarray(y_true) - np.asarray(y_pred)) * 100
    return {
        "mae_bps": float(mean_absolute_error(y_true, y_pred) * 100),
        "rmse_bps": float(np.sqrt(mean_squared_error(y_true, y_pred)) * 100),
        "r2": float(r2_score(y_true, y_pred)),
        "within_10bps_pct": float((err_bps <= 10).mean() * 100),
        "within_25bps_pct": float((err_bps <= 25).mean() * 100),
    }


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba)),
        "base_rate": float(y_true.mean()),
        "mean_predicted": float(proba.mean()),
    }
    out["calibration_gap"] = abs(out["mean_predicted"] - out["base_rate"])
    return out


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    """A price band is only useful if it actually contains the price and is
    narrow enough to be credible to a customer."""
    y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
    covered = (y_true >= lower) & (y_true <= upper)
    width_bps = (upper - lower) * 100
    return {
        "coverage_pct": float(covered.mean() * 100),
        "mean_width_bps": float(width_bps.mean()),
        "median_width_bps": float(np.median(width_bps)),
        "inverted_bands": float((upper < lower).sum()),
    }
