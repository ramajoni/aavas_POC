"""Controlled retraining.

Never retrain on every new loan. Retraining is triggered by evidence - drift,
volume, or elapsed time - runs into a challenger slot, and only reaches
production through the champion/challenger gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..config import model_config
from ..governance import audit, store
from ..logging_utils import get_logger
from . import drift_detection

log = get_logger(__name__)

MODEL_TRAINERS = {
    "stage1": "aavas.models.stage1.train",
    "stage2": "aavas.models.stage2.train",
    "conversion": "aavas.models.conversion.train",
    "prepayment": "aavas.models.prepayment.train",
    "risk": "aavas.models.risk.train",
    "override": "aavas.models.override.train",
}


@dataclass
class RetrainingDecision:
    should_retrain: bool
    triggers: list[str]
    detail: dict

    def as_dict(self) -> dict:
        return {"should_retrain": self.should_retrain, "triggers": self.triggers, "detail": self.detail}


def assess(reference: pd.DataFrame, current: pd.DataFrame,
           last_trained: datetime | None = None, max_age_days: int = 90) -> RetrainingDecision:
    cfg = model_config()["drift"]
    triggers: list[str] = []
    detail: dict = {}

    drift_table = drift_detection.feature_drift(reference, current)
    alerts = drift_table[drift_table["status"] == "alert"]["feature"].tolist()
    detail["input_drift_alerts"] = alerts
    if alerts:
        triggers.append(f"input drift on {len(alerts)} feature(s): {', '.join(alerts[:4])}")

    conversion = drift_detection.conversion_drift(reference, current)
    detail["conversion_drift"] = conversion
    if conversion.get("status") == "alert":
        triggers.append(f"conversion curve shifted by up to {conversion['max_shift_pts']:.1f} pts")

    new_decisions = store.count("decisions")
    detail["decisions_since_training"] = new_decisions
    if new_decisions >= cfg["performance_window"]:
        triggers.append(f"{new_decisions} decisions accumulated since the last training run")

    if last_trained is not None:
        age = datetime.now(timezone.utc) - last_trained
        detail["model_age_days"] = age.days
        if age > timedelta(days=max_age_days):
            triggers.append(f"champion is {age.days} days old")

    return RetrainingDecision(bool(triggers), triggers, detail)


def run(models: list[str] | None = None, actor: str = "retraining_job") -> dict:
    """Retrain the named models into the current registry version."""
    import importlib

    from ..governance import model_registry

    results: dict[str, dict] = {}
    for name in (models or list(MODEL_TRAINERS)):
        module_path = MODEL_TRAINERS.get(name)
        if module_path is None:
            log.warning("no trainer registered for %s", name)
            continue
        module = importlib.import_module(module_path)
        card = module.train()
        model_registry.register(card, role="challenger")
        results[name] = {"version": card.version, "metrics": card.metrics}
        audit.record(actor, "retrain_model", "model", f"{name}:{card.version}",
                     {"metrics": card.metrics})
    model_registry.clear_cache()
    return results
