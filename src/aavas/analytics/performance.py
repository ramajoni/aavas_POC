"""Model, drift and operational performance for the governance dashboard.

Three different senses of "performance", kept apart on purpose because they fail
independently:

* model performance  - is the model still accurate and calibrated?
* drift              - is the world still the one it was fitted on?
* operational        - is the engine being used, how fast, and how often is it
                       overridden or blocked by policy?

A model can be accurate and still be wrong for the market, and it can be right
about everything while nobody in a branch actually uses it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ..config import model_config, versions
from ..governance import model_registry, store
from ..learning import drift_detection
from ..logging_utils import get_logger
from .portfolio import frame

log = get_logger(__name__)

# The one or two numbers that decide whether each model is healthy, and the
# direction that counts as good. Everything else is on the model card.
HEADLINE_METRICS = {
    "stage1": [("coverage_pct", "Band coverage", "%", "higher", 75.0),
               ("mean_width_bps", "Mean band width", "bps", "lower", 100.0),
               ("median_mae_bps", "Anchor error", "bps", "lower", 35.0)],
    "stage2": [("mae_bps", "Pricing error", "bps", "lower", 20.0),
               ("within_25bps_pct", "Within 25 bps", "%", "higher", 80.0),
               ("r2", "R squared", "", "higher", 0.90)],
    "conversion": [("auc", "AUC", "", "higher", 0.65),
                   ("calibration_gap", "Calibration gap", "", "lower", 0.02),
                   ("s2d_drop_per_25bps_pts", "Conversion cost of +25 bps", "pts", "", None)],
    "prepayment": [("auc", "AUC", "", "higher", 0.60),
                   ("base_rate", "Observed BT-out rate", "", "", None)],
    "risk": [("auc", "AUC", "", "higher", 0.65),
             ("base_rate", "Observed default rate", "", "", None)],
    "override": [("auc", "AUC", "", "higher", 0.60),
                 ("magnitude_mae_bps", "Override size error", "bps", "lower", 15.0)],
}


def _safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _status(value: float | None, direction: str, threshold: float | None) -> str:
    """Traffic light. 'unknown' when there is no defensible threshold - better
    than inventing one and colouring a number green."""
    if value is None or threshold is None or not direction:
        return "unknown"
    if direction == "higher":
        return "ok" if value >= threshold else "warn"
    return "ok" if value <= threshold else "warn"


# ------------------------------------------------------------ model performance


def model_performance() -> dict:
    """Per-model headline metrics, provenance and - prominently - limitations."""
    bundle = model_registry.load_bundle()
    cards = bundle.cards()

    models = []
    for name, card in cards.items():
        metrics = card.get("metrics", {})
        headline = []
        for key, label, unit, direction, threshold in HEADLINE_METRICS.get(name, []):
            value = _safe(metrics.get(key))
            headline.append({
                "key": key, "label": label, "value": value, "unit": unit,
                "direction": direction, "threshold": threshold,
                "status": _status(value, direction, threshold),
            })
        models.append({
            "name": name,
            "learner": card.get("learner"),
            "version": card.get("version"),
            "target": card.get("target"),
            "trained_at": card.get("trained_at"),
            "n_train": card.get("n_train"),
            "n_test": card.get("n_test"),
            "training_window": card.get("training_window", {}),
            "headline": headline,
            "all_metrics": {k: _safe(v) for k, v in metrics.items()},
            "limitations": card.get("limitations", []),
            "feature_count": len(card.get("features", [])),
            "status": "warn" if any(h["status"] == "warn" for h in headline) else "ok",
        })

    models.sort(key=lambda m: m["name"])
    return {
        "models": models,
        "missing": bundle.missing or [],
        "versions": versions(),
        "registry": [
            {"name": row["name"], "version": row["version"], "role": row["role"],
             "registered_at": row["registered_at"]}
            for row in model_registry.registered()
        ],
        "limitation_count": sum(len(m["limitations"]) for m in models),
        "note": (
            "Every model here is fitted on synthetic data. These metrics demonstrate that the "
            "pipeline works; they are not expected production performance. The limitations listed "
            "against each model are part of its output, not a footnote."
        ),
    }


# --------------------------------------------------------------------- drift


@lru_cache(maxsize=1)
def drift(split: float = 0.75) -> dict:
    """Input drift and, more dangerously, conversion drift.

    The reference window is the older part of the book; the current window is the
    most recent. Input drift says the mix of files changed. Conversion drift says
    the same file converts differently now - usually because a competitor moved -
    and no input-distribution monitor would catch it.
    """
    df = frame().sort_values("application_date")
    cut = int(len(df) * split)
    reference, current = df.iloc[:cut], df.iloc[cut:]

    features = ["bureau_score", "ltv", "foir", "sanctioned_loan_amount", "verified_monthly_income",
                "competitor_rate", "competitor_gap_bps", "sales_target_pct", "banking_behaviour_score",
                "deviation_count", "tenure_years", "applicant_age"]
    features = [f for f in features if f in df.columns]

    table = drift_detection.feature_drift(reference, current, features)
    rows = [{"feature": r["feature"], "psi": _safe(r["psi"]), "status": r["status"]}
            for r in table.to_dict(orient="records")]

    conversion = drift_detection.conversion_drift(reference, current)
    cfg = model_config()["drift"]

    worst = next((r for r in rows if r["status"] == "alert"),
                 next((r for r in rows if r["status"] == "warn"), None))
    return {
        "reference_window": {"from": str(reference["application_date"].min().date()),
                             "to": str(reference["application_date"].max().date()),
                             "n": int(len(reference))},
        "current_window": {"from": str(current["application_date"].min().date()),
                           "to": str(current["application_date"].max().date()),
                           "n": int(len(current))},
        "input_drift": rows,
        "conversion_drift": conversion,
        "thresholds": {"psi_warn": cfg["psi_warn"], "psi_alert": cfg["psi_alert"]},
        "overall_status": (
            "alert" if conversion.get("status") == "alert" or any(r["status"] == "alert" for r in rows)
            else "warn" if conversion.get("status") == "warn" or any(r["status"] == "warn" for r in rows)
            else "ok"
        ),
        "worst_feature": worst,
        "note": (
            "Input drift (PSI) says the mix of applications changed. Conversion drift says the same "
            "application converts differently than it used to - that is the one that quietly "
            "invalidates the pricing curve while every input distribution still looks normal."
        ),
    }


def clear_cache() -> None:
    drift.cache_clear()


# --------------------------------------------------------------- operational


def operations(limit: int = 500) -> dict:
    """How the engine is actually being used, from the decision and override tables."""
    decisions = store.query(
        "SELECT decision_id, loan_id, stage, created_at, recommended_rate, card_rate, "
        "conversion_probability, expected_value_inr, required_approval, policy_version, "
        "elapsed_ms FROM decisions ORDER BY created_at DESC LIMIT ?", (limit,))
    overrides = store.query("SELECT * FROM overrides ORDER BY created_at DESC LIMIT ?", (limit,))
    outcomes = store.query("SELECT * FROM outcomes ORDER BY updated_at DESC LIMIT ?", (limit,))

    latencies = [d["elapsed_ms"] for d in decisions if d.get("elapsed_ms") is not None]
    approvals: dict[str, int] = {}
    for decision in decisions:
        level = decision.get("required_approval")
        if level:
            approvals[level] = approvals.get(level, 0) + 1

    reasons: dict[str, int] = {}
    outside_authority = 0
    for override in overrides:
        reasons[override["reason"]] = reasons.get(override["reason"], 0) + 1
        if override.get("within_authority") == 0:
            outside_authority += 1

    by_stage: dict[str, int] = {}
    for decision in decisions:
        by_stage[decision["stage"]] = by_stage.get(decision["stage"], 0) + 1

    return {
        "decisions_total": store.count("decisions"),
        "overrides_total": store.count("overrides"),
        "outcomes_total": store.count("outcomes"),
        "audit_entries": store.count("audit_log"),
        "decisions_by_stage": by_stage,
        "latency_ms": {
            "n": len(latencies),
            "p50": _safe(np.percentile(latencies, 50)) if latencies else None,
            "p95": _safe(np.percentile(latencies, 95)) if latencies else None,
            "max": _safe(max(latencies)) if latencies else None,
        },
        "approval_mix": approvals,
        "override_reason_mix": reasons,
        "overrides_outside_authority": outside_authority,
        "override_rate_on_recorded_decisions": _safe(
            len(overrides) / len(decisions)) if decisions else None,
        "recent_decisions": [
            {k: _safe(v) for k, v in decision.items()} for decision in decisions[:40]
        ],
        "policy_versions_in_use": sorted({d["policy_version"] for d in decisions if d.get("policy_version")}),
        "note": (
            "Counts come from the governance store, so they reflect decisions this deployment "
            "actually made - not the historical training data."
        ),
    }


# ------------------------------------------------------- champion / challenger


def governance_status() -> dict:
    """What is in production, under what policy, and what is waiting to be promoted."""
    registered = model_registry.registered()
    champions = [r for r in registered if r["role"] == "champion"]
    challengers = [r for r in registered if r["role"] == "challenger"]
    cfg = model_config()["champion_challenger"]

    return {
        "versions": versions(),
        "champions": [{"name": r["name"], "version": r["version"], "registered_at": r["registered_at"]}
                      for r in champions],
        "challengers": [{"name": r["name"], "version": r["version"], "registered_at": r["registered_at"]}
                        for r in challengers],
        "promotion_gates": {
            "shadow_min_decisions": cfg["shadow_min_decisions"],
            "promote_if_better_by_pct": cfg["promote_if_better_by_pct"],
            "max_conversion_drop_pct": cfg["max_conversion_drop_pct"],
            "require_human_signoff": cfg["require_human_signoff"],
        },
        "note": (
            "A challenger is promoted on replayed risk-adjusted value, not on AUC, and never "
            "without human sign-off."
        ),
    }
