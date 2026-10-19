"""Persist a pricing decision with everything needed to defend it later."""

from __future__ import annotations

import uuid
from typing import Any

from ..config import versions
from ..logging_utils import get_logger
from . import agent_registry, audit, store

log = get_logger(__name__)


def new_decision_id() -> str:
    return f"DEC-{uuid.uuid4().hex[:12].upper()}"


def log_decision(loan_id: str, stage: str, payload: dict[str, Any], actor: str = "pricing_engine") -> str:
    # A manifest id stamped on a decision that is not on file would be a dangling
    # reference, so the snapshot is published before the decision that names it.
    # Idempotent and memoised per process.
    agent_registry.publish(actor="decision")

    decision_id = payload.get("decision_id") or new_decision_id()
    recommendation = payload.get("recommendation", {})
    policy_check = payload.get("policy", {})
    economics = recommendation.get("economics", {})
    band = payload.get("band") or policy_check.get("permissible_band") or {}

    store.insert("decisions", {
        "decision_id": decision_id,
        "loan_id": loan_id,
        "stage": stage,
        "created_at": store.now(),
        "recommended_rate": recommendation.get("recommended_rate") or payload.get("indicative_rate"),
        "card_rate": policy_check.get("card_rate"),
        "min_rate": band.get("min_rate"),
        "max_rate": band.get("max_rate"),
        "conversion_probability": economics.get("conversion_probability"),
        "expected_value_inr": economics.get("expected_value_inr"),
        "required_approval": policy_check.get("required_approval"),
        "policy_version": versions()["policy_version"],
        "model_versions": payload.get("model_versions", {}),
        # How long the decision took to produce. A pricing engine a branch has to
        # wait on is a pricing engine a branch stops using.
        "elapsed_ms": payload.get("elapsed_ms"),
        "payload": payload,
    })
    audit.record(actor, f"{stage}_decision", "loan", loan_id,
                 {"decision_id": decision_id,
                  "rate": recommendation.get("recommended_rate") or payload.get("indicative_rate")})
    return decision_id


def get_decision(decision_id: str) -> dict[str, Any] | None:
    row = store.fetch_one("SELECT * FROM decisions WHERE decision_id = ?", (decision_id,))
    if row:
        row["payload"] = store.decode(row.get("payload"))
        row["model_versions"] = store.decode(row.get("model_versions"))
    return row


def latest_for_loan(loan_id: str, stage: str | None = None) -> dict[str, Any] | None:
    if stage:
        row = store.fetch_one(
            "SELECT * FROM decisions WHERE loan_id = ? AND stage = ? ORDER BY created_at DESC LIMIT 1",
            (loan_id, stage))
    else:
        row = store.fetch_one(
            "SELECT * FROM decisions WHERE loan_id = ? ORDER BY created_at DESC LIMIT 1", (loan_id,))
    if row:
        row["payload"] = store.decode(row.get("payload"))
        row["model_versions"] = store.decode(row.get("model_versions"))
    return row


def recent(limit: int = 200, stage: str | None = None) -> list[dict[str, Any]]:
    if stage:
        rows = store.query(
            "SELECT * FROM decisions WHERE stage = ? ORDER BY created_at DESC LIMIT ?", (stage, limit))
    else:
        rows = store.query("SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?", (limit,))
    for row in rows:
        row["payload"] = store.decode(row.get("payload"))
    return rows
