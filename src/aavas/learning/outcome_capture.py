"""Outcome capture - closing the loop from decision to realized yield.

A pricing decision is not scored at sanction. It is scored when the loan has
disbursed, run, and either survived or transferred out. This module records the
outcome against the decision that produced it, so every later analysis compares
what the engine expected with what the book actually earned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import policy
from ..governance import audit, decision_logger, store
from ..logging_utils import get_logger
from ..optimization import yield_calculator

log = get_logger(__name__)


# Why a file was lost. A closed list, because free text cannot be aggregated and
# "rejected" tells an underwriter nothing they can act on.
LOSS_REASONS = {
    "competitor_lower_rate": "A competitor offered a lower rate and won the file.",
    "competitor_faster": "A competitor disbursed sooner; rate was not the binding factor.",
    "customer_deferred": "The customer postponed the purchase.",
    "property_fell_through": "The transaction itself did not proceed.",
    "eligibility_shortfall": "The customer did not qualify for the amount needed.",
    "documentation": "The file stalled on documentation.",
    "no_reason_recorded": "Lost, and nobody recorded why.",
}


def capture(loan_id: str, disbursed: bool, sanctioned_rate: float | None = None,
            customer_response: str | None = None, bt_out: bool = False,
            delinquent: bool = False, months_to_exit: int | None = None,
            realized_yield_pct: float | None = None, actor: str = "lms_feed",
            decision_id: str | None = None, lost_to: str | None = None,
            lost_to_rate: float | None = None, loss_reason: str | None = None,
            **extra: Any) -> dict:
    """Record what actually happened to a file, against the decision it followed.

    ``decision_id`` is the join the learning loop needs. When the caller does not
    supply one the most recent final assessment for the file is used, and the
    record says which of the two happened - a file priced six times has six
    candidate decisions, and silently picking the newest is a guess worth
    labelling as one.
    """
    if decision_id:
        decision = decision_logger.get_decision(decision_id)
        if decision is None:
            raise ValueError(f"no decision {decision_id!r} on record to attach this outcome to")
        if decision.get("loan_id") != loan_id:
            raise ValueError(
                f"decision {decision_id!r} belongs to loan {decision.get('loan_id')!r}, "
                f"not {loan_id!r}")
        decision_source = "supplied"
    else:
        decision = decision_logger.latest_for_loan(loan_id, stage="post_assessment")
        decision_id = decision.get("decision_id") if decision else None
        decision_source = "inferred_latest" if decision else "none_on_record"

    if loss_reason and loss_reason not in LOSS_REASONS:
        raise ValueError(f"unknown loss_reason {loss_reason!r}; expected one of "
                         f"{sorted(LOSS_REASONS)}")
    if disbursed is False and customer_response == "rejected" and not loss_reason:
        # Not an error - but the record says the reason is missing rather than
        # leaving a blank that reads as "no reason existed".
        loss_reason = "no_reason_recorded"

    realized_value_pct = extra.pop("realized_value_pct", None)

    if realized_value_pct is None and disbursed and sanctioned_rate is not None:
        realized_value_pct = _realized_value(
            sanctioned_rate=sanctioned_rate, months_to_exit=months_to_exit,
            delinquent=delinquent, decision=decision)

    payload = {
        "loan_id": loan_id, "decision_id": decision_id, "decision_source": decision_source,
        "disbursed": disbursed, "sanctioned_rate": sanctioned_rate,
        "customer_response": customer_response, "bt_out": bt_out, "delinquent": delinquent,
        "months_to_exit": months_to_exit, "realized_yield_pct": realized_yield_pct,
        "realized_value_pct": realized_value_pct,
        "lost_to": lost_to, "lost_to_rate": lost_to_rate, "loss_reason": loss_reason,
        "loss_reason_label": LOSS_REASONS.get(loss_reason or ""), **extra,
    }
    # One outcome per decision, not per file. Re-reporting the same decision
    # replaces it; a second decision on the same file gets its own row.
    outcome_id = f"OUT-{decision_id}" if decision_id else f"OUT-LOAN-{loan_id}"
    store.insert("outcomes", {
        "outcome_id": outcome_id, "loan_id": loan_id, "decision_id": decision_id,
        "updated_at": store.now(), "customer_response": customer_response,
        "disbursed": bool(disbursed), "sanctioned_rate": sanctioned_rate, "bt_out": bool(bt_out),
        "delinquent": bool(delinquent), "months_to_exit": months_to_exit,
        "realized_yield_pct": realized_yield_pct, "realized_value_pct": realized_value_pct,
        "lost_to": lost_to, "lost_to_rate": lost_to_rate, "loss_reason": loss_reason,
        "payload": payload,
    })
    audit.record(actor, "capture_outcome", "loan", loan_id,
                 {"decision_id": decision_id, "decision_source": decision_source,
                  "disbursed": disbursed, "bt_out": bt_out, "loss_reason": loss_reason,
                  "realized_yield_pct": realized_yield_pct})

    comparison = compare_to_decision(loan_id, payload, decision)
    payload["vs_expected"] = comparison
    return payload


def _realized_value(sanctioned_rate: float, months_to_exit: int | None,
                    delinquent: bool, decision: dict | None) -> float:
    """Lifetime value actually earned, on the same basis as the expected value."""
    econ = policy()["economics"]
    life = (months_to_exit or 96) / 12.0
    expected_loss = 0.0
    if decision:
        expected_loss = float((decision.get("payload", {}).get("risk", {}) or {}).get("expected_loss_pct", 0.0))
    result = yield_calculator.compute(
        rate=sanctioned_rate, conversion_probability=1.0,
        expected_loss_pct=expected_loss * (2.0 if delinquent else 1.0),
        expected_life_years=life, loan_amount=100.0, tenure_years=20.0)
    return round(result.lifetime_value_pct, 3)


def compare_to_decision(loan_id: str, outcome: dict, decision: dict | None = None) -> dict | None:
    """Expected versus realized, for the file and for the learning loop."""
    decision = decision or decision_logger.latest_for_loan(loan_id, stage="post_assessment")
    if not decision:
        return None

    payload = decision.get("payload") or {}
    recommendation = payload.get("recommendation", {})
    economics = recommendation.get("economics", {})
    expected_conversion = economics.get("conversion_probability")
    recommended_rate = recommendation.get("recommended_rate")
    sanctioned_rate = outcome.get("sanctioned_rate")

    comparison = {
        "decision_id": decision.get("decision_id"),
        "recommended_rate": recommended_rate,
        "sanctioned_rate": sanctioned_rate,
        "rate_delta_bps": round((sanctioned_rate - recommended_rate) * 100, 1)
        if (sanctioned_rate is not None and recommended_rate is not None) else None,
        "expected_conversion": expected_conversion,
        "converted": bool(outcome.get("disbursed")),
        "expected_value_inr": economics.get("expected_value_inr"),
    }
    if outcome.get("realized_value_pct") is not None and outcome.get("disbursed"):
        loan_amount = (payload.get("twin", {}) or {}).get("loan_amount")
        if loan_amount:
            realized_inr = outcome["realized_value_pct"] / 100.0 * loan_amount
            comparison["realized_value_inr"] = round(realized_inr, 0)
            if comparison["expected_value_inr"] is not None:
                comparison["value_surprise_inr"] = round(realized_inr - comparison["expected_value_inr"], 0)
    return comparison


def backfill_from_dataset(df: pd.DataFrame, limit: int | None = None) -> int:
    """Load historical outcomes from the lifecycle dataset into the store."""
    subset = df if limit is None else df.head(limit)
    rows = []
    for row in subset.itertuples(index=False):
        rows.append({
            "loan_id": row.loan_id, "updated_at": store.now(),
            "customer_response": row.customer_response, "disbursed": bool(row.disbursed),
            "sanctioned_rate": float(row.sanctioned_rate), "bt_out": bool(row.bt_out_24m),
            "delinquent": bool(row.delinquent_12m),
            "months_to_exit": int(row.months_to_exit),
            "realized_yield_pct": None if pd.isna(row.realized_yield_pct) else float(row.realized_yield_pct),
            "realized_value_pct": float(row.realized_value_pct),
            "payload": {"source": "historical_dataset"},
        })
    with store.connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO outcomes (loan_id, updated_at, customer_response, disbursed, "
            "sanctioned_rate, bt_out, delinquent, months_to_exit, realized_yield_pct, "
            "realized_value_pct, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [tuple(store._encode(v) for v in r.values()) for r in rows],
        )
    log.info("backfilled %s outcomes", len(rows))
    return len(rows)
