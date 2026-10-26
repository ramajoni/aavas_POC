"""Offers the relationship manager sends, and lower rates the area manager approves.

Screens 04-06 of the wireframes. The limits are relative to the system
recommendation on the file as last priced:

* at or above the recommendation, up to the policy ceiling: the RM sends it;
* down to ``rm_bps`` below it: the RM sends it with a reason, recorded as a
  human override;
* lower: the RM asks the area manager, who may approve down to ``am_bps`` below
  the recommendation; lower still is the credit head's, and is refused here.

Nothing here changes the recommendation. An approved or reasoned lower rate goes
through the same override capture as before, so the policy checks (evidence for
a competitor claim, the absolute floor) still apply and the learning loop sees it.
"""

from __future__ import annotations

import json
import secrets
from typing import Any, Callable

from ..config import policy
from ..governance import audit, decision_logger, store
from ..ingestion.schemas import Stage

COMPETITOR = "competitor_lower_rate"


class OfferError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# ----------------------------------------------------------------- the file

def priced(loan_id: str) -> dict[str, Any]:
    decision = decision_logger.latest_for_loan(loan_id, stage=Stage.POST_ASSESSMENT.value)
    if decision is None:
        raise OfferError(409, "this file has not been priced yet")
    payload = dict(decision["payload"])
    payload.setdefault("decision_id", decision["decision_id"])
    return payload


def limits(payload: dict[str, Any], rm_bps: float, am_bps: float) -> dict[str, float]:
    rec = float(payload["recommendation"]["recommended_rate"])
    envelope = payload.get("envelope") or {}
    floor, ceiling = float(envelope.get("min_rate", 0)), float(envelope.get("max_rate", 100))
    # No offer above the top of the range the customer was quoted at Stage 1.
    quoted = (payload.get("indicative_band") or {}).get("max_rate")
    quoted_cap = quoted is not None and float(quoted) < ceiling and float(quoted) >= rec - 1e-9
    if quoted_cap:
        ceiling = float(quoted)
    return {"recommended": rec, "ceiling": ceiling, "floor": floor, "quoted_cap": quoted_cap,
            "reason_floor": round(max(rec - rm_bps / 100, floor), 2),
            "manager_floor": round(max(rec - am_bps / 100, floor), 2)}


def _nearest(payload: dict[str, Any], rate: float) -> dict[str, Any] | None:
    table = (payload.get("recommendation") or {}).get("scenario_table") or []
    return min(table, key=lambda s: abs(s["rate"] - rate)) if table else None


def headroom(payload: dict[str, Any], loan_id: str | None = None) -> dict[str, Any]:
    """How high a rate this customer is still likely to accept (7 in 10 similar
    customers do), capped at what may be offered: the top of the range quoted at
    Stage 1, then the policy ceiling. ``capped_by`` says which limit bound it."""
    sensitivity = payload.get("price_sensitivity") or {}
    rate = sensitivity.get("rate_for_70pct_conversion") if sensitivity.get("available", True) else None
    quoted = (payload.get("indicative_band") or {}).get("max_rate")
    if quoted is None and loan_id:
        # Decisions priced before the quoted range was recorded on them.
        pre = decision_logger.latest_for_loan(loan_id, stage=Stage.PRE_ASSESSMENT.value)
        quoted = (((pre or {}).get("payload") or {}).get("band") or {}).get("max_rate")
    ceiling = (payload.get("envelope") or {}).get("max_rate")
    rec = (payload.get("recommendation") or {}).get("recommended_rate")
    capped_by = None
    for limit, label in ((quoted, "quoted_range"), (ceiling, "policy")):
        # A quote that policy already overrode (the recommendation sits above it)
        # no longer caps anything: the offer may still go at the recommendation.
        if label == "quoted_range" and limit is not None and rec is not None and float(limit) < float(rec) - 1e-9:
            continue
        if limit is not None and (rate is None or float(limit) < float(rate)):
            rate, capped_by = float(limit), label
    return {"rate": round(float(rate), 2) if rate is not None else None, "capped_by": capped_by,
            "quoted_max": float(quoted) if quoted is not None else None}


def analysis(payload: dict[str, Any], rate: float, loan_amount: float | None,
             loan_id: str | None = None) -> dict[str, Any]:
    """What approving ``rate`` gives up and buys, for the area manager."""
    rec = float(payload["recommendation"]["recommended_rate"])
    at_rec, at_rate = _nearest(payload, rec), _nearest(payload, rate)
    life = (at_rate or {}).get("expected_life_years")
    per_year = round((rec - rate) / 100 * loan_amount) if loan_amount else None
    room = headroom(payload, loan_id or payload.get("loan_id"))
    return {
        # Interest given up: the rate reduction on the loan amount.
        "cost_per_year": per_year,
        "cost_over_life": round(per_year * life) if per_year is not None and life else None,
        "expected_life_years": round(life, 1) if life else None,
        "acceptance_at_recommended": (at_rec or {}).get("conversion_probability"),
        "acceptance_at_requested": (at_rate or {}).get("conversion_probability"),
        "headroom_rate": room["rate"], "headroom_capped_by": room["capped_by"],
        "breakeven_rate": (payload.get("risk") or {}).get("breakeven_rate"),
    }


def _loan_amount(payload: dict[str, Any]) -> float | None:
    snap = payload.get("application_snapshot") or {}
    return snap.get("sanctioned_loan_amount") or snap.get("requested_loan_amount")


def _evidenced(payload: dict[str, Any], reason: str) -> bool:
    snap = payload.get("application_snapshot") or {}
    return reason == COMPETITOR and bool(snap.get("competitor_evidence_verified"))


def _check_reason(reason: str | None) -> str:
    allowed = policy()["override"]["allowed_reasons"]
    if not reason or reason not in allowed:
        raise OfferError(422, f"give a reason for a rate below the recommendation: one of {allowed}")
    return reason


def _policy_refused(record: dict[str, Any]) -> str | None:
    check = record.get("policy_check") or {}
    if check and check.get("allowed") is False:
        return "; ".join(check.get("violations") or []) or "the rate is outside policy"
    return None


# -------------------------------------------------------------------- offers

def send_offer(loan_id: str, rate: float, *, reason: str | None, note: str | None, sent_by: str,
               rate_authority: str | None, rm_bps: float,
               capture_override: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    payload = priced(loan_id)
    lim = limits(payload, rm_bps, 0)
    rate = round(float(rate), 2)
    if rate > lim["ceiling"]:
        raise OfferError(422, f"{rate:.2f}% is above the top of the range quoted to the customer "
                              f"({lim['ceiling']:.2f}%)" if lim["quoted_cap"] else
                              f"{rate:.2f}% is above the policy ceiling of {lim['ceiling']:.2f}%")
    if rate < lim["reason_floor"]:
        raise OfferError(409, f"{rate:.2f}% is below your authority (down to {lim['reason_floor']:.2f}%): "
                              f"send it to the area manager")
    route, used_reason = "direct", None
    if rate < lim["recommended"]:
        used_reason = _check_reason(reason)
        record = capture_override(loan_id=loan_id, human_rate=rate, reason=used_reason,
                                  approved_by=f"{sent_by}:{rate_authority}" if rate_authority else sent_by,
                                  reason_detail=note, evidence_verified=_evidenced(payload, used_reason))
        refused = _policy_refused(record)
        if refused:
            raise OfferError(422, f"not sent: {refused}")
        route = "with_reason"
    return _record_offer(payload, loan_id, rate, route, used_reason, note, sent_by)


def _record_offer(payload, loan_id, rate, route, reason, note, sent_by, request_id=None) -> dict[str, Any]:
    with store.connect() as conn:
        conn.execute("UPDATE offers SET status = 'superseded' WHERE loan_id = ? AND status = 'sent'",
                     (loan_id,))
    row = {"offer_id": f"OFR-{secrets.token_hex(5).upper()}", "loan_id": loan_id,
           "decision_id": payload["decision_id"], "rate": rate,
           "recommended_rate": float(payload["recommendation"]["recommended_rate"]), "route": route,
           "reason": reason, "note": note, "request_id": request_id, "sent_by": sent_by,
           "sent_at": store.now(), "status": "sent", "customer_decided_at": None}
    store.insert("offers", row)
    audit.record(sent_by, "sent_offer", "application", loan_id,
                 {"offer_id": row["offer_id"], "rate": rate, "route": route, "reason": reason})
    return row


def offers(loan_id: str | None = None, sent_by: str | None = None) -> list[dict[str, Any]]:
    clauses, params = [], []
    if loan_id:
        clauses.append("loan_id = ?"); params.append(loan_id)
    if sent_by:
        clauses.append("sent_by = ?"); params.append(sent_by)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return store.query(f"SELECT * FROM offers{where} ORDER BY sent_at DESC", tuple(params))


def customer_decision(offer_id: str, *, accepted: bool, final_rate: float | None,
                      loss_reason: str | None, lost_to: str | None, lost_to_rate: float | None,
                      actor: str, capture_outcome: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    offer = store.fetch_one("SELECT * FROM offers WHERE offer_id = ?", (offer_id,))
    if offer is None:
        raise OfferError(404, "no such offer")
    if offer["status"] != "sent":
        raise OfferError(409, f"this offer is {offer['status']}, not awaiting the customer")
    try:
        outcome = capture_outcome(
            loan_id=offer["loan_id"], disbursed=accepted,
            sanctioned_rate=final_rate if final_rate else offer["rate"],
            customer_response="accepted" if accepted else "rejected",
            decision_id=offer["decision_id"], loss_reason=None if accepted else (loss_reason or None),
            lost_to=lost_to, lost_to_rate=lost_to_rate)
    except ValueError as exc:
        raise OfferError(422, str(exc)) from exc
    status = "accepted" if accepted else "not_accepted"
    with store.connect() as conn:
        conn.execute("UPDATE offers SET status = ?, customer_decided_at = ? WHERE offer_id = ?",
                     (status, store.now(), offer_id))
    audit.record(actor, "recorded_customer_decision", "application", offer["loan_id"],
                 {"offer_id": offer_id, "status": status})
    return {**offer, "status": status, "outcome": outcome}


# ----------------------------------------------------------- rate requests

def request_lower_rate(loan_id: str, rate: float, *, reason: str, note: str | None,
                       requested_by: str, rm_bps: float, am_bps: float) -> dict[str, Any]:
    payload = priced(loan_id)
    lim = limits(payload, rm_bps, am_bps)
    rate = round(float(rate), 2)
    if rate >= lim["recommended"]:
        raise OfferError(422, "no approval is needed at or above the recommendation: send it")
    if rate >= lim["reason_floor"]:
        raise OfferError(422, f"{rate:.2f}% is within your authority: send it yourself with a reason")
    if rate < lim["floor"]:
        raise OfferError(422, f"{rate:.2f}% is below the policy floor of {lim['floor']:.2f}%")
    used_reason = _check_reason(reason)
    if store.fetch_one("SELECT request_id FROM rate_requests WHERE loan_id = ? AND status = 'pending'",
                       (loan_id,)):
        raise OfferError(409, "this file already has a request waiting for the area manager")
    snap = payload.get("application_snapshot") or {}
    from . import applications
    submitted = applications.get(loan_id)
    name = (submitted or {}).get("applicant_name") or snap.get("name")
    amount = _loan_amount(payload)
    row = {"request_id": f"RRQ-{secrets.token_hex(5).upper()}", "loan_id": loan_id,
           "decision_id": payload["decision_id"], "applicant_name": name, "loan_amount": amount,
           "requested_rate": rate, "recommended_rate": lim["recommended"], "reason": used_reason,
           "note": note, "evidence_verified": int(_evidenced(payload, used_reason)),
           "analysis": json.dumps(analysis(payload, rate, amount)), "requested_by": requested_by,
           "requested_at": store.now(), "status": "pending", "decided_by": None, "decided_at": None,
           "approved_rate": None, "decision_reason": None, "decision_note": None}
    store.insert("rate_requests", row)
    audit.record(requested_by, "requested_lower_rate", "application", loan_id,
                 {"request_id": row["request_id"], "rate": rate, "reason": used_reason})
    return _shape(row)


def _shape(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["analysis"] = store.decode(row.get("analysis")) if isinstance(row.get("analysis"), str) else row.get("analysis")
    out["evidence_verified"] = bool(row.get("evidence_verified"))
    return out


def requests(status: str | None = None, requested_by: str | None = None,
             loan_id: str | None = None) -> list[dict[str, Any]]:
    clauses, params = [], []
    for column, value in (("status", status), ("requested_by", requested_by), ("loan_id", loan_id)):
        if value:
            clauses.append(f"{column} = ?"); params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return [_shape(r) for r in store.query(f"SELECT * FROM rate_requests{where} "
                                           f"ORDER BY requested_at DESC", tuple(params))]


def request(request_id: str) -> dict[str, Any] | None:
    row = store.fetch_one("SELECT * FROM rate_requests WHERE request_id = ?", (request_id,))
    return _shape(row) if row else None


def decide(request_id: str, *, approve: bool, rate: float | None, reason: str | None,
           note: str | None, decided_by: str, rate_authority: str | None, am_bps: float,
           capture_override: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    row = request(request_id)
    if row is None:
        raise OfferError(404, "no such request")
    if row["status"] != "pending":
        raise OfferError(409, f"this request was already {row['status']}")
    if not (note or "").strip():
        raise OfferError(422, "add a note to your decision")
    offer = None
    if approve:
        decision = decision_logger.get_decision(row["decision_id"])
        payload = dict(decision["payload"]) if decision else priced(row["loan_id"])
        payload.setdefault("decision_id", row["decision_id"])
        lim = limits(payload, 0, am_bps)
        approved = round(float(rate if rate is not None else row["requested_rate"]), 2)
        if approved >= lim["recommended"]:
            raise OfferError(422, "that is not below the recommendation: decline and keep the recommended rate")
        if approved < lim["manager_floor"]:
            raise OfferError(403, f"{approved:.2f}% is below your limit (down to {lim['manager_floor']:.2f}%): "
                                  f"it needs the credit head")
        used_reason = _check_reason(reason or row["reason"])
        record = capture_override(loan_id=row["loan_id"], human_rate=approved, reason=used_reason,
                                  approved_by=f"{decided_by}:{rate_authority}" if rate_authority else decided_by,
                                  reason_detail=note, evidence_verified=_evidenced(payload, used_reason))
        refused = _policy_refused(record)
        if refused:
            raise OfferError(422, f"cannot approve: {refused}")
        offer = _record_offer(payload, row["loan_id"], approved, "manager_approved", used_reason,
                              note, row["requested_by"], request_id=request_id)
        status, final_reason = "approved", used_reason
    else:
        approved, status, final_reason = None, "declined", reason or "hold_recommended_rate"
    with store.connect() as conn:
        conn.execute("UPDATE rate_requests SET status = ?, decided_by = ?, decided_at = ?, approved_rate = ?, "
                     "decision_reason = ?, decision_note = ? WHERE request_id = ?",
                     (status, decided_by, store.now(), approved, final_reason, note, request_id))
    audit.record(decided_by, f"{status}_rate_request", "application", row["loan_id"],
                 {"request_id": request_id, "approved_rate": approved})
    return {**request(request_id), "offer": offer}
