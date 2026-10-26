"""The area manager's view of the area: disbursements against target, and how each
relationship manager prices.

**Disbursements, month to date.** A file counts when the customer accepts the
offer: in this service that is the point the loan is sanctioned at a rate and
handed to the loan system to disburse. The amount is the assessed loan amount on
the decision the offer was priced on (the requested amount until it is verified).
Each RM is set against their monthly target (``config/access.yaml``: the role's
``monthly_disbursement_target_inr``, or a per-user ``monthly_target_inr``) and
against pace - the share of the month gone.

**Pricing patterns.** The same RM's lower-rate requests (how many, how often
approved, how deep) and offers (how many, how often taken, how far below the
recommendation), so a manager sees who discounts and whether it converts.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timezone
from typing import Any

from ..governance import decision_logger


def _num(value: Any, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _curve(payload: dict[str, Any], table: list[dict[str, Any]], breakeven: float | None,
           ceiling: float | None) -> list[dict[str, Any]]:
    envelope = payload.get("envelope") or {}
    low = min(x for x in (breakeven, envelope.get("min_rate")) if x is not None) if (breakeven or envelope.get("min_rate")) else None
    high = envelope.get("max_rate")
    if low is not None and high is not None:
        try:
            wide = wide_curve(payload, round(float(low) - 0.25, 1), float(high))
            if wide:
                return wide
        except Exception:                                          # noqa: BLE001 - the table still serves
            pass
    return [{"rate": s["rate"], "acceptance": s.get("conversion_probability"),
             "expected_value_inr": s.get("expected_value_inr")} for s in table]


def loan_amount(decision_id: str | None, cache: dict[str, float | None]) -> float | None:
    if not decision_id:
        return None
    if decision_id not in cache:
        found = decision_logger.get_decision(decision_id) or {}
        snap = (found.get("payload") or {}).get("application_snapshot") or {}
        cache[decision_id] = snap.get("sanctioned_loan_amount") or snap.get("requested_loan_amount")
    return cache[decision_id]


def area(rms: list[dict[str, Any]], offers: list[dict[str, Any]], requests: list[dict[str, Any]],
         now: datetime | None = None) -> dict[str, Any]:
    """``rms``: [{username, name, target_inr}]. Offers and requests: every row on file."""
    now = now or datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    days = calendar.monthrange(now.year, now.month)[1]
    pace = now.day / days
    in_month = lambda iso: bool(iso) and str(iso)[:7] == month          # noqa: E731
    amounts: dict[str, float | None] = {}

    rows = []
    for rm in rms:
        name = rm["username"]
        mine = [o for o in offers if o.get("sent_by") == name]
        accepted = [o for o in mine if o.get("status") == "accepted" and in_month(o.get("customer_decided_at"))]
        disbursed = sum(loan_amount(o.get("decision_id"), amounts) or 0 for o in accepted)
        sent_month = [o for o in mine if in_month(o.get("sent_at")) and o.get("status") != "superseded"]
        answered = [o for o in sent_month if o.get("status") in ("accepted", "not_accepted")]
        below = [(o["recommended_rate"] - o["rate"]) * 100 for o in sent_month
                 if o.get("recommended_rate") is not None and o.get("rate") is not None]
        asks = [r for r in requests if r.get("requested_by") == name and in_month(r.get("requested_at"))]
        decided = [r for r in asks if r.get("status") in ("approved", "declined")]
        depth = [(r["recommended_rate"] - r["requested_rate"]) * 100 for r in asks
                 if r.get("recommended_rate") is not None and r.get("requested_rate") is not None]
        target = rm.get("target_inr")
        rows.append({
            "username": name, "name": rm.get("name") or name, "target_inr": target,
            "disbursed_inr": round(disbursed), "files": len(accepted),
            "pct_of_target": _num(disbursed / target) if target else None,
            "expected_by_now_inr": round(target * pace) if target else None,
            "on_pace": (disbursed >= target * pace) if target else None,
            "offers_sent": len(sent_month),
            "acceptance_rate": _num(sum(o["status"] == "accepted" for o in answered) / len(answered)) if answered else None,
            "avg_offer_below_recommendation_bps": _num(sum(below) / len(below), 1) if below else None,
            "requests": len(asks), "approved": sum(r["status"] == "approved" for r in asks),
            "declined": sum(r["status"] == "declined" for r in asks),
            "pending": sum(r["status"] == "pending" for r in asks),
            "approval_rate": _num(sum(r["status"] == "approved" for r in decided) / len(decided)) if decided else None,
            "avg_discount_asked_bps": _num(sum(depth) / len(depth), 1) if depth else None,
        })
    rows.sort(key=lambda r: -(r["disbursed_inr"] or 0))
    target = sum(r["target_inr"] or 0 for r in rows)
    disbursed = sum(r["disbursed_inr"] for r in rows)
    return {
        "month": month, "as_of": now.isoformat(timespec="seconds"), "day": now.day, "days_in_month": days,
        "pace": _num(pace),
        "area": {"target_inr": target or None, "disbursed_inr": disbursed,
                 "files": sum(r["files"] for r in rows),
                 "pct_of_target": _num(disbursed / target) if target else None,
                 "expected_by_now_inr": round(target * pace) if target else None},
        "rms": rows,
        "note": ("A file counts when the customer accepts the offer, at its assessed loan amount; "
                 "the loan system records the payout itself."),
    }


def wide_curve(payload: dict[str, Any], low: float, high: float, step: float = 0.05) -> list[dict[str, Any]]:
    """The acceptance curve from break-even to the policy ceiling. The decision's own
    scenario table only covers the range it was allowed to search (the quoted range),
    which is too narrow to show the floor and the ceiling on one picture."""
    import numpy as np
    import pandas as pd

    from ..governance import model_registry
    from ..optimization import price_optimizer

    bundle = model_registry.load_bundle()
    snap = payload.get("application_snapshot") or {}
    if bundle.conversion is None or bundle.prepayment is None or not snap:
        return []
    rates = np.round(np.arange(low, high + 1e-9, step), 2)
    frame = pd.DataFrame([snap])
    scored = price_optimizer.evaluate_scenarios(
        application=frame, rates=rates, conversion_model=bundle.conversion,
        prepayment_model=bundle.prepayment,
        expected_loss_pct=float((payload.get("risk") or {}).get("expected_loss_pct") or 0.0),
        loan_amount=float(snap.get("sanctioned_loan_amount") or snap.get("requested_loan_amount") or 0),
        tenure_years=float(snap.get("tenure_years") or snap.get("requested_tenure_years") or 20))
    return [{"rate": float(r["rate"]), "acceptance": float(r["conversion_probability"]),
             "expected_value_inr": float(r["expected_value_inr"])} for _, r in scored.iterrows()]


def build_up(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The floor, built from transparent components: what the loan costs to fund,
    run and hold capital against, plus the credit loss it is expected to carry."""
    value = (payload.get("recommendation") or {}).get("value_decomposition") \
        or (payload.get("explanation") or {}).get("value_decomposition") or {}
    parts = [("Cost of funds", value.get("less_funding_cost")),
             ("Operating cost (servicing)", value.get("less_servicing_cost")),
             ("Capital charge", value.get("less_capital_charge")),
             ("Expected credit loss", value.get("less_expected_credit_loss"))]
    if all(v is None for _, v in parts):
        return None
    rows = [{"label": label, "pct": round(abs(float(v)), 3)} for label, v in parts if v is not None]
    floor = round(sum(r["pct"] for r in rows), 3)
    return {"components": rows, "breakeven_rate": floor,
            "recommended_rate": value.get("customer_rate"),
            "margin_over_breakeven_pct": round(float(value["customer_rate"]) - floor, 3)
            if value.get("customer_rate") is not None else None}


def deep_dive(request: dict[str, Any], payload: dict[str, Any], room: dict[str, Any]) -> dict[str, Any]:
    """One file's acceptance curve and the three rates that frame the decision."""
    rec = payload.get("recommendation") or {}
    table = rec.get("scenario_table") or []
    risk = payload.get("risk") or {}
    sensitivity = payload.get("price_sensitivity") or {}
    explanation = payload.get("explanation") or {}
    return {
        "request_id": request.get("request_id"), "loan_id": request.get("loan_id"),
        "applicant_name": request.get("applicant_name"), "loan_amount": request.get("loan_amount"),
        "requested_rate": request.get("requested_rate"),
        "recommended_rate": rec.get("recommended_rate") or request.get("recommended_rate"),
        "breakeven_rate": risk.get("breakeven_rate"),
        "ceiling_rate": room.get("rate"), "ceiling_capped_by": room.get("capped_by"),
        "quoted_max": room.get("quoted_max"),
        "policy": {k: (payload.get("envelope") or {}).get(k) for k in ("min_rate", "max_rate", "card_rate")},
        "risk": {"grade": risk.get("risk_grade"), "annual_pd": risk.get("pd"),
                 "expected_loss_pct": risk.get("expected_loss_pct")},
        "curve": _curve(payload, table, risk.get("breakeven_rate"), room.get("rate")),
        "build_up": build_up(payload),
        "sensitivity": {"label": sensitivity.get("sensitivity_label"),
                        "points_lost_per_25bps": sensitivity.get("points_lost_per_25bps"),
                        "acceptance_at_recommended": sensitivity.get("conversion_at_recommended")},
        "reasons": [{"label": c.get("label"), "direction": c.get("direction"), "impact_bps": c.get("impact_bps"),
                     "value": c.get("value")} for c in (explanation.get("reason_codes") or [])[:6]],
        "headline": explanation.get("headline"),
    }
