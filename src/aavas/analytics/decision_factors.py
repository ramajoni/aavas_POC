"""The factors an area manager weighs on a lower-rate request, each with its signal.

The decision-support screen lists these rows, and the decision memory records them
when the manager decides. Both read the signal from here, so what a manager saw and
what memory says they saw cannot drift apart. A signal is read against the RM's ask:

    for       supports giving the lower rate
    clear     nothing on this count stands in the way
    verify    a fact to check before approving
    caution   needs a second look
    against   argues against the lower rate
    neutral   information only

``case`` is the same file reduced to the few fields that make two requests alike.
It deliberately leaves out the applicant's name, age and anything else that would
let memory sort people rather than files.

Pure: it reads the request, the decision it was priced on, the file, and the
checklist and briefing the screen already fetches. Nothing here reaches a store.
"""

from __future__ import annotations

from typing import Any

SIGNALS = ("for", "clear", "verify", "caution", "against", "neutral")

# (group, key, label) in the order the screen shows them.
FACTORS: list[tuple[str, str, str]] = [
    ("need", "competitor", "Other lender’s offer"),
    ("need", "response", "Effect of the cut"),
    ("need", "negotiation", "Bargaining"),
    ("need", "reason", "What the customer cares about"),
    ("cost", "interest", "Interest the bank loses"),
    ("cost", "depth", "Size of the cut"),
    ("cost", "margin", "Profit margin left"),
    ("cost", "credit", "Credit quality"),
    ("branch", "policy", "Policy"),
    ("branch", "precedent", "Rates on similar loans"),
    ("branch", "track", "Past cuts in this branch"),
    ("branch", "month", "Cuts approved this month"),
    ("branch", "target", "RM’s monthly target"),
    ("branch", "rm", "RM’s past requests"),
    ("branch", "relationship", "Relationship"),
]
LABELS = {key: label for _, key, label in FACTORS}

BIG_LOSS_INR = 500_000          # interest given up over the loan above this argues against
GAIN_FOR_PTS = 5                # a cut must win at least this many acceptance points
MARGIN_CLEAR_PCT, MARGIN_THIN_PCT = 1.0, 0.5
STRONG_GRADES, WEAK_GRADES = ("A+", "A"), ("C", "D")


def sales(applicant: dict[str, Any]) -> dict[str, Any]:
    """The sales context, whether the file nests it or a priced snapshot carries it flat."""
    flat = {k: applicant.get(k) for k in ("competitor_name", "competitor_rate", "competitor_evidence_verified",
                                          "customer_negotiating", "prior_negotiation_rounds",
                                          "existing_relationship") if applicant.get(k) is not None}
    return {**flat, **(applicant.get("sales_context") or {})}


def _gain_pts(analysis: dict[str, Any]) -> int | None:
    at_req, at_rec = analysis.get("acceptance_at_requested"), analysis.get("acceptance_at_recommended")
    if at_req is None or at_rec is None:
        return None
    return round(float(at_req) * 100) - round(float(at_rec) * 100)


def _margin(request: dict[str, Any], payload: dict[str, Any]) -> float | None:
    analysis = request.get("analysis") or {}
    breakeven = (payload.get("risk") or {}).get("breakeven_rate")
    breakeven = breakeven if breakeven is not None else analysis.get("breakeven_rate")
    return None if breakeven is None else round(float(request["requested_rate"]) - float(breakeven), 2)


def _checks_clear(applicant: dict[str, Any]) -> bool:
    return all(applicant.get(k) in (0, "0") for k in ("legal_flag", "technical_flag"))


def factors(request: dict[str, Any], payload: dict[str, Any], applicant: dict[str, Any],
            checklist: dict[str, Any] | None, briefing: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every factor with a short value and its signal. A factor whose data is missing
    is neutral, not omitted: absence is part of what the manager saw."""
    checklist, briefing = checklist or {}, briefing or {}
    x = request.get("analysis") or {}
    sc = sales(applicant)
    ask, rec = float(request["requested_rate"]), float(request["recommended_rate"])
    evidenced = bool(request.get("evidence_verified") or sc.get("competitor_evidence_verified"))
    out: dict[str, tuple[Any, str]] = {}

    quote = sc.get("competitor_rate")
    out["competitor"] = ((sc.get("competitor_name") or "a lender", quote, evidenced),
                         ("for" if evidenced else "verify") if quote else "neutral")
    gain = _gain_pts(x)
    out["response"] = (gain, "neutral" if gain is None else "for" if gain >= GAIN_FOR_PTS else "against")
    out["negotiation"] = (bool(sc.get("customer_negotiating")),
                          "caution" if sc.get("customer_negotiating") else "neutral")
    hints = [k for k in ((checklist.get("reason_hints") or {}).get("found") or []) if k != "rate"]
    out["reason"] = (hints, "caution" if hints else "neutral")

    cost = x.get("cost_full_term") if x.get("cost_full_term") is not None else x.get("cost_over_life")
    out["interest"] = (cost, "against" if (cost or 0) > BIG_LOSS_INR else "caution")
    depth = round((rec - ask) * 100)
    deepest = (((briefing.get("overrides") or {}).get("similar_here")) or {}).get("deepest_bps")
    out["depth"] = (depth, "against" if deepest is not None and depth > deepest else "clear")
    margin = _margin(request, payload)
    out["margin"] = (margin, "neutral" if margin is None else "clear" if margin >= MARGIN_CLEAR_PCT
                     else "caution" if margin >= MARGIN_THIN_PCT else "against")
    grade = (payload.get("risk") or {}).get("risk_grade")
    out["credit"] = (grade, "against" if grade in WEAK_GRADES else "verify" if not _checks_clear(applicant)
                     else "clear" if grade in STRONG_GRADES else "caution")

    below = bool(x.get("below_policy_floor")) or (payload.get("policy") or {}).get("allowed") is False
    out["policy"] = ("outside" if below else "within", "against" if below else "clear")
    prec = checklist.get("precedent") or {}
    share = prec.get("share_at_or_below")
    out["precedent"] = (share, ("against" if share < 0.1 else "caution" if share < 0.25 else "clear")
                        if prec.get("files") and share is not None else "neutral")
    paid = next((s for s in briefing.get("signals") or [] if s.get("topic") == "did_it_pay"), None)
    out["track"] = ((paid or {}).get("verdict"),
                    {"good": "for", "bad": "against"}.get(paid.get("tone"), "caution") if paid else "neutral")
    month = checklist.get("this_month") or {}
    approved = (month.get("branch") or {}).get("approved") or 0
    out["month"] = (approved, "caution" if approved >= 3 else "clear")
    mine = month.get("rm_disbursed")
    out["target"] = ((mine or {}).get("pct_of_target"),
                     ("clear" if mine.get("on_pace") else "caution") if mine else "neutral")
    rm = ((briefing.get("performance") or {}).get("rm")) or {}
    keen = (rm.get("requests") or 0) > 3 and (rm.get("approval_rate") or 0) > 0.8
    out["rm"] = (rm.get("approval_rate"), "caution" if keen else "neutral")
    existing = "Existing customer" in (checklist.get("relationship") or [])
    out["relationship"] = (existing, "for" if existing else "neutral")

    return [{"group": group, "key": key, "label": label, "value": out[key][0], "signal": out[key][1]}
            for group, key, label in FACTORS]


def lean(rows: list[dict[str, Any]]) -> str:
    """The screen's headline reading of the balance: for, against or balanced."""
    tally = {s: sum(1 for r in rows if r["signal"] == s) for s in SIGNALS}
    if tally["against"] > tally["for"]:
        return "against"
    if tally["for"] > tally["against"] and tally["caution"] <= tally["for"]:
        return "for"
    return "balanced"


def case(request: dict[str, Any], payload: dict[str, Any], applicant: dict[str, Any]) -> dict[str, Any]:
    """The request reduced to what makes two requests alike, for recall."""
    snap = payload.get("application_snapshot") or {}
    sc = sales({**snap, **applicant})
    ask, rec = float(request["requested_rate"]), float(request["recommended_rate"])
    quote = sc.get("competitor_rate")
    margin = _margin(request, payload)
    return {
        "reason": request.get("reason"),
        "discount_bps": round((rec - ask) * 100, 1),
        "evidence_verified": bool(request.get("evidence_verified") or sc.get("competitor_evidence_verified")),
        "competitor": bool(quote),
        "gap_to_competitor_bps": round((ask - float(quote)) * 100, 1) if quote else None,
        "acceptance_gain_pts": _gain_pts(request.get("analysis") or {}),
        "margin_pct": margin,
        "risk_grade": (payload.get("risk") or {}).get("risk_grade"),
        "segment": applicant.get("segment") or snap.get("segment"),
        "city": applicant.get("city") or snap.get("city"),
    }


def minimal_case(request: dict[str, Any]) -> dict[str, Any]:
    """What can be said about a request from its own row, when the rest could not be read."""
    return {"reason": request.get("reason"),
            "discount_bps": round((float(request["recommended_rate"]) - float(request["requested_rate"])) * 100, 1),
            "evidence_verified": bool(request.get("evidence_verified")),
            "acceptance_gain_pts": _gain_pts(request.get("analysis") or {})}
