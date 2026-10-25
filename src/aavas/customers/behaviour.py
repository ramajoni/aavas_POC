"""How one customer is behaving, and how customers like them behaved.

Three parts, all read from the database:

* **signals**: what this customer has told us, from the file as last priced
  (``decisions.application_snapshot``) or, before pricing, as they applied;
* **timeline**: what they have done, from the decisions, applications, documents,
  overrides and outcomes recorded against the loan id;
* **cohort**: what similar customers in the loan book did, by how our rate sat
  against their competitor quote. The cohort matches on segment, negotiation,
  quote evidence, channel and relationship, and drops the last of those in turn
  until it holds at least ``MIN_COHORT`` loans, saying which ones it kept.

Context for the relationship manager, never a price input.
"""

from __future__ import annotations

from typing import Any

from ..config import policy
from ..governance import decision_logger, store
from ..ingestion.schemas import Stage
from ..intake import applications
from ..intelligence.behaviour import GAP_BANDS, RESPONSES
from . import loan_book

MIN_COHORT = 300       # before splitting by price gap
MIN_CELL = 30          # a band with fewer loans is suppressed

SEGMENTS = {"salaried_formal": "salaried (formal)", "salaried_informal": "salaried (informal)",
            "self_employed_professional": "self-employed professional",
            "self_employed_non_professional": "self-employed non-professional"}
CHANNELS = {"branch_walkin": "branch walk-in", "dsa": "DSA", "connector": "connector",
            "digital": "digital"}

# (column, describe(value)) in the order they are given up when the cohort is thin.
MATCH_ORDER = [
    ("segment", lambda v: SEGMENTS.get(v, str(v).replace("_", " "))),
    ("customer_negotiating", lambda v: "negotiating" if v else "not negotiating"),
    ("competitor_evidence_verified", lambda v: "documented quote" if v else "verbal quote only"),
    ("channel", lambda v: CHANNELS.get(v, str(v))),
    ("existing_relationship", lambda v: "existing customer" if v else "new to us"),
]


class NotFound(Exception):
    pass


# ------------------------------------------------------------------- the profile

def profile(loan_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The customer's fields, and the final assessment they came from (if priced)."""
    decision = decision_logger.latest_for_loan(loan_id, stage=Stage.POST_ASSESSMENT.value)
    if decision and (decision["payload"] or {}).get("application_snapshot"):
        return dict(decision["payload"]["application_snapshot"]), decision
    submitted = applications.get(loan_id)
    if submitted:
        a = dict(submitted["applicant"])
        return {**a, **(a.pop("sales_context", None) or {})}, None
    raise NotFound(loan_id)


def _flag(value: Any) -> int:
    return 1 if value in (True, 1, "1", "true", "True") else 0


def signals(p: dict[str, Any], decision: dict[str, Any] | None) -> dict[str, Any]:
    payload = (decision or {}).get("payload") or {}
    sensitivity = payload.get("price_sensitivity") or {}
    sales = payload.get("sales_context") or {}
    return {
        "segment": p.get("segment"), "segment_label": SEGMENTS.get(p.get("segment"), p.get("segment")),
        "channel": p.get("channel"), "channel_label": CHANNELS.get(p.get("channel"), p.get("channel")),
        "city": p.get("city"),
        "competitor_name": p.get("competitor_name"), "competitor_rate": p.get("competitor_rate"),
        "competitor_evidence_verified": bool(_flag(p.get("competitor_evidence_verified"))),
        "customer_negotiating": bool(_flag(p.get("customer_negotiating"))),
        "prior_negotiation_rounds": int(p.get("prior_negotiation_rounds") or 0),
        "existing_relationship": bool(_flag(p.get("existing_relationship"))),
        "days_to_month_end": p.get("days_to_month_end"),
        "sales_target_pct": p.get("sales_target_pct"),
        "recommended_rate": (payload.get("recommendation") or {}).get("recommended_rate"),
        "price_sensitivity": {k: sensitivity.get(k) for k in
                              ("sensitivity_label", "points_lost_per_25bps",
                               "conversion_at_recommended")} if sensitivity.get("available") else None,
        "max_discount_on_quote_bps": sales.get("max_discount_on_this_claim_bps"),
        "priced_at": (decision or {}).get("created_at"),
    }


# ------------------------------------------------------------------- the timeline

def timeline(loan_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for d in store.query("SELECT stage, created_at, min_rate, max_rate, recommended_rate "
                         "FROM decisions WHERE loan_id = ? ORDER BY created_at", (loan_id,)):
        if d["stage"] == Stage.PRE_ASSESSMENT.value:
            events.append({"at": d["created_at"], "kind": "indicative_offer",
                           "text": f"Saw an indicative offer of {d['min_rate']:.2f}–{d['max_rate']:.2f}%"
                           if d["min_rate"] is not None else "Requested an indicative offer"})
        elif d["stage"] == Stage.POST_ASSESSMENT.value:
            events.append({"at": d["created_at"], "kind": "final_assessment",
                           "text": f"File priced: {d['recommended_rate']:.2f}% recommended"
                           if d["recommended_rate"] is not None else "File priced"})
    submitted = applications.get(loan_id)
    if submitted:
        events.append({"at": submitted["created_at"], "kind": "applied", "text": "Applied online"})
    types = applications.document_types()
    for doc in applications.documents(loan_id):
        events.append({"at": doc["uploaded_at"], "kind": "document",
                       "text": f"Uploaded {types.get(doc['document_type'], doc['document_type']).split(' (')[0].lower()}"})
    for o in store.query("SELECT created_at, human_rate, reason FROM overrides WHERE loan_id = ? "
                         "ORDER BY created_at", (loan_id,)):
        events.append({"at": o["created_at"], "kind": "override",
                       "text": f"Rate proposed at {o['human_rate']:.2f}% ({o['reason'].replace('_', ' ')})"})
    for o in store.query("SELECT updated_at, customer_response, disbursed FROM outcomes "
                         "WHERE loan_id = ? ORDER BY updated_at", (loan_id,)):
        response = o["customer_response"] or ("took the loan" if o["disbursed"] else "outcome recorded")
        events.append({"at": o["updated_at"], "kind": "outcome", "text": f"Customer {response}"})
    # A file re-priced ten times is one event that happened ten times, not ten events.
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for e in sorted(events, key=lambda e: e["at"]):
        key = (e["kind"], e["text"])
        if key in merged:
            merged[key]["count"] += 1
            merged[key]["last_at"] = e["at"]
        else:
            merged[key] = {**e, "count": 1, "last_at": e["at"]}
    return list(merged.values())


# ----------------------------------------------------------------------- cohort

_BAND_CASE = ("CASE WHEN competitor_gap_bps <= -25 THEN 0 WHEN competitor_gap_bps <= 25 THEN 1 "
              "WHEN competitor_gap_bps <= 75 THEN 2 ELSE 3 END")


def _where(p: dict[str, Any], columns: list[str]) -> tuple[str, tuple]:
    clauses, params = [], []
    for column in columns:
        value = p.get(column)
        if column in ("customer_negotiating", "competitor_evidence_verified", "existing_relationship"):
            value = _flag(value)
        clauses.append(f"{column} = ?")
        params.append(value)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", tuple(params)


def cohort(p: dict[str, Any]) -> dict[str, Any]:
    matched = [c for c, _ in MATCH_ORDER if p.get(c) is not None]
    while True:
        where, params = _where(p, matched)
        n = loan_book.query(f"SELECT COUNT(*) AS n FROM loan_book{where}", params)[0]["n"]
        if n >= MIN_COHORT or not matched:
            break
        matched.pop()

    responses = ", ".join(f"SUM(customer_response = '{r}') AS {r}" for r in RESPONSES)
    rows = loan_book.query(
        f"SELECT {_BAND_CASE} AS band, COUNT(*) AS n, SUM(disbursed) AS won, "
        f"AVG((ai_recommended_rate - sales_offered_rate) * 100) AS avg_discount_bps, "
        f"SUM(CASE WHEN disbursed = 1 AND bt_out_24m = 1 THEN 1 ELSE 0 END) AS moved, {responses} "
        f"FROM loan_book{where} GROUP BY band", params)
    by_band = {r["band"]: r for r in rows}
    bands = []
    for i, (label, lo, hi) in enumerate(GAP_BANDS):
        r = by_band.get(i) or {"n": 0}
        cell = {"label": label, "lo_bps": None if lo == float("-inf") else lo,
                "hi_bps": None if hi == float("inf") else hi, "n": int(r["n"]),
                "suppressed": r["n"] < MIN_CELL}
        if not cell["suppressed"]:
            cell.update({
                "win_rate": round(r["won"] / r["n"], 4),
                "avg_discount_bps": round(r["avg_discount_bps"], 1),
                "moved_away_24m": round(r["moved"] / r["won"], 4) if r["won"] >= MIN_CELL else None,
                "responses": {k: round((r[k] or 0) / r["n"], 4) for k in RESPONSES},
            })
        bands.append(cell)
    describe = dict(MATCH_ORDER)
    return {"n": int(n), "matched_on": [describe[c](p.get(c) if c not in (
                "customer_negotiating", "competitor_evidence_verified", "existing_relationship")
                else _flag(p.get(c))) for c in matched],
            "relaxed": [c.replace("_", " ") for c, _ in MATCH_ORDER
                        if c not in matched and p.get(c) is not None],
            "by_gap": bands}


def _rate(where: str, params: tuple) -> dict[str, Any] | None:
    r = loan_book.query(f"SELECT COUNT(*) AS n, AVG(disbursed) AS win, "
                        f"AVG((ai_recommended_rate - sales_offered_rate) * 100) AS discount "
                        f"FROM loan_book{where}", params)[0]
    return None if r["n"] < MIN_CELL else {"n": int(r["n"]), "win_rate": round(r["win"], 4),
                                          "avg_discount_bps": round(r["discount"], 1)}


def readouts(s: dict[str, Any]) -> list[str]:
    """Plain sentences about this customer, each computed from the loan book."""
    out: list[str] = []
    segment = SEGMENTS.get(s["segment"], s["segment"])
    rounds = min(s["prior_negotiation_rounds"], 3)
    none = _rate(" WHERE segment = ? AND prior_negotiation_rounds = 0", (s["segment"],))
    same = _rate(" WHERE segment = ? AND " + ("prior_negotiation_rounds >= 3" if rounds == 3 else
                                              "prior_negotiation_rounds = ?"),
                 (s["segment"],) if rounds == 3 else (s["segment"], rounds))
    if rounds and none and same:
        stage = f"{rounds}+ rounds" if rounds == 3 else f"{rounds} round{'s' if rounds > 1 else ''}"
        out.append(f"Has negotiated {s['prior_negotiation_rounds']} round{'s' if rounds > 1 else ''}. "
                   f"{segment.capitalize()} customers after {stage} took the loan "
                   f"{same['win_rate']:.0%} of the time, against {none['win_rate']:.0%} for those who "
                   f"did not negotiate, with an average discount of {same['avg_discount_bps']:.1f} bps "
                   f"against {none['avg_discount_bps']:.1f}.")
    if s["competitor_rate"] and not s["competitor_evidence_verified"]:
        cap = s["max_discount_on_quote_bps"] or policy()["override"]["max_unevidenced_discount_bps"]
        docs = _rate(" WHERE segment = ? AND competitor_evidence_verified = 1", (s["segment"],))
        verbal = _rate(" WHERE segment = ? AND competitor_evidence_verified = 0", (s["segment"],))
        tail = (f" Similar customers with a documented quote took the loan {docs['win_rate']:.0%} "
                f"of the time; with a verbal one, {verbal['win_rate']:.0%}." if docs and verbal else "")
        out.append(f"The {s['competitor_rate']:.2f}% quote is verbal: policy allows at most {cap} bps "
                   f"of discount on it until the sanction letter is seen.{tail}")
    ps = s["price_sensitivity"]
    if ps and ps.get("points_lost_per_25bps") is not None:
        out.append(f"The conversion model rates this customer {ps['sensitivity_label']}: about "
                   f"{ps['points_lost_per_25bps']:.1f} acceptance points lost for every 25 bps added.")
    if s["days_to_month_end"] is not None and s["days_to_month_end"] <= 3:
        out.append("It is the last three days of the month. Across the book, discounts are deeper "
                   "now while the win rate barely moves: price the customer, not the target.")
    if s["existing_relationship"]:
        out.append("An existing customer: across the book they take the loan more often than "
                   "new customers at the same price.")
    return out


def analyse(loan_id: str) -> dict[str, Any]:
    p, decision = profile(loan_id)
    s = signals(p, decision)
    meta = loan_book.ensure_loaded()
    return {"loan_id": loan_id, "signals": s, "timeline": timeline(loan_id), "cohort": cohort(p),
            "readouts": readouts(s),
            "loan_book": {"rows": meta["rows"], "loaded_at": meta["loaded_at"],
                          "table": loan_book.NAME, "store": str(store.database_path().name)},
            "settings": {"min_cohort": MIN_COHORT, "min_cell": MIN_CELL}}
