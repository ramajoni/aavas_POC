"""Where each file stands, for the relationship manager's worklist.

A file moves through one stage at a time:

    in_progress            the documents verification needs are not all in
    ready                  verified: waiting for the RM to send a final offer
    pending_rate_approval  the RM asked the area manager for a lower rate
    offer_sent             waiting for the customer
    accepted / not_accepted

The stage is read from the store - documents, rate requests, offers - and never
from what a browser remembers, so a file sits in exactly one list for everyone
who looks at it: once the RM asks for a lower rate it leaves "Ready for final
offer" for "Pending rate approval", whoever asked.

A verified file carries the engine's suggested offer: the recommended rate on its
latest final assessment and how likely the customer is to accept at that rate.
A file that changed since it was last priced is priced again first.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Callable

from ..governance import decision_logger, store
from ..ingestion.schemas import Stage
from . import applications, offers

# The same split as the browser's payload(): what a quote needs, and what the
# final assessment adds. Until a figure is verified the customer's declared one
# stands in for it - never a made-up one.
_PRE_KEYS = ("loan_id", "segment", "city", "channel", "property_type", "declared_monthly_income",
             "requested_loan_amount", "requested_tenure_years", "applicant_age",
             "declared_obligations", "estimated_property_value", "bureau_score", "new_to_credit",
             "sales_context")
_FINAL_KEYS = ("verified_monthly_income", "appraised_property_value", "sanctioned_loan_amount",
               "tenure_years", "bureau_vintage_months", "delinquency_count_24m",
               "banking_behaviour_score", "deviation_count", "legal_flag", "technical_flag")
_PROVISIONAL = {"verified_monthly_income": "declared_monthly_income",
                "appraised_property_value": "estimated_property_value",
                "sanctioned_loan_amount": "requested_loan_amount",
                "tenure_years": "requested_tenure_years"}

STAGES = ("in_progress", "ready", "pending_rate_approval", "offer_sent", "accepted", "not_accepted")


def _given(value: Any) -> bool:
    return value is not None and value != ""


def required_documents() -> list[str]:
    configured = applications.settings().get("required_for_verification")
    return list(configured) if configured else [t for t in applications.document_types() if t != "other"]


def verification(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Which required documents are in, and when the last of them arrived: the
    moment the file became complete enough to verify."""
    first: dict[str, str] = {}
    for doc in sorted(docs, key=lambda d: d["uploaded_at"]):
        first.setdefault(doc["document_type"], doc["uploaded_at"])
    need = required_documents()
    missing = [t for t in need if t not in first]
    return {"verified": not missing, "missing": missing,
            "verified_at": max(first[t] for t in need) if need and not missing else None}


def stage(loan_id: str, verified: bool, offer_rows: list[dict[str, Any]],
          request_rows: list[dict[str, Any]]) -> str:
    """``offer_rows`` and ``request_rows``: this file's, newest first."""
    live = [o for o in offer_rows if o.get("status") != "superseded"]
    if live:
        status = live[0]["status"]
        if status == "sent":
            return "offer_sent"
        if status in ("accepted", "not_accepted"):
            return status
    if any(r.get("status") == "pending" for r in request_rows):
        return "pending_rate_approval"
    return "ready" if verified else "in_progress"


def stage2_row(applicant: dict[str, Any]) -> dict[str, Any]:
    """The final-assessment request for a stored file, built as the browser builds it."""
    out: dict[str, Any] = {}
    for key in (*_PRE_KEYS, *_FINAL_KEYS):
        value = applicant.get(key)
        if not _given(value) and key in _PROVISIONAL:
            value = applicant.get(_PROVISIONAL[key])
        if key == "sales_context":
            out[key] = {k: v for k, v in (value or {}).items() if _given(v)}
        elif _given(value):
            out[key] = value
    return out


def suggested_offer(row: dict[str, Any], verified_at: str | None,
                    price: Callable[[dict[str, Any]], dict[str, Any]] | None) -> dict[str, Any] | None:
    """The engine's suggested offer for a verified file: one rate, and how likely
    the customer is to accept it. Priced again when the file changed since."""
    loan_id = row["loan_id"]
    latest = decision_logger.latest_for_loan(loan_id, stage=Stage.POST_ASSESSMENT.value)
    since = max(row.get("updated_at") or "", verified_at or "")
    stale = latest is None or latest["created_at"] < since
    payload = decision_id = priced_at = None
    if stale and price is not None:
        try:
            payload = price(stage2_row(row["applicant"] or {}))
            decision_id, priced_at = payload.get("decision_id"), store.now()
        except Exception:                                          # noqa: BLE001 - the RM opens it to price
            payload = None
    if payload is None and latest:
        payload, decision_id, priced_at = latest["payload"], latest["decision_id"], latest["created_at"]
    if not payload:
        return None
    rec = payload.get("recommendation") or {}
    snap = payload.get("application_snapshot") or {}
    applicant = row["applicant"] or {}
    return {"rate": rec.get("recommended_rate"),
            "acceptance": (payload.get("price_sensitivity") or {}).get("conversion_at_recommended"),
            "sanctioned_amount": snap.get("sanctioned_loan_amount") or snap.get("requested_loan_amount"),
            "amount_verified": _given(applicant.get("sanctioned_loan_amount")),
            "decision_id": decision_id, "priced_at": priced_at}


def annotate(summaries: list[dict[str, Any]], rows: dict[str, dict[str, Any]],
             price: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Add each file's stage, what is still due and, once verified, the suggested offer.
    ``rows``: the stored applications by loan id (with the decoded applicant)."""
    all_offers = offers.offers()
    all_requests = offers.requests()
    out = []
    for summary in summaries:
        loan_id = summary["loan_id"]
        check = verification(applications.documents(loan_id))
        mine_offers = [o for o in all_offers if o["loan_id"] == loan_id]
        mine_requests = [r for r in all_requests if r["loan_id"] == loan_id]
        where = stage(loan_id, check["verified"], mine_offers, mine_requests)
        extra = {"stage": where, "verified_at": check["verified_at"],
                 "missing_documents": check["missing"], "suggested": None}
        if where == "ready" and loan_id in rows:
            extra["suggested"] = suggested_offer(rows[loan_id], check["verified_at"], price)
        approved = next((r for r in mine_requests if r.get("status") == "approved"), None)
        declined = next((r for r in mine_requests if r.get("status") == "declined"), None)
        if where == "ready" and declined:
            extra["manager_declined"] = {"requested_rate": declined.get("requested_rate"),
                                         "decided_at": declined.get("decided_at")}
        if approved:
            extra["manager_approved_rate"] = approved.get("approved_rate")
        out.append({**summary, **extra})
    return out


# ------------------------------------------------------------- in progress

def _days_ago(iso: str | None, now: datetime) -> int | None:
    if not iso:
        return None
    return max((now - datetime.fromisoformat(iso)).days, 0)


def nudges(refs: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    rows = store.query("SELECT * FROM customer_nudges ORDER BY sent_at DESC")
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if refs is None or row["ref"] in refs:
            out.setdefault(row["ref"], []).append(row)
    return out


def missing_text(types: list[str]) -> str:
    labels = applications.document_types()
    short = applications.settings().get("document_short_names") or {}
    names = [short.get(t) or labels.get(t, t).split(" (")[0].lower() for t in types]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def in_progress(files: list[dict[str, Any]], now: datetime | None = None) -> list[dict[str, Any]]:
    """Files short of verification, plus the sample leads that stand in for the rest
    of a real pipeline. ``files``: annotated summaries in the in_progress stage."""
    now = now or datetime.now(timezone.utc)
    rows = []
    for f in files:
        last = f.get("updated_at") or f.get("created_at")
        rows.append({
            "ref": f["loan_id"], "loan_id": f["loan_id"], "sample": False,
            "name": f.get("applicant_name"), "city": f.get("city"), "segment": f.get("segment"),
            "loan_amount": f.get("loan_amount"), "stage": "documents_pending",
            "missing": f.get("missing_documents") or [], "quoted": f.get("indicative"),
            "source": f.get("source"), "created_by": f.get("created_by"),
            "customer_login": f.get("customer_login"), "documents": f.get("documents"),
            "last_activity_at": last, "days_since_activity": _days_ago(last, now)})
    for lead in applications.settings().get("sample_in_progress") or []:
        days = int(lead.get("days_since_activity") or 0)
        rows.append({**lead, "loan_id": None, "sample": True, "missing": lead.get("missing") or [],
                     "last_activity_at": None, "days_since_activity": days})
    sent = nudges([r["ref"] for r in rows])
    for row in rows:
        mine = sent.get(row["ref"]) or []
        row["detail"] = _detail(row)
        row["nudges"] = len(mine)
        row["last_nudge"] = ({"channel": mine[0]["channel"], "sent_at": mine[0]["sent_at"],
                              "sent_by": mine[0]["sent_by"]} if mine else None)
    rows.sort(key=lambda r: -(r["days_since_activity"] or 0))
    return rows


def _detail(row: dict[str, Any]) -> str:
    if row["stage"] == "documents_pending":
        return (f"Waiting for {missing_text(row['missing'])}" if row["missing"]
                else "Documents in, being verified")
    if row["stage"] == "quote_not_applied":
        q = row.get("quoted") or {}
        band = f" ({q['min_rate']:.2f}%–{q['max_rate']:.2f}%)" if q.get("min_rate") else ""
        return f"Saw the indicative range{band} and has not applied"
    if row["stage"] == "dropped_off":
        return f"Stopped at {row.get('stopped_at') or 'the application form'}"
    return ""


def nudge_message(row: dict[str, Any], rm_name: str) -> str:
    templates = applications.settings().get("nudge_messages") or {}
    q = row.get("quoted") or {}
    return (templates.get(row["stage"]) or "Hi {first}, this is {rm} from Aavas about your home loan.").format(
        first=(row.get("name") or "there").split()[0], rm=rm_name,
        missing=missing_text(row["missing"]) if row.get("missing") else "your documents",
        quoted=f"{q['min_rate']:.2f}%–{q['max_rate']:.2f}%" if q.get("min_rate") else "on our site")


def record_nudge(row: dict[str, Any], channel: str, sent_by: str, rm_name: str) -> dict[str, Any]:
    channels = applications.settings().get("nudge_channels") or ["sms"]
    if channel not in channels:
        raise applications.IntakeError(422, f"channel must be one of {channels}")
    nudge = {"nudge_id": f"NDG-{secrets.token_hex(5).upper()}", "ref": row["ref"], "channel": channel,
             "message": nudge_message(row, rm_name), "sent_by": sent_by, "sent_at": store.now()}
    store.insert("customer_nudges", nudge)
    return nudge
