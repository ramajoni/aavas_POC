"""Fair Practices Code duties outside the KFS.

HFC Directions para 135: when a loan is not given, the reasons are communicated in
writing. A declined file's decision records the policy's own reason; this turns it
into words an applicant can read, with the grievance route beside it.
"""

from __future__ import annotations

from typing import Any

from ..config import load
from ..governance import audit, decision_logger
from ..ingestion.schemas import Stage


class NoDecline(Exception):
    pass


def plain_reason(internal: str) -> tuple[str, str]:
    """(category, applicant-facing sentence) for a policy's no-offer reason."""
    reasons = load("fair_practices")["rejection"]["plain_reasons"]
    text = (internal or "").lower()
    if "bureau score" in text:
        key = "bureau"
    elif "risk grade" in text:
        key = "grade"
    elif "ceiling" in text:
        key = "ceiling"
    elif "afford" in text or "foir" in text:
        key = "affordability"
    else:
        key = "other"
    return key, reasons[key]


def rejection_note(loan_id: str, *, requested_by: str) -> dict[str, Any]:
    decision = decision_logger.latest_for_loan(loan_id, stage=Stage.POST_ASSESSMENT.value)
    payload = (decision or {}).get("payload") or {}
    if payload.get("decision") != "decline":
        raise NoDecline(f"{loan_id} was not declined: there is no rejection to explain")
    fp = load("fair_practices")
    internal = (payload.get("policy") or {}).get("violations") or [payload.get("envelope", {}).get("no_offer_reason")]
    reasons = [plain_reason(r) for r in internal if r]
    g = fp["grievance"]
    note = {
        "loan_id": loan_id, "decision_id": decision["decision_id"], "decided_at": decision.get("created_at"),
        "policy_version": (payload.get("policy") or {}).get("policy_version"),
        "in_writing": True,
        "reasons": [{"category": k, "text": t} for k, t in reasons],
        "what_next": ("You may ask us to review this decision, or for more detail on the reasons, through our "
                      f"grievance process: {g['officer']}. We acknowledge within {g['acknowledge_within_days']} "
                      f"days and reply in full within {g['final_response_within_days']} days; with no reply in "
                      f"{g['escalate_after_days']} days you may approach the {g['escalation_body']}."),
        "non_discrimination": " ".join(fp["non_discrimination"]["statement"].split()),
        "use_of_models": " ".join(fp["ai_disclosure"]["text"].split()),
        "basis": "HFC Directions 2025, para 135",
    }
    audit.record(requested_by, "rejection_note_issued", "application", loan_id,
                 {"decision_id": note["decision_id"], "categories": [k for k, _ in reasons]})
    return note
