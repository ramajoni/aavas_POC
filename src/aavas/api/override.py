"""POST /pricing/override - capture the human decision.

The recommendation is never replaced. Both numbers, the reason, the approver and
the authority check are stored so the learning loop can later ask whether the
override created value.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..config import policy
from ..logging_utils import get_logger
from .contracts import OverrideRequest
from .deps import get_orchestrator

log = get_logger(__name__)
router = APIRouter(prefix="/pricing", tags=["human decision"])


@router.post("/override", summary="Record a human override of the recommendation")
def override(request: OverrideRequest, orchestrator=Depends(get_orchestrator)) -> dict:
    allowed = policy()["override"]["allowed_reasons"]
    if request.reason not in allowed:
        raise HTTPException(
            status_code=422,
            detail={"error": f"'{request.reason}' is not an allowed override reason",
                    "allowed_reasons": allowed})
    try:
        record = orchestrator.capture_override(
            loan_id=request.loan_id, human_rate=request.human_rate, reason=request.reason,
            approved_by=request.approved_by, reason_detail=request.reason_detail,
            evidence_verified=request.evidence_verified)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:                        # noqa: BLE001
        log.exception("override capture failed for %s", request.loan_id)
        raise HTTPException(status_code=500, detail=f"override capture failed: {exc}") from exc

    if record.get("within_authority") is False:
        record["warning"] = (
            f"this override needs {record['required_approval']} authority; "
            "it is recorded but must not be released to the customer without that approval"
        )
    return record


@router.get("/override/reasons", summary="Allowed override reasons and evidence rules")
def reasons() -> dict:
    ovr = policy()["override"]
    return {
        "allowed_reasons": ovr["allowed_reasons"],
        "evidence_required_reasons": ovr["evidence_required_reasons"],
        "max_unevidenced_discount_bps": ovr["max_unevidenced_discount_bps"],
        "approval_authority": policy()["approval_authority"],
    }
