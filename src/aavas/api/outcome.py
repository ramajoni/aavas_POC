"""POST /pricing/outcome - close the loop from decision to realized yield."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..ingestion import loader
from ..logging_utils import get_logger
from .contracts import OutcomeRequest
from .deps import get_orchestrator

log = get_logger(__name__)
router = APIRouter(prefix="/pricing", tags=["outcome and learning"])


@router.get("/outcome/loss-reasons", summary="Why a file can be recorded as lost")
def loss_reasons() -> dict:
    """A closed list, because free text cannot be aggregated.

    Registered before nothing conflicting, and kept beside the capture endpoint so
    a caller can populate a dropdown from the same service that validates it.
    """
    from ..learning.outcome_capture import LOSS_REASONS

    return {
        "reasons": [{"code": code, "label": label} for code, label in LOSS_REASONS.items()],
        "note": ("A rejected file with no reason is recorded as 'no_reason_recorded' rather "
                 "than left blank - a blank reads as 'no reason existed', which is a "
                 "different claim."),
    }


@router.post("/outcome", summary="Record what actually happened to a file")
def outcome(request: OutcomeRequest, orchestrator=Depends(get_orchestrator)) -> dict:
    try:
        return orchestrator.capture_outcome(
            loan_id=request.loan_id, disbursed=request.disbursed,
            sanctioned_rate=request.sanctioned_rate, customer_response=request.customer_response,
            bt_out=request.bt_out, delinquent=request.delinquent,
            months_to_exit=request.months_to_exit, realized_yield_pct=request.realized_yield_pct,
            decision_id=request.decision_id, loss_reason=request.loss_reason,
            lost_to=request.lost_to, lost_to_rate=request.lost_to_rate)
    except ValueError as exc:
        # A mismatched decision id or an unknown loss reason is the caller's error,
        # not a server fault, and must not read as one.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:                        # noqa: BLE001
        log.exception("outcome capture failed for %s", request.loan_id)
        raise HTTPException(status_code=500, detail=f"outcome capture failed: {exc}") from exc


@router.get("/learning/override-review", summary="Did human overrides create or destroy value?",
            tags=["outcome and learning"])
def override_review(orchestrator=Depends(get_orchestrator)) -> dict:
    try:
        df = loader.load_training_table()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return orchestrator.learning.review_overrides(df)


@router.get("/learning/questions", summary="The standing review questions, answered from data")
def learning_questions(orchestrator=Depends(get_orchestrator)) -> dict:
    try:
        df = loader.load_training_table()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"answers": orchestrator.learning.questions(df),
            "recommendation_quality": orchestrator.learning.recommendation_quality(df)}
