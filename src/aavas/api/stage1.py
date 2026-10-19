"""POST /pricing/pre-assessment - the early, credible band."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..logging_utils import get_logger
from .contracts import Stage1Request
from .deps import get_orchestrator

log = get_logger(__name__)
router = APIRouter(prefix="/pricing", tags=["stage 1 - pre-assessment"])


@router.post("/pre-assessment", summary="Indicative rate band and eligibility")
def pre_assessment(request: Stage1Request, orchestrator=Depends(get_orchestrator)) -> dict:
    try:
        return orchestrator.pre_assessment(request.to_row())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:                        # noqa: BLE001
        log.exception("pre-assessment failed for %s", request.loan_id)
        raise HTTPException(status_code=500, detail=f"pre-assessment failed: {exc}") from exc
