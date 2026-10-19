"""POST /pricing/final-assessment - the risk-adjusted recommendation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..governance import audit, decision_logger
from ..logging_utils import get_logger
from .contracts import Stage2Request
from .deps import get_orchestrator, get_policy_retriever

log = get_logger(__name__)
router = APIRouter(prefix="/pricing", tags=["stage 2 - final assessment"])


@router.post("/final-assessment", summary="Yield-optimal permissible price with reason codes")
def final_assessment(request: Stage2Request, include_scenarios: bool = Query(
        False, description="Return the full scenario table (large)"),
        orchestrator=Depends(get_orchestrator)) -> dict:
    try:
        result = orchestrator.process(request.to_row())
    except Exception as exc:                        # noqa: BLE001
        log.exception("final assessment failed for %s", request.loan_id)
        raise HTTPException(status_code=500, detail=f"final assessment failed: {exc}") from exc

    if not include_scenarios:
        result["recommendation"].pop("scenario_table", None)
    return result


@router.get("/decision/{decision_id}", summary="Retrieve a recorded decision")
def get_decision(decision_id: str) -> dict:
    decision = decision_logger.get_decision(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail=f"no decision {decision_id}")
    return decision


@router.get("/policy/explain", summary="Cite the policy clause behind a limit")
def explain_policy(question: str = Query(..., min_length=3, examples=["branch manager discount limit"]), retriever=Depends(get_policy_retriever)) -> dict:
    return {"question": question, "clauses": retriever.cite(question, k=3)}


@router.get("/audit/{loan_id}", summary="Full decision history for one file", tags=["governance"])
def loan_audit(loan_id: str) -> dict:
    history = audit.loan_history(loan_id)
    if not history["decisions"] and not history["audit_entries"]:
        raise HTTPException(status_code=404, detail=f"no history for {loan_id}")
    return history
