"""GET /pricing/explain/{decision_id} - one loan's price, explained feature by feature.

The reason codes on the pricing screen are the top few. This is the rest: SHAP
weight for every feature of the rate, risk and conversion models, the bridge
from a typical customer's rate to the recommendation, and on request an
AI-written explanation of the same figures.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..governance import decision_logger
from ..xai.loan_explanation import NotExplainable
from ..xai.narrative import NarrativeFailed, NarrativeUnavailable
from .deps import get_loan_explainer, get_narrator

router = APIRouter(prefix="/pricing", tags=["loan explainability"])


def _detail(decision_id: str, explainer) -> dict:
    decision = decision_logger.get_decision(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail=f"no decision {decision_id}")
    try:
        return explainer.explain(decision)
    except NotExplainable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/explain/{decision_id}",
            summary="Every feature's SHAP weight in the rate, risk and conversion models for one decision")
def explain_decision(decision_id: str, explainer=Depends(get_loan_explainer),
                     narrator=Depends(get_narrator)) -> dict:
    detail = _detail(decision_id, explainer)
    detail["narrative"] = narrator.status()
    return detail


@router.post("/explain/{decision_id}/narrative",
             summary="A plain-language explanation of the decision, written by Claude from its figures")
def narrate_decision(decision_id: str,
                     refresh: bool = Query(False, description="Write a new explanation instead of the cached one"),
                     explainer=Depends(get_loan_explainer), narrator=Depends(get_narrator)) -> dict:
    detail = _detail(decision_id, explainer)
    try:
        return narrator.generate(detail, refresh=refresh)
    except NarrativeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NarrativeFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
