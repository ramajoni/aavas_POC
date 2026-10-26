"""The online competitor check on one file.

    POST /competitors/check     a quoted rate and its evidence -> a score out of 100,
                                and what the lender advertises online
    GET  /competitors/lenders   the lender names a relationship manager can pick from

The search behind a check runs in the background (``intelligence/market_rates.py``).
While it does, the response says ``researching`` and scores the quote on its
evidence alone; the screen asks again until the search is done. Nothing here
moves a recommendation - the score says how far the quote can justify a discount.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..governance import audit
from ..intelligence import market_rates
from .security import Principal, current_principal

router = APIRouter(tags=["competitor check"])


class CheckRequest(BaseModel):
    loan_id: str | None = Field(None, max_length=40)
    competitor_name: str | None = Field(None, max_length=120)
    competitor_rate: float = Field(..., gt=0, lt=30)
    evidence_seen: bool = False
    bureau_score: int | None = None
    new_to_credit: bool = False
    segment: str | None = Field(None, max_length=60)
    refresh: bool = Field(False, description="search again even when recent findings exist")


@router.post("/competitors/check", summary="Score a competitor's quoted rate against the market")
def check(request: CheckRequest, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    result = market_rates.check(
        request.competitor_name, request.competitor_rate, request.evidence_seen,
        {"bureau_score": request.bureau_score, "new_to_credit": request.new_to_credit,
         "segment": request.segment},
        refresh=request.refresh, actor=principal.name)
    if request.refresh:
        audit.record(principal.name, "requested_competitor_check", "application",
                     request.loan_id or "-",
                     {"lender": result["lender"], "quote": request.competitor_rate,
                      "evidence_seen": request.evidence_seen, "state": result["state"]})
    return result


@router.get("/competitors/lenders", summary="Lender names to pick a competitor from")
def lenders() -> dict[str, Any]:
    return {"lenders": market_rates.lenders()}
