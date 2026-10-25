"""GET /customers/{loan_id}/behaviour: how this customer is behaving, and how
customers like them behaved, from the loan book in the database.

For the relationship manager working the file. Context only: nothing here is
read by the pricing path.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..customers import behaviour
from .competitors import clean

router = APIRouter(prefix="/customers", tags=["customer behaviour"])


@router.get("/{loan_id}/behaviour", summary="One customer's behaviour, against similar customers")
def customer_behaviour(loan_id: str) -> dict[str, Any]:
    try:
        return clean(behaviour.analyse(loan_id))
    except behaviour.NotFound as exc:
        raise HTTPException(404, "price this file, or have the borrower apply, before analysing "
                                 "their behaviour") from exc
