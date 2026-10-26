"""Proof that the engine learns.

    GET  /learning/continuous   each continuous-learning claim with its evidence: the
                                decision / override / outcome loop, the monitors, the
                                learning-curve backtest, and the promotion gate
    GET  /learning/federated    a federated run computed live (not recorded), laid out
                                as evidence: before, after, centralised, personalised
    POST /learning/federated    the same, and the round is recorded with its content hash

Drift and retraining checks are run from ``POST /governance/monitors/run``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..analytics import learning_proof
from ..governance import audit
from .competitors import clean
from .security import Principal, current_principal

router = APIRouter(prefix="/learning", tags=["learning proof"])

STRATEGIES = ("fedavg", "fedprox")


class FederatedRunIn(BaseModel):
    strategy: str = Field("fedavg", description="fedavg or fedprox")


@router.get("/continuous", summary="Evidence for continuous learning")
def continuous() -> dict[str, Any]:
    return clean(learning_proof.continuous())


@router.get("/federated", summary="Evidence for federated learning (computed, not recorded)")
def federated(strategy: str = "fedavg") -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise HTTPException(422, f"strategy must be one of {', '.join(STRATEGIES)}")
    return clean(learning_proof.federated(strategy))


@router.post("/federated", summary="Run and record a federated round")
def record_federated(body: FederatedRunIn,
                     principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    if body.strategy not in STRATEGIES:
        raise HTTPException(422, f"strategy must be one of {', '.join(STRATEGIES)}")
    result = learning_proof.federated(body.strategy, record=True)
    audit.record(principal.name, "ran_federated_round", "federated", result["run_id"] or "-",
                 {"strategy": body.strategy, "final_mae_bps": result["mae_after_bps"]})
    return clean(result)
