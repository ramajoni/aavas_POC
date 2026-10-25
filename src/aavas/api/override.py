"""POST /pricing/override - capture the human decision.

The recommendation is never replaced. Both numbers, the reason, the approver and
the authority check are stored so the learning loop can later ask whether the
override created value.

POST /pricing/uplift - before the decision: what would this discount buy? The
pricing uplift model answers in acceptance points with an interval, and - when
expected loss and expected life are on record - in rupees, from the same yield
calculator the optimiser prices with. Advisory only: it records nothing and
approves nothing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..config import policy
from ..governance import decision_logger
from ..ingestion.schemas import Stage
from ..logging_utils import get_logger
from . import security
from .contracts import OverrideRequest, Stage2Request
from .deps import get_orchestrator
from .security import Principal, current_principal

log = get_logger(__name__)
router = APIRouter(prefix="/pricing", tags=["human decision"])


@router.post("/override", summary="Record a human override of the recommendation")
def override(request: OverrideRequest, orchestrator=Depends(get_orchestrator),
             principal: Principal = Depends(current_principal)) -> dict:
    allowed = policy()["override"]["allowed_reasons"]
    approved_by = request.approved_by
    if principal.mode == security.SESSION:
        # Signed in, the approver is who signed in, at their role's authority. A
        # name typed into the request would let anyone claim anyone's limit.
        approved_by = (f"{principal.name}:{principal.rate_authority}"
                       if principal.rate_authority else principal.name)
    if request.reason not in allowed:
        raise HTTPException(
            status_code=422,
            detail={"error": f"'{request.reason}' is not an allowed override reason",
                    "allowed_reasons": allowed})
    try:
        record = orchestrator.capture_override(
            loan_id=request.loan_id, human_rate=request.human_rate, reason=request.reason,
            approved_by=approved_by, reason_detail=request.reason_detail,
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


# ------------------------------------------------------------------- uplift


class UpliftRequest(BaseModel):
    """A proposed rate against a recommendation: a recorded decision, or a file."""

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"decision_id": "DEC-0123456789AB", "proposed_rate": 11.85}]})

    decision_id: str | None = Field(
        None, description="A recorded final-assessment decision: its application snapshot, "
                          "recommended rate and expected loss are used")
    application: Stage2Request | None = Field(
        None, description="Instead of decision_id: the assessed file")
    recommended_rate: float | None = Field(
        None, gt=0, lt=30, description="With an application: the recommendation the discount is "
                                       "measured from")
    proposed_rate: float = Field(..., gt=0, lt=30, description="The rate the RM is considering")
    curve_bps: list[float] | None = Field(
        None, description="Discounts to draw the curve at; the configured grid when omitted")


def _from_decision(decision_id: str) -> dict[str, Any]:
    decision = decision_logger.get_decision(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail=f"no decision {decision_id}")
    payload = decision.get("payload") or {}
    snapshot = payload.get("application_snapshot")
    recommendation = payload.get("recommendation") or {}
    if decision.get("stage") != Stage.POST_ASSESSMENT.value or not snapshot \
            or recommendation.get("recommended_rate") is None:
        raise HTTPException(status_code=422, detail=(
            f"{decision_id} is not a final-assessment decision with an application snapshot; "
            "the uplift is measured from a risk-adjusted recommendation"))
    return {"source": "decision", "decision_id": decision_id, "loan_id": decision.get("loan_id"),
            "frame": pd.DataFrame([snapshot]),
            "recommended_rate": float(recommendation["recommended_rate"]),
            "expected_loss_pct": (payload.get("risk") or {}).get("expected_loss_pct"),
            "expected_life_years": (recommendation.get("economics") or {}).get("expected_life_years")}


def _from_application(request: UpliftRequest, bundle) -> dict[str, Any]:
    if request.recommended_rate is None:
        raise HTTPException(status_code=422, detail=(
            "with an application payload, recommended_rate is required: the discount is measured "
            "from it (or send the decision_id of a final assessment)"))
    frame = pd.DataFrame([request.application.to_row()])
    expected_loss = None
    risk_model = getattr(bundle, "risk", None)
    if risk_model is not None:
        # The model's grade, as the pricing pipeline would assign it - not the caller's.
        assessed = risk_model.assess(frame).iloc[0]
        frame["risk_grade"] = assessed["risk_grade"]
        expected_loss = float(assessed["expected_loss_pct"])
    return {"source": "application", "decision_id": None, "loan_id": request.application.loan_id,
            "frame": frame, "recommended_rate": float(request.recommended_rate),
            "expected_loss_pct": expected_loss, "expected_life_years": None}


@router.post("/uplift", summary="What a proposed discount buys in acceptance, and what it is worth")
def uplift(request: UpliftRequest, orchestrator=Depends(get_orchestrator)) -> dict:
    bundle = orchestrator.bundle
    model = getattr(bundle, "uplift", None)
    if model is None:
        raise HTTPException(status_code=503, detail={
            "error": "the pricing uplift model is not trained, so no estimate is given",
            "how_to_fix": "python scripts/train_all.py --only uplift  "
                          "(search first: python -m aavas.models.uplift.train --search)"})
    if (request.decision_id is None) == (request.application is None):
        raise HTTPException(status_code=422, detail="send exactly one of decision_id or application")

    case = (_from_decision(request.decision_id) if request.decision_id
            else _from_application(request, bundle))
    discount = round((case["recommended_rate"] - request.proposed_rate) * 100, 1)
    if discount < 0:
        raise HTTPException(status_code=422, detail=(
            f"{request.proposed_rate}% is {-discount:g} bps above the recommendation "
            f"({case['recommended_rate']}%). The uplift model covers discounts only - it was "
            "learned from discount overrides against none, premiums excluded."))
    try:
        assessment = model.assess(
            case["frame"], discount, recommended_rate=case["recommended_rate"],
            expected_loss_pct=case["expected_loss_pct"],
            prepayment_model=getattr(bundle, "prepayment", None),
            expected_life_years=case["expected_life_years"])
        curve = model.curve(case["frame"], request.curve_bps,
                            recommended_rate=case["recommended_rate"])
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot assess this file: {exc}") from exc
    except Exception as exc:                        # noqa: BLE001
        log.exception("uplift assessment failed for %s", case["loan_id"])
        raise HTTPException(status_code=500, detail=f"uplift assessment failed: {exc}") from exc

    card = model.card
    return {
        "decision_id": case["decision_id"],
        "loan_id": case["loan_id"],
        "source": case["source"],
        **assessment,
        "curve": curve.drop(columns=["row"]).to_dict(orient="records"),
        "model": {"name": "uplift", "version": getattr(card, "version", None),
                  "learner": getattr(card, "learner", None),
                  "trained_at": getattr(card, "trained_at", None)},
        "caveat": ("An observational estimate from override history, adjusted for which files "
                   "get discounted; it is advice for the negotiation, not an approval, and "
                   "recording the override is still required."),
    }
