"""Offers and rate approvals (wireframes 04-06).

    POST /pricing/offers                                 RM sends an offer
    GET  /pricing/offers                                 offers (RM: their own)
    POST /pricing/offers/{offer_id}/customer-decision    RM records accepted / not accepted
    POST /pricing/rate-requests                          RM asks for a lower rate
    GET  /pricing/rate-requests                          requests (RM: their own)
    GET  /pricing/rate-requests/{request_id}             one request
    GET  /pricing/rate-requests/{request_id}/briefing    what the area manager reads first
    GET  /pricing/rate-requests/{request_id}/deep-dive   acceptance curve: floor, recommendation, ceiling
    GET  /pricing/area/disbursements                     month to date against target, by RM
    GET  /pricing/my/disbursements                       the same, for the signed-in RM
    POST /pricing/rate-requests/{request_id}/decide      area manager approves or declines

The route rules in config/access.yaml let both roles reach this prefix; which of
them may do what is decided here, from the role's limits in that file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..analytics import approval_briefing, area_performance, portfolio
from ..governance import decision_logger, model_registry
from ..intake import applications as intake
from ..intake import offers
from ..logging_utils import get_logger
from . import access
from . import competitors as competitor_api
from .deps import get_orchestrator
from .security import Principal, current_principal

router = APIRouter(prefix="/pricing", tags=["offers and approvals"])
log = get_logger(__name__)

RM, AM, ADMIN = "relationship_manager", "area_manager", "administrator"


def _limit(role: str, key: str) -> float:
    value = access.role_config(role).get(key)
    if value is None:
        raise HTTPException(500, f"config/access.yaml gives {role} no {key}")
    return float(value)


def _require(principal: Principal, *roles: str) -> None:
    if not (set(principal.roles) & set(roles)):
        raise HTTPException(403, "your role cannot do this")


def _fail(exc: offers.OfferError) -> HTTPException:
    return HTTPException(exc.status, str(exc))


def _named(row: dict[str, Any]) -> dict[str, Any]:
    """Who asked, by name: the queue is read by people, not by usernames."""
    found = access.user(row.get("requested_by") or "")
    return {**row, "requested_by_name": found.name if found else row.get("requested_by")}


def _own_only(principal: Principal) -> bool:
    return principal.has(RM) and not (principal.has(AM) or principal.has(ADMIN))


class OfferIn(BaseModel):
    loan_id: str = Field(..., min_length=1, max_length=64)
    rate: float = Field(..., gt=0, lt=30)
    reason: str | None = Field(None, max_length=64)
    note: str | None = Field(None, max_length=500)


class CustomerDecisionIn(BaseModel):
    accepted: bool
    final_rate: float | None = Field(None, gt=0, lt=30)
    loss_reason: str | None = Field(None, max_length=64)
    lost_to: str | None = Field(None, max_length=120)
    lost_to_rate: float | None = Field(None, gt=0, lt=30)


class RateRequestIn(BaseModel):
    loan_id: str = Field(..., min_length=1, max_length=64)
    rate: float = Field(..., gt=0, lt=30)
    reason: str = Field(..., min_length=1, max_length=64)
    note: str | None = Field(None, max_length=500)


class DecisionIn(BaseModel):
    approve: bool
    rate: float | None = Field(None, gt=0, lt=30)
    reason: str | None = Field(None, max_length=64)
    note: str = Field(..., min_length=1, max_length=500)


@router.post("/offers", summary="Send an offer within your authority")
def send_offer(body: OfferIn, principal: Principal = Depends(current_principal),
               orchestrator=Depends(get_orchestrator)) -> dict[str, Any]:
    _require(principal, RM, ADMIN)
    try:
        return offers.send_offer(body.loan_id, body.rate, reason=body.reason, note=body.note,
                                 sent_by=principal.name, rate_authority=principal.rate_authority,
                                 rm_bps=_limit(RM, "offer_below_recommendation_bps"),
                                 capture_override=orchestrator.capture_override)
    except offers.OfferError as exc:
        raise _fail(exc) from exc


@router.get("/offers", summary="Offers sent")
def list_offers(loan_id: str | None = Query(None),
                principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    return {"offers": offers.offers(loan_id=loan_id,
                                    sent_by=principal.name if _own_only(principal) else None)}


@router.post("/offers/{offer_id}/customer-decision", summary="Record what the customer decided")
def customer_decision(offer_id: str, body: CustomerDecisionIn,
                      principal: Principal = Depends(current_principal),
                      orchestrator=Depends(get_orchestrator)) -> dict[str, Any]:
    _require(principal, RM, ADMIN)
    try:
        return offers.customer_decision(offer_id, accepted=body.accepted, final_rate=body.final_rate,
                                        loss_reason=body.loss_reason, lost_to=body.lost_to,
                                        lost_to_rate=body.lost_to_rate, actor=principal.name,
                                        capture_outcome=orchestrator.capture_outcome)
    except offers.OfferError as exc:
        raise _fail(exc) from exc


@router.post("/rate-requests", summary="Ask the area manager for a lower rate")
def request_lower_rate(body: RateRequestIn,
                       principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require(principal, RM, ADMIN)
    try:
        return offers.request_lower_rate(body.loan_id, body.rate, reason=body.reason, note=body.note,
                                         requested_by=principal.name,
                                         rm_bps=_limit(RM, "offer_below_recommendation_bps"),
                                         am_bps=_limit(AM, "approve_below_recommendation_bps"))
    except offers.OfferError as exc:
        raise _fail(exc) from exc


@router.get("/rate-requests", summary="Rate requests (an RM sees their own)")
def list_requests(status: str | None = Query(None), loan_id: str | None = Query(None),
                  principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    rows = offers.requests(status=status, loan_id=loan_id,
                           requested_by=principal.name if _own_only(principal) else None)
    if _own_only(principal):
        for row in rows:
            row.pop("analysis", None)       # the cost and acceptance analysis is the manager's
    else:
        for row in rows:
            if row.get("status") == "pending":
                _refresh_headroom(row)
    return {"requests": [_named(r) for r in rows],
            "limits": {"relationship_manager_bps": _limit(RM, "offer_below_recommendation_bps"),
                       "area_manager_bps": _limit(AM, "approve_below_recommendation_bps")}}


@router.get("/rate-requests/{request_id}", summary="One rate request")
def get_request(request_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    row = offers.request(request_id)
    if row is None or (_own_only(principal) and row["requested_by"] != principal.name):
        raise HTTPException(404, "no such request")
    if _own_only(principal):
        row.pop("analysis", None)
    return _named(row)


def _refresh_headroom(row: dict[str, Any]) -> None:
    """Requests raised before the quoted range capped the headroom: recompute it, and
    the break-even, from the decision they were priced on."""
    decision = decision_logger.get_decision(row["decision_id"]) if row.get("decision_id") else None
    if not decision:
        return
    payload = decision.get("payload") or {}
    room = offers.headroom(payload, row["loan_id"])
    row["analysis"] = {**(row.get("analysis") or {}), "headroom_rate": room["rate"],
                       "headroom_capped_by": room["capped_by"],
                       "breakeven_rate": (payload.get("risk") or {}).get("breakeven_rate")}


@router.get("/rate-requests/{request_id}/deep-dive",
            summary="One file's acceptance curve, with the floor, recommendation and ceiling")
def deep_dive(request_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require(principal, AM, ADMIN)
    row = offers.request(request_id)
    if row is None:
        raise HTTPException(404, "no such request")
    decision = decision_logger.get_decision(row["decision_id"]) if row.get("decision_id") else None
    if not decision:
        raise HTTPException(404, "the decision this request was priced on is not on file")
    payload = decision.get("payload") or {}
    return competitor_api.clean(area_performance.deep_dive(row, payload, offers.headroom(payload, row["loan_id"])))


def _rms(only: str | None = None) -> list[dict[str, Any]]:
    default = access.role_config(RM).get("monthly_disbursement_target_inr")
    staff = (access.config().get("users") or {})
    return [{"username": u.username, "name": u.name,
             "target_inr": (staff.get(u.username) or {}).get("monthly_target_inr", default)}
            for u in access.users() if u.role == RM and (only is None or u.username == only)]


@router.get("/area/disbursements", summary="Disbursements month to date against target, by RM")
def disbursements(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require(principal, AM, ADMIN)
    return area_performance.area(_rms(), offers.offers(), offers.requests())


@router.get("/my/disbursements", summary="My disbursements month to date against target")
def my_disbursements(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require(principal, RM)
    return area_performance.area(_rms(principal.name), offers.offers(sent_by=principal.name),
                                 offers.requests(requested_by=principal.name))


@lru_cache(maxsize=1)
def _propensity(files: int) -> np.ndarray | None:
    """How likely each historical file was to be overridden: what makes the
    'paid for itself' comparison like-for-like. Scored once per process."""
    model = model_registry.load_bundle().override
    if model is None:
        return None
    return model.predict(portfolio.frame())["override_probability"].to_numpy()


@router.get("/rate-requests/{request_id}/briefing",
            summary="Branch overrides, competitors and performance, before approving")
def briefing(request_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require(principal, AM, ADMIN)
    row = offers.request(request_id)
    if row is None:
        raise HTTPException(404, "no such request")
    stored = intake.get(row["loan_id"])
    applicant = (stored or {}).get("applicant")
    if not applicant:
        try:
            applicant = offers.priced(row["loan_id"]).get("application_snapshot") or {}
        except offers.OfferError:
            applicant = {}
    history = portfolio.frame()
    try:
        analysis = competitor_api.historical()["analysis"]
    except Exception as exc:                        # noqa: BLE001 - the rest still helps
        log.warning("competitor history unavailable for the briefing: %s", exc)
        analysis = None
    out = approval_briefing.briefing(
        row, applicant, history=history, propensity=_propensity(len(history)), analysis=analysis,
        rm_requests=offers.requests(requested_by=row["requested_by"]),
        rm_offers=offers.offers(sent_by=row["requested_by"]))
    out["requested_by_name"] = _named(row)["requested_by_name"]
    return competitor_api.clean(out)


@router.post("/rate-requests/{request_id}/decide", summary="Approve or decline a lower rate")
def decide(request_id: str, body: DecisionIn, principal: Principal = Depends(current_principal),
           orchestrator=Depends(get_orchestrator)) -> dict[str, Any]:
    _require(principal, AM, ADMIN)
    try:
        return offers.decide(request_id, approve=body.approve, rate=body.rate, reason=body.reason,
                             note=body.note, decided_by=principal.name,
                             rate_authority=principal.rate_authority or "regional_manager",
                             am_bps=_limit(AM, "approve_below_recommendation_bps"),
                             capture_override=orchestrator.capture_override)
    except offers.OfferError as exc:
        raise _fail(exc) from exc
