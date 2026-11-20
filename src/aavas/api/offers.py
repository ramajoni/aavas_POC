"""Offers and rate approvals (wireframes 04-06).

    POST /pricing/offers                                 RM sends an offer
    GET  /pricing/offers                                 offers (RM: their own)
    POST /pricing/offers/{offer_id}/customer-decision    RM records accepted / not accepted
    POST /pricing/rate-requests                          RM asks for a lower rate
    GET  /pricing/rate-requests                          requests (RM: their own)
    GET  /pricing/rate-requests/{request_id}             one request
    GET  /pricing/rate-requests/{request_id}/briefing    what the area manager reads first
    GET  /pricing/rate-requests/{request_id}/deep-dive   acceptance curve: floor, recommendation, ceiling
    GET  /pricing/rate-requests/{request_id}/scenarios   each rate: the model's prediction and this branch's
    GET  /pricing/rate-requests/{request_id}/history     the branch's upselling, competitors and sales notes
    GET  /pricing/rate-requests/{request_id}/precedents  the past negotiations most like this file's conversation
    GET  /pricing/rate-requests/{request_id}/suggestion  a suggested decision and the facts behind it
    GET  /pricing/rate-requests/{request_id}/checklist   the manager's checks: alternatives, precedent, this month
    GET  /pricing/area/disbursements                     month to date against target, by RM
    GET  /pricing/my/disbursements                       the same, for the signed-in RM
    POST /pricing/rate-requests/{request_id}/decide      area manager approves or declines

The route rules in config/access.yaml let both roles reach this prefix; which of
them may do what is decided here, from the role's limits in that file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..analytics import (approval_briefing, area_performance, branch_history, branch_scenarios,
                         decision_support, manager_checks, portfolio, precedents)
from ..agents.manager_memory_agent import ManagerMemoryAgent
from ..governance import decision_logger, model_registry
from ..intake import applications as intake
from ..intake import offers
from ..logging_utils import get_logger
from . import access
from . import competitors as competitor_api
from .deps import get_note_retriever, get_orchestrator
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


def _where(row: dict[str, Any]) -> dict[str, Any]:
    """The file's branch city and segment, so a manager can filter the queue by them:
    from the stored application, or the snapshot the request was priced on."""
    found = intake.get(row["loan_id"]) if row.get("loan_id") else None
    applicant = (found or {}).get("applicant") or {}
    if not applicant.get("city") and row.get("decision_id"):
        decision = decision_logger.get_decision(row["decision_id"]) or {}
        applicant = (decision.get("payload") or {}).get("application_snapshot") or {}
    return {"city": applicant.get("city"), "segment": applicant.get("segment")}


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
    # Decline the discount and price above the recommendation instead.
    uplift: bool = False
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


@router.get("/offers/acceptance", summary="How likely the customer is to accept at the rates given")
def offer_acceptance(loan_id: str = Query(..., min_length=1, max_length=64),
                     rates: str = Query(..., description="Comma-separated rates in per cent, up to 10"),
                     principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """The final acceptance rate the RM offers at: one number per rate, scored on the
    file as last priced, the same way every other screen scores it."""
    _require(principal, RM, AM, ADMIN)
    try:
        values = [round(float(r), 2) for r in rates.split(",") if r.strip()][:10]
    except ValueError as exc:
        raise HTTPException(422, "rates must be numbers, separated by commas") from exc
    if not values or any(not 0 < r < 30 for r in values):
        raise HTTPException(422, "give one or more rates between 0 and 30")
    try:
        payload = offers.priced(loan_id)
    except offers.OfferError as exc:
        raise _fail(exc) from exc
    scored = offers.acceptance_at(payload, values)
    if scored is None:
        raise HTTPException(503, "acceptance could not be scored for this file")
    return {"loan_id": loan_id, "decision_id": payload["decision_id"],
            "recommended_rate": payload["recommendation"]["recommended_rate"],
            "acceptance": [{"rate": r, "acceptance": a} for r, a in zip(values, scored)]}


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
            row.update(_where(row))
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
    elif row.get("status") == "pending":
        # The same view as the list: a review reopened after a refresh reads it from here.
        _refresh_headroom(row)
        row.update(_where(row))
    return _named(row)


def _indicative(payload: dict[str, Any], loan_id: str) -> dict[str, Any] | None:
    """The range quoted at application: recorded on the decision, or for files priced
    before it was, the file's latest Stage 1 quote."""
    band = payload.get("indicative_band") or {}
    if band.get("min_rate") is None:
        pre = decision_logger.latest_for_loan(loan_id, stage="pre_assessment")
        if not pre:
            return None
        band = {"min_rate": pre.get("min_rate"), "max_rate": pre.get("max_rate"),
                "decision_id": pre["decision_id"], "quoted_at": pre.get("created_at")}
    return {k: band.get(k) for k in ("min_rate", "max_rate", "decision_id", "quoted_at")}


def _refresh_headroom(row: dict[str, Any]) -> None:
    """Requests raised before the quoted range capped the headroom: recompute it, the
    break-even and the full-term cost, from the decision they were priced on."""
    decision = decision_logger.get_decision(row["decision_id"]) if row.get("decision_id") else None
    if not decision:
        return
    payload = decision.get("payload") or {}
    room = offers.headroom(payload, row["loan_id"])
    row["analysis"] = {**(row.get("analysis") or {}), "headroom_rate": room["rate"],
                       "headroom_capped_by": room["capped_by"],
                       "breakeven_rate": (payload.get("risk") or {}).get("breakeven_rate"),
                       # The whole value given up, for requests raised before it was recorded.
                       **offers.full_term(payload, float(row["requested_rate"]), row.get("loan_amount"))}
    row["analysis"]["indicative_band"] = _indicative(payload, row["loan_id"])
    row["analysis"].update(offers.floors(payload, float(row["requested_rate"]),
                                         _limit(AM, "approve_below_recommendation_bps")))
    exact = offers.acceptance_at(payload, [float(row["recommended_rate"]), float(row["requested_rate"])])
    if exact:
        row["analysis"].update(acceptance_at_recommended=exact[0], acceptance_at_requested=exact[1])


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
    out = area_performance.deep_dive(row, payload, offers.headroom(payload, row["loan_id"]))
    # The same curve for this branch: the model moved by how far the branch's recent
    # files beat or fell short of it, beyond what every branch shows.
    check = branch_scenarios.calibration((payload.get("application_snapshot") or {}).get("city"))
    br = check["branch"]
    shift = (br["shift"] - check["all_branches"]["shift"]) if br else 0.0
    for point in out["curve"]:
        if point.get("acceptance") is not None:
            point["acceptance_branch"] = round(float(branch_scenarios._sigmoid(
                branch_scenarios._logit(point["acceptance"]) + shift)), 4)
    out["branch"] = {"city": check["city"], "files": br["files"] if br else None,
                     "gap_pts": round(br["gap_pts"] - check["all_branches"]["gap_pts"], 1) if br else None,
                     "shift": round(shift, 4), "note": check["note"]}
    return competitor_api.clean(out)


def _scenario_inputs(request_id: str) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any] | None]:
    """The request, the decision it was priced on, the branch's files and its matched discount effect."""
    row = offers.request(request_id)
    if row is None:
        raise HTTPException(404, "no such request")
    decision = decision_logger.get_decision(row["decision_id"]) if row.get("decision_id") else None
    if not decision:
        raise HTTPException(404, "the decision this request was priced on is not on file")
    payload = decision.get("payload") or {}
    history = portfolio.frame()
    branch, _ = approval_briefing.scope(history, (payload.get("application_snapshot") or {}).get("city"))
    propensity = _propensity(len(history))
    effect = None
    if propensity is not None:
        at_branch = propensity[history.index.get_indexer(branch.index)]
        effect = (approval_briefing.adjusted_effect(branch, at_branch, row.get("reason"), str(row.get("reason")))
                  if row.get("reason") else None) or approval_briefing.adjusted_effect(branch, at_branch, None, "all discounts")
    return row, payload, branch, effect


@router.get("/rate-requests/{request_id}/scenarios",
            summary="Each rate: what the model predicts, and how this branch differs")
def request_scenarios(request_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require(principal, AM, ADMIN)
    row, payload, _, effect = _scenario_inputs(request_id)
    room = offers.headroom(payload, row["loan_id"])
    return competitor_api.clean(branch_scenarios.scenarios(row, payload, ceiling=room["rate"], branch_effect=effect))


@router.get("/rate-requests/{request_id}/suggestion",
            summary="A suggested decision and the facts behind it")
def request_suggestion(request_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require(principal, AM, ADMIN)
    row, payload, branch, effect = _scenario_inputs(request_id)
    room = offers.headroom(payload, row["loan_id"])
    scen = branch_scenarios.scenarios(row, payload, ceiling=room["rate"], branch_effect=effect)
    lim = offers.limits({**payload, "recommendation": {**(payload.get("recommendation") or {}),
                                                        "recommended_rate": scen["recommended_rate"]}},
                        0, _limit(AM, "approve_below_recommendation_bps"))
    depth = (scen["recommended_rate"] - float(row["requested_rate"])) * 100
    similar = approval_briefing.similar_decisions(branch, row.get("reason"), depth)
    return competitor_api.clean(decision_support.suggest(
        row, scen, floor_rate=lim["manager_floor"], similar=similar,
        evidence_verified=bool(row.get("evidence_verified")),
        uplift_ceiling=min(room["rate"], lim["ceiling"]) if room["rate"] is not None else None))


@router.get("/rate-requests/{request_id}/checklist",
            summary="The manager's checks: alternatives to a rate cut, precedent, this month, relationship")
def request_checklist(request_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require(principal, AM, ADMIN)
    row, payload, branch, effect = _scenario_inputs(request_id)
    snap = payload.get("application_snapshot") or {}
    stored = intake.get(row["loan_id"])
    applicant = {**snap, **((stored or {}).get("applicant") or {})}
    rec = float((payload.get("recommendation") or {}).get("recommended_rate") or row["recommended_rate"])
    ask = float(row["requested_rate"])
    loan = float(row.get("loan_amount") or snap.get("sanctioned_loan_amount") or snap.get("requested_loan_amount") or 0)
    years = float(snap.get("tenure_years") or snap.get("requested_tenure_years") or 20)
    try:
        suggested = request_suggestion(request_id, principal)
    except HTTPException:
        suggested = None
    rm = row["requested_by"]
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    mine = area_performance.area(_rms(rm), offers.offers(sent_by=rm), offers.requests(requested_by=rm))["rms"]
    risk = payload.get("risk") or {}
    options = manager_checks.alternatives(
        loan=loan, years=years, age=applicant.get("applicant_age"), ask=ask, rec=rec,
        counter=(suggested or {}).get("rate") if (suggested or {}).get("action") == "counter" else None)
    return competitor_api.clean({
        "request_id": request_id, "city": snap.get("city"), "loan_amount": loan, "tenure_years": years,
        "alternatives": options,
        "precedent": manager_checks.precedent(snap.get("city"), snap.get("segment"), risk.get("risk_grade"), ask),
        "this_month": {**manager_checks.month_to_date(offers.requests(), snap.get("city"), rm, month, _where),
                       "rm_disbursed": mine[0] if mine else None},
        "relationship": manager_checks.relationship(applicant),
        "reason_hints": manager_checks.reason_hints(applicant.get("conversation_notes"), row.get("note")),
        "suggestion": suggested and {k: suggested[k] for k in ("action", "rate", "headline")},
        "fees": {"our_processing_fee_pct": options["processing_fee_pct"]},
    })


@router.get("/rate-requests/{request_id}/history",
            summary="The branch's upselling, competitor strength and sales notes, and the file's own notes")
def request_history(request_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require(principal, AM, ADMIN)
    row, applicant, grade = _history_context(request_id)
    return competitor_api.clean(branch_history.history(
        row, applicant, risk_grade=grade, retriever=get_note_retriever()))


@router.get("/rate-requests/{request_id}/precedents",
            summary="The past negotiations most similar to this file's conversation, with what happened")
def request_precedents(request_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """Context for the area manager only: it never changes the recommendation or a price."""
    _require(principal, AM, ADMIN)
    row, applicant, _ = _history_context(request_id)
    return competitor_api.clean(precedents.similar_negotiations(row, applicant, get_note_retriever()))


def _history_context(request_id: str) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    row = offers.request(request_id)
    if row is None:
        raise HTTPException(404, "no such request")
    decision = decision_logger.get_decision(row["decision_id"]) if row.get("decision_id") else None
    payload = (decision or {}).get("payload") or {}
    stored = intake.get(row["loan_id"])
    # The stored file for its notes and sales context; the priced snapshot fills what it lacks.
    applicant = {**(payload.get("application_snapshot") or {}), **((stored or {}).get("applicant") or {})}
    return row, applicant, (payload.get("risk") or {}).get("risk_grade")


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


def _memory_inputs(request_id: str, principal: Principal) -> tuple[dict[str, Any], dict[str, Any],
                                                                   dict[str, Any], dict | None, dict | None]:
    """The request and what the decision-support screen reads for it: the decision it was
    priced on, the file, the manager's checklist and the branch briefing."""
    row = offers.request(request_id)
    if row is None:
        raise HTTPException(404, "no such request")
    decision = decision_logger.get_decision(row["decision_id"]) if row.get("decision_id") else None
    payload = (decision or {}).get("payload") or {}
    stored = intake.get(row["loan_id"])
    applicant = {**(payload.get("application_snapshot") or {}), **((stored or {}).get("applicant") or {})}
    try:
        checklist = request_checklist(request_id, principal)
    except HTTPException:
        checklist = None
    try:
        brief = briefing(request_id, principal)
    except HTTPException:
        brief = None
    return row, payload, applicant, checklist, brief


@router.get("/rate-requests/{request_id}/memory",
            summary="The factors and their signals, and how you decided requests like this before")
def request_memory(request_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require(principal, AM, ADMIN)
    agent = ManagerMemoryAgent()
    seen = agent.observe(*_memory_inputs(request_id, principal))
    try:
        memory = {"available": True, **agent.recall(seen, manager=principal.name, request_id=request_id)}
    except Exception as exc:                        # noqa: BLE001 - the factors still serve the screen
        log.warning("decision memory unavailable for %s: %s", request_id, exc)
        memory = {"available": False, "reason": "Decision memory could not be read on this server."}
    return competitor_api.clean({"request_id": request_id, "factors": seen["factors"], "lean": seen["lean"],
                                 "suggestion": seen["suggestion"], "memory": memory})


@router.post("/rate-requests/{request_id}/decide", summary="Approve, decline or uplift a requested rate")
def decide(request_id: str, body: DecisionIn, principal: Principal = Depends(current_principal),
           orchestrator=Depends(get_orchestrator)) -> dict[str, Any]:
    _require(principal, AM, ADMIN)
    # What the manager is deciding on, frozen before the decision changes it: the factors,
    # the suggestion and the past cases memory put in front of them. Memory never stands
    # between a manager and a decision, so a failure here is recorded, not raised.
    pending = offers.request(request_id)
    seen, unavailable, recalled = None, None, []
    if pending and pending.get("status") == "pending":
        agent = ManagerMemoryAgent()
        try:
            seen = agent.observe(*_memory_inputs(request_id, principal))
        except Exception as exc:                    # noqa: BLE001
            unavailable = f"the factors could not be read when the decision was taken ({exc})"
            log.warning("decision memory for %s: %s", request_id, unavailable)
        if seen is not None:
            try:
                recalled = agent.recalled_ids(agent.recall(seen, manager=principal.name, request_id=request_id))
            except Exception as exc:                # noqa: BLE001
                log.warning("decision memory recall for %s failed: %s", request_id, exc)
    try:
        result = offers.decide(request_id, approve=body.approve, rate=body.rate, reason=body.reason,
                               note=body.note, decided_by=principal.name,
                               rate_authority=principal.rate_authority or "regional_manager",
                               am_bps=_limit(AM, "approve_below_recommendation_bps"),
                               capture_override=orchestrator.capture_override, uplift=body.uplift)
    except offers.OfferError as exc:
        raise _fail(exc) from exc
    try:
        memory = orchestrator.remember_manager_decision(request=pending, decided=result, observation=seen,
                                                        unavailable=unavailable, recalled=recalled)
    except Exception as exc:                        # noqa: BLE001 - the decision is already taken
        log.warning("decision %s stands but was not remembered: %s", request_id, exc)
        memory = {"recorded": False, "reason": "The decision stands, but it could not be added to memory."}
    return {**result, "memory": memory}


# ------------------------------------------------ Fair Practices Code documents

class KFSIn(BaseModel):
    language: str = Field("en", min_length=2, max_length=8)


def _kfs_offer(offer_id: str, principal: Principal) -> dict[str, Any]:
    from ..governance import store
    row = store.fetch_one("SELECT offer_id, sent_by FROM offers WHERE offer_id = ?", (offer_id,))
    if row is None:
        raise HTTPException(404, f"no offer {offer_id}")
    if _own_only(principal) and row["sent_by"] != principal.name:
        raise HTTPException(403, "a relationship manager gives the KFS for their own offers")
    return row


@router.post("/offers/{offer_id}/kfs", summary="Give the borrower the Key Facts Statement for an offer")
def issue_kfs(offer_id: str, body: KFSIn = KFSIn(),
              principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """RBC Directions para 29: the APR, every charge and the repayment schedule, before
    the contract, binding for three working days. A still-valid KFS for the same rate is
    returned again, not replaced."""
    from ..intake import kfs
    _require(principal, RM, AM, ADMIN)
    _kfs_offer(offer_id, principal)
    try:
        return kfs.issue(offer_id, issued_by=principal.name, language=body.language)
    except kfs.KFSError as exc:
        raise HTTPException(exc.status, exc.detail) from exc


@router.get("/offers/{offer_id}/kfs", summary="Key Facts Statements given for an offer")
def list_kfs(offer_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    from ..intake import kfs
    _require(principal, RM, AM, ADMIN)
    _kfs_offer(offer_id, principal)
    return {"statements": kfs.statements(offer_id=offer_id)}


@router.post("/offers/{offer_id}/kfs/{kfs_id}/acknowledge",
             summary="Record that the borrower acknowledged the KFS after it was explained")
def acknowledge_kfs(offer_id: str, kfs_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    from ..intake import kfs
    _require(principal, RM, ADMIN)
    _kfs_offer(offer_id, principal)
    held = [s for s in kfs.statements(offer_id=offer_id) if s["kfs_id"] == kfs_id]
    if not held:
        raise HTTPException(404, f"no KFS {kfs_id} for offer {offer_id}")
    return kfs.acknowledge(kfs_id, recorded_by=principal.name)


@router.get("/rejection-note/{loan_id}", summary="The reasons for a rejection, in writing (HFC Directions 135)")
def rejection_note(loan_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    from ..intake import fair_practices
    _require(principal, RM, AM, ADMIN)
    if _own_only(principal):
        row = intake.get(loan_id) or {}
        logged = {x for x in ((row.get("applicant") or {}).get("created_by"), row.get("owner")) if x}
        if principal.name not in logged:
            raise HTTPException(403, "a relationship manager gets the rejection note for files they logged")
    try:
        return fair_practices.rejection_note(loan_id, requested_by=principal.name)
    except fair_practices.NoDecline as exc:
        raise HTTPException(409, str(exc)) from exc
