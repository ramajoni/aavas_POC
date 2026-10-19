"""Change requests, sign-off, and the one place a change is actually applied.

The endpoints below are the design-time half the system was missing. The shape
that matters: **approving a request does not change anything.** Approval produces
an authorisation; a separate call applies it, and the apply path re-checks that
an unapplied approval exists for that exact artefact and action. Splitting the two
is what makes "approved but not yet released" a state the system can be in, which
is the state a change-management process actually lives in.

Every write here carries the caller's principal, and every response carries the
authentication mode. With auth disabled the mode says so, because a signature
from an unverified caller is a name in a text field.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from ..governance import (agent_registry, approvals, knowledge_registry, model_registry,
                          monitors, prompt_registry)
from ..logging_utils import get_logger
from . import security
from .security import Principal, current_principal

log = get_logger(__name__)
router = APIRouter(prefix="/governance", tags=["governance"])


def _envelope(payload: dict[str, Any], principal: Principal) -> dict[str, Any]:
    return {**payload, "principal": principal.as_dict(), "auth": security.status()}


@router.get("", summary="Who may approve what, and what is outstanding")
def index(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    return _envelope({"summary": approvals.summary()}, principal)


@router.get("/approvals", summary="Change requests, newest first")
def listing(state: str | None = None, limit: int = 100,
            principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    return _envelope({"approvals": approvals.listing(state=state, limit=limit)}, principal)


@router.get("/approvals/{approval_id}", summary="One change request with its signatures")
def detail(approval_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    record = approvals.get(approval_id)
    if not record:
        raise HTTPException(404, f"no approval {approval_id!r}")
    return _envelope({"approval": record}, principal)


@router.post("/approvals", summary="Raise a change request; gates run now, not at approval")
def raise_request(
    artefact_type: str = Body(..., embed=True),
    artefact_id: str = Body(..., embed=True),
    action: str = Body("publish", embed=True),
    summary: str = Body("", embed=True),
    override_justification: str | None = Body(None, embed=True),
    principal: Principal = Depends(current_principal),
) -> dict[str, Any]:
    try:
        record = approvals.request(
            artefact_type=artefact_type, artefact_id=artefact_id, action=action,
            requested_by=principal.name, summary=summary,
            override_justification=override_justification)
    except (ValueError, PermissionError) as exc:
        # A failed gate is the caller's problem to resolve, not a server fault.
        raise HTTPException(422, str(exc)) from exc
    return _envelope({"approval": record}, principal)


@router.post("/approvals/{approval_id}/sign", summary="Sign a change request as one role")
def sign(
    approval_id: str,
    role: str = Body(..., embed=True),
    verdict: str = Body("approve", embed=True),
    comment: str = Body("", embed=True),
    principal: Principal = Depends(current_principal),
) -> dict[str, Any]:
    if not principal.has(role):
        raise HTTPException(403, f"you do not hold the role {role!r}; you hold {principal.roles}")
    try:
        record = approvals.sign(approval_id, role=role, principal=principal.name,
                                verdict=verdict, comment=comment)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _envelope({"approval": record}, principal)


@router.post("/approvals/{approval_id}/withdraw", summary="Withdraw your own request")
def withdraw(approval_id: str,
             principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    try:
        record = approvals.withdraw(approval_id, principal=principal.name)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    return _envelope({"approval": record}, principal)


@router.post("/approvals/{approval_id}/apply",
             summary="Apply an approved change. Approval alone changes nothing.")
def apply(approval_id: str,
          principal: Principal = Depends(security.require_roles("technology", "model_risk"))
          ) -> dict[str, Any]:
    record = approvals.get(approval_id)
    if not record:
        raise HTTPException(404, f"no approval {approval_id!r}")
    if record["state"] != approvals.APPROVED:
        raise HTTPException(
            409, f"approval {approval_id} is {record['state']}. Only a fully signed request "
                 f"can be applied; awaiting {record['awaiting_roles']}")

    artefact_type, artefact_id = record["artefact_type"], record["artefact_id"]
    try:
        if artefact_type == "model" and record["action"] == "promote":
            name, _, version = artefact_id.partition(":")
            model_registry.promote(name, version, actor=principal.name)
            detail = {"promoted": artefact_id}
        elif artefact_type == "agent_manifest":
            published = agent_registry.publish(actor=principal.name)
            detail = {"published": published}
        else:
            # Config and connector changes are file edits. The approval is the
            # record that the change was authorised; this endpoint does not write
            # to a config file, and says so rather than implying it did.
            raise HTTPException(
                501, f"a {artefact_type} change is applied by editing the file and restarting. "
                     f"This approval authorises that edit and is the record it was authorised; "
                     f"the API does not write configuration.")
    except HTTPException:
        raise
    except Exception as exc:                                    # noqa: BLE001
        log.exception("applying %s failed", approval_id)
        raise HTTPException(500, f"applying the change failed: {exc}") from exc

    approvals.mark_applied(approval_id, principal=principal.name, detail=detail)
    return _envelope({"approval": approvals.get(approval_id), "applied": detail}, principal)


@router.get("/monitors", summary="Which monitors have run, and which never have")
def monitor_status(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    return _envelope({**monitors.status(),
                      "history": monitors.history(limit=25)}, principal)


@router.post("/monitors/run", summary="Run the monitors now and record that they ran")
def monitor_run(monitor: str | None = Body(None, embed=True),
                principal: Principal = Depends(
                    security.require_roles("model_risk", "technology"))) -> dict[str, Any]:
    runners = {"drift": monitors.run_drift, "fairness": monitors.run_fairness,
               "retraining": monitors.run_retraining}
    if monitor and monitor not in runners:
        raise HTTPException(404, f"no runnable monitor {monitor!r}; expected one of "
                                 f"{sorted(runners)}")
    results = {monitor: runners[monitor](actor=principal.name)} if monitor \
        else monitors.run_all(actor=principal.name)
    return _envelope({"results": results, "status": monitors.status()}, principal)


@router.get("/registries", summary="Prompts, knowledge indexes and federated rounds on file")
def registries(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    from ..governance import store

    prompt_registry.register_declared(actor="api")
    return _envelope({
        "prompts": {"registered": prompt_registry.registered(),
                    "current": prompt_registry.current_ids()},
        "knowledge": {"indexes": knowledge_registry.registered()},
        "federated": {"rounds": store.query(
            "SELECT run_id, created_at, strategy, proximal_mu, rounds, participants, "
            "global_version, final_mae_bps FROM federated_rounds ORDER BY created_at DESC "
            "LIMIT 25")},
        "models": {"registered": [
            {k: v for k, v in row.items() if k != "card"} for row in model_registry.registered()]},
        "agent_sets": {"manifests": agent_registry.registered(limit=25),
                       "current": agent_registry.current_id()},
    }, principal)


@router.get("/roles", summary="The roles, and what each one signs off")
def roles(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    return _envelope({"roles": approvals.roles(),
                      "artefacts": approvals.config().get("artefacts") or {},
                      "separation_of_duties": approvals.config().get("separation_of_duties") or {},
                      "gate_override": approvals.config().get("gate_override") or {}},
                     principal)


@router.get("/gates/{artefact_type}/{artefact_id}",
            summary="Run the gates for a change without raising a request")
def gates(artefact_type: str, artefact_id: str,
          principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    try:
        results = approvals.run_gates(artefact_type, artefact_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return _envelope({"artefact_type": artefact_type, "artefact_id": artefact_id,
                      "gates": results, "clear": approvals.gates_clear(results)}, principal)
