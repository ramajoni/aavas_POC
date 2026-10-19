"""The agent control plane.

The rest of the API answers what the engine decided. This answers *who decided
it*: which agents exist, which the pipeline actually calls, what each one
contributes, and - for any decision on record - the order they ran in and what
each produced.

Two things it deliberately shows rather than hides:

* **Dormant agents.** Three are implemented and not wired in. An agent that
  exists but never runs is a fact about the system, and a roster that quietly
  omitted it would be describing a pipeline that is not the one running.
* **Degraded agents.** An agent whose model artefact is missing still runs; the
  engine falls back rather than failing. That is the difference between "ready"
  and "the number you are looking at came from a fallback".
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..agents import capabilities as agent_capabilities
from ..agents import packages as agent_packages
from ..agents import roster as agent_roster
from ..governance import decision_logger
from ..logging_utils import get_logger
from .deps import get_orchestrator

log = get_logger(__name__)
router = APIRouter(prefix="/agents", tags=["agents"])


def _loaded_models(orchestrator) -> set[str]:
    bundle = orchestrator.bundle
    return {name for name in bundle.versions() if name not in (bundle.missing or [])}


def _trace_of(decision: dict[str, Any]) -> dict[str, Any] | None:
    payload = decision.get("payload") or {}
    trace = payload.get("trace")
    if not trace:
        return None
    return {
        "decision_id": decision.get("decision_id"),
        "loan_id": decision.get("loan_id"),
        "stage": decision.get("stage"),
        "created_at": decision.get("created_at") or decision.get("logged_at"),
        "elapsed_ms": payload.get("elapsed_ms"),
        "recommended_rate": (payload.get("recommendation") or {}).get("recommended_rate"),
        **trace,
    }


@router.get("", summary="Which agents exist, which run, and what each decides")
def index(orchestrator=Depends(get_orchestrator)) -> dict[str, Any]:
    loaded = _loaded_models(orchestrator)
    return {
        "summary": agent_roster.summary(loaded),
        "pipeline": agent_roster.PIPELINE,
        "agents": agent_roster.roster(loaded),
        "capabilities": agent_capabilities.summary(),
        "models_loaded": sorted(loaded),
        "models_missing": sorted(orchestrator.bundle.missing or []),
        "note": ("The pipeline is the final-assessment path. Pre-assessment runs the "
                 "underwriting and pre-assessment agents only; the learning agent runs "
                 "after a human decision, not during one."),
    }


@router.get("/packages", summary="The reusable agent packages and their boundaries")
def packages() -> dict[str, Any]:
    """Four of the agents are whole subsystems rather than pipeline steps.

    Each declares its public surface, its dependencies, the config it owns and
    what it would take to lift it out - and `tests/test_agent_boundaries.py`
    checks the declaration against the real import graph in both directions, so
    none of this is documentation that can quietly go stale.
    """
    return {
        **agent_packages.summary(),
        "packages": agent_packages.listing(),
        "note": ("The four were nearly independent already. What coupled them was three pure "
                 "functions living wherever they were first written - column coercion in the "
                 "feature package, PSI in the learning package, fairness in the model "
                 "package. Those are kernel primitives now and the peer coupling is gone."),
    }


@router.get("/tools", summary="What the agents call that lives outside this process")
def tools() -> dict[str, Any]:
    """Availability is answered only where it is actually probed.

    Model artefacts come from the loaded bundle and connectors from the registry.
    Everything else reports 'unknown': a tool this endpoint has not probed is not
    a tool it can call healthy."""
    try:
        from ..connectors.registry import status as connector_status
        connectors = connector_status()
    except Exception as exc:                                    # noqa: BLE001
        log.warning("connector status unavailable: %s", exc)
        connectors = None

    from ..governance import model_registry
    bundle = model_registry.load_bundle()
    loaded = {n for n in bundle.versions() if n not in (bundle.missing or [])}

    rows = agent_capabilities.tools(loaded, connectors)
    return {
        "tools": rows,
        "summary": agent_capabilities.summary(),
        "used_by": {t["name"]: [a["name"] for a in agent_roster.AGENTS if t["name"] in a["tools"]]
                    for t in rows},
        # Registered says an agent can call it. Permissioned says whether it may,
        # and what the call is allowed to do.
        "permissions": agent_capabilities.tool_permissions(),
        "writers": agent_capabilities.writers(),
        "note": ("A tool is something outside this process that can be absent: a model "
                 "artefact, a store, an MCP server, an optional library, an LLM endpoint. "
                 "Each names what makes it unavailable and what runs instead. Tools whose "
                 "availability is not probed here report 'unknown' rather than a guess."),
    }


@router.get("/skills", summary="What the agents work out for themselves")
def skills() -> dict[str, Any]:
    rows = agent_capabilities.skills()
    return {
        "skills": rows,
        "used_by": {s["name"]: [a["name"] for a in agent_roster.AGENTS if s["name"] in a["skills"]]
                    for s in rows},
        "note": ("A skill is deterministic logic with no external dependency, so it has no "
                 "availability and no fallback - nothing can make it unavailable. Each names "
                 "the config that governs it, because a skill whose thresholds can move is "
                 "only deterministic with respect to a version of that file. "
                 "tests/test_capabilities.py enforces the distinction: a module that reaches "
                 "a store, a connector or a model artefact is rejected as a skill."),
    }


@router.get("/traces", summary="Recent decisions that carry a step trace")
def traces(limit: int = 20) -> dict[str, Any]:
    rows = []
    for decision in decision_logger.recent(limit=max(limit, 1) * 3):
        trace = _trace_of(decision)
        if not trace:
            continue
        rows.append({k: v for k, v in trace.items() if k != "steps"} |
                    {"steps": len(trace.get("steps") or [])})
        if len(rows) >= limit:
            break
    return {"available": bool(rows), "traces": rows,
            "note": ("Decisions priced before the trace existed carry none. Price one "
                     "from the assessment workspace to produce a trace."
                     if not rows else "")}


@router.get("/trace/latest", summary="The most recent decision's step trace")
def latest_trace() -> dict[str, Any]:
    # Registered before /trace/{decision_id}: 'latest' would otherwise be read
    # as a decision id and 404.
    for decision in decision_logger.recent(limit=50):
        trace = _trace_of(decision)
        if trace:
            return trace
    raise HTTPException(404, "no decision on record carries a step trace. Price one to "
                             "produce it - decisions made before the trace existed have none.")


@router.get("/trace/{decision_id}", summary="One decision's step trace")
def trace(decision_id: str) -> dict[str, Any]:
    decision = decision_logger.get_decision(decision_id)
    if not decision:
        raise HTTPException(404, f"no decision {decision_id!r} on record")
    found = _trace_of(decision)
    if not found:
        raise HTTPException(404, f"decision {decision_id!r} was priced before the step trace "
                                 f"existed, so it carries none")
    return found
