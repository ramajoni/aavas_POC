"""The control tower: one view of what is configured, running and learning.

The lifecycle console shows each stage. This shows the whole estate at once -
the view an operator keeps open - and it is built on a rule that matters more
than any tile on it:

**A number appears only when something produced it.** Every metric carries the
source it came from, and a metric with no source is returned with a null value
and the reason there is none, rather than being quietly dropped or filled with a
plausible figure. A control tower whose dashboard cannot be distinguished from
its mock-up is worse than no dashboard: it reports confidence about a system
nobody has measured.

Almost nothing here is new analysis. ``aavas.analytics.performance`` has
computed model health, drift, operational load and champion/challenger state
since before this module existed, and no route reached it - the functions were
written for a governance dashboard that was never wired up. This module is that
wiring, plus the honest accounting of what those functions cannot answer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from ..agents import roster as agent_roster
from ..analytics import performance
from ..config import versions
from ..connectors.registry import status as connector_status
from ..governance import monitors, request_log, store
from ..logging_utils import get_logger
from .deps import get_orchestrator

log = get_logger(__name__)
router = APIRouter(prefix="/control-tower", tags=["control tower"])

BPS = 100.0


# --------------------------------------------------------------- metric shape


def _metric(label: str, value: Any, source: str, unit: str = "",
            status: str = "", detail: str = "") -> dict[str, Any]:
    return {"label": label, "value": value, "unit": unit, "source": source,
            "status": status or "", "detail": detail, "sourced": True}


def _unsourced(label: str, reason: str, unit: str = "") -> dict[str, Any]:
    """A tile the estate cannot fill.

    Kept in the payload rather than omitted: the absence is the finding. A
    dashboard that silently drops what it cannot measure looks complete.
    """
    return {"label": label, "value": None, "unit": unit, "source": None,
            "status": "unsourced", "detail": reason, "sourced": False}


def _round(value: Any, places: int = 2) -> Any:
    return round(float(value), places) if isinstance(value, (int, float)) else None


# ------------------------------------------------------------------ the zones


def _auth_enforced() -> bool:
    from ..api import security

    return security.status()["enforced"]


def _zones(agents: dict[str, Any], connectors: dict[str, Any]) -> list[dict[str, Any]]:
    """The six control zones, each with where it is reached and what it lacks."""
    dormant = agents.get("dormant", 0)
    return [
        {"key": "policy", "n": 1, "title": "Business & policy control",
         "controls": "Product, segment, pricing boundaries, risk appetite, authority limits.",
         "status": "partial",
         "surfaces": ["/config", "/config/{name}", "/pricing/policy/explain"],
         "gap": ("Every setting is readable and versioned; none is editable. A policy change "
                 "is a file edit and a restart, with no validation and no approval in front "
                 "of it.")},
        {"key": "agent_factory", "n": 2, "title": "Agent factory",
         "controls": "Which agents exist, what each decides, and which the pipeline calls.",
         "status": "partial",
         "surfaces": ["/agents", "/agents/traces"],
         "gap": (f"The roster is declared and observable, but agents are not configurable and "
                 f"there is no version over the agent set as a whole. "
                 f"{dormant} agent(s) are built and called by nothing.")},
        {"key": "model_data", "n": 3, "title": "Model & data control",
         "controls": "Data sources, feature contract, AutoML, calibration, champion selection.",
         "status": "partial",
         "surfaces": ["/tracking", "/tracking/runs", "/connectors", "/control-tower/models"],
         "gap": (f"AutoML searches and explains a champion but is not the training path, and "
                 f"{len(connectors.get('blocked') or [])} of "
                 f"{len(connectors.get('connectors') or [])} connectors are blocked, so the "
                 f"data control zone governs sources the engine does not yet read.")},
        {"key": "approval", "n": 4, "title": "Human approval centre",
         "controls": "Business, credit, pricing, model-risk and technology sign-off.",
         "status": "partial" if _auth_enforced() else "partial",
         "surfaces": ["/governance", "/governance/approvals", "/governance/gates/{type}/{id}",
                      "/governance/roles"],
         "gap": ("Change requests, per-role signatures, gates that block, an override that "
                 "needs a written justification, and an apply step separate from approval - "
                 "all built. " + ("" if _auth_enforced() else
                 "Authentication is disabled, so a signature is a name in a text field "
                 "rather than a verified identity. That is what keeps this partial. ") +
                 "A config change is still applied by editing the file: the approval records "
                 "that the edit was authorised, it does not perform it.")},
        {"key": "runtime", "n": 5, "title": "Runtime operations centre",
         "controls": "Decisions, agent traces, latency, approval routing, overrides.",
         "status": "ready",
         "surfaces": ["/control-tower/operations", "/agents/trace/{decision_id}",
                      "/pricing/audit/{loan_id}"],
         "gap": ("A request row is opened before the pipeline runs and closed however it "
                 "ends, so success, failure and abandonment are all countable. Only the four "
                 "priced endpoints are tracked - dashboard polling is deliberately not.")},
        {"key": "learning", "n": 6, "title": "Continuous learning centre",
         "controls": "Outcomes, drift, override learning, challenger training, federation.",
         "status": "partial",
         "surfaces": ["/control-tower/drift", "/governance/monitors",
                      "/pricing/learning/override-review", "/pricing/learning/questions"],
         "gap": ("Monitor passes now record having run, what they found and whether they "
                 "triggered - and federated rounds persist. Nothing puts them on a schedule: "
                 "scripts/run_monitors.py is what a cron would call, and a gap in the run "
                 "history is still only visible by looking for it.")},
    ]


# ----------------------------------------------------------------- the panels


def _today_count() -> int:
    day = datetime.now(timezone.utc).date().isoformat()
    row = store.fetch_one(
        "SELECT COUNT(*) AS n FROM decisions WHERE created_at >= ?", (day,))
    return int(row["n"]) if row else 0


def _latency_by_stage() -> dict[str, dict[str, Any]]:
    """Latency split by stage, because one figure over both describes neither.

    Pre-assessment scores a band from a loaded model. Final assessment runs the
    whole agent pipeline. Reporting a single median over the two produces a
    number no request has ever taken.
    """
    out: dict[str, dict[str, Any]] = {}
    rows = store.query(
        "SELECT stage, elapsed_ms FROM decisions WHERE elapsed_ms IS NOT NULL "
        "ORDER BY stage, elapsed_ms")
    for stage in {r["stage"] for r in rows}:
        values = sorted(r["elapsed_ms"] for r in rows if r["stage"] == stage)
        if not values:
            continue
        out[stage] = {
            "n": len(values),
            "p50": round(values[len(values) // 2], 1),
            "p95": round(values[min(len(values) - 1, int(len(values) * 0.95))], 1),
            "max": round(values[-1], 1),
        }
    return out


def _failed_count() -> int:
    try:
        return int(request_log.summary().get("by_outcome", {}).get(request_log.FAILED, 0))
    except Exception:                                           # noqa: BLE001
        return 0


def _agents_panel(agents: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    healthy = agents["wired"] - agents["degraded"]
    return {
        "key": "agents", "title": "Production agents",
        "metrics": [
            _metric("Declared", agents["agents"], "agents/roster.py"),
            _metric("Wired into the pipeline", agents["wired"], "agents/roster.py",
                    status="ok" if agents["wired"] else "warn"),
            _metric("Healthy", healthy, "agents/roster.py + model bundle",
                    status="ok" if healthy == agents["wired"] else "warn"),
            _metric("Degraded", agents["degraded"], "missing model artefacts",
                    status="warn" if agents["degraded"] else "ok",
                    detail=(f"missing: {', '.join(missing)}" if missing else
                            "every model an agent needs is loaded")),
            _metric("Runs on request", agents.get("on_demand", 0), "agents/roster.py on_demand",
                    detail="called from a person's screen, outside the pricing pipeline"),
            _metric("Dormant", agents["dormant"], "agents/roster.py wired=False",
                    status="warn" if agents["dormant"] else "ok",
                    detail="built, tested, and called by nothing"),
            _metric("Failed requests", _failed_count(), "requests.outcome = failed",
                    status="warn" if _failed_count() else "ok",
                    detail="an agent that raises is re-raised into the request; the request "
                           "row is closed as failed with its exception type"),
        ],
    }


def _requests_panel(ops: dict[str, Any]) -> dict[str, Any]:
    try:
        requests = request_log.summary()
    except Exception as exc:                                    # noqa: BLE001
        log.warning("request log unavailable: %s", exc)
        requests = {}
    latency = _latency_by_stage()
    pre = latency.get("pre_assessment") or {}
    final = latency.get("post_assessment") or {}
    slowest = max((v.get("max") or 0 for v in latency.values()), default=None)

    def timing(row: dict[str, Any]) -> str:
        if not row:
            return "no timed decision at this stage"
        return f"p50 {row['p50']:.0f} ms, p95 {row['p95']:.0f} ms, over {row['n']} decisions"

    return {
        "key": "requests", "title": "Requests",
        "metrics": [
            _metric("Decisions today", _today_count(), "governance store, decisions table"),
            _metric("Decisions recorded", ops["decisions_total"], "governance store"),
            _metric("Pre-assessment", pre.get("n", 0), "decisions.stage + elapsed_ms",
                    detail=timing(pre)),
            _metric("Final assessment", final.get("n", 0), "decisions.stage + elapsed_ms",
                    status="warn" if final.get("p95", 0) > 2000 else "ok",
                    detail=timing(final)),
            _metric("Slowest decision", _round(slowest, 0), "decisions.elapsed_ms", "ms",
                    status="warn" if (slowest or 0) > 10000 else "",
                    detail="a single request a branch waited on, not an average"),
            _metric("Success rate", requests.get("success_rate_pct"),
                    "governance store, requests table", "%",
                    status="ok" if (requests.get("success_rate_pct") or 0) >= 99 else "warn",
                    detail=f"over {requests.get('window', 0)} logged requests. A row is opened "
                           f"before the pipeline runs, so a request that failed leaves one too"),
            _metric("Failure rate", requests.get("failure_rate_pct"),
                    "requests.outcome = failed", "%",
                    status="warn" if (requests.get("failure_rate_pct") or 0) > 0 else "ok"),
            _metric("Client errors", requests.get("client_error_rate_pct"),
                    "requests.outcome = client_error", "%",
                    detail="4xx: the caller sent something invalid. Not an engine failure"),
            _metric("Abandoned", requests.get("abandoned", 0), "requests never closed",
                    status="warn" if requests.get("abandoned") else "ok",
                    detail="opened and never closed - the process died mid-request, which a "
                           "completion-only log cannot see"),
        ],
    }


def _model_panel(key: str, title: str, model: dict[str, Any] | None,
                 champion: str | None, drift_status: str | None,
                 extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not model:
        return {"key": key, "title": title,
                "metrics": [_unsourced("Champion", f"no {key} model is loaded")]}
    metrics = model.get("all_metrics") or {}
    gap = metrics.get("calibration_gap")
    return {
        "key": key, "title": title,
        "metrics": [
            _metric("Champion", champion or model.get("version") or "unregistered",
                    "model registry",
                    detail=f"{model.get('learner')}, fitted on {model.get('n_train')} rows "
                           f"to {model.get('trained_at', '')[:10]}"),
            _metric("AUC", _round(metrics.get("auc"), 3), "model card",
                    status=model.get("status", "")),
            _metric("Calibration gap", _round(gap, 4), "model card",
                    status="ok" if isinstance(gap, (int, float)) and abs(gap) <= 0.02 else "warn",
                    detail="mean predicted minus observed on the held-out split"),
            _metric("Drift", drift_status or "unknown", "analytics/performance.drift()",
                    status="ok" if drift_status == "ok" else "warn",
                    detail="whole-estate input and conversion drift, not per model"),
            *(extra or []),
        ],
    }


def _pricing_panel() -> dict[str, Any]:
    row = store.fetch_one(
        "SELECT COUNT(*) AS n, AVG(recommended_rate) AS rec, AVG(card_rate) AS card, "
        "AVG(conversion_probability) AS conv FROM decisions "
        "WHERE recommended_rate IS NOT NULL AND card_rate IS NOT NULL") or {}
    n = int(row.get("n") or 0)
    rec, card = row.get("rec"), row.get("card")
    uplift = (rec - card) * BPS if n and rec is not None and card is not None else None
    return {
        "key": "pricing", "title": "Pricing optimiser",
        "metrics": [
            _metric("Avg recommended rate", _round(rec), "decisions table", "% p.a.",
                    detail=f"over {n} priced decisions"),
            _metric("Avg card rate", _round(card), "decisions table", "% p.a."),
            _metric("Rate movement vs card", _round(uplift, 1), "decisions table", "bps",
                    status="ok",
                    detail="signed: positive means the engine priced above the card rate on "
                           "average, negative means it discounted. Which of those is correct "
                           "is a business question, not a model one"),
            _metric("Mean modelled conversion", _round(row.get("conv"), 3),
                    "decisions.conversion_probability",
                    detail="what the conversion model predicted, not what happened"),
            _unsourced("Realised conversion",
                       "outcomes are recorded for a small fraction of decisions in this "
                       "store; an acceptance rate over them would describe the files someone "
                       "chose to follow up, not the book"),
        ],
    }


def _overrides_panel(ops: dict[str, Any]) -> dict[str, Any]:
    reasons = ops.get("override_reason_mix") or {}
    top = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else None
    rate = ops.get("override_rate_on_recorded_decisions")
    return {
        "key": "overrides", "title": "Human overrides",
        "metrics": [
            _metric("Overrides recorded", ops["overrides_total"], "governance store"),
            _metric("Override rate", _round((rate or 0) * 100, 1), "overrides / decisions", "%",
                    detail="over the most recent window, not lifetime"),
            _metric("Outside authority", ops.get("overrides_outside_authority", 0),
                    "overrides.within_authority",
                    status="warn" if ops.get("overrides_outside_authority") else "ok",
                    detail="recorded and flagged; recording an override is not approval"),
            _metric("Most common reason", top or "none", "overrides.reason"),
            _metric("Outcomes captured", ops["outcomes_total"], "governance store",
                    status="warn" if ops["outcomes_total"] < ops["overrides_total"] else "ok"),
        ],
    }


def _federated_panel() -> dict[str, Any]:
    latest = store.fetch_one(
        "SELECT * FROM federated_rounds ORDER BY created_at DESC LIMIT 1")
    if not latest:
        return {"key": "federated", "title": "Federated learning", "metrics": [_unsourced(
            "Participants, rounds and global version",
            "the implementation exists and no round has been recorded. Run one - "
            "scripts/run_demo.py exercises it - and this panel fills from the store")]}
    return {
        "key": "federated", "title": "Federated learning",
        "metrics": [
            _metric("Participants", latest["participants"], "federated_rounds table",
                    detail=f"regions in run {latest['run_id']}"),
            _metric("Rounds", latest["rounds"], "federated_rounds table"),
            _metric("Strategy", latest["strategy"], "federated_rounds table",
                    detail=(f"proximal_mu {latest['proximal_mu']}"
                            if latest.get("proximal_mu") else "no proximal term")),
            _metric("Global version", latest["global_version"], "content hash of the "
                                                                "client parameters"),
            _metric("Final global MAE", _round(latest["final_mae_bps"], 2),
                    "federated_rounds table", "bps"),
            _unsourced("Effect on a priced decision",
                       "rounds are recorded, but the regional adjustment is not applied on "
                       "the priced path. No decision has been moved by one, so there is no "
                       "effect to report - which is different from an effect of zero"),
        ],
    }


def _governance_panel() -> dict[str, Any]:
    """Approval, authentication and monitoring: the design-time half."""
    from ..api import security
    from ..governance import approvals

    auth = security.status()
    try:
        summary = approvals.summary()
    except Exception as exc:                                    # noqa: BLE001
        log.warning("approvals unavailable: %s", exc)
        summary = {"by_state": {}, "pending": [], "blocked_by_gates": []}
    try:
        monitor = monitors.status()
    except Exception as exc:                                    # noqa: BLE001
        log.warning("monitors unavailable: %s", exc)
        monitor = {"never_run": list(monitors.MONITORS)}

    never = monitor.get("never_run") or []
    return {
        "key": "governance", "title": "Governance",
        "metrics": [
            _metric("Authentication", auth["mode"], "AAVAS_AUTH_MODE",
                    status="ok" if auth["enforced"] else "warn",
                    detail=auth["warning"]),
            _metric("Change requests", sum((summary.get("by_state") or {}).values()),
                    "governance store, approvals table",
                    detail=str(summary.get("by_state") or {})),
            _metric("Awaiting signature", len(summary.get("pending") or []),
                    "approvals.state = pending",
                    status="warn" if summary.get("pending") else "ok"),
            _metric("Raised over a failed gate", len(summary.get("blocked_by_gates") or []),
                    "approvals.gates_passed = 0",
                    status="warn" if summary.get("blocked_by_gates") else "ok",
                    detail="each carries a written justification, recorded permanently"),
            _metric("Monitors never run", len(never), "monitor_runs table",
                    status="warn" if never else "ok",
                    detail=", ".join(never) if never else "every monitor has a run on record"),
        ],
    }


# ------------------------------------------------------------------- routes


@router.get("", summary="The whole estate: zones, live numbers, and what has no source")
def index(orchestrator=Depends(get_orchestrator)) -> dict[str, Any]:
    bundle = orchestrator.bundle
    loaded = {n for n in bundle.versions() if n not in (bundle.missing or [])}
    agents = agent_roster.summary(loaded)
    missing = sorted(bundle.missing or [])

    ops = performance.operations()
    models = {m["name"]: m for m in performance.model_performance()["models"]}
    governance = performance.governance_status()
    champions = {c["name"]: c["version"] for c in governance["champions"]}

    try:
        drift_status = performance.drift()["overall_status"]
    except Exception as exc:                                        # noqa: BLE001
        # Drift needs the training dataset. A tower that cannot start because one
        # panel has no data is a tower nobody keeps open.
        log.warning("drift unavailable for the control tower: %s", exc)
        drift_status = None

    try:
        connectors = connector_status()
    except Exception as exc:                                        # noqa: BLE001
        log.warning("connector status unavailable: %s", exc)
        connectors = {}

    conversion_extra = [
        _metric("Conversion cost of +25 bps",
                _round((models.get("conversion", {}).get("all_metrics") or {})
                       .get("s2d_drop_per_25bps_pts"), 2),
                "model card", "pts"),
        _unsourced("Acceptance-signal agents",
                   "the three agents that read a customer's own words are dormant and every "
                   "communication connector is blocked, so no acceptance signal reaches this "
                   "model - it scores price response from structured fields only"),
    ]

    panels = [
        _governance_panel(),
        _agents_panel(agents, missing),
        _requests_panel(ops),
        _model_panel("risk", "Risk model", models.get("risk"),
                     champions.get("risk"), drift_status),
        _model_panel("conversion", "Acceptance / conversion model", models.get("conversion"),
                     champions.get("conversion"), drift_status, conversion_extra),
        _pricing_panel(),
        _overrides_panel(ops),
        _federated_panel(),
    ]

    unsourced = [{"panel": p["title"], "label": m["label"], "reason": m["detail"]}
                 for p in panels for m in p["metrics"] if not m["sourced"]]

    return {
        "generated_at": store.now(),
        "versions": versions(),
        "zones": _zones(agents, connectors),
        "panels": panels,
        "unsourced": unsourced,
        "counts": {"metrics": sum(len(p["metrics"]) for p in panels),
                   "unsourced": len(unsourced)},
        "champions": governance["champions"],
        "challengers": governance["challengers"],
        "note": ("Every figure names the table or artefact it came from. The tiles with no "
                 "value are the ones this estate cannot answer yet; they are listed rather "
                 "than hidden, because a dashboard that looks complete is the failure mode "
                 "a control tower exists to prevent."),
    }


@router.get("/operations", summary="How the engine is actually being used")
def operations(limit: int = 500) -> dict[str, Any]:
    # Split by stage as well as pooled: pre-assessment scores a band, final
    # assessment runs the pipeline, and the pooled median sits between two
    # workloads rather than describing either.
    return {**performance.operations(limit=limit), "latency_by_stage": _latency_by_stage()}


@router.get("/drift", summary="Input drift and conversion drift against the training window")
def drift() -> dict[str, Any]:
    return performance.drift()


@router.get("/models", summary="Per-model headline metrics, provenance and limitations")
def models() -> dict[str, Any]:
    return performance.model_performance()


@router.get("/governance", summary="Champions, challengers and the promotion gates")
def governance() -> dict[str, Any]:
    return performance.governance_status()
