"""/agentic - the Agentic OS: rebuild the models as a workflow of agents.

    GET  /agentic/overview          config, step definitions, readiness, recent runs, champions
    POST /agentic/policy/check      run the policy consistency checks and sample cases now
    POST /agentic/runs              start a run with the saved config/agentic_os.yaml
    GET  /agentic/runs              recent runs
    GET  /agentic/runs/{id}         one run, with its log
    POST /agentic/runs/{id}/promote make a local run's challengers the champions
    GET  /agentic/serving           the SageMaker Serverless endpoint's state
    POST /agentic/serving/invoke    call the endpoint with rows, for one model

The settings themselves are edited through /config/agentic_os, which versions and
audits every change like any other configuration file.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..agentic import runs, workflow
from ..config import load
from ..governance import audit
from ..models import base as model_base
from ..paths import ROOT
from .security import Principal, require_roles

router = APIRouter(prefix="/agentic", tags=["agentic OS"])
OPERATORS = ("administrator", "model_validation")

STEPS = [
    {"name": "data", "title": "Data",
     "description": "Build a synthetic book under the policy in force, or take the live dataset, then check "
                    "every loan obeys the policy before anything trains on it.",
     "settings": ["data.source", "data.rows", "data.seed", "data.checks"]},
    {"name": "policy", "title": "Policy",
     "description": "Run the policy file's consistency checks and its sample data. Training is blocked "
                    "while either fails.",
     "settings": ["policy.require_consistency_checks", "policy.require_sample_cases"]},
    {"name": "training", "title": "Training",
     "description": "Train the models with the chosen engine: in-process, an Amazon SageMaker training job, "
                    "or the AutoGluon Assistant agent. Results are compared with the champions.",
     "settings": ["training.engine", "training.models", "training.register_as", "training.sagemaker",
                  "training.autogluon_assistant"]},
    {"name": "serving", "title": "Serving",
     "description": "Serve the result: promote in-process after approval, or deploy a SageMaker Serverless "
                    "endpoint and check it answers exactly as training recorded.",
     "settings": ["serving.mode", "serving.promote_on_success", "serving.sagemaker_serverless"]},
]


def _who(p: Principal) -> str:
    return p.name or "unknown"


def _readiness(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        report = workflow.policy_report()
        out["policy"] = {"ok": report["consistency"]["ok"] and not report["sample_cases"]["failed"],
                         "version": report["policy_version"], "status": report["status"],
                         "consistency": f"{report['consistency']['passed']}/{report['consistency']['checks']}",
                         "sample_cases": f"{report['sample_cases']['passed']}/{report['sample_cases']['cases']}"}
    except Exception as exc:                                         # noqa: BLE001 - reported, not raised
        out["policy"] = {"ok": False, "error": str(exc)}
    out["boto3"] = {"ok": importlib.util.find_spec("boto3") is not None}
    cli = ROOT / cfg["training"]["autogluon_assistant"]["cli"]
    out["autogluon_assistant"] = {"ok": cli.exists(), "cli": str(cfg["training"]["autogluon_assistant"]["cli"])}
    out["bash"] = {"ok": shutil.which("bash") is not None}
    return out


@router.get("/overview", summary="Config, steps, readiness, runs and champions")
def overview(principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    cfg = load("agentic_os")
    champions = workflow.model_cards(model_base.registry_dir())
    return {
        "config": cfg, "version": cfg.get("agentic_os_config_version"), "steps": STEPS,
        "readiness": _readiness(cfg), "active_run": runs.active(), "runs": runs.all_runs(20),
        "champions": [{"model": n, "learner": c.get("learner"), "trained_at": c.get("trained_at"),
                       "metrics": {k: c.get("metrics", {}).get(k) for k in workflow.HEADLINE.get(n, [])}}
                      for n, c in champions.items()],
    }


@router.post("/policy/check", summary="Run the policy checks and sample cases now")
def policy_check(principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    return workflow.policy_report()


@router.post("/runs", summary="Start a run with the saved Agentic OS config")
def start_run(principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    try:
        run = workflow.start(_who(principal))
    except workflow.StepError as exc:
        raise HTTPException(409, str(exc)) from exc
    audit.record(_who(principal), "agentic_run_started", "agentic_run", run.run_id,
                 {"engine": run.record["engine"], "serving": run.record["serving_mode"],
                  "config_version": run.record["config_version"]})
    return run.as_dict()


@router.get("/runs", summary="Recent runs")
def list_runs(principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    return {"active_run": runs.active(), "runs": runs.all_runs(50)}


@router.get("/runs/{run_id}", summary="One run, with its log")
def get_run(run_id: str, principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    return run.as_dict(with_log=True)


@router.post("/runs/{run_id}/resume", summary="Retry a failed run from its first unfinished step")
def resume_run(run_id: str, principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    if run.record.get("status") != "failed":
        raise HTTPException(409, f"only a failed run resumes; run {run_id} is {run.record.get('status')}")
    if not runs.launch(run, workflow.resume):
        raise HTTPException(409, f"run {runs.active()} is still executing; one run at a time")
    audit.record(_who(principal), "agentic_run_resumed", "agentic_run", run_id, {})
    return run.as_dict()


@router.post("/runs/{run_id}/promote", summary="Make a local run's challengers the champions")
def promote(run_id: str, principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    roles = set(load("agentic_os").get("approval", {}).get("promote_roles", OPERATORS))
    if not roles & set(principal.roles):
        raise HTTPException(403, f"promotion needs one of {sorted(roles)}")
    if run.record.get("promotion"):
        raise HTTPException(409, f"run {run_id} was already promoted")
    try:
        record = workflow.promote(run, _who(principal))
    except workflow.StepError as exc:
        raise HTTPException(409, str(exc)) from exc
    audit.record(_who(principal), "agentic_models_promoted", "agentic_run", run_id, record)
    return run.as_dict()


@router.get("/serving", summary="The SageMaker Serverless endpoint's state")
def serving_state(principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    from ..agentic import serving
    return serving.status(load("agentic_os"))


class InvokeRequest(BaseModel):
    model: str = Field(..., examples=["stage2"])
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=50)
    rate: float | None = None


@router.post("/serving/invoke", summary="Call the endpoint for one model")
def serving_invoke(request: InvokeRequest,
                   principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    from ..agentic import serving
    try:
        return serving.invoke(load("agentic_os"), request.model_dump(exclude_none=True))
    except serving.ServingError as exc:
        raise HTTPException(502, str(exc)) from exc
