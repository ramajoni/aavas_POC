"""/monitoring - data, model and concept drift of the pricing models.

    GET  /monitoring/drift           the latest drift report, with earlier runs
    POST /monitoring/drift/run       detect drift now (optionally a simulated scenario) and ship it
    GET  /monitoring/alerts          alerts OpenObserve has sent, newest first
    POST /monitoring/alert-webhook   where OpenObserve posts an alert (shared token, no sign-in)

Detection lives in aavas.monitoring.drift (settings: config/monitoring.yaml); the
findings go to OpenObserve, whose scheduled alerts call the webhook back.
"""

from __future__ import annotations

import hmac
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from ..governance import audit
from ..logging_utils import get_logger
from ..monitoring import drift, openobserve
from .security import Principal, require_roles

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
OPERATORS = ("administrator", "model_validation")
log = get_logger(__name__)


def _alerts_file():
    return drift.reports_dir() / "alerts.jsonl"


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    flagged = {k: [f for f in report.get(k, []) if f.get("status") != "ok"] for k in ("data", "model", "concept")}
    return {key: report.get(key) for key in ("run_id", "generated_at", "scenario", "config_version", "windows",
                                             "status", "counts", "thresholds")} | {"flagged": flagged}


@router.get("/drift")
def latest_drift(include_findings: bool = False,
                 principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    folder = drift.reports_dir()
    latest = folder / "latest.json"                    # the live book; simulated runs are listed below
    if not latest.exists():
        return {"latest": None, "runs": [], "openobserve_url": _o2_url(),
                "detail": "no drift run yet: POST /monitoring/drift/run or scripts/run_drift.py"}
    report = json.loads(latest.read_text(encoding="utf-8"))
    runs = []
    for path in sorted(folder.glob("2*.json"), reverse=True)[:20]:
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        runs.append({"run_id": r.get("run_id"), "generated_at": r.get("generated_at"),
                     "scenario": r.get("scenario"), "status": r.get("status")})
    out = {"latest": report if include_findings else _summary(report), "runs": runs, "openobserve_url": _o2_url()}
    return out


class DriftRun(BaseModel):
    scenario: str = "none"
    ship: bool = True


@router.post("/drift/run")
def run_drift(body: DriftRun, principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    if body.scenario not in drift.SCENARIOS:
        raise HTTPException(422, f"scenario must be one of {', '.join(drift.SCENARIOS)}")
    report = drift.report(body.scenario)
    drift.save(report)
    shipped: dict[str, Any] | str = "not requested"
    if body.ship:
        client = openobserve.Client()
        shipped = openobserve.ship(report, client) if client.health() else f"OpenObserve at {client.url} unreachable"
    audit.record(principal.name or "unknown", "drift_run", "drift_run", report["run_id"],
                 {"scenario": body.scenario, "status": report["status"]})
    return {"report": _summary(report), "shipped": shipped}


@router.get("/alerts")
def alerts(limit: int = 50, principal: Principal = Depends(require_roles(*OPERATORS))) -> dict[str, Any]:
    path = _alerts_file()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    items = [json.loads(line) for line in lines[-max(1, min(limit, 500)):] if line.strip()]
    return {"alerts": list(reversed(items)), "total": len(lines)}


@router.post("/alert-webhook")
async def alert_webhook(request: Request,
                        x_aavas_webhook_token: str | None = Header(default=None)) -> dict[str, Any]:
    openobserve.load_env_file()
    expected = os.environ.get("AAVAS_O2_WEBHOOK_TOKEN", "")
    if not expected:                                   # never accept unauthenticated alerts
        raise HTTPException(503, "alert webhook is not configured (AAVAS_O2_WEBHOOK_TOKEN)")
    if not x_aavas_webhook_token or not hmac.compare_digest(x_aavas_webhook_token, expected):
        raise HTTPException(401, "bad or missing webhook token")
    raw = (await request.body())[:64_000].decode("utf-8", errors="replace")
    try:
        payload: Any = json.loads(raw)
    except ValueError:                                 # the template substitutes rows as text,
        name = re.search(r'"alert"\s*:\s*"([^"]+)"', raw)  # which can break the JSON
        payload = {"raw": raw, "alert": name.group(1) if name else None}
    entry = {"received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "alert": payload.get("alert") if isinstance(payload, dict) else None, "payload": payload}
    path = _alerts_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    audit.record("openobserve", "drift_alert_received", "drift_alert", str(entry["alert"] or "unknown"),
                 {"received_at": entry["received_at"]})
    log.warning("drift alert from OpenObserve: %s", entry["alert"])
    return {"received": True}


def _o2_url() -> str:
    openobserve.load_env_file()
    return os.environ.get("AAVAS_O2_URL", "http://localhost:5080")
