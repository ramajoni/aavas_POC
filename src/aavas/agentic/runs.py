"""Run records: one JSON file and one workspace per run, under reports/agentic_runs/.

Only one run executes at a time: two runs training into the registry together would
leave it holding a mix of both.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..paths import REPORTS_DIR

RUNS_DIR = REPORTS_DIR / "agentic_runs"
_LOCK = threading.Lock()
_ACTIVE: dict[str, threading.Thread] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    def __init__(self, record: dict[str, Any]):
        self.record = record

    # ------------------------------------------------------------- identity
    @property
    def run_id(self) -> str:
        return self.record["run_id"]

    @property
    def workspace(self) -> Path:
        return RUNS_DIR / self.run_id

    @property
    def config(self) -> dict[str, Any]:
        return self.record["config"]

    # ------------------------------------------------------------- persistence
    def save(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        path = self.workspace / "run.json"
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps(self.record, indent=1, default=str), encoding="utf-8")
        partial.replace(path)

    def log(self, message: str) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        with (self.workspace / "run.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{now()}  {message.rstrip()}\n")

    def log_tail(self, lines: int = 200) -> list[str]:
        path = self.workspace / "run.log"
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]

    # ------------------------------------------------------------- steps
    def step(self, name: str) -> dict[str, Any]:
        return next(s for s in self.record["steps"] if s["name"] == name)

    def start_step(self, name: str) -> None:
        s = self.step(name)
        s.update(status="running", started_at=now())
        self.log(f"[{name}] started")
        self.save()

    def finish_step(self, name: str, summary: str, details: dict[str, Any] | None = None,
                    status: str = "succeeded") -> None:
        s = self.step(name)
        s.update(status=status, finished_at=now(), summary=summary, details=details or {})
        self.log(f"[{name}] {status}: {summary}")
        self.save()

    def fail_step(self, name: str, error: str, details: dict[str, Any] | None = None) -> None:
        s = self.step(name)
        s.update(status="failed", finished_at=now(), summary=error, details=details or {})
        self.log(f"[{name}] FAILED: {error}")
        self.record["status"] = "failed"
        self.record["finished_at"] = now()
        self.save()

    def as_dict(self, with_log: bool = False) -> dict[str, Any]:
        out = dict(self.record)
        if with_log:
            out["log"] = self.log_tail()
        return out


def create(config: dict[str, Any], started_by: str) -> Run:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    record = {
        "run_id": run_id, "status": "queued", "started_by": started_by, "created_at": now(),
        "finished_at": None, "config_version": config.get("agentic_os_config_version"),
        "engine": config["training"]["engine"], "serving_mode": config["serving"]["mode"],
        "config": config,
        "steps": [{"name": n, "status": "pending", "started_at": None, "finished_at": None,
                   "summary": "", "details": {}} for n in config["workflow"]],
        "promotion": None,
    }
    run = Run(record)
    run.save()
    return run


def get(run_id: str) -> Run | None:
    if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", run_id or ""):    # an id, never a path
        return None
    path = RUNS_DIR / run_id / "run.json"
    if not path.exists():
        return None
    return Run(json.loads(path.read_text(encoding="utf-8")))


def all_runs(limit: int = 50) -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    records = []
    for path in sorted(RUNS_DIR.glob("*/run.json"), reverse=True)[:limit]:
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        r.pop("config", None)
        records.append(r)
    return records


def active() -> str | None:
    with _LOCK:
        for run_id, thread in list(_ACTIVE.items()):
            if not thread.is_alive():
                _ACTIVE.pop(run_id)
        return next(iter(_ACTIVE), None)


def launch(run: Run, target) -> bool:
    """Start ``target(run)`` in the background; False when another run is executing."""
    with _LOCK:
        for run_id, thread in list(_ACTIVE.items()):
            if thread.is_alive():
                return False
            _ACTIVE.pop(run_id)
        thread = threading.Thread(target=target, args=(run,), name=f"agentic-{run.run_id}", daemon=True)
        _ACTIVE[run.run_id] = thread
        thread.start()
        return True
