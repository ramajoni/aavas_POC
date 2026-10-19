"""Monitoring passes that record having run.

Drift computed when somebody opens a page is a report. A monitor is something
that runs on a cadence, records that it ran, and says what it found - including
when it found nothing, because a gap in the run history is itself a finding.

Each pass writes a row before it starts and closes it however it ends, for the
same reason the request log does: a monitor that only writes on success cannot
record the run that failed.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from ..logging_utils import get_logger
from . import audit, store

log = get_logger(__name__)

MONITORS = {
    "drift": "Input drift (PSI) and conversion drift against the training window.",
    "retraining": "Whether the evidence justifies retraining: drift, conversion shift, "
                  "decisions accumulated, champion age.",
    "override_review": "Whether human overrides created or destroyed value, by reason.",
    "fairness": "Group fairness of each registered champion, re-read from its card.",
}


def new_run_id() -> str:
    return f"MON-{uuid.uuid4().hex[:10].upper()}"


@contextmanager
def recorded(monitor: str, actor: str = "scheduler") -> Iterator[dict[str, Any]]:
    """Open a run row, hand back a mutable result, close it whatever happens."""
    if monitor not in MONITORS:
        raise ValueError(f"unknown monitor {monitor!r}; expected one of {sorted(MONITORS)}")
    run_id = new_run_id()
    started = time.perf_counter()
    store.insert("monitor_runs", {
        "run_id": run_id, "monitor": monitor, "started_at": store.now(),
        "status": "running", "actor": actor,
    })
    result: dict[str, Any] = {"run_id": run_id, "status": "ok", "triggered": False,
                              "findings": {}}
    try:
        yield result
    except Exception as exc:                                    # noqa: BLE001
        result["status"] = "failed"
        result["findings"] = {"error": f"{type(exc).__name__}: {exc}"}
        _close(run_id, result, started)
        raise
    _close(run_id, result, started)
    audit.record(actor, f"monitor_{monitor}", "monitor", run_id,
                 {"status": result["status"], "triggered": result["triggered"]})


def _close(run_id: str, result: dict[str, Any], started: float) -> None:
    with store.connect() as conn:
        conn.execute(
            "UPDATE monitor_runs SET finished_at = ?, status = ?, triggered = ?, findings = ? "
            "WHERE run_id = ?",
            (store.now(), result["status"], int(bool(result["triggered"])),
             store._encode(result["findings"]), run_id))
    log.info("monitor run %s: %s (%.0f ms)", run_id, result["status"],
             (time.perf_counter() - started) * 1000)


# ------------------------------------------------------------------ the passes


def run_drift(actor: str = "scheduler") -> dict[str, Any]:
    from ..analytics import performance

    with recorded("drift", actor) as result:
        performance.clear_cache()
        report = performance.drift()
        result["status"] = report["overall_status"]
        result["triggered"] = report["overall_status"] in ("warn", "alert")
        result["findings"] = {
            "overall": report["overall_status"],
            "worst_feature": report.get("worst_feature"),
            "conversion": report["conversion_drift"].get("status"),
            "max_shift_pts": report["conversion_drift"].get("max_shift_pts"),
        }
    return result


def run_fairness(actor: str = "scheduler") -> dict[str, Any]:
    """Re-read fairness off every registered champion's card.

    Cheap, and the one monitor that would have caught the finding that the models
    in production had never been fairness-evaluated at all.
    """
    from . import model_registry

    with recorded("fairness", actor) as result:
        failing, unmeasured, passing = [], [], []
        for row in model_registry.registered():
            if row["role"] != "champion":
                continue
            card = store.decode(row.get("card")) or {}
            fairness = card.get("fairness")
            label = f"{row['name']}:{row['version']}"
            if fairness is None or fairness.get("passed") is None:
                unmeasured.append(label)
            elif fairness.get("passed"):
                passing.append(label)
            else:
                failing.append({"model": label,
                                "reasons": fairness.get("reasons"),
                                "gaps": fairness.get("gaps", {}).get("calibration_gap")})
        result["status"] = "alert" if failing else "warn" if unmeasured else "ok"
        result["triggered"] = bool(failing)
        result["findings"] = {"failing": failing, "unmeasured": unmeasured, "passing": passing}
    return result


def run_retraining(actor: str = "scheduler") -> dict[str, Any]:
    from ..analytics.portfolio import frame
    from ..learning import retraining

    with recorded("retraining", actor) as result:
        df = frame().sort_values("application_date")
        cut = int(len(df) * 0.75)
        decision = retraining.assess(df.iloc[:cut], df.iloc[cut:])
        result["status"] = "warn" if decision.should_retrain else "ok"
        result["triggered"] = bool(decision.should_retrain)
        result["findings"] = decision.as_dict()
    return result


def run_all(actor: str = "scheduler") -> dict[str, Any]:
    out = {}
    for name, fn in (("drift", run_drift), ("fairness", run_fairness),
                     ("retraining", run_retraining)):
        try:
            out[name] = fn(actor)
        except Exception as exc:                                # noqa: BLE001
            log.exception("monitor %s failed", name)
            out[name] = {"status": "failed", "findings": {"error": str(exc)}}
    return out


def history(monitor: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    if monitor:
        rows = store.query("SELECT * FROM monitor_runs WHERE monitor = ? ORDER BY started_at "
                           "DESC LIMIT ?", (monitor, limit))
    else:
        rows = store.query("SELECT * FROM monitor_runs ORDER BY started_at DESC LIMIT ?", (limit,))
    for row in rows:
        row["findings"] = store.decode(row.get("findings"))
        row["triggered"] = bool(row.get("triggered"))
    return rows


def status() -> dict[str, Any]:
    """Last run per monitor, and - the point - which have never run at all."""
    out = {}
    for name, description in MONITORS.items():
        rows = history(name, limit=1)
        last = rows[0] if rows else None
        out[name] = {
            "description": description,
            "last_run": last["started_at"] if last else None,
            "last_status": last["status"] if last else None,
            "triggered": last["triggered"] if last else None,
            "never_run": last is None,
        }
    return {"monitors": out,
            "never_run": [n for n, v in out.items() if v["never_run"]],
            "note": ("A monitor with no run history has never run. Nothing here is on a "
                     "schedule yet - these are recorded passes, invoked by "
                     "scripts/run_monitors.py or POST /governance/monitors/run, which is "
                     "the difference between a monitor and a report.")}
