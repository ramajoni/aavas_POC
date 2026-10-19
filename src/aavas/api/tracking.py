"""Experiment tracking and the live dashboards, for the admin view.

Two things a model-risk reviewer asks that the rest of the API cannot answer:

* **What has been tried?**  The AutoML search writes one parent run per search
  and one nested run per candidate. That history is the record of what was
  considered and rejected, not only what shipped.
* **Where are the dashboards?**  MLflow's UI and the OmniXAI dashboards are
  separate servers with their own lifecycles. The admin view has to say whether
  they are up, where, and how to start them when they are not.

Two deliberate choices:

* **The run history is read straight off the file store, not through mlflow.**
  The API must not grow a hard dependency on a tracking library - tracking is
  optional everywhere else in this repo and an unstarted server must never break
  a page. The store layout (``meta.yaml`` + ``metrics/`` + ``params/`` +
  ``tags/``) is plain text. A remote ``http://`` store is reported as such and
  the reviewer is sent to the MLflow server itself, which is the thing that
  serves it.
* **Reachability is probed with a TCP connect from the API host.** It says a
  server is listening, not that the browser can reach it - those differ the
  moment the API runs in a container - so the payload labels it as seen from the
  API host and the UI repeats that.
"""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from fastapi import APIRouter

from ..config import env, model_config
from ..logging_utils import get_logger
from ..paths import MLRUNS_DIR, OMNIXAI_DIR, resolve

log = get_logger(__name__)
router = APIRouter(prefix="/tracking", tags=["tracking"])

MLFLOW_UI_DEFAULT = "http://127.0.0.1:5000"
# scripts/run_omnixai.py hands out one port per served model from this one.
OMNIXAI_BASE_PORT = 8765
OMNIXAI_PORT_SPAN = 6
DASHBOARD_MANIFEST = OMNIXAI_DIR / "ui" / "dashboards.json"
PROBE_TIMEOUT = 0.25

# mlflow's RunStatus enum, which the file store writes as an integer.
RUN_STATUS = {1: "running", 2: "scheduled", 3: "finished", 4: "failed", 5: "killed"}

MLFLOW_START = ("docker compose --profile tracking up   # or, without docker: "
                "mlflow ui --backend-store-uri file:./mlruns --port 5000")
OMNIXAI_START = "make xai   # one dashboard process per model, about 1.5 GB each"


# ------------------------------------------------------------------ the store


def _tracking_cfg() -> dict[str, Any]:
    return model_config().get("tracking", {}) or {}


def _uri() -> str:
    return env("MLFLOW_TRACKING_URI") or _tracking_cfg().get("uri", "file:./mlruns")


def _store_dir() -> Path | None:
    """The file store behind the configured URI, or None if it is remote."""
    uri = _uri()
    if uri.startswith(("http://", "https://")):
        return None
    path = uri[len("file:"):] if uri.startswith("file:") else uri
    if path.startswith("//"):                   # file:///abs/path, file://host/abs
        path = "/" + path.lstrip("/")
    directory = Path(path) if path.startswith("/") else resolve(path)
    if directory.exists():
        return directory
    return MLRUNS_DIR if MLRUNS_DIR.exists() else None


def _stamp(ms: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def _read_leaf(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_metric(path: Path) -> float | None:
    """A metric file is one ``timestamp value step`` line per logged step."""
    text = _read_leaf(path)
    if not text:
        return None
    try:
        return float(text.strip().splitlines()[-1].split()[1])
    except (IndexError, ValueError):
        return None


def _read_run(directory: Path) -> dict[str, Any] | None:
    meta_path = directory / "meta.yaml"
    if not meta_path.exists():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:                    # noqa: BLE001
        log.debug("tracking: unreadable run at %s (%s)", directory, exc)
        return None
    if meta.get("lifecycle_stage") == "deleted":
        # mlflow soft-deletes; a run someone removed in the UI must not come
        # back through this door.
        return None

    tags = {p.name: _read_leaf(p) for p in sorted((directory / "tags").glob("*"))
            if p.is_file()}
    metrics = {p.name: _read_metric(p) for p in sorted((directory / "metrics").glob("*"))
               if p.is_file()}
    params = {p.name: _read_leaf(p) for p in sorted((directory / "params").glob("*"))
              if p.is_file()}

    start, end = meta.get("start_time"), meta.get("end_time")
    duration = None
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        duration = round((end - start) / 1000, 2)

    return {
        "run_id": meta.get("run_id") or directory.name,
        "name": meta.get("run_name") or tags.get("mlflow.runName") or directory.name,
        "status": RUN_STATUS.get(meta.get("status"), str(meta.get("status"))),
        "started_at": _stamp(start),
        "duration_sec": duration,
        "parent_run_id": tags.get("mlflow.parentRunId"),
        "user": meta.get("user_id") or tags.get("mlflow.user"),
        # mlflow's own bookkeeping tags say nothing about the experiment.
        "tags": {k: v for k, v in tags.items() if not k.startswith("mlflow.")},
        "metrics": {k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in metrics.items() if v is not None},
        "params": params,
    }


def _experiments() -> list[dict[str, Any]]:
    store = _store_dir()
    if store is None:
        return []
    out = []
    for directory in sorted(store.iterdir()):
        meta_path = directory / "meta.yaml"
        if not directory.is_dir() or directory.name == ".trash" or not meta_path.exists():
            continue
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):                       # noqa: BLE001
            continue
        if meta.get("lifecycle_stage") == "deleted":
            continue
        out.append({"experiment_id": str(meta.get("experiment_id") or directory.name),
                    "name": meta.get("name") or directory.name,
                    "path": directory,
                    "runs": sum(1 for d in directory.iterdir()
                                if d.is_dir() and (d / "meta.yaml").exists())})
    return out


def _runs(limit: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in _experiments():
        for directory in experiment["path"].iterdir():
            if not directory.is_dir():
                continue
            run = _read_run(directory)
            if run:
                run["experiment"] = experiment["name"]
                rows.append(run)
    rows.sort(key=lambda r: r["started_at"] or "", reverse=True)
    return rows[:limit]


def _tree(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parent searches with their candidates nested, which is how they were run."""
    by_id = {r["run_id"]: r for r in rows}
    children: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        parent = row.get("parent_run_id")
        if parent and parent in by_id:
            children.setdefault(parent, []).append(row)
    tops = [r for r in rows if not (r.get("parent_run_id") and r["parent_run_id"] in by_id)]
    for top in tops:
        kids = sorted(children.get(top["run_id"], []), key=lambda r: r["started_at"] or "")
        top["children"] = kids
        top["child_count"] = len(kids)
    return tops


# ---------------------------------------------------------------- the servers


def _listening(url: str, timeout: float = PROBE_TIMEOUT) -> bool:
    parts = urlsplit(url)
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _mlflow_ui_url() -> str:
    configured = env("AAVAS_MLFLOW_UI_URL")
    if configured:
        return configured.rstrip("/")
    uri = _uri()
    return uri.rstrip("/") if uri.startswith(("http://", "https://")) else MLFLOW_UI_DEFAULT


def _omnixai_targets() -> list[dict[str, Any]]:
    """Which dashboard is on which port.

    ``scripts/run_omnixai.py`` writes a manifest when it serves, because the
    dashboards themselves do not advertise which model they hold - without it
    all an admin can see is that *something* is listening on 8766.
    """
    import json

    if DASHBOARD_MANIFEST.exists():
        try:
            with open(DASHBOARD_MANIFEST, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            entries = [{"model": d.get("model"), "url": d.get("url")}
                       for d in manifest.get("dashboards", []) if d.get("url")]
            if entries:
                return entries
        except (OSError, ValueError) as exc:                    # noqa: BLE001
            log.debug("tracking: unreadable dashboard manifest (%s)", exc)
    # No manifest: the ports are still worth probing, just not nameable.
    return [{"model": None, "url": f"http://127.0.0.1:{OMNIXAI_BASE_PORT + i}"}
            for i in range(OMNIXAI_PORT_SPAN)]


def _services() -> list[dict[str, Any]]:
    targets = [{"name": "MLflow", "kind": "mlflow", "model": None,
                "url": _mlflow_ui_url(), "start": MLFLOW_START,
                "what": "Every AutoML search, its candidates, parameters and metrics."}]
    named = False
    for entry in _omnixai_targets():
        named = named or entry["model"] is not None
        label = f"OmniXAI · {entry['model']}" if entry["model"] else f"OmniXAI · {entry['url'].rsplit(':', 1)[-1]}"
        targets.append({"name": label, "kind": "omnixai", "model": entry["model"],
                        "url": entry["url"], "start": OMNIXAI_START,
                        "what": "Data, local, global, prediction and what-if tabs for one model."})

    with ThreadPoolExecutor(max_workers=8) as pool:
        reachable = list(pool.map(lambda t: _listening(t["url"]), targets))
    for target, up in zip(targets, reachable):
        target["reachable"] = up

    if named:
        # The manifest names them, so a dashboard that is *down* is worth showing:
        # "stage2's dashboard is not running" is the answer to the question.
        return targets
    # Without a manifest all that exists is a port range, and an unanswered port
    # is noise rather than a dashboard.
    return [t for t in targets if t["kind"] != "omnixai" or t["reachable"]]


# ----------------------------------------------------------------- endpoints


@router.get("", summary="Tracking store, run counts and dashboard reachability")
def index() -> dict[str, Any]:
    cfg = _tracking_cfg()
    store = _store_dir()
    experiments = _experiments()
    rows = _runs(limit=1000) if store is not None else []
    parents = [r for r in rows if not r.get("parent_run_id")]
    last = max((r["started_at"] for r in rows if r["started_at"]), default=None)

    return {
        "mlflow": {
            "tracking_uri": _uri(),
            "store": str(store) if store else None,
            "store_kind": "file" if store else "remote",
            "writes_enabled": bool(cfg.get("enabled", False)),
            "configured_experiment": cfg.get("experiment", "aavas-pricing-poc"),
            "experiments": [{k: v for k, v in e.items() if k != "path"} for e in experiments],
            "runs": len(rows),
            "searches": len(parents),
            "candidates": len(rows) - len(parents),
            "last_run_at": last,
        },
        "services": _services(),
        "note": ("Reachability is a TCP connect from the API host, so a server behind a "
                 "container boundary can be up and still unreachable from this browser."),
    }


@router.get("/services", summary="Are the dashboards up, and where")
def services() -> dict[str, Any]:
    found = _services()
    return {"services": found,
            "up": sum(1 for s in found if s["reachable"]),
            "checked_from": "the API host"}


@router.get("/runs", summary="AutoML searches with their candidates")
def runs(limit: int = 200, flat: bool = False) -> dict[str, Any]:
    store = _store_dir()
    if store is None:
        return {"available": False,
                "reason": (f"the tracking store is remote ({_uri()}); read its history in the "
                           f"MLflow UI rather than through this API."),
                "searches": []}
    rows = _runs(limit=limit)
    if not rows:
        return {"available": False,
                "reason": (f"no runs under {store}. Enable tracking "
                           f"(AAVAS_TRACKING_ENABLED=1) and run an AutoML search."),
                "searches": []}
    return {"available": True, "store": str(store), "runs": len(rows),
            "searches": rows if flat else _tree(rows)}
