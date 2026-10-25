"""Every configuration the engine reads, in one place.

The repository rule is that nothing in ``src`` hard-codes a rate, spread, cost or
threshold a business user would reasonably want to change - it all lives in
``config/*.yaml``. The consequence is that the six files *are* the product's
behaviour, and until now the only way to see them was to open them.

This serves them with three things a text editor does not give you: the version
each file declares, which environment variables are currently overriding it, and
a search across all of them at once.

**Editable by a signed-in administrator only.** A change applies to the next
priced decision, so it is checked, must carry a new version, is test-priced and
rolled back if pricing fails, and is recorded with both texts so it can be undone
(``governance/config_changes.py``). With authentication disabled nobody can edit:
a configuration change by an unidentified caller is the one thing this must not
allow. Everyone else with access reads.

Secrets are never served. ``ANTHROPIC_API_KEY`` is reported as set or unset and
never carries a value.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .. import config as config_module
from ..config import _FILES, load
from ..governance import config_changes
from ..logging_utils import get_logger
from . import security
from .security import Principal, current_principal

log = get_logger(__name__)
router = APIRouter(prefix="/config", tags=["configuration"])

# What each file governs, in the words of its own header.
DESCRIPTIONS = {
    "policy": "Credit policy: card rates by grade, the permissible band, deviation and "
              "override rules, approval authority and the fair-pricing review threshold.",
    "model": "Per-model learner and hyper-parameters, the data split, the LGD table the "
             "risk model prices with, and the tracking connection.",
    "features": "The feature contract per model - which columns each may see, the monotone "
                "constraints, and the human-authored reason label for every feature.",
    "acceptance": "Loan-acceptance scoring: which data sources are enabled, the signal "
                  "vocabulary, evidence reliability tiers and the governance rules.",
    "automl": "The AutoML search: candidate scope per contract, HPO strategy and budget, "
              "calibration and ensemble methods, ranking weights and the hard gates.",
    "federated": "Federated learning: rounds, clients, aggregation strategy and the "
                 "regional personalisation settings.",
    "connectors": "External systems - WhatsApp, Gmail, Outlook, Salesforce, Oracle Fusion - "
                  "which layer each one feeds, its MCP transport, and the consent it needs.",
    "governance": "Who may approve a change to the system itself, which gates must pass "
                  "first, and whether a failed gate can be overridden and by whom.",
    "competitors": "The competitor registry: canonical lenders, the aliases branches write "
                   "them under, which bare names are ambiguous, and the fuzzy-match bound.",
    "intake": "Online applications: the document types a borrower is asked for, the file "
              "formats accepted (judged by contents), and the size and count limits.",
}

# The environment variables the code actually reads. Declared rather than
# scanned: a variable nobody reads is not configuration, and a reader that
# guessed from prefixes would report both.
ENV_VARS = [
    {"name": "AAVAS_STORE_URL", "controls": "Governance store DSN (sqlite or postgres)."},
    {"name": "AAVAS_PG_DSN", "controls": "Postgres/pgvector DSN for the knowledge store."},
    {"name": "AAVAS_VECTOR_BACKEND", "controls": "Vector backend; falls back to in-memory TF-IDF."},
    {"name": "AAVAS_SEED", "controls": "Global RNG seed. Overrides the config value."},
    {"name": "AAVAS_LOG_LEVEL", "controls": "Log level for every aavas logger."},
    {"name": "AAVAS_TRACKING_ENABLED", "controls": "Turns MLflow logging on. Off by default."},
    {"name": "AAVAS_LLM_ENABLED", "controls": "Allows the acceptance agents to call Claude."},
    {"name": "AAVAS_LLM_MODEL", "controls": "Which Claude model the acceptance agents use."},
    {"name": "AAVAS_MLFLOW_UI_URL", "controls": "Where the admin view links the MLflow UI."},
    {"name": "MLFLOW_TRACKING_URI", "controls": "Tracking store. Overrides model_config.tracking.uri."},
    {"name": "ANTHROPIC_API_KEY", "controls": "Claude credential.", "secret": True},
    {"name": "AAVAS_AUTH_MODE", "controls": "session (default, sign-in required), apikey or disabled."},
    {"name": "AAVAS_SESSION_SECRET", "controls": "Signs session tokens; unset, restarts sign everyone out.",
     "secret": True},
    {"name": "AAVAS_ACCESS_FILE", "controls": "Users, roles and route rules. Default config/access.yaml."},
    {"name": "AAVAS_UPLOAD_DIR", "controls": "Where borrower documents are stored. Default data/uploads."},
]

VERSION_KEYS = ("policy_version", "model_config_version", "feature_config_version",
                "acceptance_config_version", "automl_config_version",
                "federated_config_version", "connectors_config_version",
                "governance_config_version", "competitors_config_version",
                "intake_config_version")


def _path(name: str):
    return config_module.config_path(name)


def _version(payload: dict[str, Any]) -> str | None:
    for key in VERSION_KEYS:
        if key in payload:
            return str(payload[key])
    return None


_flatten = config_changes.flatten


def _env_state() -> list[dict[str, Any]]:
    import os

    rows = []
    for var in ENV_VARS:
        raw = os.environ.get(var["name"])
        rows.append({
            "name": var["name"], "controls": var["controls"],
            "set": raw is not None,
            # A secret is reported as present, never as a value. The key pasted
            # into this session is treated as burned; it must not be echoed by
            # anything, least of all an endpoint a browser calls.
            "value": None if var.get("secret") or raw is None else raw,
            "secret": bool(var.get("secret")),
        })
    return rows


def _file_meta(name: str) -> dict[str, Any]:
    path = _path(name)
    payload = load(name)
    stat = path.stat()
    leaves = _flatten(payload)
    return {
        "name": name,
        "file": path.name,
        "description": DESCRIPTIONS.get(name, ""),
        "version": _version(payload),
        "sections": sorted(k for k in payload if not k.endswith("_version")),
        "settings": len(leaves),
        "lines": sum(1 for _ in path.open("r", encoding="utf-8")),
        "bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                            .isoformat(timespec="seconds"),
        "locked": config_changes.LOCKED.get(name),
    }


def can_edit(principal: Principal) -> bool:
    """A signed-in administrator, or the technology role under API keys. Never an
    unauthenticated caller, whatever roles the disabled mode hands it."""
    if not principal.authenticated:
        return False
    if principal.mode == security.SESSION:
        return principal.has("administrator")
    return principal.has("technology")


def editor(principal: Principal = Depends(current_principal)) -> Principal:
    if not can_edit(principal):
        raise HTTPException(403, "only a signed-in administrator can change configuration")
    return principal


def _reload() -> None:
    """Drop every cache that holds configuration or something built from it."""
    from ..intelligence import registry
    from . import deps

    config_module.reload()
    registry.reset()
    deps.reset()


def _smoke() -> None:
    """Price a sample application end to end with the configuration as written."""
    from .contracts import Stage1Request, Stage2Request
    from .deps import get_orchestrator
    from .examples import FINAL_ASSESSMENT, PRE_ASSESSMENT

    for name in _FILES:
        load(name)
    orchestrator = get_orchestrator()
    orchestrator.pre_assessment(Stage1Request(**{**PRE_ASSESSMENT, "loan_id": "CONFIG-CHECK"}).to_row(),
                                persist=False)
    result = orchestrator.process(Stage2Request(**{**FINAL_ASSESSMENT, "loan_id": "CONFIG-CHECK"}).to_row(),
                                  persist=False)
    if not (result.get("recommendation") or {}).get("recommended_rate"):
        raise ValueError("the sample application produced no recommended rate")
    config_module.versions()


class ConfigEdit(BaseModel):
    raw: str = Field(..., min_length=1, max_length=500_000)
    version: str | None = Field(None, max_length=64,
                                description="The new version; replaces the file's version line")


def _rejected(exc: config_changes.ChangeRejected) -> HTTPException:
    return HTTPException(exc.status, str(exc))


@router.get("", summary="Every configuration file the engine reads")
def index(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    files = [_file_meta(name) for name in sorted(_FILES)]
    env = _env_state()
    editable = can_edit(principal)
    return {
        "files": files,
        "settings": sum(f["settings"] for f in files),
        "environment": env,
        "overrides_active": [e["name"] for e in env if e["set"]],
        "editable": editable,
        "note": ("Editable by you. A change applies to the next priced decision: it must parse, "
                 "keep every section and carry a new version; a sample application is priced "
                 "with it and the change is rolled back if that fails; every change is recorded "
                 "and can be undone." if editable else
                 "Read-only for your role. Only a signed-in administrator can change configuration, "
                 "and every change is checked, test-priced and recorded."),
    }


@router.get("/search", summary="Find a setting across every config file")
def search(q: str, limit: int = 100) -> dict[str, Any]:
    # Registered before /{name}: 'search' is a single segment and would
    # otherwise be read as a config name.
    needle = q.strip().lower()
    if not needle:
        return {"query": q, "matches": [], "note": "empty query"}

    matches = []
    for name in sorted(_FILES):
        for path, value in _flatten(load(name)):
            if needle in path.lower() or needle in str(value).lower():
                matches.append({"config": name, "file": _FILES[name],
                                "path": path, "value": value})
                if len(matches) >= limit:
                    break
    return {"query": q, "matches": matches, "truncated": len(matches) >= limit}


@router.get("/{name}", summary="One configuration file, parsed and raw")
def detail(name: str, raw: bool = True) -> dict[str, Any]:
    if name not in _FILES:
        raise HTTPException(404, f"unknown config {name!r}; expected one of {sorted(_FILES)}")
    payload = load(name)
    meta = _file_meta(name)
    body = {**meta, "content": payload,
            "flat": [{"path": p, "value": v} for p, v in _flatten(payload)]}
    if raw:
        # The comments are the reasoning. A parsed view drops every one of them,
        # and in these files the comment is often the control - "never learn
        # that more expensive converts better" is not expressible in YAML.
        body["raw"] = _path(name).read_text(encoding="utf-8")
    return body


@router.post("/{name}/preview", summary="Check a proposed edit without applying it")
def preview(name: str, edit: ConfigEdit, principal: Principal = Depends(editor)) -> dict[str, Any]:
    try:
        return config_changes.check(name, edit.raw, edit.version)
    except config_changes.ChangeRejected as exc:
        raise _rejected(exc) from exc


@router.put("/{name}", summary="Apply an edit to one configuration file")
def update(name: str, edit: ConfigEdit, principal: Principal = Depends(editor)) -> dict[str, Any]:
    try:
        return config_changes.apply(name, edit.raw, version=edit.version, actor=principal.name,
                                    reload=_reload, smoke=_smoke)
    except config_changes.ChangeRejected as exc:
        raise _rejected(exc) from exc


@router.get("/{name}/history", summary="Changes made to one configuration file")
def history(name: str) -> dict[str, Any]:
    if name not in _FILES:
        raise HTTPException(404, f"unknown config {name!r}; expected one of {sorted(_FILES)}")
    return {"name": name, "changes": config_changes.history(name)}


@router.post("/{name}/history/{change_id}/restore",
             summary="Put a file back as it was before one change")
def restore(name: str, change_id: str, principal: Principal = Depends(editor)) -> dict[str, Any]:
    row = config_changes.change(change_id)
    if row is None or row["name"] != name:
        raise HTTPException(404, f"no change {change_id} on {name}")
    try:
        return config_changes.restore(change_id, actor=principal.name, reload=_reload, smoke=_smoke)
    except config_changes.ChangeRejected as exc:
        raise _rejected(exc) from exc


@router.get("/{name}/raw", summary="One configuration file as written, comments included")
def raw(name: str) -> dict[str, Any]:
    if name not in _FILES:
        raise HTTPException(404, f"unknown config {name!r}; expected one of {sorted(_FILES)}")
    return {"name": name, "file": _FILES[name],
            "raw": _path(name).read_text(encoding="utf-8")}

