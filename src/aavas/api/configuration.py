"""Every configuration the engine reads, in one place.

The repository rule is that nothing in ``src`` hard-codes a rate, spread, cost or
threshold a business user would reasonably want to change - it all lives in
``config/*.yaml``. The consequence is that the six files *are* the product's
behaviour, and until now the only way to see them was to open them.

This serves them read-only, with three things a text editor does not give you:
the version each file declares, which environment variables are currently
overriding it, and a search across all of them at once.

**Read-only on purpose.** Editing these values from a browser would change the
priced path - authority limits, rate floors, fairness gates - with no approval
path behind it, and this repo has no such path yet (see
docs/agentic-os-flow-gap.md section 2). What the endpoint does instead is show
exactly what is in force, which is the part a reviewer needs.

Secrets are never served. ``ANTHROPIC_API_KEY`` is reported as set or unset and
never carries a value.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import _FILES, load
from ..logging_utils import get_logger
from ..paths import CONFIG_DIR

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
]

VERSION_KEYS = ("policy_version", "model_config_version", "feature_config_version",
                "acceptance_config_version", "automl_config_version",
                "federated_config_version", "connectors_config_version",
                "governance_config_version")


def _path(name: str):
    return CONFIG_DIR / _FILES[name]


def _version(payload: dict[str, Any]) -> str | None:
    for key in VERSION_KEYS:
        if key in payload:
            return str(payload[key])
    return None


def _flatten(payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Every leaf as a dotted path, which is what makes the search useful."""
    out: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.extend(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(payload, list):
        scalars = all(not isinstance(v, (dict, list)) for v in payload)
        if scalars:
            out.append((prefix, payload))
        else:
            for index, value in enumerate(payload):
                out.extend(_flatten(value, f"{prefix}[{index}]"))
    else:
        out.append((prefix, payload))
    return out


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
    }


@router.get("", summary="Every configuration file the engine reads")
def index() -> dict[str, Any]:
    files = [_file_meta(name) for name in sorted(_FILES)]
    env = _env_state()
    return {
        "files": files,
        "settings": sum(f["settings"] for f in files),
        "environment": env,
        "overrides_active": [e["name"] for e in env if e["set"]],
        "editable": False,
        "note": ("Read-only. Changing these values changes the priced path - authority "
                 "limits, rate floors, fairness gates - and this repo has no approval path "
                 "for a config change yet. Edit the file, restart the API, and the version "
                 "each file declares is what lands on every decision."),
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


@router.get("/{name}/raw", summary="One configuration file as written, comments included")
def raw(name: str) -> dict[str, Any]:
    if name not in _FILES:
        raise HTTPException(404, f"unknown config {name!r}; expected one of {sorted(_FILES)}")
    return {"name": name, "file": _FILES[name],
            "raw": _path(name).read_text(encoding="utf-8")}

