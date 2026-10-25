"""Editing a configuration file from the admin console.

The files in ``config/`` are the product's behaviour, so a change made here is
held to four rules before it reaches the priced path:

1. **It must parse** as a YAML mapping and keep every top-level section the
   file had. Deleting a section is a code change, not a configuration change.
2. **It must carry a new version.** The version a file declares is stamped on
   every decision, so content that changes under an unchanged version would make
   the audit name the wrong configuration. A version string also may never be
   reused for different content.
3. **It must still price.** After the file is written, the caller's ``smoke``
   check prices a sample application with it; if that fails the previous file
   is written back and the change is rejected.
4. **It is recorded.** Who changed what, both versions, both texts and every
   setting that moved go into ``config_changes``; any change can be undone from
   that record, and the undo is itself a recorded change.

The caller supplies ``reload`` (drop every cache that holds configuration) and
``smoke``: both live in the API layer, which owns the orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
from typing import Any, Callable

import yaml

from .. import config
from . import audit, store

_lock = threading.Lock()

# Files the console never writes, and why. Each has a control of its own that a
# direct edit would walk around.
LOCKED = {
    "features": "The feature contract decides what the models may see. A feature is added "
                "through feature admission at Feature Intelligence, signed by model risk and "
                "credit policy, not by editing this file.",
    "governance": "This file says who must approve what. An editor that could rewrite its own "
                  "approval rules would make them no control at all.",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS config_changes (
    change_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    file            TEXT NOT NULL,
    kind            TEXT NOT NULL,
    changed_at      TEXT NOT NULL,
    changed_by      TEXT NOT NULL,
    before_version  TEXT,
    after_version   TEXT,
    before_sha      TEXT NOT NULL,
    after_sha       TEXT NOT NULL,
    before_raw      TEXT NOT NULL,
    after_raw       TEXT NOT NULL,
    changed         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_config_changes_name ON config_changes(name, changed_at);
"""


class ChangeRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------- helpers

def flatten(payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Every leaf as a dotted path; a list of scalars is one leaf."""
    out: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.extend(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(payload, list):
        if all(not isinstance(v, (dict, list)) for v in payload):
            out.append((prefix, payload))
        else:
            for index, value in enumerate(payload):
                out.extend(flatten(value, f"{prefix}[{index}]"))
    else:
        out.append((prefix, payload))
    return out


def diff(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    old, new = dict(flatten(before)), dict(flatten(after))
    rows = []
    for path in sorted(set(old) | set(new)):
        if path not in new:
            rows.append({"path": path, "change": "removed", "before": old[path], "after": None})
        elif path not in old:
            rows.append({"path": path, "change": "added", "before": None, "after": new[path]})
        elif old[path] != new[path]:
            rows.append({"path": path, "change": "changed", "before": old[path], "after": new[path]})
    return rows


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def version_key(payload: dict[str, Any]) -> str | None:
    keys = [k for k in payload if str(k).endswith("_version")]
    return keys[0] if keys else None


def set_version(raw: str, key: str, value: str) -> str:
    """Rewrite the top-level ``key: value`` line, keeping any trailing comment."""
    pattern = re.compile(rf"^({re.escape(key)}\s*:\s*)([^#\n]*?)(\s*#.*)?$", re.MULTILINE)
    if not pattern.search(raw):
        raise ChangeRejected(422, f"the file has no top-level {key} line to set the version on")
    return pattern.sub(lambda m: f'{m.group(1)}"{value}"{m.group(3) or ""}', raw, count=1)


def _ensure_table() -> None:
    with store.connect() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- the checks

def _parse(raw: str) -> dict[str, Any]:
    try:
        proposed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ChangeRejected(422, f"the file is not valid YAML{where}: "
                                  f"{getattr(exc, 'problem', None) or exc}") from exc
    if not isinstance(proposed, dict):
        raise ChangeRejected(422, "the file must be a mapping of sections, as it was")
    return proposed


def check(name: str, raw: str, version: str | None = None) -> dict[str, Any]:
    """Validate a proposed text without writing it. Returns what would change."""
    if name not in config._FILES:
        raise ChangeRejected(404, f"unknown config {name!r}")
    if name in LOCKED:
        raise ChangeRejected(403, f"{config._FILES[name]} cannot be edited here. {LOCKED[name]}")
    path = config.config_path(name)
    current_raw = path.read_text(encoding="utf-8")
    current = yaml.safe_load(current_raw) or {}
    key = version_key(current)
    if key is None:
        raise ChangeRejected(422, f"{path.name} declares no version, so it cannot be edited here")
    proposed = _parse(raw)
    if version is not None and version.strip():
        raw = set_version(raw, key, version.strip())
        proposed = _parse(raw)
    missing = [k for k in current if k not in proposed]
    if missing:
        raise ChangeRejected(422, f"sections cannot be removed here: {', '.join(map(str, missing))}")

    before_version, after_version = str(current.get(key)), str(proposed.get(key))
    if sha(raw) == sha(current_raw):
        raise ChangeRejected(422, "nothing has changed")
    if after_version == before_version:
        raise ChangeRejected(422, f"give this change a new {key}: the version is stamped on every "
                                  f"decision, so it must change when the content does")
    _ensure_table()
    reused = store.query(
        "SELECT change_id FROM config_changes WHERE name = ? AND "
        "((after_version = ? AND after_sha != ?) OR (before_version = ? AND before_sha != ?)) LIMIT 1",
        (name, after_version, sha(raw), after_version, sha(raw)))
    if reused:
        raise ChangeRejected(422, f"version {after_version} has already named different content; "
                                  f"choose another")
    return {"name": name, "file": path.name, "version_key": key,
            "before_version": before_version, "after_version": after_version,
            "raw": raw, "changes": diff(current, proposed)}


# ------------------------------------------------------------------ applying

def apply(name: str, raw: str, *, actor: str, version: str | None = None, kind: str = "edit",
          reload: Callable[[], None], smoke: Callable[[], None]) -> dict[str, Any]:
    with _lock:
        prepared = check(name, raw, version)
        path = config.config_path(name)
        before_raw = path.read_text(encoding="utf-8")
        _write(path, prepared["raw"])
        reload()
        try:
            smoke()
        except Exception as exc:                               # noqa: BLE001
            _write(path, before_raw)
            reload()
            raise ChangeRejected(
                422, f"rolled back: pricing a sample application failed with this change "
                     f"({type(exc).__name__}: {exc})") from exc

        row = {
            "change_id": f"CFG-{secrets.token_hex(6).upper()}", "name": name, "file": path.name,
            "kind": kind, "changed_at": store.now(), "changed_by": actor,
            "before_version": prepared["before_version"], "after_version": prepared["after_version"],
            "before_sha": sha(before_raw), "after_sha": sha(prepared["raw"]),
            "before_raw": before_raw, "after_raw": prepared["raw"],
            "changed": json.dumps(prepared["changes"], default=str),
        }
        store.insert("config_changes", row)
        audit.record(actor, "restored_config" if kind == "restore" else "edited_config", "config",
                     name, {"change_id": row["change_id"], "file": path.name,
                            "before_version": row["before_version"],
                            "after_version": row["after_version"],
                            "settings_changed": len(prepared["changes"])})
        return summary(row)


def _write(path, text: str) -> None:
    partial = path.with_name(f".{path.name}.writing")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def restore(change_id: str, *, actor: str, reload: Callable[[], None],
            smoke: Callable[[], None]) -> dict[str, Any]:
    """Put back the file as it was before ``change_id``."""
    row = change(change_id)
    if row is None:
        raise ChangeRejected(404, f"no change {change_id}")
    return apply(row["name"], row["before_raw"], actor=actor, kind="restore",
                 reload=reload, smoke=smoke)


# ------------------------------------------------------------------- reading

def summary(row: dict[str, Any]) -> dict[str, Any]:
    changed = json.loads(row["changed"]) if isinstance(row["changed"], str) else row["changed"]
    return {k: row[k] for k in ("change_id", "name", "file", "kind", "changed_at", "changed_by",
                                "before_version", "after_version")} | {"changes": changed}


def history(name: str, limit: int = 50) -> list[dict[str, Any]]:
    _ensure_table()
    rows = store.query("SELECT * FROM config_changes WHERE name = ? ORDER BY changed_at DESC "
                       "LIMIT ?", (name, limit))
    return [summary(r) for r in rows]


def change(change_id: str) -> dict[str, Any] | None:
    _ensure_table()
    return store.fetch_one("SELECT * FROM config_changes WHERE change_id = ?", (change_id,))
