"""The agent set as a registered object, not only as code.

The agent, skill and tool declarations describe the system as it is now. That is
enough to render a control plane and not enough to audit one: asked what an agent
looked like at the time of a decision, a code declaration returns today's answer
with no way to know it differs. This module closes that gap by snapshotting the
declaration into the governance store and stamping the snapshot's id onto every
decision.

**The id is derived, not declared.** It is a content hash over the agent, skill
and tool declarations plus the pipeline order - so it cannot drift from the code
it names. A hand-bumped version number is a promise someone has to keep on every
edit, and the one edit where they forget is the one that matters: two different
rosters would then share a version, and the audit would say they were the same.
A hash simply changes.

What the manifest deliberately does **not** cover is config. Policy, feature and
model-config versions are stamped separately on the same decision, because "which
roster ran" and "under which rules" are two questions and folding them together
makes a config edit look like an agent change.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from ..logging_utils import get_logger

log = get_logger(__name__)

PREFIX = "ASET"


def _declaration() -> dict[str, Any]:
    """Everything about the agent set that changes how a decision is produced."""
    from ..agents import capabilities, roster

    return {
        "pipeline": [dict(step) for step in roster.PIPELINE],
        "agents": [
            {"name": a["name"], "module": a["module"], "cls": a["cls"], "wired": a["wired"],
             "stages": list(a["stages"]), "models": list(a["models"]),
             "tools": list(a["tools"]), "skills": list(a["skills"])}
            for a in roster.AGENTS
        ],
        "skills": [{"name": s["name"], "module": s["module"], "governed_by": s["governed_by"]}
                   for s in capabilities.SKILLS],
        "tools": [{"name": t["name"], "kind": t["kind"], "module": t["module"]}
                  for t in capabilities.TOOLS],
    }


def _digest(declaration: dict[str, Any]) -> str:
    # sort_keys so the hash depends on content, not on dict insertion order.
    blob = json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


@lru_cache(maxsize=1)
def manifest() -> dict[str, Any]:
    """The current agent set, with the id derived from its content."""
    declaration = _declaration()
    digest = _digest(declaration)
    period = datetime.now(timezone.utc).strftime("%Y.%m")
    return {
        "manifest_id": f"{PREFIX}-{period}-{digest[:8]}",
        "content_hash": digest,
        "agents": len(declaration["agents"]),
        "wired": sum(1 for a in declaration["agents"] if a["wired"]),
        "skills": len(declaration["skills"]),
        "tools": len(declaration["tools"]),
        "pipeline_steps": len(declaration["pipeline"]),
        "declaration": declaration,
    }


def current_id() -> str:
    return str(manifest()["manifest_id"])


def clear_cache() -> None:
    manifest.cache_clear()
    _published.clear()


_published: set[str] = set()


def publish(actor: str = "startup") -> str:
    """Record the current manifest if it is not already on file.

    Idempotent and memoised per process: a manifest id stamped on a decision that
    is not in the store would be a dangling reference, so this runs before the
    first decision rather than on a schedule.
    """
    from . import store

    current = manifest()
    manifest_id = current["manifest_id"]
    if manifest_id in _published:
        return manifest_id

    existing = store.fetch_one(
        "SELECT manifest_id FROM agent_manifests WHERE manifest_id = ?", (manifest_id,))
    if not existing:
        store.insert("agent_manifests", {
            "manifest_id": manifest_id,
            "content_hash": current["content_hash"],
            "created_at": store.now(),
            "agents": current["agents"],
            "wired": current["wired"],
            "skills": current["skills"],
            "tools": current["tools"],
            "manifest": current["declaration"],
        })
        from . import audit
        audit.record(actor, "publish_agent_manifest", "agent_set", manifest_id,
                     {"content_hash": current["content_hash"], "agents": current["agents"]})
        log.info("registered agent manifest %s (%s agents, %s skills, %s tools)",
                 manifest_id, current["agents"], current["skills"], current["tools"])
    _published.add(manifest_id)
    return manifest_id


def get(manifest_id: str) -> dict[str, Any] | None:
    """One manifest as it was recorded. This is what makes the agent set
    queryable as of a past moment rather than as of now."""
    from . import store

    row = store.fetch_one(
        "SELECT * FROM agent_manifests WHERE manifest_id = ?", (manifest_id,))
    if row:
        row["manifest"] = store.decode(row.get("manifest"))
    return row


def registered(limit: int = 50) -> list[dict[str, Any]]:
    from . import store

    rows = store.query(
        "SELECT manifest_id, content_hash, created_at, agents, wired, skills, tools "
        "FROM agent_manifests ORDER BY created_at DESC LIMIT ?", (limit,))
    current = current_id()
    for row in rows:
        row["is_current"] = row["manifest_id"] == current
    return rows


def diff(left_id: str, right_id: str) -> dict[str, Any]:
    """What changed between two agent sets.

    An audit that can only say "the roster differs" has not explained anything.
    """
    left, right = get(left_id), get(right_id)
    if not left or not right:
        missing = [i for i, m in ((left_id, left), (right_id, right)) if not m]
        raise KeyError(f"not on file: {', '.join(missing)}")

    def names(manifest_row: dict[str, Any], key: str) -> set[str]:
        return {item["name"] for item in (manifest_row["manifest"] or {}).get(key, [])}

    out: dict[str, Any] = {"from": left_id, "to": right_id, "changed": {}}
    for key in ("agents", "skills", "tools"):
        before, after = names(left, key), names(right, key)
        if before != after:
            out["changed"][key] = {"added": sorted(after - before),
                                   "removed": sorted(before - after)}

    before_wired = {a["name"] for a in (left["manifest"] or {}).get("agents", []) if a["wired"]}
    after_wired = {a["name"] for a in (right["manifest"] or {}).get("agents", []) if a["wired"]}
    if before_wired != after_wired:
        out["changed"]["wired"] = {"wired_in": sorted(after_wired - before_wired),
                                   "made_dormant": sorted(before_wired - after_wired)}
    out["identical"] = not out["changed"]
    return out
