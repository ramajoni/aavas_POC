"""Feature candidates put forward for admission, and the record of what happened.

A proposal is not an admission. The contract in ``feature_config.yaml`` is widened
only by an approved change; this table records that a candidate was tested, by
whom, against which dataset, and what the tests said - so a feature that *is*
eventually admitted arrives with its evidence rather than with an assertion.

It also records the ones that were declined, which is the more useful half. The
interesting output of feature discovery is rarely the feature that got in; it is
the list of strong, stable, independent candidates that nobody could write an
honest sentence about.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..logging_utils import get_logger
from . import audit, store

log = get_logger(__name__)

PROPOSED, ADMITTED, DECLINED = "proposed", "admitted", "declined"


def propose(candidate: dict[str, Any], stage: str, proposed_by: str,
            dataset_id: str | None = None) -> dict[str, Any]:
    """Record a tested candidate. Refusals are stored with it, not filtered out."""
    proposal_id = f"FEAT-{uuid.uuid4().hex[:10].upper()}"
    store.insert("feature_proposals", {
        "proposal_id": proposal_id, "name": candidate["name"], "stage": stage,
        "recipe": candidate["recipe"], "inputs": candidate["inputs"],
        "generator": candidate.get("generator"), "proposed_at": store.now(),
        "proposed_by": proposed_by, "dataset_id": dataset_id,
        "tests": candidate.get("tests") or {},
        "refusals": [r["code"] for r in candidate.get("refusals") or []],
        "state": PROPOSED,
    })
    audit.record(proposed_by, "propose_feature", "feature", candidate["name"],
                 {"proposal_id": proposal_id, "stage": stage,
                  "refusals": [r["code"] for r in candidate.get("refusals") or []]})
    log.info("feature %s proposed for %s (%s)", candidate["name"], stage, proposal_id)
    return get(proposal_id)


def get(proposal_id: str) -> dict[str, Any] | None:
    row = store.fetch_one("SELECT * FROM feature_proposals WHERE proposal_id = ?",
                          (proposal_id,))
    return _hydrate(row) if row else None


def by_name(name: str, stage: str | None = None) -> dict[str, Any] | None:
    if stage:
        row = store.fetch_one(
            "SELECT * FROM feature_proposals WHERE name = ? AND stage = ? "
            "ORDER BY proposed_at DESC LIMIT 1", (name, stage))
    else:
        row = store.fetch_one(
            "SELECT * FROM feature_proposals WHERE name = ? ORDER BY proposed_at DESC LIMIT 1",
            (name,))
    return _hydrate(row) if row else None


def _hydrate(row: dict[str, Any]) -> dict[str, Any]:
    row["inputs"] = store.decode(row.get("inputs")) or []
    row["tests"] = store.decode(row.get("tests")) or {}
    row["refusals"] = store.decode(row.get("refusals")) or []
    return row


def listing(state: str | None = None, stage: str | None = None,
            limit: int = 200) -> list[dict[str, Any]]:
    clauses, params = [], []
    if state:
        clauses.append("state = ?")
        params.append(state)
    if stage:
        clauses.append("stage = ?")
        params.append(stage)
    where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
    params.append(limit)
    rows = store.query(f"SELECT * FROM feature_proposals {where}ORDER BY proposed_at DESC "
                       f"LIMIT ?", tuple(params))
    return [_hydrate(row) for row in rows]


def set_state(proposal_id: str, state: str, actor: str, note: str = "") -> dict[str, Any] | None:
    if state not in (PROPOSED, ADMITTED, DECLINED):
        raise ValueError(f"unknown state {state!r}")
    with store.connect() as conn:
        conn.execute("UPDATE feature_proposals SET state = ? WHERE proposal_id = ?",
                     (state, proposal_id))
    audit.record(actor, f"feature_{state}", "feature", proposal_id, {"note": note})
    return get(proposal_id)


def summary() -> dict[str, Any]:
    rows = listing(limit=1000)
    by_state: dict[str, int] = {}
    blocked_on_label = 0
    for row in rows:
        by_state[row["state"]] = by_state.get(row["state"], 0) + 1
        if row["refusals"] == ["no_reason_label"]:
            blocked_on_label += 1
    return {
        "proposals": len(rows), "by_state": by_state,
        "blocked_only_on_a_reason_label": blocked_on_label,
        "note": ("A proposal is evidence that a candidate was tested, not permission to use "
                 "it. Admission is an approved change to feature_config.yaml."),
    }
