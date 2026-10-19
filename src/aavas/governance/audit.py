"""Append-only audit trail.

Every recommendation, override, approval and outcome update writes an entry. The
business statement makes this non-negotiable: interest-rate decisions affect
customers, so who decided what, on what basis, and under which model and policy
version, has to survive the conversation.
"""

from __future__ import annotations

from typing import Any

from ..config import versions
from ..logging_utils import get_logger
from . import store

log = get_logger(__name__)


def record(actor: str, action: str, entity_type: str, entity_id: str,
           detail: dict[str, Any] | None = None) -> None:
    payload = dict(detail or {})
    payload.setdefault("versions", versions())
    store.insert("audit_log", {
        "created_at": store.now(), "actor": actor, "action": action,
        "entity_type": entity_type, "entity_id": entity_id, "detail": payload,
    })


def trail(entity_type: str | None = None, entity_id: str | None = None, limit: int = 100) -> list[dict]:
    sql = "SELECT * FROM audit_log"
    params: list[Any] = []
    clauses = []
    if entity_type:
        clauses.append("entity_type = ?")
        params.append(entity_type)
    if entity_id:
        clauses.append("entity_id = ?")
        params.append(entity_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY entry_id DESC LIMIT ?"
    params.append(limit)

    rows = store.query(sql, tuple(params))
    for row in rows:
        row["detail"] = store.decode(row.get("detail"))
    return rows


def loan_history(loan_id: str) -> dict[str, Any]:
    """Everything that happened to one file, in order - the view a reviewer or
    an auditor actually asks for."""
    decisions = store.query(
        "SELECT * FROM decisions WHERE loan_id = ? ORDER BY created_at", (loan_id,))
    overrides = store.query(
        "SELECT * FROM overrides WHERE loan_id = ? ORDER BY created_at", (loan_id,))
    outcome = store.fetch_one("SELECT * FROM outcomes WHERE loan_id = ?", (loan_id,))

    for row in decisions:
        row["payload"] = store.decode(row.get("payload"))
        row["model_versions"] = store.decode(row.get("model_versions"))
    for row in overrides:
        row["sales_context"] = store.decode(row.get("sales_context"))
        row["payload"] = store.decode(row.get("payload"))
    if outcome:
        outcome["payload"] = store.decode(outcome.get("payload"))

    return {
        "loan_id": loan_id,
        "decisions": decisions,
        "overrides": overrides,
        "outcome": outcome,
        "audit_entries": trail(entity_id=loan_id),
    }
