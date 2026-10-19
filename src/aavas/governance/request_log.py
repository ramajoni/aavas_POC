"""Every request that reached a priced endpoint, written before the work starts.

The decisions table holds what the engine produced. It cannot answer how often
the engine *failed*, because a request that raised never wrote a row - so a
success rate computed over decisions is 100% by construction, and the control
tower refused to report one.

This closes that. A row is opened before the pipeline runs and closed when it
finishes, whatever way it finishes:

* ``ok``           - 2xx
* ``client_error`` - 4xx, the caller sent something invalid
* ``failed``       - 5xx or an unhandled exception
* ``abandoned``    - opened and never closed; the process died mid-request

``abandoned`` is the reason the row is opened first. A log written only on
completion cannot record the failure that stopped completion happening.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..logging_utils import get_logger
from . import store

log = get_logger(__name__)

# Only the endpoints that produce or change a priced decision. Logging every GET
# would bury the signal under dashboard polling.
TRACKED_PREFIXES = ("/pricing/pre-assessment", "/pricing/final-assessment",
                    "/pricing/override", "/pricing/outcome")

OK, CLIENT_ERROR, FAILED, ABANDONED = "ok", "client_error", "failed", "abandoned"


def tracked(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in TRACKED_PREFIXES)


def new_request_id() -> str:
    return f"REQ-{uuid.uuid4().hex[:12].upper()}"


def open_request(route: str, method: str, actor: str | None = None) -> str:
    request_id = new_request_id()
    store.insert("requests", {
        "request_id": request_id, "route": route, "method": method,
        "started_at": store.now(), "outcome": ABANDONED, "actor": actor,
    })
    return request_id


def close_request(request_id: str, *, status_code: int, elapsed_ms: float,
                  decision_id: str | None = None, loan_id: str | None = None,
                  error_type: str | None = None, error_detail: str | None = None) -> None:
    outcome = OK if status_code < 400 else CLIENT_ERROR if status_code < 500 else FAILED
    with store.connect() as conn:
        conn.execute(
            "UPDATE requests SET finished_at = ?, status_code = ?, outcome = ?, "
            "elapsed_ms = ?, decision_id = ?, loan_id = ?, error_type = ?, error_detail = ? "
            "WHERE request_id = ?",
            (store.now(), int(status_code), outcome, float(elapsed_ms), decision_id, loan_id,
             error_type, (error_detail or "")[:500], request_id))


def summary(limit: int = 2000) -> dict[str, Any]:
    """Success, failure and abandonment over the most recent window."""
    rows = store.query(
        "SELECT route, outcome, status_code, elapsed_ms, error_type FROM requests "
        "ORDER BY started_at DESC LIMIT ?", (limit,))
    total = len(rows)
    counts: dict[str, int] = {}
    by_route: dict[str, dict[str, int]] = {}
    errors: dict[str, int] = {}
    for row in rows:
        outcome = row["outcome"] or ABANDONED
        counts[outcome] = counts.get(outcome, 0) + 1
        bucket = by_route.setdefault(row["route"], {})
        bucket[outcome] = bucket.get(outcome, 0) + 1
        if row.get("error_type"):
            errors[row["error_type"]] = errors.get(row["error_type"], 0) + 1

    served = total - counts.get(ABANDONED, 0)
    return {
        "window": total,
        "by_outcome": counts,
        "by_route": by_route,
        "error_types": errors,
        # Deliberately over requests, not decisions. That is the whole point.
        "success_rate_pct": round(100.0 * counts.get(OK, 0) / served, 2) if served else None,
        "failure_rate_pct": round(100.0 * counts.get(FAILED, 0) / served, 2) if served else None,
        "client_error_rate_pct": round(100.0 * counts.get(CLIENT_ERROR, 0) / served, 2)
        if served else None,
        "abandoned": counts.get(ABANDONED, 0),
        "note": ("Rates are over requests that reached a priced endpoint, not over decisions. "
                 "An 'abandoned' row was opened and never closed - the process died mid-request, "
                 "which is exactly the failure a completion-only log cannot see."),
    }


def recent(limit: int = 50, outcome: str | None = None) -> list[dict[str, Any]]:
    if outcome:
        return store.query("SELECT * FROM requests WHERE outcome = ? ORDER BY started_at DESC "
                           "LIMIT ?", (outcome, limit))
    return store.query("SELECT * FROM requests ORDER BY started_at DESC LIMIT ?", (limit,))
