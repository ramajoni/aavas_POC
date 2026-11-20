"""Decision memory: what each area manager decided on a lower-rate request, with the
factors as they stood, what the system suggested, and - joined when read - what the
customer then did.

One episode per decided request, written once by the learning agent at the moment the
manager decides, and never edited. The factors are frozen then because the screen
recomputes them from a book that keeps moving: an episode that re-derived them later
would describe a decision nobody took.

The outcome is deliberately not stored on the episode. It is joined when memory is
read, from the offer the decision led to and the outcome row, so a customer who
answers next week updates every later recall without anyone rewriting memory.

Memory informs the manager; it does not move the suggestion. The suggestion is the
rate worth the most to the bank. An agent that nudged it toward what a manager
usually does would learn the manager's habits, mistakes included - the trap the
whole engine was built to avoid.
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from ..governance import store
from ..logging_utils import get_logger

log = get_logger(__name__)

FORMAT = "MEM-1"
CUT, HOLD, UPLIFT = ("approve", "counter"), "hold", "uplift"


def action_taken(request: dict[str, Any], decided: dict[str, Any]) -> tuple[str, float | None]:
    """The manager's decision in the suggestion's vocabulary, and the rate it set."""
    status = decided.get("status")
    if status == "declined":
        return HOLD, float(request["recommended_rate"])
    approved = decided.get("approved_rate")
    if status == "uplifted":
        return UPLIFT, approved
    if status == "approved":
        same = approved is not None and abs(float(approved) - float(request["requested_rate"])) < 0.005
        return ("approve" if same else "counter"), approved
    raise ValueError(f"request {request.get('request_id')} is not decided ({status!r})")


def followed(suggestion: dict[str, Any] | None, action: str, rate: float | None) -> bool | None:
    """Did the manager take the suggested decision? None when there was none to take."""
    if not suggestion or not suggestion.get("action"):
        return None
    if suggestion["action"] != action:
        return False
    if action == HOLD or rate is None or suggestion.get("rate") is None:
        return True
    return abs(float(suggestion["rate"]) - float(rate)) < 0.005


def remember(*, request: dict[str, Any], decided: dict[str, Any], observation: dict[str, Any] | None,
             unavailable: str | None = None, recalled: list[str] | None = None) -> dict[str, Any]:
    """Write the episode for a decided request. Append-only: a request has one episode,
    and a second call for it returns the first rather than replacing it."""
    action, rate = action_taken(request, decided)
    observation = observation or {}
    suggestion = observation.get("suggestion")
    episode = {
        "episode_id": f"MEM-{uuid.uuid4().hex[:12]}",
        "request_id": request["request_id"], "loan_id": request["loan_id"],
        "decision_id": request.get("decision_id"),
        "manager": decided.get("decided_by") or "unknown",
        "decided_at": decided.get("decided_at") or store.now(),
        "action": action, "requested_rate": float(request["requested_rate"]),
        "recommended_rate": float(request["recommended_rate"]), "decided_rate": rate,
        "reason": decided.get("decision_reason"), "note": decided.get("decision_note"),
        "requested_by": request.get("requested_by"), "request_reason": request.get("reason"),
        "suggested_action": (suggestion or {}).get("action"), "suggested_rate": (suggestion or {}).get("rate"),
        "followed_suggestion": followed(suggestion, action, rate),
        "lean": observation.get("lean"),
        "case_features": observation.get("case") or {},
        "factors": observation.get("factors") or None,
        "factors_unavailable": None if observation.get("factors") else (unavailable or "not observed"),
        "recalled": list(recalled or []),
        "format": FORMAT, "recorded_at": store.now(),
    }
    try:
        store.append("decision_memory", episode)
    except sqlite3.IntegrityError:
        existing = store.fetch_one("SELECT episode_id FROM decision_memory WHERE request_id = ?",
                                   (request["request_id"],))
        log.info("request %s is already in memory as %s", request["request_id"], existing and existing["episode_id"])
        return {"recorded": False, "episode_id": existing and existing["episode_id"],
                "reason": "this request is already in memory; an episode is never replaced"}
    return {"recorded": True, "episode_id": episode["episode_id"], "action": action,
            "followed_suggestion": episode["followed_suggestion"],
            "factors_recorded": episode["factors"] is not None}


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("case_features", "factors", "recalled"):
        out[key] = store.decode(out.get(key))
    if out.get("followed_suggestion") is not None:
        out["followed_suggestion"] = bool(out["followed_suggestion"])
    return out


def _outcome(episode: dict[str, Any], offers: list[dict[str, Any]], outcome: dict[str, Any] | None) -> dict[str, Any]:
    """What the customer did after this decision: the latest live offer on the file made
    by the decision or after it, and the outcome row's reason if they walked."""
    after = [o for o in offers if o.get("request_id") == episode["request_id"]
             or str(o.get("sent_at") or "") >= str(episode["decided_at"])]
    live = [o for o in after if o.get("status") != "superseded"] or after
    offer = max(live, key=lambda o: str(o.get("sent_at") or ""), default=None)
    if offer is None:
        return {"state": "no_offer", "text": "No offer sent yet"}
    state = {"accepted": "accepted", "not_accepted": "not_accepted"}.get(offer.get("status"), "awaiting")
    out = {"state": state, "offered_rate": offer.get("rate"), "decided_at": offer.get("customer_decided_at")}
    if state == "not_accepted" and outcome:
        out.update(lost_to=outcome.get("lost_to"), lost_to_rate=outcome.get("lost_to_rate"),
                   loss_reason=outcome.get("loss_reason"))
    rate = f"{float(offer['rate']):.2f}%" if offer.get("rate") is not None else "the offered rate"
    out["text"] = {"accepted": f"Customer accepted {rate}",
                   "not_accepted": f"Customer turned down {rate}" + (f" for {out.get('lost_to')}" if out.get("lost_to") else ""),
                   "awaiting": f"Offer of {rate} sent; customer has not answered"}[state]
    return out


def episodes(manager: str | None = None, exclude_request: str | None = None) -> list[dict[str, Any]]:
    """Episodes, newest first, each with its outcome joined. Three queries on one
    connection however many episodes there are."""
    clauses, params = [], []
    if manager:
        clauses.append("manager = ?"); params.append(manager)
    if exclude_request:
        clauses.append("request_id != ?"); params.append(exclude_request)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with store.connect() as conn:
        rows = [_decode(dict(r)) for r in conn.execute(
            f"SELECT * FROM decision_memory{where} ORDER BY decided_at DESC", tuple(params)).fetchall()]
        loans = sorted({r["loan_id"] for r in rows})
        offers: dict[str, list[dict[str, Any]]] = {}
        outcomes: dict[str, dict[str, Any]] = {}
        if loans:
            marks = ", ".join("?" for _ in loans)
            for o in conn.execute(f"SELECT loan_id, request_id, rate, status, sent_at, customer_decided_at "
                                  f"FROM offers WHERE loan_id IN ({marks})", loans).fetchall():
                offers.setdefault(o["loan_id"], []).append(dict(o))
            for o in conn.execute(f"SELECT loan_id, lost_to, lost_to_rate, loss_reason, updated_at "
                                  f"FROM outcomes WHERE loan_id IN ({marks}) ORDER BY updated_at", loans).fetchall():
                outcomes[o["loan_id"]] = dict(o)            # the latest wins
    for row in rows:
        row["outcome"] = _outcome(row, offers.get(row["loan_id"], []), outcomes.get(row["loan_id"]))
    return rows


def count() -> int:
    return store.count("decision_memory")
