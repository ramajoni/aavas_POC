"""Competitor rates found online, kept per lender, and the check built on them.

The market research agent takes a minute or two per lender - longer than a page
should wait - so a search runs in the background and the screen asks again until
it is done. Its findings are kept for ``online_research.max_age_days``: a lender's
rate card changes every few weeks, and a relationship manager opening ten files
that quote HDFC should not start ten searches.

A customer who names no lender, or only "a local bank", is compared with the
market as a whole (``MARKET``): the lowest rates the usual competitors advertise.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from ..agents.market_research_agent import MARKET, MarketResearchAgent, ResearchFailed
from ..config import load
from ..governance import audit, store
from ..logging_utils import get_logger
from . import registry

log = get_logger(__name__)

TABLE = "market_rate_research"
CATCH_ALL = {"Local Bank"}          # a registry name that is not one lender
MARKET_LABEL = "lenders in the market"

_running: set[str] = set()
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="market-research")
_agent: MarketResearchAgent | None = None


def settings() -> dict[str, Any]:
    return load("market_research")["online_research"]


def agent() -> MarketResearchAgent:
    global _agent
    if _agent is None:
        _agent = MarketResearchAgent()
    return _agent


def set_agent(replacement: MarketResearchAgent | None) -> None:
    """Tests: a stand-in agent, or None to go back to the real one."""
    global _agent
    _agent = replacement


def _submit(job: Callable[..., None], *args: Any) -> None:
    _executor.submit(job, *args)


# ---------------------------------------------------------------- the lender

def research_key(competitor_name: str | None) -> tuple[str, str, list[str]]:
    """(cache key, how to name it on screen, other spellings to search under)."""
    reg = registry.default_registry()
    canonical = reg.resolve(competitor_name) if competitor_name else None
    if not canonical or canonical in CATCH_ALL:
        return MARKET, MARKET_LABEL, []
    aliases = (reg.config.get("lenders", {}).get(canonical) or {}).get("aliases") or []
    return canonical, canonical, [str(a) for a in aliases]


def lenders() -> list[dict[str, str]]:
    """Lender names a relationship manager can pick from, catch-alls last."""
    reg = registry.default_registry()
    rows = [{"name": name, "type": str(spec.get("type") or "")}
            for name, spec in (reg.config.get("lenders") or {}).items()]
    return sorted(rows, key=lambda r: (r["name"] in CATCH_ALL, r["name"].lower()))


# ----------------------------------------------------------------- the cache

def latest(key: str) -> dict[str, Any] | None:
    row = store.fetch_one(f"SELECT * FROM {TABLE} WHERE lender = ? "
                          "ORDER BY researched_at DESC LIMIT 1", (key,))
    if row is None:
        return None
    payload = store.decode(row["payload"])
    return {**(payload if isinstance(payload, dict) else {}), "status": row["status"],
            "researched_at": row["researched_at"], "requested_by": row.get("requested_by")}


def _age(row: dict[str, Any]) -> float:
    """Seconds since the row was researched."""
    then = datetime.fromisoformat(row["researched_at"])
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


def lookup(competitor_name: str | None, *, refresh: bool = False,
           actor: str | None = None) -> dict[str, Any]:
    """The research for this lender, starting a search when there is none fresh.

    ``state`` is ``ready`` (fresh findings), ``researching`` (a search is running;
    any older findings come with it), ``not_found`` or ``failed`` (the last search,
    recently), or ``unavailable`` (no search can run here).
    """
    cfg = settings()
    key, label, aliases = research_key(competitor_name)
    out: dict[str, Any] = {"lender": key, "label": label}
    row = latest(key)
    usable = row if row and row["status"] == "ok" else None
    with _lock:
        running = key in _running
    if running:
        return {**out, "state": "researching", "research": usable}

    fresh = usable is not None and _age(usable) <= float(cfg["max_age_days"]) * 86400
    recent_miss = (row is not None and row["status"] != "ok"
                   and _age(row) <= float(cfg["retry_after_minutes"]) * 60)
    if not refresh and fresh:
        return {**out, "state": "ready", "research": usable}
    if not refresh and recent_miss:
        return {**out, "state": row["status"], "research": None,
                "reason": row.get("error") or row.get("notes") or None}

    status = agent().status()
    if not status["available"]:
        return {**out, "state": "unavailable", "research": usable, "reason": status["reason"]}
    start(key, aliases, actor)
    return {**out, "state": "researching", "research": usable}


def start(key: str, aliases: list[str], actor: str | None) -> bool:
    """Queue a search for ``key`` unless one is already running."""
    with _lock:
        if key in _running:
            return False
        _running.add(key)
    try:
        _submit(_run, key, aliases, actor)
    except Exception:                                  # noqa: BLE001 - never leave it marked running
        with _lock:
            _running.discard(key)
        raise
    return True


def _run(key: str, aliases: list[str], actor: str | None) -> None:
    try:
        try:
            result = agent().research(key, aliases)
        except ResearchFailed as exc:
            result = {"lender": key, "status": "failed", "error": str(exc)}
        except Exception as exc:                        # noqa: BLE001 - recorded, not raised in a thread
            log.exception("market research for %s failed", key)
            result = {"lender": key, "status": "failed",
                      "error": f"The fact check could not be completed ({type(exc).__name__})."}
        save(key, result, actor)
    finally:
        with _lock:
            _running.discard(key)


def save(key: str, result: dict[str, Any], actor: str | None) -> None:
    store.insert(TABLE, {
        "research_id": f"MR-{uuid.uuid4().hex[:12].upper()}", "lender": key,
        "status": result.get("status") or "failed", "researched_at": store.now(),
        "published_min": result.get("published_min"), "published_max": result.get("published_max"),
        "payload": result, "requested_by": actor})
    audit.record(actor or "system", "researched_competitor_rates", "competitor", key,
                 {"status": result.get("status"), "published_min": result.get("published_min"),
                  "published_max": result.get("published_max"),
                  "source_quality": result.get("source_quality"),
                  "searches": result.get("searches"), "error": result.get("error")})


# ----------------------------------------------------------------- the check

def check(competitor_name: str | None, quote: float, evidence_seen: bool,
          profile: dict[str, Any] | None = None, *, refresh: bool = False,
          actor: str | None = None) -> dict[str, Any]:
    """A quoted rate scored against what its lender advertises (or would, once found)."""
    found = lookup(competitor_name, refresh=refresh, actor=actor)
    research = found.get("research")
    scored = agent().assess(quote, evidence_seen, profile, research, found["label"],
                            plural=found["lender"] == MARKET)
    return {"state": found["state"], "lender": found["lender"], "lender_label": found["label"],
            "reason": found.get("reason"), "research": public(research), "score": scored}


def public(research: dict[str, Any] | None) -> dict[str, Any] | None:
    """What a screen shows: the range, its sources, and how old it is."""
    if not research:
        return None
    return {
        "researched_at": research.get("researched_at"),
        "age_days": round(_age(research) / 86400, 1) if research.get("researched_at") else None,
        "published_min": research.get("published_min"), "published_max": research.get("published_max"),
        "source_quality": research.get("source_quality"),
        "best_rate_min_credit_score": research.get("best_rate_min_credit_score"),
        "notes": research.get("notes") or "",
        "rates": [{k: r.get(k) for k in ("lender", "min_rate", "max_rate", "applies_to", "as_of",
                                         "source_url", "source_type")}
                  for r in research.get("rates") or [] if r.get("source_seen")],
    }
