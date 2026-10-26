"""Market research agent - what a competitor advertises today, from the internet.

"Another lender offered me less" is the most common reason a relationship
manager asks to go below the recommendation, and the hardest to check from
inside the branch. This agent searches the web for the lender's currently
published home-loan rates - its own website first, then financial news and
comparison sites - and keeps only rates it can tie to a page that search
actually returned. ``assess`` then scores the claim against them
(``intelligence/claim_score.py``).

It is outside the priced path by design: it runs when a person opens a file with
a competitor quote, its findings are kept per lender (``intelligence/market_rates.py``),
and nothing it returns moves a recommendation. Without an API key, or when the
search fails, the claim is still scored - on its evidence and the customer's
profile - and says the fact check did not run.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..config import env, load
from ..intelligence import claim_score
from ..logging_utils import get_logger
from ..paths import ROOT

log = get_logger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
KEY_NAME = "ANTHROPIC_API_KEY"
MARKET = "market"                  # research key when the customer named no lender
RATE_LIMITS = (4.0, 25.0)          # any recorded rate outside this is a misread
MAX_TURNS = 6

SYSTEM = """You check what home-loan lenders in India advertise as their current home-loan \
interest rates. A loan officer at Aavas Financiers, an affordable-housing finance company, \
uses your findings to judge whether a customer's claim that another lender quoted them a \
rate is believable.

How to research:
- Search the web. Prefer the lender's own website: its home-loan interest-rate page, rate \
card or most-important-terms document. Then reputable financial news and rate-comparison sites.
- Fetch a page when the search result does not show the rate itself.
- Record only rates you saw on a page during this research, each with that page's URL. \
Never record a rate from memory.
- Record the date the page says the rates apply from when it gives one; otherwise leave \
as_of empty.
- Where the lender publishes different rates by credit score, employment type, loan amount \
or a time-limited offer, record each as its own entry and say what it applies to.
- Home loans to buy or build a house only - not loans against property or top-ups.

When you have finished, call record_findings once. If you could not find a published rate, \
call it with found set to false and say why in notes."""

FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "rates": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "lender": {"type": "string"},
                "min_rate": {"type": "number", "description": "% a year"},
                "max_rate": {"type": ["number", "null"], "description": "% a year; null if only one rate"},
                "applies_to": {"type": "string",
                               "description": "who gets it, e.g. 'salaried, credit score 800+'"},
                "as_of": {"type": "string", "description": "date the page gives, or empty"},
                "source_url": {"type": "string"},
                "source_type": {"type": "string",
                                "enum": ["lender_website", "news", "comparison_site", "other"]},
            },
            "required": ["lender", "min_rate", "max_rate", "applies_to", "as_of", "source_url",
                         "source_type"],
            "additionalProperties": False}},
        "best_rate_min_credit_score": {
            "type": ["integer", "null"],
            "description": "credit score the lender asks for its lowest rate, if published"},
        "notes": {"type": "string"},
    },
    "required": ["found", "rates", "best_rate_min_credit_score", "notes"],
    "additionalProperties": False,
}


class ResearchUnavailable(RuntimeError):
    """No key or no SDK: the fact check cannot run at all."""


class ResearchFailed(RuntimeError):
    """The search ran and did not produce usable findings."""


class MarketResearchAgent:
    def __init__(self, client=None, model: str | None = None, env_file: Path | None = None):
        self._client = client
        self.model = model or env("AAVAS_LLM_RESEARCH_MODEL") or DEFAULT_MODEL
        self.env_file = env_file if env_file is not None else ROOT / ".env"

    # ------------------------------------------------------------ availability
    def _api_key(self) -> str | None:
        return os.environ.get(KEY_NAME) or _read_env_file(self.env_file, KEY_NAME)

    def status(self) -> dict[str, Any]:
        if self._client is not None:
            return {"available": True, "reason": None, "model": self.model}
        if not self._api_key():
            return {"available": False, "model": self.model,
                    "reason": "Fact checking is not set up on this server."}
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return {"available": False, "model": self.model,
                    "reason": "Fact checking is not installed on this server."}
        return {"available": True, "reason": None, "model": self.model}

    def _get_client(self):
        if self._client is None:
            status = self.status()
            if not status["available"]:
                raise ResearchUnavailable(status["reason"])
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key())
        return self._client

    # ---------------------------------------------------------------- research
    def research(self, lender: str, aliases: list[str] | None = None,
                 today: date | None = None) -> dict[str, Any]:
        """What ``lender`` (or, for ``MARKET``, the usual competitors) advertises today."""
        cfg = load("market_research")["online_research"]
        client = self._get_client()
        today = today or date.today()
        if lender == MARKET:
            ask = ("The customer did not name the lender. Find the home-loan interest rates these "
                   "lenders currently advertise in India: " + ", ".join(cfg["market_lenders"])
                   + ". Record each lender's rates as separate entries; each one's lowest rate "
                     "matters most.")
        else:
            known = [a for a in (aliases or []) if a.lower() != lender.lower()][:6]
            ask = (f"Find the home-loan interest rates {lender} currently advertises in India"
                   + (f" (branches also write it as: {', '.join(known)})" if known else "") + ".")
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Today is {today.isoformat()}. {ask}"}]
        tools = [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": int(cfg["max_searches"])},
            {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": int(cfg["max_fetches"])},
            {"name": "record_findings", "strict": True, "input_schema": FINDINGS_SCHEMA,
             "description": "Record the published home-loan rates you found, each with the page "
                            "it came from. Call once, when the research is finished."},
        ]

        seen: set[str] = set()
        searches, nudged = 0, False
        for _ in range(MAX_TURNS):
            try:
                response = client.beta.messages.create(
                    model=self.model, max_tokens=16000, system=SYSTEM, messages=messages,
                    tools=tools, output_config={"effort": "high"},
                    # A safety classifier can decline a request; "default" re-runs it
                    # on Anthropic's recommended fallback model instead of failing.
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            except Exception as exc:                      # noqa: BLE001 - reported, never raised past
                log.warning("market research for %s failed: %s", lender, exc)
                raise ResearchFailed(_plain(exc)) from exc
            seen |= _source_urls(response.content)
            usage = getattr(getattr(response, "usage", None), "server_tool_use", None)
            searches += int(getattr(usage, "web_search_requests", 0) or 0)

            findings = next((b.input for b in response.content
                             if getattr(b, "type", None) == "tool_use"
                             and getattr(b, "name", None) == "record_findings"), None)
            if findings is not None:
                return summarise(lender, findings, seen, searches,
                                 getattr(response, "model", None) or self.model)
            if response.stop_reason == "refusal":
                raise ResearchFailed("The fact check was declined.")
            if response.stop_reason == "max_tokens":
                raise ResearchFailed("The fact check was cut off before it finished.")
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue                                   # the server resumes the same turn
            if nudged:
                break
            messages.append({"role": "user", "content": "Call record_findings now with what you found."})
            nudged = True
        raise ResearchFailed("The fact check ended without recording any findings.")

    # ------------------------------------------------------------------ assess
    def assess(self, quote: float, evidence_seen: bool, profile: dict[str, Any] | None,
               research: dict[str, Any] | None, lender_label: str,
               plural: bool = False) -> dict[str, Any]:
        return claim_score.score(quote, evidence_seen, profile, research, lender_label=lender_label,
                                 plural=plural)


def _plain(exc: Exception) -> str:
    """Why the search failed, in words for the person looking at the file; the
    provider's own message stays in the server log."""
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return "The fact-checking key was rejected. An administrator needs to replace it."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "The fact-checking key is not allowed to search the web."
    if isinstance(exc, anthropic.RateLimitError):
        return "Fact checking is busy. Try again in a few minutes."
    if isinstance(exc, anthropic.BadRequestError) and "credit balance" in str(exc).lower():
        return ("The fact-checking account has run out of credit. An administrator needs to add "
                "credit (Anthropic Console, Plans & Billing).")
    if isinstance(exc, anthropic.APIConnectionError):
        return "The fact-checking service could not be reached."
    return "The fact check could not be completed."


def summarise(lender: str, findings: dict[str, Any], seen_urls: set[str], searches: int,
              model: str) -> dict[str, Any]:
    """Findings -> the published range, keeping only rates tied to a page the search returned."""
    lo_limit, hi_limit = RATE_LIMITS
    rates = []
    for r in findings.get("rates") or []:
        low, high = r.get("min_rate"), r.get("max_rate")
        if not isinstance(low, (int, float)) or not lo_limit <= low <= hi_limit:
            continue
        if not isinstance(high, (int, float)) or not lo_limit <= high <= hi_limit or high < low:
            high = None
        url = str(r.get("source_url") or "").strip()
        rates.append({"lender": str(r.get("lender") or lender), "min_rate": round(float(low), 2),
                      "max_rate": round(float(high), 2) if high is not None else None,
                      "applies_to": str(r.get("applies_to") or ""), "as_of": str(r.get("as_of") or ""),
                      "source_url": url, "source_type": r.get("source_type") or "other",
                      # A URL the model wrote but no search or fetch returned is not a source.
                      "source_seen": _normalise(url) in seen_urls})
    used = [r for r in rates if r["source_seen"]]
    official = [r for r in used if r["source_type"] == "lender_website"]
    basis = official or used
    need = findings.get("best_rate_min_credit_score")
    return {
        "lender": lender, "status": "ok" if basis else "not_found",
        "published_min": min(r["min_rate"] for r in basis) if basis else None,
        "published_max": max((r["max_rate"] or r["min_rate"]) for r in basis) if basis else None,
        "source_quality": "lender_website" if official else "secondary" if used else None,
        "best_rate_min_credit_score": need if isinstance(need, int) and 300 <= need <= 900 else None,
        "rates": rates, "notes": str(findings.get("notes") or ""),
        "searches": searches, "model": model,
        "researched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _source_urls(content: Any) -> set[str]:
    """Every URL a search or fetch returned in this response, normalised."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "url" and isinstance(item, str):
                    found.add(_normalise(item))
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for block in content or []:
        kind = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if not kind or not str(kind).endswith("_tool_result"):
            continue
        walk(block.model_dump() if hasattr(block, "model_dump") else block)
    return found


def _normalise(url: str) -> str:
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    return f"{host}{parts.path.rstrip('/')}"


def _read_env_file(path: Path, key: str) -> str | None:
    """One key from a dotenv-style file; nothing else in it is read."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip().removeprefix("export ").lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip("'\"") or None
    return None
