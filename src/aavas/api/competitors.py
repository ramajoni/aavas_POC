"""Sales-note extraction and competitor intelligence.

    POST /notes/extract          the notes on the file being priced -> claims, signals, conflicts
    GET  /notes/summary          historical extraction coverage + quality vs latent truth
    GET  /competitors            league table, emerging lenders, unrecognised names
    GET  /competitors/behaviour  how customers behave around a competitor offer
    GET  /competitors/{name}     one lender in depth (alias or canonical name)
    POST /competitors/context    application + notes -> per-loan competitor context

Everything here is context for a human. None of it is read by the pricing path:
an extracted quote, a benchmark or a win rate never moves a recommendation (see
``agents/sales_context_agent.py`` for how a competitor claim may and may not).

The historical tables are built once per process from the cached extraction
(``data/processed/note_signals.csv``, rebuilt when the notes file or the registry
changes); the first call after a restart may take a few seconds.
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..config import model_config
from ..ingestion import loader
from ..intelligence import competitors as analysis_module
from ..customers import loan_book
from ..intelligence.behaviour import Behaviour
from ..intelligence import note_extraction, quality, registry
from ..logging_utils import get_logger
from ..paths import resolve
from .deps import get_note_retriever

log = get_logger(__name__)
router = APIRouter(tags=["competitor intelligence"])

# ------------------------------------------------------------------ examples
EXTRACT_EXAMPLE = {
    "loan_id": "DEMO-FLOW-001",
    "notes": [
        "Walk-in enquiry for a resale flat in Jaipur, salaried. Needs disbursement before registration next week.",
        "Customer showed HDFC Bank sanction letter at 8.65%, also says LICHFL offering 8.9 pc verbally.",
        "Customer bargaining on every point, wants 8.75% or close.",
    ],
    "sales_context": {"competitor_name": "HDFC", "competitor_rate": 8.9,
                      "competitor_evidence_verified": False},
}
CONTEXT_EXAMPLE = {
    "application": {"loan_id": "AAV0009223", "city": "Rajkot", "state": "Gujarat",
                    "segment": "self_employed_non_professional", "risk_grade": "D", "our_rate": 13.23,
                    "sales_context": {"competitor_name": "Shubham HF", "competitor_rate": 12.47,
                                      "competitor_evidence_verified": True}},
    "notes": ["Walk-in enquiry for construction on own plot in Rajkot, small business owner.",
              "Customer showed Shubham HF sanction letter at 12.47%. Requesting rate match to retain the file."],
}


# ----------------------------------------------------------------- contracts
class NoteSalesContext(BaseModel):
    """The structured competitor fields the notes are checked against. ``None``
    means not captured - different from ``false``."""

    competitor_name: str | None = None
    competitor_rate: float | None = Field(None, ge=0, le=40)
    competitor_evidence_verified: bool | None = None


class NotesExtractRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXTRACT_EXAMPLE]})

    loan_id: str | None = None
    notes: list[str] = Field(..., min_length=1, max_length=50,
                             description="Notes in the order they were written; the last is the latest")
    sales_context: NoteSalesContext = Field(default_factory=NoteSalesContext)


class ApplicationContext(BaseModel):
    loan_id: str | None = None
    city: str | None = None
    state: str | None = None
    segment: str | None = None
    risk_grade: str | None = None
    our_rate: float | None = Field(None, ge=0, le=40,
                                   description="The rate under consideration (recommended or proposed); "
                                               "without it the gap and win rate at that gap are not computed")
    policy_card_rate: float | None = Field(None, ge=0, le=40)
    sales_context: NoteSalesContext = Field(default_factory=NoteSalesContext)


class CompetitorContextRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [CONTEXT_EXAMPLE]})

    application: ApplicationContext
    notes: list[str] = Field(default_factory=list, max_length=50)


# --------------------------------------------------------------------- cache
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


def _key() -> tuple:
    """What the tables depend on: the notes, the registry, and the loan book."""
    mtime = lambda p: p.stat().st_mtime if p.exists() else None          # noqa: E731
    loans = resolve(model_config()["data"]["processed_path"])
    return (mtime(note_extraction.notes_path()), registry.config_fingerprint(), mtime(loans))


def historical() -> dict[str, Any]:
    """Extraction tables and analysis, built once per process (and again when
    the notes file or the registry changes)."""
    key = _key()
    with _LOCK:
        if _STATE.get("key") != key:
            started = time.perf_counter()
            registry.reset()
            _extractor.cache_clear()
            signals, mentions = note_extraction.load_history()
            _STATE.clear()
            _STATE.update(key=key, signals=signals, mentions=mentions,
                          analysis=analysis_module.build(mentions=mentions),
                          built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                          build_seconds=round(time.perf_counter() - started, 2))
            log.info("competitor intelligence ready in %.1fs", _STATE["build_seconds"])
        return _STATE


def _quality(state: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        if "quality" not in state:
            state["quality"] = quality.quality_report(state["signals"], loader.load_notes())
        return state["quality"]


@lru_cache(maxsize=1)
def _extractor() -> note_extraction.NoteExtractor:
    # Current-loan extraction may use the acceptance agent's LLM for customer
    # signals when it is switched on and keyed; the batch never does.
    return note_extraction.NoteExtractor(allow_llm=True)


def clean(value: Any) -> Any:
    """JSON-safe: numpy scalars to Python, NaN to null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


# ----------------------------------------------------------------- endpoints
@router.post("/notes/extract", summary="Read the notes on the file being priced")
def extract_notes(request: NotesExtractRequest) -> dict[str, Any]:
    result = note_extraction.extract_current(
        request.notes, request.sales_context.model_dump(), extractor=_extractor())
    return clean({"loan_id": request.loan_id, **result})


@router.get("/notes/summary", summary="Historical extraction coverage and quality")
def notes_summary() -> dict[str, Any]:
    state = historical()
    signals, mentions = state["signals"], state["mentions"]
    recognised = mentions[(mentions["match"] != registry.UNRECOGNISED) & mentions["competitor"].notna()
                          & (mentions["competitor"] != "")]
    with_mentions = signals["n_mentions"].fillna(0) > 0
    coverage = {
        "notes": int(len(signals)),
        "loans": int(signals["loan_id"].nunique()),
        "notes_with_competitor_mentions": int(with_mentions.sum()),
        "share_of_notes": round(float(with_mentions.mean()), 4) if len(signals) else None,
        "mentions": int(len(recognised)),
        "quoted_mentions": int(recognised["rate"].notna().sum()),
        "fuzzy_matches": int((recognised["match"] == registry.FUZZY).sum()),
        "ambiguous_alias_matches": int((recognised["match"] == registry.AMBIGUOUS).sum()),
        "unrecognised_mentions": int((mentions["match"] == registry.UNRECOGNISED).sum()),
        "notes_with_customer_ask": int(signals["customer_ask_rate"].notna().sum()),
        "notes_with_our_rate": int(signals["our_rate"].notna().sum()),
        "evidence": {k: int(v) for k, v in recognised["evidence"].value_counts().items()},
        "by_competitor": {k: int(v) for k, v in recognised["competitor"].value_counts().items()},
        "customer_signals": {name: int(signals[name].notna().sum())
                             for name in ("customer_urgency", "rate_sensitivity", "customer_intent",
                                          "required_amount") if name in signals},
    }
    return clean({
        "coverage": coverage,
        "quality": _quality(state),
        "method": {"competitors": "rule: registry aliases + rate patterns + evidence cues",
                   "customer_signals": "rule (the historical batch never calls the LLM)",
                   "extractor_version": str(signals["extractor_version"].iloc[0]) if len(signals) else None,
                   "registry_version": registry.default_registry().version},
        "cache": {"signals": str(note_extraction.PROCESSED_DIR / note_extraction.SIGNALS_FILE),
                  "mentions": str(note_extraction.PROCESSED_DIR / note_extraction.MENTIONS_FILE),
                  "built_at": state["built_at"], "build_seconds": state["build_seconds"]},
    })


@router.get("/competitors", summary="Competitor league table, emerging and unrecognised lenders")
def competitor_summary() -> dict[str, Any]:
    state = historical()
    reg = registry.default_registry()
    return clean({**state["analysis"].summary(),
                  "registry": {"version": reg.version, "lenders": len(reg.lenders),
                               "self_names": reg.self_names},
                  "built_at": state["built_at"]})


@lru_cache(maxsize=1)
def _behaviour(loaded: str) -> dict[str, Any]:
    """Cached per loaded version of the loan book: a reload recomputes it."""
    settings = analysis_module.load_config().get("analysis") or {}
    # Read from the loan book table in the database, not the CSV.
    return clean(Behaviour(loan_book.frame(),
                           min_n_rate=int(settings.get("min_n_rate", 30))).summary())


# Declared before /competitors/{name}, which would otherwise read "behaviour" as a lender.
@router.get("/competitors/behaviour", summary="How customers behave around a competitor offer")
def customer_behaviour() -> dict[str, Any]:
    return _behaviour(loan_book.ensure_loaded()["fingerprint"])


@router.get("/competitors/{name}", summary="One competitor in depth")
def competitor_profile(name: str) -> dict[str, Any]:
    state = historical()
    profile = state["analysis"].profile(name)
    if profile is None:
        known = sorted(state["analysis"].lc["competitor"].unique().tolist())
        raise HTTPException(status_code=404, detail={
            "error": f"'{name}' is not a competitor the registry recognises in the notes",
            "known": known})
    return clean(profile)


@router.post("/competitors/context", summary="Per-loan competitor context for the reviewer")
def competitor_context(request: CompetitorContextRequest) -> dict[str, Any]:
    state = historical()
    application = request.application.model_dump()
    result = state["analysis"].context_for(application, request.notes, extractor=_extractor(),
                                           retriever=get_note_retriever())
    return clean(result)
