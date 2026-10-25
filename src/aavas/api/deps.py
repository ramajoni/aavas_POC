"""Shared dependencies. The orchestrator and its artefacts load once per process."""

from __future__ import annotations

import threading
from functools import lru_cache

from ..agents.orchestrator import PricingOrchestrator
from ..governance import model_registry
from ..ingestion import loader
from ..knowledge.policy_indexer import PolicyRetriever
from ..knowledge.sales_note_retriever import SalesNoteRetriever
from ..logging_utils import get_logger
from ..xai.loan_explanation import LoanExplainer
from ..xai.narrative import LoanNarrator

log = get_logger(__name__)
_RETRIEVER_LOCK = threading.Lock()


def get_note_retriever() -> SalesNoteRetriever | None:
    """Single-flight: concurrent first requests (the workspace fires several on
    load) wait for one build instead of each indexing 46k notes."""
    with _RETRIEVER_LOCK:
        return _build_note_retriever()


@lru_cache(maxsize=1)
def _build_note_retriever() -> SalesNoteRetriever | None:
    """The comparable-negotiation index, over every note.

    It used to index ``notes.head(5000)``. With ~2.3 notes per loan and the file
    ordered by loan id, that was ~2,200 loans from the first three months of the
    book - every "comparable" came from late 2024. The whole corpus (46k short
    notes) indexes in about a second with the in-memory TF-IDF store, so the
    slice bought nothing. Outcomes are joined so a comparable says what happened.
    """
    try:
        notes = loader.load_notes()
        if not len(notes):
            return None
        try:
            outcomes = loader.load_training_table()
        except Exception as exc:                    # noqa: BLE001 - outcomes enrich, they are not required
            log.warning("sales note outcomes unavailable: %s", exc)
            outcomes = None
        return SalesNoteRetriever().index(notes, outcomes)
    except Exception as exc:                        # noqa: BLE001 - retrieval is optional context
        log.warning("sales note index unavailable: %s", exc)
        return None


@lru_cache(maxsize=1)
def get_orchestrator() -> PricingOrchestrator:
    return PricingOrchestrator(note_retriever=get_note_retriever())


@lru_cache(maxsize=1)
def get_policy_retriever() -> PolicyRetriever:
    return PolicyRetriever()


@lru_cache(maxsize=1)
def get_loan_explainer() -> LoanExplainer:
    # The same bundle the orchestrator priced with, so the page explains those models.
    return LoanExplainer(get_orchestrator().bundle)


@lru_cache(maxsize=1)
def get_narrator() -> LoanNarrator:
    return LoanNarrator()


def reset() -> None:
    _build_note_retriever.cache_clear()
    get_orchestrator.cache_clear()
    get_policy_retriever.cache_clear()
    get_loan_explainer.cache_clear()
    get_narrator.cache_clear()
    model_registry.clear_cache()
