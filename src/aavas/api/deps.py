"""Shared dependencies. The orchestrator and its artefacts load once per process."""

from __future__ import annotations

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


@lru_cache(maxsize=1)
def get_orchestrator() -> PricingOrchestrator:
    retriever = None
    try:
        notes = loader.load_notes()
        if len(notes):
            retriever = SalesNoteRetriever().index(notes.head(5000))
    except Exception as exc:                        # noqa: BLE001 - retrieval is optional context
        log.warning("sales note index unavailable: %s", exc)
    return PricingOrchestrator(note_retriever=retriever)


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
    get_orchestrator.cache_clear()
    get_policy_retriever.cache_clear()
    get_loan_explainer.cache_clear()
    get_narrator.cache_clear()
    model_registry.clear_cache()
