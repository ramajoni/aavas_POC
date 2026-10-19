"""Vector store for the unstructured side of pricing.

What belongs here: sales negotiation notes, competitor observations, credit
notes, branch commentary, the pricing policy itself. What does not: income, LTV,
bureau score, loan amount. Numeric features belong in the feature store where
they can be validated, versioned and monotone-constrained. Putting them in a
vector database buys nothing and loses all of that.

The default backend is in-memory TF-IDF, which needs no server and is honest
about what it is: lexical retrieval over a small corpus. ``pgvector`` is the
phase-2 backend; the interface is identical so swapping is a config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import env
from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    document: Document
    score: float

    def as_dict(self) -> dict:
        return {"doc_id": self.document.doc_id, "score": round(self.score, 4),
                "text": self.document.text, "metadata": self.document.metadata}


class VectorStore(ABC):
    @abstractmethod
    def add(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[SearchHit]: ...

    @abstractmethod
    def __len__(self) -> int: ...


class InMemoryVectorStore(VectorStore):
    """TF-IDF + cosine similarity. Adequate for a POC corpus of tens of thousands
    of short notes, and it keeps the demo runnable with no infrastructure."""

    def __init__(self, max_features: int = 20000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", ngram_range=(1, 2), min_df=1)
        self.documents: list[Document] = []
        self._matrix = None

    def add(self, documents: list[Document]) -> None:
        self.documents.extend(documents)
        if self.documents:
            self._matrix = self.vectorizer.fit_transform([d.text for d in self.documents])

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[SearchHit]:
        if self._matrix is None or not self.documents:
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity(self.vectorizer.transform([query]), self._matrix).ravel()
        order = np.argsort(scores)[::-1]

        hits: list[SearchHit] = []
        for index in order:
            if scores[index] <= 0:
                break
            document = self.documents[index]
            if filters and any(document.metadata.get(key) != value for key, value in filters.items()):
                continue
            hits.append(SearchHit(document, float(scores[index])))
            if len(hits) >= k:
                break
        return hits

    def __len__(self) -> int:
        return len(self.documents)


class PgVectorStore(VectorStore):
    """Phase-2 backend. Kept as a real seam rather than a stub comment so the
    swap is a configuration change, not a rewrite."""

    def __init__(self, dsn: str, table: str = "knowledge_documents"):
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "pgvector backend requires psycopg and pgvector: pip install 'psycopg[binary]' pgvector"
            ) from exc
        self.dsn = dsn
        self.table = table
        raise NotImplementedError(
            "pgvector backend is phase 2. Set AAVAS_VECTOR_BACKEND=memory for the POC."
        )

    def add(self, documents: list[Document]) -> None: ...
    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[SearchHit]: ...
    def __len__(self) -> int: return 0


def get_store(backend: str | None = None) -> VectorStore:
    choice = (backend or env("AAVAS_VECTOR_BACKEND", "memory") or "memory").lower()
    if choice == "pgvector":
        return PgVectorStore(env("AAVAS_PG_DSN", "") or "")
    return InMemoryVectorStore()
