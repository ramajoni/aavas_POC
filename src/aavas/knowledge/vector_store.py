"""Vector store for the unstructured side of pricing.

What belongs here: sales negotiation notes, competitor observations, credit
notes, branch commentary, the pricing policy itself. What does not: income, LTV,
bureau score, loan amount. Numeric features belong in the feature store where
they can be validated, versioned and monotone-constrained. Putting them in a
vector database buys nothing and loses all of that.

The default backend is in-memory TF-IDF, which needs no server and is honest
about what it is: lexical retrieval over a small corpus. ``pgvector`` is the
phase-2 backend; the interface is identical so swapping is a config change.

``milvus`` is Milvus Lite (embedded, one local file) with BAAI/bge-small-en-v1.5
embeddings, selected by AAVAS_VECTOR_BACKEND=milvus. Where it cannot start (Milvus
Lite has no Windows build) ``get_store`` logs one warning and serves the TF-IDF
store; ``backend_name`` says which is live.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import env
from ..logging_utils import get_logger
from ..paths import ROOT

log = get_logger(__name__)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
MEMORY_BACKEND_NAME = "memory (tf-idf)"


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
    #: Which backend actually serves, for the UI and /agents/tools.
    backend_name: str = "unknown"
    #: Set by ``get_store`` when a requested backend could not start.
    fallback_from: str | None = None
    fallback_reason: str | None = None

    @abstractmethod
    def add(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[SearchHit]: ...

    @abstractmethod
    def __len__(self) -> int: ...


class InMemoryVectorStore(VectorStore):
    """TF-IDF + cosine similarity. Adequate for a POC corpus of tens of thousands
    of short notes, and it keeps the demo runnable with no infrastructure."""

    backend_name = MEMORY_BACKEND_NAME

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



# --------------------------------------------------------------------- Milvus


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


class Embedder:
    """BAAI/bge-small-en-v1.5 through fastembed (ONNX, CPU, no PyTorch).

    Loaded lazily and once per process: the model is ~130 MB on disk and takes a
    second or two to open, and both retrieval indexes share it."""

    def __init__(self, model_name: str = EMBED_MODEL, cache_dir: str | None = None):
        self.model_name = model_name
        self.dim = EMBED_DIM
        self.cache_dir = cache_dir or env("AAVAS_EMBED_CACHE", "data/vector/embed_cache") or "data/vector/embed_cache"
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                cache = _resolve(self.cache_dir)
                cache.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(self.model_name, cache_dir=str(cache))
        return self._model

    def ready(self) -> None:
        """Load the model now so a failure surfaces at start, not at first search."""
        self._load()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load()
        vectors = np.asarray(list(model.embed(list(texts))), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


_EMBEDDER: Embedder | None = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> Embedder:
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = Embedder()
        return _EMBEDDER


_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def milvus_filter(filters: dict | None) -> str:
    """Equality filters on metadata keys -> a Milvus boolean expression.

    Keys are restricted to identifiers and values to str/bool/int/float, so a
    filter can never carry expression syntax of its own."""
    parts = []
    for key, value in (filters or {}).items():
        if not _KEY.match(str(key)):
            raise ValueError(f"unsupported filter key {key!r}")
        if isinstance(value, bool):
            literal = "true" if value else "false"
        elif isinstance(value, (int, float)) and math.isfinite(value):
            literal = repr(value)
        elif isinstance(value, str):
            literal = json.dumps(value)
        else:
            raise ValueError(f"unsupported filter value for {key!r}: {value!r}")
        parts.append(f'metadata["{key}"] == {literal}')
    return " and ".join(parts)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return None if str(value) in ("NaT", "nan", "None") else str(value)


class MilvusVectorStore(VectorStore):
    """Milvus Lite (embedded) with bge-small embeddings, COSINE similarity.

    One collection per store name. The first ``add()`` in a process fixes the
    corpus: if the collection already holds exactly that corpus (same document
    ids and texts, same model - a SHA-256 fingerprint kept in a small
    ``<name>__meta`` collection) nothing is re-embedded; otherwise the collection
    is rebuilt. Later ``add()`` calls in the same process append (upsert by id)."""

    BATCH = 256

    def __init__(self, name: str = "sales_notes", uri: str | None = None,
                 embedder: Embedder | None = None, client: Any = None):
        self.name = name
        self.meta_name = f"{name}__meta"
        self.embedder = embedder or get_embedder()
        self.uri = uri or env("AAVAS_MILVUS_URI", "data/vector/aavas_milvus.db") or "data/vector/aavas_milvus.db"
        if client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise RuntimeError("milvus backend needs pymilvus: pip install 'pymilvus>=2.5'") from exc
            path = _resolve(self.uri)
            path.parent.mkdir(parents=True, exist_ok=True)
            client = MilvusClient(uri=str(path))
        self.client = client
        self.backend_name = f"milvus ({self.embedder.model_name.split('/')[-1]})"
        self._added = False
        self.reused = False          # True when the last add() found its corpus already indexed

    # ------------------------------------------------------------- internals
    def _fingerprint(self, documents: list[Document]) -> str:
        h = hashlib.sha256(self.embedder.model_name.encode("utf-8"))
        for d in documents:
            h.update(b"\x00" + d.doc_id.encode("utf-8") + b"\x01" + d.text.encode("utf-8"))
        return h.hexdigest()

    def _create(self) -> None:
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.embedder.dim)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("metadata", DataType.JSON)
        index = self.client.prepare_index_params()
        index.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(self.name, schema=schema, index_params=index)

    def _stored_fingerprint(self) -> str | None:
        if not self.client.has_collection(self.meta_name):
            return None
        rows = self.client.get(self.meta_name, ids=["fingerprint"], output_fields=["text"])
        return rows[0].get("text") if rows else None

    def _write_fingerprint(self, fingerprint: str) -> None:
        if not self.client.has_collection(self.meta_name):
            self.client.create_collection(self.meta_name, dimension=2, primary_field_name="id",
                                          id_type="string", max_length=64, vector_field_name="vector",
                                          metric_type="COSINE", auto_id=False)
        self.client.upsert(self.meta_name, data=[{"id": "fingerprint", "vector": [1.0, 0.0], "text": fingerprint}])

    def _rows(self) -> int:
        if not self.client.has_collection(self.name):
            return 0
        return int(self.client.get_collection_stats(self.name).get("row_count", 0))

    # ------------------------------------------------------------------- API
    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        first = not self._added
        self._added = True
        fingerprint = self._fingerprint(documents) if first else None
        if first:
            if (self.client.has_collection(self.name) and self._stored_fingerprint() == fingerprint
                    and self._rows() == len(documents)):
                self.reused = True
                log.info("milvus %s already holds this corpus (%s documents); not re-embedding",
                         self.name, len(documents))
                return
            self.reused = False
            for stale in (self.name, self.meta_name):
                if self.client.has_collection(stale):
                    self.client.drop_collection(stale)
            self._create()
        elif not self.client.has_collection(self.name):
            self._create()
        for start in range(0, len(documents), self.BATCH):
            batch = documents[start:start + self.BATCH]
            vectors = self.embedder.embed([d.text for d in batch])
            self.client.upsert(self.name, data=[
                {"id": d.doc_id, "vector": vectors[i].tolist(), "text": d.text,
                 "metadata": _json_safe(d.metadata)} for i, d in enumerate(batch)])
        if first:
            self._write_fingerprint(fingerprint)
        log.info("milvus %s indexed %s documents", self.name, len(documents))

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[SearchHit]:
        if not (query or "").strip() or not self.client.has_collection(self.name):
            return []
        vector = self.embedder.embed([query])[0].tolist()
        found = self.client.search(
            self.name, data=[vector], limit=k, filter=milvus_filter(filters),
            output_fields=["text", "metadata"], search_params={"metric_type": "COSINE"})
        hits = []
        for item in (found[0] if found else []):
            entity = item.get("entity") or {}
            hits.append(SearchHit(
                Document(str(item["id"]), entity.get("text", ""), dict(entity.get("metadata") or {})),
                float(item.get("distance", 0.0))))
        return hits

    def __len__(self) -> int:
        return self._rows()


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


def get_store(backend: str | None = None, name: str = "default") -> VectorStore:
    """The vector store for ``name`` (one Milvus collection per name).

    ``milvus`` that cannot start - pymilvus or milvus-lite missing (always so on
    Windows), the file unopenable, the embedding model unavailable - logs one
    warning and serves in-memory TF-IDF; ``backend_name`` says which one is live."""
    choice = (backend or env("AAVAS_VECTOR_BACKEND", "memory") or "memory").lower()
    if choice == "pgvector":
        return PgVectorStore(env("AAVAS_PG_DSN", "") or "")
    if choice == "milvus":
        try:
            store = MilvusVectorStore(name=name)
            store.embedder.ready()
            return store
        except Exception as exc:                                # noqa: BLE001 - any start failure falls back
            memory = InMemoryVectorStore()
            memory.fallback_from = "milvus"
            memory.fallback_reason = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("AAVAS_VECTOR_BACKEND=milvus but Milvus is unavailable (%s); "
                        "serving in-memory TF-IDF for %s", memory.fallback_reason, name)
            return memory
    return InMemoryVectorStore()

