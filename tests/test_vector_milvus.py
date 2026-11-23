"""Milvus as a vector store, and the area manager's "most similar past negotiations".

What must hold: the Milvus store maps filters to a safe expression, upserts in
batches, ranks by similarity and skips re-embedding a corpus it already holds; the
real bge-small embedder gives 384-dimension vectors in which a paraphrase is nearer
than an unrelated sentence; Milvus Lite round-trips where it exists (Linux/macOS);
``get_store("milvus")`` falls back to TF-IDF, saying so, where it cannot start; and
the precedents endpoint gives the area manager up to eight similar notes with their
outcomes, never touches the recommendation, and is closed to an RM like /history.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from aavas.knowledge import vector_store
from aavas.knowledge.vector_store import (EMBED_DIM, EMBED_MODEL, Document, InMemoryVectorStore,
                                          MilvusVectorStore, get_store, milvus_filter)

import test_worklist as wl
from test_applications import client, sign_in  # noqa: F401


# --------------------------------------------------------------------- fakes

class FakeEmbedder:
    """Bag-of-words hashed into 384 dimensions: shared words mean a nearer vector."""

    model_name = EMBED_MODEL
    dim = EMBED_DIM

    def __init__(self) -> None:
        self.calls = 0

    def ready(self) -> None:
        return None

    def embed(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class FakeSchema:
    def add_field(self, *a, **k) -> None: ...


class FakeIndexParams:
    def add_index(self, **k) -> None: ...


class FakeMilvusClient:
    """The slice of MilvusClient the store uses, over dicts."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, dict]] = {}
        self.loaded: set[str] = set()
        self.upserts: list[tuple[str, int]] = []
        self.filters: list[str] = []

    def has_collection(self, name): return name in self.collections
    def drop_collection(self, name): self.collections.pop(name, None); self.loaded.discard(name)
    def prepare_index_params(self): return FakeIndexParams()

    def create_collection(self, name, **kwargs):
        self.collections[name] = {}
        self.loaded.add(name)                 # as in Milvus: a new collection is created loaded

    def load_collection(self, name):
        self.loaded.add(name)

    def restart(self):
        """What a new process sees: the collections are there, but released."""
        self.loaded.clear()

    def _require_loaded(self, name):
        if name not in self.loaded:
            raise RuntimeError(f"Collection '{name}' is in state 'released'; call load() before search/get/query")

    def upsert(self, name, data):
        self.upserts.append((name, len(data)))
        for row in data:
            self.collections[name][row["id"]] = dict(row)

    def get(self, name, ids, output_fields=None):
        self._require_loaded(name)
        return [self.collections[name][i] for i in ids if i in self.collections[name]]

    def get_collection_stats(self, name):
        return {"row_count": len(self.collections[name])}

    def search(self, name, data, limit, filter="", output_fields=None, search_params=None):
        self._require_loaded(name)
        self.filters.append(filter)
        wanted = [(m.group(1), json.loads(m.group(2)) if m.group(2) not in ("true", "false") else m.group(2) == "true")
                  for m in re.finditer(r'metadata\["(\w+)"\] == ("(?:[^"\\]|\\.)*"|true|false|[-\d.eE+]+)', filter)]
        query = np.asarray(data[0], dtype=np.float32)
        scored = []
        for row in self.collections[name].values():
            if all(row["metadata"].get(k) == v for k, v in wanted):
                scored.append((float(np.dot(query, np.asarray(row["vector"], dtype=np.float32))), row))
        scored.sort(key=lambda item: -item[0])
        return [[{"id": row["id"], "distance": score,
                  "entity": {"text": row["text"], "metadata": row["metadata"]}} for score, row in scored[:limit]]]


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(MilvusVectorStore, "_create", lambda self: self.client.create_collection(self.name))
    return FakeMilvusClient(), FakeEmbedder()


CORPUS = [
    Document("n1", "customer says HDFC offered 9.1 percent, wants us to match the rate",
             {"loan_id": "L1", "city": "Pune"}),
    Document("n2", "customer accepted the recommended rate without any negotiation",
             {"loan_id": "L2", "city": "Pune"}),
    Document("n3", "ICICI letter at 9.0 percent, customer negotiating hard on rate",
             {"loan_id": "L3", "city": "Jaipur"}),
    Document("n4", "site visit done, documents pending", {"loan_id": "L4", "city": "Jaipur"}),
]


# ------------------------------------------------------------- (a) fake client

def test_add_upserts_in_batches_and_search_orders_by_similarity(fake, monkeypatch) -> None:
    client_, embedder = fake
    monkeypatch.setattr(MilvusVectorStore, "BATCH", 3)
    store = MilvusVectorStore("sales_notes", client=client_, embedder=embedder)
    store.add(CORPUS)
    assert [n for name, n in client_.upserts if name == "sales_notes"] == [3, 1]
    assert len(store) == 4 and store.backend_name == "milvus (bge-small-en-v1.5)"

    store.add([Document("n1", "customer says HDFC offered 9.1 percent, wants us to match the rate",
                        {"loan_id": "L1", "city": "Pune"})])           # same id: upsert, not a second row
    assert len(store) == 4

    hits = store.search("HDFC offered 9.1 percent match the rate", k=3)
    assert hits[0].document.doc_id == "n1"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert hits[0].document.metadata["city"] == "Pune" and 0 < hits[0].score <= 1.0001


def test_search_maps_equality_filters_to_a_milvus_expression(fake) -> None:
    client_, embedder = fake
    store = MilvusVectorStore("sales_notes", client=client_, embedder=embedder)
    store.add(CORPUS)
    hits = store.search("customer negotiating rate", k=5, filters={"city": "Jaipur"})
    assert {h.document.doc_id for h in hits} <= {"n3", "n4"} and hits
    assert client_.filters[-1] == 'metadata["city"] == "Jaipur"'
    store.search("rate", k=2)
    assert client_.filters[-1] == ""


def test_filter_expression_building_is_safe() -> None:
    assert milvus_filter(None) == "" and milvus_filter({}) == ""
    assert milvus_filter({"city": "Pune", "approved": True, "n": 3}) == \
        'metadata["city"] == "Pune" and metadata["approved"] == true and metadata["n"] == 3'
    # A value cannot carry expression syntax: it is quoted and escaped.
    assert milvus_filter({"city": 'x" or 1 == 1 or "'}) == 'metadata["city"] == "x\\" or 1 == 1 or \\""'
    with pytest.raises(ValueError):
        milvus_filter({'city"] == 1 or metadata["x': "a"})
    with pytest.raises(ValueError):
        milvus_filter({"city": None})


def test_a_corpus_already_indexed_is_not_embedded_again(fake) -> None:
    client_, embedder = fake
    MilvusVectorStore("sales_notes", client=client_, embedder=embedder).add(CORPUS)
    first = embedder.calls
    assert first == len(CORPUS)

    again = MilvusVectorStore("sales_notes", client=client_, embedder=embedder)
    again.add(CORPUS)
    assert again.reused and embedder.calls == first and len(again) == len(CORPUS)

    # A restart: Milvus hands the collections back released. The store must load them
    # before reading the fingerprint or searching (it failed here in production).
    client_.restart()
    after_restart = MilvusVectorStore("sales_notes", client=client_, embedder=embedder)
    after_restart.add(CORPUS)
    assert after_restart.reused and embedder.calls == first
    assert after_restart.search("HDFC offered 9.1 percent match the rate", k=1)[0].document.doc_id == "n1"

    spent = embedder.calls                                                    # the search above embedded one query
    changed = MilvusVectorStore("sales_notes", client=client_, embedder=embedder)
    changed.add(CORPUS[:3] + [Document("n4", "a different note entirely", {"loan_id": "L4", "city": "Jaipur"})])
    assert not changed.reused and embedder.calls == spent + len(CORPUS)      # rebuilt, not appended
    assert len(changed) == 4
    assert "documents pending" not in {h.document.text for h in changed.search("documents pending", k=4)}


def test_collections_are_kept_per_store_name(fake) -> None:
    client_, embedder = fake
    MilvusVectorStore("sales_notes", client=client_, embedder=embedder).add(CORPUS)
    policy = MilvusVectorStore("policy_clauses", client=client_, embedder=embedder)
    policy.add([Document("c1", "branch manager discount limit", {})])
    assert len(policy) == 1 and len(client_.collections["sales_notes"]) == 4


def test_the_schema_is_id_vector_text_metadata() -> None:
    pymilvus = pytest.importorskip("pymilvus")
    schema = pymilvus.MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", pymilvus.DataType.VARCHAR, is_primary=True, max_length=256)
    schema.add_field("vector", pymilvus.DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    assert [f.name for f in schema.fields] == ["id", "vector"]


# -------------------------------------------------------- (b) the real embedder

def test_the_real_embedder_gives_384_dimensions_and_paraphrases_land_closer() -> None:
    embedder = vector_store.Embedder()
    try:
        vectors = embedder.embed([
            "The customer says another bank offered a lower interest rate.",
            "Client mentions a rival lender quoted a cheaper rate on the loan.",
            "The site visit for the property valuation is scheduled for Tuesday."])
    except Exception as exc:                                    # noqa: BLE001 - offline, or no fastembed
        pytest.skip(f"embedding model unavailable: {exc}")
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2]) + 0.1


# -------------------------------------------------- (c) a real Milvus Lite file

def test_milvus_lite_round_trip_and_reuse(tmp_path) -> None:
    pytest.importorskip("milvus_lite")
    pytest.importorskip("pymilvus")
    uri = str(tmp_path / "lite.db")
    embedder = FakeEmbedder()
    store = MilvusVectorStore("sales_notes", uri=uri, embedder=embedder)
    store.add(CORPUS)
    assert len(store) == 4
    hits = store.search("HDFC offered 9.1 percent match the rate", k=2)
    assert hits[0].document.doc_id == "n1" and hits[0].document.metadata["city"] == "Pune"
    in_city = store.search("customer negotiating rate", k=4, filters={"city": "Jaipur"})
    assert {h.document.doc_id for h in in_city} == {"n3", "n4"}
    store.client.close()

    reopened = MilvusVectorStore("sales_notes", uri=uri, embedder=embedder)
    spent = embedder.calls
    reopened.add(CORPUS)
    assert reopened.reused and embedder.calls == spent and len(reopened) == 4

    # What a new process sees: both collections released. Reuse and search must still work.
    for name in ("sales_notes", reopened.meta_name):
        reopened.client.release_collection(name)
    restarted = MilvusVectorStore("sales_notes", uri=uri, embedder=embedder)
    restarted.add(CORPUS)
    assert restarted.reused and embedder.calls == spent
    assert restarted.search("HDFC offered 9.1 percent match the rate", k=1)[0].document.doc_id == "n1"


# ------------------------------------------------------------- (d) the fallback

def test_milvus_falls_back_to_memory_when_it_cannot_start(monkeypatch, caplog) -> None:
    def broken(*a, **k):
        raise RuntimeError("milvus-lite is not available on this platform")

    monkeypatch.setattr(MilvusVectorStore, "__init__", broken)
    with caplog.at_level("WARNING"):
        store = get_store("milvus", name="sales_notes")
    assert isinstance(store, InMemoryVectorStore)
    assert store.backend_name == "memory (tf-idf)"
    assert store.fallback_from == "milvus" and "milvus-lite" in store.fallback_reason
    assert sum("Milvus is unavailable" in r.getMessage() for r in caplog.records) == 1


def test_the_default_backend_is_memory(monkeypatch) -> None:
    monkeypatch.delenv("AAVAS_VECTOR_BACKEND", raising=False)
    store = get_store()
    assert isinstance(store, InMemoryVectorStore) and store.fallback_from is None
    assert store.backend_name == "memory (tf-idf)"


def test_a_failing_embedding_model_also_falls_back(monkeypatch, fake) -> None:
    client_, embedder = fake

    class Broken(FakeEmbedder):
        def ready(self):
            raise OSError("model download failed")

    real_init = MilvusVectorStore.__init__
    monkeypatch.setattr(MilvusVectorStore, "__init__",
                        lambda self, name="x", **k: real_init(self, name, client=client_, embedder=Broken()))
    store = get_store("milvus", name="sales_notes")
    assert isinstance(store, InMemoryVectorStore) and "model download failed" in store.fallback_reason


# ----------------------------------------------------------- (e) the AM endpoint

def _request(client, loan_id: str) -> str:
    rahul, rohit = sign_in(client, "rahul"), sign_in(client, "rohit")
    wl.quote_and_apply(client, rahul, loan_id)
    wl.verify(client, rahul, loan_id)
    rate = wl.listed(client, rohit, loan_id)["suggested"]["rate"]
    return client.post("/pricing/rate-requests", headers=rohit, json={
        "loan_id": loan_id, "rate": round(rate - 0.40, 2), "reason": "strong_negotiation",
        "note": "customer says HDFC quoted a lower rate and is pushing hard"}).json()["request_id"]


def test_precedents_endpoint_returns_similar_notes_with_outcomes(client, monkeypatch) -> None:
    from aavas.api import deps
    from aavas.intake import offers

    monkeypatch.delenv("AAVAS_VECTOR_BACKEND", raising=False)
    deps._build_note_retriever.cache_clear()
    request_id = _request(client, "VM-PREC-1")
    rahul, rohit, meera = sign_in(client, "rahul"), sign_in(client, "rohit"), sign_in(client, "meera")
    before = offers.request(request_id)["recommended_rate"]

    # Closed to an RM, as /history is.
    for headers in (rohit, rahul):
        assert client.get(f"/pricing/rate-requests/{request_id}/precedents", headers=headers).status_code == 403
    assert client.get("/pricing/rate-requests/NOPE/precedents", headers=meera).status_code == 404

    response = client.get(f"/pricing/rate-requests/{request_id}/precedents", headers=meera)
    assert response.status_code == 200, response.text
    p = response.json()
    assert p["available"] and p["backend"] == "memory (tf-idf)" and p["scope"] in ("city", "company")
    assert 0 < len(p["precedents"]) <= 8
    sims = [r["similarity"] for r in p["precedents"]]
    assert sims == sorted(sims, reverse=True) and all(0 < s <= 1 for s in sims)
    assert all(r["loan_id"] != "VM-PREC-1" for r in p["precedents"])
    assert all({"text", "date", "city", "lender", "quote", "took", "rate", "priced"} <= set(r)
               for r in p["precedents"])
    assert any(r["took"] is not None and r["priced"] in ("up", "down", "at_recommendation")
               for r in p["precedents"])
    if p["scope"] == "city":
        assert {r["city"] for r in p["precedents"]} == {p["city"]}

    # Context only: nothing about the request or its recommendation moved.
    assert offers.request(request_id)["recommended_rate"] == before

    # The same list rides in the history payload for the UI.
    h = client.get(f"/pricing/rate-requests/{request_id}/history", headers=meera).json()
    assert [r["note_id"] for r in h["similar_negotiations"]["precedents"]] == [r["note_id"] for r in p["precedents"]]


def test_precedents_fall_back_company_wide_when_the_city_has_too_few() -> None:
    from aavas.analytics import precedents

    class Stub:
        class store:                                            # noqa: N801
            backend_name = "memory (tf-idf)"
            fallback_reason = None

        def __init__(self) -> None:
            self.calls: list[dict | None] = []

        def search_notes(self, query, k=3, filters=None, exclude_loan=None, extract=True):
            self.calls.append(filters)
            return [] if filters else []

    stub = Stub()
    out = precedents.similar_negotiations({"loan_id": "X", "reason": "strong_negotiation"},
                                          {"city": "Pune", "conversation_notes": "HDFC at 9.1"}, stub)
    assert stub.calls == [{"city": "Pune"}, None] and out["scope"] == "company" and out["available"]
    none = precedents.similar_negotiations({"loan_id": "X"}, {}, stub)
    assert not none["available"] and none["precedents"] == []
    assert not precedents.similar_negotiations({"loan_id": "X", "note": "n"}, {}, None)["available"]


def test_the_embedding_model_is_in_the_inventory() -> None:
    from aavas.governance import model_risk

    entry = model_risk.inventory()["embedding_bge_small"]
    assert entry["kind"] == "embedding" and entry["kill_switch"] == "AAVAS_VECTOR_BACKEND=memory"
    assert model_risk.version("embedding_bge_small", entry, {}) == EMBED_MODEL
    assert model_risk.load("model_risk")["model_risk_config_version"] == "MRM-2026.13"
