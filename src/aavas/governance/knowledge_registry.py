"""What the knowledge indexes were built from.

The policy and sales-note indexes are rebuilt at every start. Without a record of
what went into them, a policy answer cannot be traced to the text that produced
it: the retriever returns a clause, and nothing says which version of the file
that clause came from or how many documents it was chosen against.

The content hash covers the documents themselves, so re-indexing an unchanged
corpus produces the same index id and does not fill the table with noise.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from ..logging_utils import get_logger
from . import store

log = get_logger(__name__)


def _digest(documents: Iterable[str]) -> str:
    sha = hashlib.sha256()
    for text in documents:
        sha.update(str(text).encode())
        sha.update(b"\x00")
    return sha.hexdigest()


def register(name: str, documents: list[str], *, backend: str = "memory",
             corpus: str = "", actor: str = "startup") -> str:
    digest = _digest(documents)
    index_id = f"IDX-{name}-{digest[:8]}"
    if store.fetch_one("SELECT index_id FROM knowledge_indexes WHERE index_id = ?", (index_id,)):
        return index_id
    store.insert("knowledge_indexes", {
        "index_id": index_id, "name": name, "built_at": store.now(), "backend": backend,
        "documents": len(documents), "corpus": corpus, "content_hash": digest,
    })
    from . import audit
    audit.record(actor, "register_knowledge_index", "knowledge", index_id,
                 {"name": name, "documents": len(documents), "backend": backend})
    log.info("registered knowledge index %s (%s documents, %s)", index_id, len(documents), backend)
    return index_id


def registered(limit: int = 50) -> list[dict[str, Any]]:
    return store.query("SELECT * FROM knowledge_indexes ORDER BY built_at DESC LIMIT ?", (limit,))


def latest(name: str) -> dict[str, Any] | None:
    return store.fetch_one(
        "SELECT * FROM knowledge_indexes WHERE name = ? ORDER BY built_at DESC LIMIT 1", (name,))
