"""The past negotiations most like this file's conversation, for the area manager.

Given a lower-rate request and its application, the file's conversation notes, the
competitor named and quoted, and the request's reason become one query against
the sales-notes vector store. The top notes come back with what happened on the
file each one was written on: did the customer take the loan, at what rate, and
was the price cut, held at the recommendation or raised.

**Context for a person, never an input to a price.** Nothing here is read by the
pricing models, the rate grid, the suggestion or the approval limits; the
recommendation on the request is untouched. Retrieval is semantic similarity
over free text, not a statistical comparable, so each result is shown with its
similarity and its outcome and left for the area manager to weigh.

Same-city notes are preferred; when fewer than MIN_IN_CITY turn up the search
widens to the whole company. The payload names the backend that served it
(Milvus with bge-small embeddings, or the in-memory TF-IDF fallback).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..ingestion import loader
from . import portfolio

TOP_K = 8
MIN_IN_CITY = 3
_PRICED = {"premium": "up", "discount": "down", "none": "at_recommendation"}


def query_for(request: dict[str, Any], applicant: dict[str, Any]) -> str:
    """The file's conversation, the competitor named and quoted, and the reason."""
    sales = {**{k: applicant.get(k) for k in ("competitor_name", "competitor_rate")},
             **(applicant.get("sales_context") or {})}
    parts = [(applicant.get("conversation_notes") or "").strip(), (request.get("note") or "").strip()]
    if request.get("reason"):
        parts.append(f"reason for lower rate: {str(request['reason']).replace('_', ' ')}")
    if sales.get("competitor_name"):
        quote = sales.get("competitor_rate")
        parts.append(f"competitor {sales['competitor_name']}"
                     + (f" quoted {float(quote):.2f}%" if quote not in (None, "") and not pd.isna(quote) else ""))
    return " ".join(p for p in parts if p)


def _num(value: Any, digits: int = 2) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), digits)


def _text(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def similar_negotiations(request: dict[str, Any], applicant: dict[str, Any], retriever,
                         *, k: int = TOP_K) -> dict[str, Any]:
    """Top ``k`` most similar past notes with their outcomes. ``retriever`` is the
    ``SalesNoteRetriever``; without one the panel says it is unavailable."""
    store = getattr(retriever, "store", None)
    out: dict[str, Any] = {
        "available": False, "precedents": [], "scope": None, "query": None,
        "backend": getattr(store, "backend_name", None), "fallback_reason": getattr(store, "fallback_reason", None),
        "note": ("Similar past negotiations, found by meaning. Context for you to weigh; they do not change "
                 "the recommendation or any price."),
    }
    if retriever is None:
        out["unavailable"] = "the sales-note index is not built"
        return out
    query = query_for(request, applicant)
    out["query"] = query
    if not query:
        out["unavailable"] = "no conversation or competitor detail on this file to compare"
        return out
    loan = request.get("loan_id")
    city = applicant.get("city")
    hits: list[dict] = []
    scope = "company"
    if city:
        hits = retriever.search_notes(query, k=k, filters={"city": city}, exclude_loan=loan, extract=False)
        scope = "city"
    if len(hits) < MIN_IN_CITY:
        hits, scope = retriever.search_notes(query, k=k, exclude_loan=loan, extract=False), "company"
    out.update(available=True, scope=scope, city=city if scope == "city" else None,
               precedents=_describe(hits))
    return out


def _describe(hits: list[dict]) -> list[dict[str, Any]]:
    if not hits:
        return []
    ids = [h["doc_id"] for h in hits]
    loans = {h["loan_id"] for h in hits}
    notes = loader.load_notes()
    by_note = notes[notes["note_id"].isin(ids)].drop_duplicates("note_id").set_index("note_id") \
        if "note_id" in notes.columns else pd.DataFrame()
    book = portfolio.frame()
    outcome = book[book["loan_id"].isin(loans)].drop_duplicates("loan_id") \
        .set_index("loan_id")[["disbursed", "sanctioned_rate", "override_direction"]]
    rows = []
    for h in hits:
        meta = h.get("metadata") or {}
        row = by_note.loc[h["doc_id"]] if h["doc_id"] in by_note.index else None
        won = outcome.loc[h["loan_id"]] if h["loan_id"] in outcome.index else None
        rows.append({
            "similarity": round(float(h["score"]), 4),
            "note_id": h["doc_id"], "loan_id": h["loan_id"],
            "text": h["text"],
            "date": _text(meta.get("note_date")) or (_text(row["note_date"]) if row is not None else None),
            "city": _text(meta.get("city")) or (_text(row["city"]) if row is not None else None),
            "lender": _text(row["competitor_name"]) if row is not None else None,
            "quote": _num(row["competitor_rate"]) if row is not None else None,
            "took": None if won is None or pd.isna(won["disbursed"]) else bool(won["disbursed"]),
            "rate": _num(won["sanctioned_rate"]) if won is not None else None,
            "priced": _PRICED.get(str(won["override_direction"])) if won is not None else None,
        })
    return rows
