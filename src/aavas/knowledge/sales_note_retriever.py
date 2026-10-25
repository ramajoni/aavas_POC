"""Retrieve comparable negotiations from the free-text record.

Section 6 of the business statement: sales teams hold competitive intelligence
that never reaches the model in a structured form. This surfaces it as *context*
for a human - "here is what happened the last three times a customer in Jaipur
showed us an HDFC letter" - without letting a free-text claim move the price.

A loan now carries several notes (enquiry, negotiation, follow-up), so the
document id is the ``note_id``; ``loan_id`` rides in the metadata and is what the
outcome join uses. Keying on the loan made later notes overwrite earlier ones in
any store that upserts by id, and made "which note matched" unanswerable.

``competitor_intelligence`` aggregates what the retrieved notes *say* - the
lender, rate and evidence read out of the text by
``intelligence.note_extraction`` - not the structured copies of the competitor
fields, which have one slot per loan and miss every second lender.
"""

from __future__ import annotations

import pandas as pd

from ..logging_utils import get_logger
from .vector_store import Document, VectorStore, get_store

log = get_logger(__name__)


class SalesNoteRetriever:
    def __init__(self, store: VectorStore | None = None):
        # Not `store or get_store()`: an empty store is falsy (it has __len__),
        # so a caller's fresh store would be silently swapped for a new one.
        self.store = store if store is not None else get_store()
        self._outcomes: pd.DataFrame | None = None
        self._extractor = None

    # ------------------------------------------------------------------ index
    def index(self, notes: pd.DataFrame, outcomes: pd.DataFrame | None = None) -> "SalesNoteRetriever":
        has_note_id = "note_id" in notes.columns
        documents = []
        for position, row in enumerate(notes.itertuples(index=False)):
            loan_id = str(row.loan_id)
            documents.append(Document(
                # Older notes files carry one note per loan and no note_id; the
                # position keeps ids unique there too.
                doc_id=str(row.note_id) if has_note_id else f"{loan_id}-{position}",
                text=str(row.note_text),
                metadata={
                    "loan_id": loan_id,
                    "note_type": getattr(row, "note_type", None),
                    "note_date": getattr(row, "note_date", None),
                    "city": getattr(row, "city", None),
                    "state": getattr(row, "state", None),
                    "override_reason": getattr(row, "override_reason", None),
                },
            ))
        self.store.add(documents)
        if outcomes is not None:
            keep = [c for c in ("loan_id", "disbursed", "override_bps", "sanctioned_rate",
                                "realized_value_pct", "bt_out_24m") if c in outcomes.columns]
            self._outcomes = outcomes[keep].drop_duplicates("loan_id").set_index("loan_id")
        log.info("indexed %s sales notes", len(documents))
        # Same reason as the policy index: a comparable negotiation quoted back to
        # a user should be traceable to the corpus it was drawn from.
        try:
            from ..governance import knowledge_registry

            self.index_id = knowledge_registry.register(
                "sales_notes", [d.text for d in documents],
                backend=type(self.store).__name__, corpus="training notes")
        except Exception as exc:                                # noqa: BLE001
            log.warning("sales-note index not registered: %s", exc)
            self.index_id = None
        return self

    # --------------------------------------------------------------- retrieval
    def similar(self, twin, k: int = 3) -> list[dict]:
        """Comparable past situations, with what actually happened to them.

        The file being priced is excluded: its own notes are not a comparable.
        Among lexical candidates, notes that carry a claim about the same lender
        rank first - an enquiry note that merely shares the city and segment is
        not a comparable negotiation."""
        query = self._query_from_twin(twin)
        loan_id = getattr(twin, "loan_id", None) or _get(twin, "loan_id")
        hits = self.search_notes(query, k=max(k * 5, 15), exclude_loan=loan_id)
        context = getattr(twin, "sales_context", None) or {}
        lender = self._canonical(context.get("competitor_name") or _get(twin, "competitor_name"))
        ranked = sorted(enumerate(hits), key=lambda item: (
            not any(c["competitor"] == lender for c in item[1].get("competitors") or []) if lender else False,
            not item[1].get("competitors"),
            item[0]))
        return [hit for _, hit in ranked[:k]]

    def search_notes(self, query: str, k: int = 3, filters: dict | None = None,
                     exclude_loan: str | None = None, extract: bool = True) -> list[dict]:
        """Top-k notes for ``query``, each with its outcome and the competitor
        claims read out of its text."""
        fetch = k + 6 if exclude_loan else k
        results = []
        for hit in self.store.search(query, k=fetch, filters=filters):
            record = hit.as_dict()
            loan_id = record["metadata"].get("loan_id")
            if exclude_loan and loan_id == str(exclude_loan):
                continue
            record["loan_id"] = loan_id
            outcome = self._outcome(loan_id)
            if outcome:
                record["outcome"] = outcome
            if extract:
                record["competitors"] = self._claims(record["text"])
            results.append(record)
            if len(results) >= k:
                break
        return results

    def competitor_intelligence(self, city: str | None = None, k: int = 20) -> dict:
        """Aggregate what the retrieved notes say about competitors, with the
        caveat that this is unverified sales commentary until evidence is attached."""
        # Lexical retrieval only finds what shares a word with the query, so the
        # query carries the vocabulary competitor claims are written in.
        query = (f"competitor rate offer offering quote quoted sanction letter verbal "
                 f"bank housing finance {city or ''}").strip()
        hits = self.store.search(query, k=k, filters={"city": city} if city else None)
        by_competitor: dict[str, dict] = {}
        for hit in hits:
            for claim in self._claims(hit.document.text):
                entry = by_competitor.setdefault(claim["competitor"],
                                                 {"mentions": 0, "evidenced": 0, "rates": [], "notes": []})
                entry["mentions"] += 1
                entry["evidenced"] += int(claim["evidence"] == "documented")
                if claim["rate"] is not None:
                    entry["rates"].append(float(claim["rate"]))
                entry["notes"].append(hit.document.doc_id)

        summary = {}
        for name, entry in by_competitor.items():
            rates = sorted(entry.pop("rates"))
            entry["quotes"] = len(rates)
            entry["median_quoted_rate"] = round(rates[len(rates) // 2], 2) if rates else None
            entry["mean_quoted_rate"] = round(sum(rates) / len(rates), 2) if rates else None
            entry["evidenced_share"] = round(entry["evidenced"] / entry["mentions"], 2)
            summary[name] = entry
        return {
            "city": city,
            "notes_read": len(hits),
            "competitors": summary,
            "source": "extracted from note text (intelligence.note_extraction), not the structured fields",
            "caveat": "Derived from unstructured sales notes. Treat as market signal, not as verified pricing.",
        }

    # ------------------------------------------------------------- internals
    def _outcome(self, loan_id: str | None) -> dict | None:
        if self._outcomes is None or loan_id is None or loan_id not in self._outcomes.index:
            return None
        outcome = self._outcomes.loc[loan_id]
        return {
            "disbursed": bool(outcome.get("disbursed", False)),
            "override_bps": float(outcome.get("override_bps", 0) or 0),
            "sanctioned_rate": float(outcome.get("sanctioned_rate", 0) or 0),
            "realized_value_pct": float(outcome.get("realized_value_pct", 0) or 0),
        }

    def _canonical(self, name) -> str | None:
        if not name:
            return None
        try:
            from ..intelligence.registry import default_registry

            return default_registry().resolve(str(name))
        except Exception:                                       # noqa: BLE001
            return None

    def _claims(self, text: str) -> list[dict]:
        """Lender, rate and evidence per lender in one note. Customer signals are
        skipped: retrieval wants the competitor claim, cheaply."""
        try:
            if self._extractor is None:
                from ..intelligence.note_extraction import NoteExtractor

                self._extractor = NoteExtractor(allow_llm=False)
            note = self._extractor.extract(text, with_signals=False)
        except Exception as exc:                                # noqa: BLE001 - context is optional
            log.warning("note extraction unavailable for retrieval: %s", exc)
            return []
        return [{"competitor": c.competitor, "alias": c.alias, "rate": c.rate,
                 "evidence": c.evidence, "evidence_cue": c.evidence_cue}
                for c in note.competitors().values()]

    def _query_from_twin(self, twin) -> str:
        parts = [
            str(_get(twin, "city") or ""),
            str(_get(twin, "segment") or "").replace("_", " "),
        ]
        # The sales agent sets ``sales_context`` after it asks for comparables,
        # so the twin's own columns are the fallback - without them the query
        # was city + segment only and matched enquiry boilerplate.
        context = getattr(twin, "sales_context", None) or {}
        name = context.get("competitor_name") or _get(twin, "competitor_name")
        rate = context.get("competitor_rate") or _get(twin, "competitor_rate")
        if name:
            parts.append(str(name))
        if context.get("customer_negotiating") or _get(twin, "customer_negotiating"):
            parts.append("customer negotiating rate")
        if rate:
            parts.append(f"competitor offered {float(rate):.2f}")
        if context.get("month_end") or _get(twin, "month_end"):
            parts.append("month end target")
        return " ".join(p for p in parts if p)


def _get(twin, key: str):
    getter = getattr(twin, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:                                       # noqa: BLE001
            return None
    return None
