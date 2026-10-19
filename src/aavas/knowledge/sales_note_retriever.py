"""Retrieve comparable negotiations from the free-text record.

Section 6 of the business statement: sales teams hold competitive intelligence
that never reaches the model in a structured form. This surfaces it as *context*
for a human - "here is what happened the last three times a customer in Jaipur
showed us an HDFC letter" - without letting a free-text claim move the price.
"""

from __future__ import annotations

import pandas as pd

from ..logging_utils import get_logger
from .vector_store import Document, VectorStore, get_store

log = get_logger(__name__)


class SalesNoteRetriever:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or get_store()
        self._outcomes: pd.DataFrame | None = None

    # ------------------------------------------------------------------ index
    def index(self, notes: pd.DataFrame, outcomes: pd.DataFrame | None = None) -> "SalesNoteRetriever":
        documents = [
            Document(
                doc_id=str(row.loan_id),
                text=str(row.note_text),
                metadata={
                    "loan_id": str(row.loan_id),
                    "city": getattr(row, "city", None),
                    "state": getattr(row, "state", None),
                    "override_reason": getattr(row, "override_reason", None),
                    "competitor_name": getattr(row, "competitor_name", None),
                    "competitor_rate": getattr(row, "competitor_rate", None),
                    "evidence_verified": bool(getattr(row, "evidence_verified", False)),
                },
            )
            for row in notes.itertuples(index=False)
        ]
        self.store.add(documents)
        if outcomes is not None:
            keep = [c for c in ("loan_id", "disbursed", "override_bps", "sanctioned_rate",
                                "realized_value_pct", "bt_out_24m") if c in outcomes.columns]
            self._outcomes = outcomes[keep].set_index("loan_id")
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
        """Comparable past situations, with what actually happened to them."""
        query = self._query_from_twin(twin)
        hits = self.store.search(query, k=k, filters=None)
        results = []
        for hit in hits:
            record = hit.as_dict()
            if self._outcomes is not None and record["doc_id"] in self._outcomes.index:
                outcome = self._outcomes.loc[record["doc_id"]]
                record["outcome"] = {
                    "disbursed": bool(outcome.get("disbursed", False)),
                    "override_bps": float(outcome.get("override_bps", 0) or 0),
                    "sanctioned_rate": float(outcome.get("sanctioned_rate", 0) or 0),
                    "realized_value_pct": float(outcome.get("realized_value_pct", 0) or 0),
                }
            results.append(record)
        return results

    def competitor_intelligence(self, city: str | None = None, k: int = 20) -> dict:
        """Aggregate what the notes say about competitors, with the caveat that
        this is unverified sales commentary until evidence is attached."""
        query = f"competitor rate offer {city or ''}".strip()
        hits = self.store.search(query, k=k)
        by_competitor: dict[str, dict] = {}
        for hit in hits:
            name = hit.document.metadata.get("competitor_name")
            if not name:
                continue
            entry = by_competitor.setdefault(name, {"mentions": 0, "evidenced": 0, "rates": []})
            entry["mentions"] += 1
            entry["evidenced"] += int(bool(hit.document.metadata.get("evidence_verified")))
            rate = hit.document.metadata.get("competitor_rate")
            if rate:
                entry["rates"].append(float(rate))

        summary = {}
        for name, entry in by_competitor.items():
            rates = entry.pop("rates")
            entry["mean_quoted_rate"] = round(sum(rates) / len(rates), 2) if rates else None
            entry["evidenced_share"] = round(entry["evidenced"] / entry["mentions"], 2)
            summary[name] = entry
        return {
            "city": city,
            "competitors": summary,
            "caveat": "Derived from unstructured sales notes. Treat as market signal, not as verified pricing.",
        }

    def _query_from_twin(self, twin) -> str:
        parts = [
            str(twin.get("city") or ""),
            str(twin.get("segment") or "").replace("_", " "),
        ]
        context = twin.sales_context or {}
        if context.get("competitor_name"):
            parts.append(str(context["competitor_name"]))
        if context.get("customer_negotiating"):
            parts.append("customer negotiating rate")
        if context.get("competitor_rate"):
            parts.append(f"competitor offered {context['competitor_rate']:.2f}")
        if context.get("month_end"):
            parts.append("month end target")
        return " ".join(p for p in parts if p)
