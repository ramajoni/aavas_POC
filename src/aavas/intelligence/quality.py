"""Extraction quality, scored against the notes' latent truth.

Measurable only because the data is synthetic. The generator wrote down, per
note, which lenders the text actually mentions and at what rate
(``true_competitors`` / ``true_competitor_rates``). A real notes file has no such
columns, and in production this report would come from a hand-labelled sample
instead. This is the only module in the package that reads ``true_*`` columns,
and it reads them *after* extraction, joined on ``note_id`` - the extractor
never sees them (``note_extraction.INPUT_COLUMNS``).

What it measures:

* competitor identification - precision / recall / F1 over (note, lender)
  pairs, micro-averaged and per lender, with the misses and false alarms as
  examples a reviewer can read;
* rates - on correctly identified mentions that carry a true rate, how often
  the extracted rate matches exactly, is wrong, or is missing;
* evidence - agreement of the documented / undocumented class with the
  structured ``evidence_verified`` flag, on mentions of the structured lender.
  That flag is itself a human entry, not ground truth, so disagreement is a
  finding about the file as much as about the extractor;
* unrecognised lender-shaped names, with counts, for an analyst to triage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .note_extraction import DOCUMENTED, UNDOCUMENTED
from .registry import CompetitorRegistry, default_registry

CAVEAT = ("Measurable only because the data is synthetic: the notes file carries the generator's "
          "latent truth. A production notes file has no such columns; score a hand-labelled sample instead.")
RATE_TOLERANCE = 0.005


def _split(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    return [part.strip() for part in text.split("|")] if text.strip() else []


def _rate(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    r = lambda v: None if v is None else round(v, 4)                   # noqa: E731
    return {"precision": r(precision), "recall": r(recall), "f1": r(f1),
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}


def quality_report(signals: pd.DataFrame, notes: pd.DataFrame,
                   registry: CompetitorRegistry | None = None, max_examples: int = 8) -> dict[str, Any]:
    """Score per-note extraction (``signals``) against the truth in ``notes``."""
    registry = registry or default_registry()
    truth_cols = ["note_id", "true_competitors", "true_competitor_rates"]
    missing = [c for c in truth_cols if c not in notes.columns]
    if missing:
        return {"available": False, "caveat": CAVEAT,
                "reason": f"notes carry no latent truth columns ({', '.join(missing)}); "
                          "extraction quality cannot be measured on this file"}

    extra = [c for c in ("competitor_name", "evidence_verified", "note_text") if c in notes.columns]
    joined = signals.merge(notes[truth_cols + extra], on="note_id", how="inner")

    unmapped_truth: Counter = Counter()
    pairs = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    rate_outcomes: Counter = Counter()
    rate_by_lender = defaultdict(Counter)
    evidence_confusion: Counter = Counter()
    evidence_unknown = 0
    exact_sets = 0
    examples = {"false_positive": [], "false_negative": [], "wrong_rate": [], "missing_rate": []}
    unaligned_notes = 0

    for row in joined.itertuples(index=False):
        predicted = _split(row.competitors)
        p_rates = dict(zip(predicted, (_rate(v) for v in _split(row.rates) + [""] * len(predicted))))
        p_evidence = dict(zip(predicted, _split(row.evidence)))

        truth: dict[str, float | None] = {}
        true_names = _split(row.true_competitors)
        true_rates = [_rate(v) for v in _split(row.true_competitor_rates)]
        # The truth lists are positional, except where the generator dropped an
        # empty rate slot (fewer rates than lenders). Those notes are scored by
        # value instead of by position, and counted.
        aligned = len(true_rates) == len(true_names) or not true_rates
        unaligned_notes += not aligned
        for position, name in enumerate(true_names):
            canonical = name if name in registry.lenders else registry.resolve(name)
            if canonical is None:
                unmapped_truth[name] += 1
                canonical = name
            rate = true_rates[position] if aligned and position < len(true_rates) else None
            truth.setdefault(canonical, rate)

        p_set, t_set = set(predicted), set(truth)
        exact_sets += p_set == t_set
        pool = [r for r in true_rates if r is not None] if not aligned else None
        for name in sorted(p_set & t_set):
            pairs[name]["tp"] += 1
            got = p_rates.get(name)
            if pool is not None:
                hit = next((r for r in pool if got is not None and abs(got - r) <= RATE_TOLERANCE), None)
                if hit is not None:
                    pool.remove(hit)
                    true_rate = hit
                else:
                    true_rate = pool[0] if (pool and got is not None) else None
            else:
                true_rate = truth[name]
            if true_rate is None:
                outcome = "rate_extracted_without_true_rate" if got is not None else None
            elif got is None:
                outcome = "missing"
            elif abs(got - true_rate) <= RATE_TOLERANCE:
                outcome = "exact"
            else:
                outcome = "wrong"
                if len(examples["wrong_rate"]) < max_examples:
                    examples["wrong_rate"].append({"note_id": row.note_id, "competitor": name,
                                                   "extracted": got, "truth": true_rate,
                                                   "text": getattr(row, "note_text", None)})
            if outcome == "missing" and len(examples["missing_rate"]) < max_examples:
                examples["missing_rate"].append({"note_id": row.note_id, "competitor": name,
                                                 "truth": true_rate,
                                                 "text": getattr(row, "note_text", None)})
            if outcome:
                rate_outcomes[outcome] += 1
                rate_by_lender[name][outcome] += 1
        if pool:
            rate_outcomes["missing"] += len(pool)
        for name in p_set - t_set:
            pairs[name]["fp"] += 1
            if len(examples["false_positive"]) < max_examples:
                examples["false_positive"].append({"note_id": row.note_id, "extracted": name,
                                                   "truth": sorted(t_set),
                                                   "text": getattr(row, "note_text", None)})
        for name in t_set - p_set:
            pairs[name]["fn"] += 1
            if len(examples["false_negative"]) < max_examples:
                examples["false_negative"].append({"note_id": row.note_id, "missed": name,
                                                   "extracted": sorted(p_set),
                                                   "text": getattr(row, "note_text", None)})

        # Evidence agreement: only the structured lender's mention is comparable,
        # because the structured flag is about that lender's quote.
        structured = registry.resolve(getattr(row, "competitor_name", None))
        if structured and structured in p_evidence and hasattr(row, "evidence_verified"):
            label = p_evidence[structured]
            if label in (DOCUMENTED, UNDOCUMENTED):
                evidence_confusion[(label, bool(row.evidence_verified))] += 1
            else:
                evidence_unknown += 1

    total = {k: sum(v[k] for v in pairs.values()) for k in ("tp", "fp", "fn")}
    per_lender = {name: {**_prf(**counts), "rate_exact": rate_by_lender[name].get("exact", 0),
                         "rate_scored": sum(rate_by_lender[name][k] for k in ("exact", "wrong", "missing"))}
                  for name, counts in sorted(pairs.items(), key=lambda kv: -(kv[1]["tp"] + kv[1]["fn"]))}

    scored_rates = sum(rate_outcomes[k] for k in ("exact", "wrong", "missing"))
    agree = evidence_confusion[(DOCUMENTED, True)] + evidence_confusion[(UNDOCUMENTED, False)]
    classified = sum(evidence_confusion.values())

    unrecognised = Counter(a for value in signals.get("unrecognised", pd.Series(dtype=str))
                           for a in _split(value))
    return {
        "available": True,
        "caveat": CAVEAT,
        "notes_scored": int(len(joined)),
        "notes_with_true_mentions": int(sum(1 for v in joined["true_competitors"] if _split(v))),
        "note_set_exact_match": round(exact_sets / len(joined), 4) if len(joined) else None,
        "competitor_identification": {"micro": _prf(**total), "per_competitor": per_lender},
        "rates": {
            "scored": scored_rates,
            "exact_match_accuracy": round(rate_outcomes["exact"] / scored_rates, 4) if scored_rates else None,
            "exact": rate_outcomes["exact"], "wrong": rate_outcomes["wrong"],
            "missing": rate_outcomes["missing"],
            "extracted_where_truth_has_none": rate_outcomes["rate_extracted_without_true_rate"],
            "definition": "on correctly identified mentions whose truth carries a rate; exact = within 0.005",
            "truth_unaligned_notes": unaligned_notes,
            "truth_unaligned_note": ("notes whose truth lists fewer rates than lenders (an empty slot was "
                                     "dropped by the generator) are scored by value, not position"),
        },
        "evidence_vs_structured_flag": {
            "classified": classified,
            "agreement": round(agree / classified, 4) if classified else None,
            "unknown_class": evidence_unknown,
            "coverage": round(classified / (classified + evidence_unknown), 4) if classified + evidence_unknown else None,
            "confusion": {
                "documented_and_flag_verified": evidence_confusion[(DOCUMENTED, True)],
                "documented_but_flag_unverified": evidence_confusion[(DOCUMENTED, False)],
                "undocumented_and_flag_unverified": evidence_confusion[(UNDOCUMENTED, False)],
                "undocumented_but_flag_verified": evidence_confusion[(UNDOCUMENTED, True)],
            },
            "definition": ("mentions of the structured lender with a documented/undocumented class, "
                           "compared with the structured evidence_verified flag (a human entry, not truth)"),
        },
        "unrecognised_lenders": [{"name": n, "notes": c} for n, c in unrecognised.most_common()],
        "truth_labels_not_in_registry": dict(unmapped_truth),
        "examples": examples,
    }
