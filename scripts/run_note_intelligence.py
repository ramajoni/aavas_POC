#!/usr/bin/env python3
"""Historical sales-note extraction, its quality, and the competitor analysis.

    python scripts/run_note_intelligence.py [--rebuild] [--loan-id AAV0009223 ...] [--json out.json]

Reads every note in ``data/raw/sales_notes.csv`` with the deterministic
extractor (the LLM is never called for the batch), caches the per-note table at
``data/processed/note_signals.csv``, scores it against the notes' latent truth -
measurable only because the data is synthetic - and prints the headline numbers
of the competitor analysis. ``--loan-id`` also prints the per-loan competitor
context a reviewer would see for that file (its notes, its structured sales
context, and the AI recommended rate as the rate under consideration).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aavas.ingestion import loader  # noqa: E402
from aavas.intelligence import competitors, note_extraction, quality  # noqa: E402
from aavas.knowledge.sales_note_retriever import SalesNoteRetriever  # noqa: E402

WIDTH = 92


def rule(title: str) -> None:
    print("\n" + "=" * WIDTH + f"\n{title}\n" + "=" * WIDTH)


def pct(value, digits: int = 1) -> str:
    return "  -  " if value is None else f"{value * 100:.{digits}f}%"


def rate(value) -> str:
    return "  -  " if value is None else f"{value:.2f}%"


def bps(value) -> str:
    return "  - " if value is None else f"{value:+.0f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sales-note extraction and competitor intelligence")
    parser.add_argument("--rebuild", action="store_true", help="re-extract even if the cache is fresh")
    parser.add_argument("--loan-id", action="append", default=[], help="print per-loan context (repeatable)")
    parser.add_argument("--json", help="write the quality report and analysis summary to this file")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    signals, mentions = note_extraction.load_history(rebuild=args.rebuild)
    extracted_in = time.perf_counter() - started
    notes = loader.load_notes()
    report = quality.quality_report(signals, notes)
    analysis = competitors.build(mentions=mentions)
    summary = analysis.summary()

    rule("HISTORICAL EXTRACTION")
    recognised = mentions[(mentions["match"] != "unrecognised") & (mentions["competitor"] != "")]
    print(f"  Notes read          {len(signals):,} ({signals['loan_id'].nunique():,} loans) in {extracted_in:.1f}s "
          f"-> {note_extraction.PROCESSED_DIR / note_extraction.SIGNALS_FILE}")
    print(f"  Extractor           {signals['extractor_version'].iloc[0]}  (rule mode; the batch never calls the LLM)")
    print(f"  Notes naming a lender {int((signals['n_mentions'] > 0).sum()):,}   mentions {len(recognised):,}   "
          f"quoted {int(recognised['rate'].notna().sum()):,}")
    print(f"  Matched by          exact {int((recognised['match'] == 'exact').sum()):,}  "
          f"ambiguous-word {int((recognised['match'] == 'ambiguous').sum()):,}  "
          f"near-spelling {int((recognised['match'] == 'fuzzy').sum()):,}  "
          f"unrecognised {int((mentions['match'] == 'unrecognised').sum()):,}")
    evidence = recognised["evidence"].value_counts()
    print(f"  Evidence            documented {evidence.get('documented', 0):,}  "
          f"undocumented {evidence.get('undocumented', 0):,}  unknown {evidence.get('unknown', 0):,}")
    print(f"  Rates by role       customer asks {int(signals['customer_ask_rate'].notna().sum()):,} notes  "
          f"our rate {int(signals['our_rate'].notna().sum()):,} notes")

    rule("EXTRACTION QUALITY  -  measurable only because the data is synthetic")
    if not report.get("available"):
        print(f"  {report['reason']}")
    else:
        micro = report["competitor_identification"]["micro"]
        print(f"  Competitor identification  precision {pct(micro['precision'], 2)}  recall {pct(micro['recall'], 2)}  "
              f"F1 {pct(micro['f1'], 2)}  (tp {micro['tp']:,} fp {micro['fp']:,} fn {micro['fn']:,})")
        r = report["rates"]
        print(f"  Rate exact match           {pct(r['exact_match_accuracy'], 2)} of {r['scored']:,}  "
              f"(wrong {r['wrong']}, missing {r['missing']}; {r['truth_unaligned_notes']} truth rows scored by value)")
        e = report["evidence_vs_structured_flag"]
        c = e["confusion"]
        print(f"  Evidence vs structured flag agreement {pct(e['agreement'])} on {e['classified']:,} classified "
              f"(coverage {pct(e['coverage'])})")
        print(f"      flag verified but note verbal/WhatsApp/no document: {c['undocumented_but_flag_verified']:,}   "
              f"note cites a letter but flag unverified: {c['documented_but_flag_unverified']:,}")
        print(f"  Unrecognised lender names  {len(report['unrecognised_lenders'])}")
        print("  Per competitor (P / R / support / rates exact):")
        for name, row in report["competitor_identification"]["per_competitor"].items():
            print(f"    {name:14s} {pct(row['precision']):>7} {pct(row['recall']):>7} {row['support']:>6,}  "
                  f"{row['rate_exact']:,}/{row['rate_scored']:,}")
        for example in report["examples"]["missing_rate"][:2]:
            print(f"  missing-rate example: \"{example['text'][:80]}\"")

    rule("COMPETITOR ANALYSIS  (context for reviewers, not a price input)")
    base = summary["base_rates"]
    print(f"  Loans {summary['coverage']['loans']:,}; with a competitor in the notes "
          f"{summary['coverage']['loans_with_competitor_mentions']:,}; book win rate {pct(base['win_rate'])}, "
          f"override rate {pct(base['override_rate'])}")
    print(f"\n  {'lender':14s} {'type':12s} {'share':>6} {'files':>6} {'median (p10-p90)':>22} {'gap':>5} "
          f"{'card':>5} {'win':>6} {'lift':>6} {'doc':>6} {'ovr':>6}  most aggressive")
    for row in summary["league"]:
        q = row["quotes"]
        dist = f"{rate(q['median'])} ({rate(q['p10'])}-{rate(q['p90'])})" if q["median"] is not None else "suppressed"
        city = row["most_aggressive_city"]
        print(f"  {row['competitor']:14s} {row['lender_type']:12s} {pct(row['share_of_loans']):>6} "
              f"{row['loans_mentioned']:>6,} {dist:>22} {bps(row['gap_to_our_rate_bps']['median']):>5} "
              f"{bps(row['gap_to_card_bps']['median']):>5} {pct(row['win_rate']['rate']):>6} "
              f"{(row['win_rate'].get('lift_pp') or 0):>+5.1f} {pct(row['evidence']['documented_share']):>6} "
              f"{pct(row['override_when_cited']['rate']):>6}  "
              f"{city['city'] + ' (' + bps(city['median_gap_bps']) + ')' if city else '-'}"
              f"{'' if row['in_structured_data'] else '   [notes only]'}")
    print("\n  Win rate by gap (competitor quote minus our sanctioned rate, bps), all lenders:")
    for bucket in summary["gap_win_curve_all"]:
        print(f"    {bucket['bucket']:>12}  {pct(bucket['rate']):>6}  n={bucket['n']:,}")
    print("\n  Who is new in our market (lenders only in the note text):")
    for row in summary["emerging"]:
        print(f"    {row['competitor']:14s} first seen {row['first_seen']}  {row['loans_mentioned']:>5,} files  "
              f"{row['cities']} cities ({', '.join(c['city'] for c in row['top_cities'])})")
    print(f"  Unrecognised lender names for review: {len(summary['unrecognised'])}")

    if args.loan_id:
        retriever = SalesNoteRetriever().index(notes, loader.load_training_table())
        book = loader.load_training_table().set_index("loan_id")
        for loan_id in args.loan_id:
            if loan_id not in book.index:
                print(f"\n  {loan_id}: not in the training table")
                continue
            loan = book.loc[loan_id]
            file_notes = notes[notes["loan_id"] == loan_id].sort_values("note_date")["note_text"].tolist()
            application = {"loan_id": loan_id, "city": loan["city"], "state": loan["state"],
                           "segment": loan["segment"], "risk_grade": loan["risk_grade"],
                           "our_rate": float(loan["ai_recommended_rate"]),
                           "sales_context": {"competitor_name": loan["competitor_name"],
                                             "competitor_rate": float(loan["competitor_rate"]),
                                             "competitor_evidence_verified": bool(loan["competitor_evidence_verified"])}}
            context = analysis.context_for(application, file_notes, retriever=retriever)
            rule(f"PER-LOAN CONTEXT  {loan_id}  |  {loan['city']}, {loan['segment']}, grade {loan['risk_grade']}, "
                 f"AI rate {loan['ai_recommended_rate']:.2f}%, card {loan['policy_card_rate']:.2f}%")
            for item in context["competitors"]:
                b = item["benchmark"]
                print(f"  {item['competitor']} quote {rate(item['quote'])} from {item['quote_source']} "
                      f"({item['evidence']}{', ' + repr(item['evidence_cue']) if item['evidence_cue'] else ''})")
                print(f"    history [{b['level']}, n={b['n']}] median {rate(b['median'])}, "
                      f"p10-p90 {rate(b['p10'])}-{rate(b['p90'])}")
                print(f"    plausibility: {item['plausibility'].get('message') or item['plausibility'].get('reason')}")
                w = item["win_rate_at_gap"]
                if w.get("available"):
                    city = w["city"] or {}
                    print(f"    gap {bps(w['gap_bps'])} bps -> past win rate {pct(w['all_cities']['rate'])} "
                          f"(n={w['all_cities']['n']}), in {loan['city']} {pct(city.get('rate'))} (n={city.get('n')})")
            for conflict in context["conflicts"]:
                print(f"  [{conflict['severity']}] {conflict['message']}")
            for hit in context["comparable_negotiations"]:
                outcome = hit.get("outcome") or {}
                print(f"  comparable: \"{hit['text'][:70]}\" -> "
                      f"{'disbursed' if outcome.get('disbursed') else 'lost'} at {rate(outcome.get('sanctioned_rate'))}")

    if args.json:
        Path(args.json).write_text(json.dumps({"quality": report, "summary": summary}, indent=2, default=str))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
