#!/usr/bin/env python3
"""End-to-end walkthrough of one customer, then the learning loop.

    python scripts/run_demo.py [--loan-id AAV0000123] [--skip-federated]

Prints the journey the business statement describes: an early band, a
risk-adjusted recommendation with reason codes, a human override, the outcome,
and what the platform learned from all of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aavas.agents.orchestrator import PricingOrchestrator  # noqa: E402
from aavas.governance import audit, store  # noqa: E402
from aavas.ingestion import loader  # noqa: E402
from aavas.knowledge.sales_note_retriever import SalesNoteRetriever  # noqa: E402

WIDTH = 92


def rule(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def money(value: float | None) -> str:
    return "-" if value is None else f"Rs {value:,.0f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aavas pricing engine walkthrough")
    parser.add_argument("--loan-id", default=None)
    parser.add_argument("--skip-federated", action="store_true")
    parser.add_argument("--fresh-store", action="store_true", help="reset the governance store first")
    args = parser.parse_args(argv)

    if args.fresh_store:
        store.reset()

    df = loader.load_training_table()
    row = (df[df["loan_id"] == args.loan_id].iloc[0] if args.loan_id
           else df[(df["competitor_rate"].notna()) & (df["customer_negotiating"])].iloc[7])
    application = row.to_dict()

    notes = loader.load_notes()
    retriever = SalesNoteRetriever().index(notes.head(5000), df.head(5000)) if len(notes) else None
    orchestrator = PricingOrchestrator(note_retriever=retriever)

    rule(f"CUSTOMER  {row['loan_id']}  |  {row['city']}, {row['state']}")
    print(f"  Segment            {row['segment'].replace('_', ' ')}  via {row['channel'].replace('_', ' ')}")
    print(f"  Declared income    Rs {row['declared_monthly_income']:,.0f}/month")
    print(f"  Verified income    Rs {row['verified_monthly_income']:,.0f}/month "
          f"({row['income_verification_haircut']:.0%} haircut)")
    print(f"  Loan / property    {money(row['sanctioned_loan_amount'])} against "
          f"{money(row['appraised_property_value'])}  (LTV {row['ltv']:.0%}, FOIR {row['foir']:.0%})")
    print(f"  Bureau             {int(row['bureau_score'])}   Tenure {int(row['tenure_years'])} years")
    print(f"  Competitor         {row['competitor_name']} at {row['competitor_rate']:.2f}% "
          f"({'evidenced' if row['competitor_evidence_verified'] else 'UNVERIFIED claim'})")
    print(f"  Commercial         sales target {row['sales_target_pct']:.0f}%, branch {row['branch_target_pct']:.0f}%, "
          f"{'month end, ' if row['month_end'] else ''}customer negotiating")

    # ------------------------------------------------------------------ stage 1
    rule("STAGE 1  -  PRE-ASSESSMENT  (self-declared information only)")
    stage1 = orchestrator.pre_assessment(application)
    band = stage1["band"]
    print(f"  Indicative range   {band['min_rate']:.2f}% - {band['max_rate']:.2f}%  "
          f"(width {band['width_bps']:.0f} bps)")
    print(f"  Confidence         {stage1['confidence']:.2f} ({stage1['confidence_label']})")
    eligible = stage1["eligible_amount"]
    if abs(eligible["max"] - eligible["min"]) < 1000:
        print(f"  Eligible amount    {money(eligible['max'])}  "
              f"(capped by {eligible['binding_constraint']}, so the rate does not move it)")
    else:
        print(f"  Eligible amount    {money(eligible['min'])} - {money(eligible['max'])}  "
              f"(limited by {eligible['binding_constraint']})")
    for caveat in stage1["caveats"]:
        print(f"  Caveat             {caveat}")

    # ------------------------------------------------------------------ stage 2
    rule("STAGE 2  -  POST-ASSESSMENT  (verified, valued, underwritten)")
    result = orchestrator.process(application)
    recommendation = result["recommendation"]
    economics = recommendation["economics"]
    risk = result["risk"]

    print(f"  Risk               grade {risk['risk_grade']}, PD {risk['pd']:.2%}, "
          f"expected loss {risk['expected_loss_pct']:.2f} pts, breakeven {risk['breakeven_rate']:.2f}%")
    print(f"  Policy card rate   {result['policy']['card_rate']:.2f}%   "
          f"permissible {result['envelope']['min_rate']:.2f}% - {result['envelope']['max_rate']:.2f}%")
    print(f"  Stage-2 anchor     {recommendation['stage2_anchor']['anchor_rate']:.2f}% "
          f"(what history priced a file like this at)")
    print(f"\n  RECOMMENDED RATE   {recommendation['recommended_rate']:.2f}%"
          f"   -> approval: {result['policy']['required_approval'].replace('_', ' ')}")
    print(f"  Expected outcome   {economics['conversion_probability']:.0%} conversion, "
          f"{economics['expected_life_years']:.1f} year expected life, "
          f"{economics['bt_out_probability']:.0%} BT-out risk")
    print(f"  Risk-adjusted value {money(economics['expected_value_inr'])} "
          f"({economics['expected_value_pct']:.2f}% of disbursal)")

    print("\n  Why not simply charge more?")
    marginal = recommendation["marginal_analysis"]
    if "next_step_bps" in marginal:
        print(f"    the next {marginal['next_step_bps']:.0f} bps costs "
              f"{abs(marginal['next_step_conversion_change_pts']):.1f} points of conversion "
              f"and changes value by {money(marginal['next_step_value_change_inr'])}")
    print(f"    the value peak across all permissible prices sits at "
          f"{marginal['unconstrained_peak_rate']:.2f}% ({money(marginal.get('value_at_peak_inr'))})")

    print("\n  Alternatives considered:")
    for name, comparison in recommendation["comparisons"].items():
        print(f"    {name:<24} {comparison['rate']:.2f}%  conversion {comparison['conversion_probability']:.0%}  "
              f"value {money(comparison['expected_value_inr'])}  "
              f"({comparison['value_vs_recommended_inr']:+,.0f} vs recommended)")

    explanation = result["explanation"]
    print(f"\n  Reason codes ({explanation['attribution_method']}):")
    for code in explanation["reason_codes"]:
        print(f"    {code['text']:<52} {code['label']}")
    share = explanation.get("commercial_vs_risk", {})
    if share:
        print(f"    commercial pressure accounts for {share['commercial_share_pct']:.0f}% of the "
              f"movement ({share['commercial_net_bps']:+.0f} bps)")

    sensitivity = result["price_sensitivity"]
    if sensitivity.get("available"):
        print(f"\n  Price sensitivity  {sensitivity['sensitivity_label']}: "
              f"{sensitivity['points_lost_per_25bps']:.1f} points of conversion per 25 bps")
        print(f"                     70% conversion needs {sensitivity['rate_for_70pct_conversion']}%, "
              f"50% needs {sensitivity['rate_for_50pct_conversion']}%")

    for note in result["notes"]:
        print(f"  Note               {note}")

    similar = (result.get("sales_context") or {}).get("similar_situations") or []
    if similar:
        print("\n  Comparable past negotiations:")
        for hit in similar[:2]:
            outcome_text = ""
            if hit.get("outcome"):
                outcome_text = (f" -> {'disbursed' if hit['outcome']['disbursed'] else 'lost'} at "
                                f"{hit['outcome']['sanctioned_rate']:.2f}%")
            print(f"    \"{hit['text'][:78]}\"{outcome_text}")

    # ----------------------------------------------------------- human decision
    rule("HUMAN DECISION  -  the salesperson overrides")
    human_rate = round(recommendation["recommended_rate"] - 0.20, 2)
    override = orchestrator.capture_override(
        loan_id=result["loan_id"], human_rate=human_rate, reason="competitor_lower_rate",
        reason_detail=f"Customer showed {row['competitor_name']} quote at {row['competitor_rate']:.2f}%",
        approved_by="r.sharma:branch_manager",
        evidence_verified=bool(row["competitor_evidence_verified"]))
    print(f"  AI recommended     {override['ai_rate']:.2f}%")
    print(f"  Human offered      {override['human_rate']:.2f}%   ({override['override_bps']:+.0f} bps)")
    print(f"  Reason             {override['reason']} - {override['reason_detail']}")
    print(f"  Authority          requires {override['required_approval'].replace('_', ' ')}; "
          f"approved by branch manager -> within authority: {override['within_authority']}")
    if override.get("policy_check") and not override["policy_check"]["allowed"]:
        for violation in override["policy_check"]["violations"]:
            print(f"  POLICY BLOCK       {violation}")

    # ----------------------------------------------------------------- outcome
    rule("OUTCOME  -  what actually happened")
    outcome = orchestrator.capture_outcome(
        loan_id=result["loan_id"], disbursed=bool(row["disbursed"]),
        sanctioned_rate=human_rate, customer_response=row["customer_response"],
        bt_out=bool(row["bt_out_24m"]), delinquent=bool(row["delinquent_12m"]),
        months_to_exit=int(row["months_to_exit"]))
    print(f"  Customer response  {outcome['customer_response']}  |  disbursed: {outcome['disbursed']}")
    if outcome["disbursed"]:
        print(f"  Behaviour          exited after {outcome['months_to_exit']} months"
              f"{' (balance transfer)' if outcome['bt_out'] else ''}")
        print(f"  Lifetime value     {outcome['realized_value_pct']:.2f}% of disbursal")
    comparison = outcome.get("vs_expected")
    if comparison:
        print(f"  Versus expected    recommended {comparison['recommended_rate']:.2f}%, "
              f"sanctioned {comparison['sanctioned_rate']:.2f}% ({comparison['rate_delta_bps']:+.0f} bps); "
              f"expected conversion {comparison['expected_conversion']:.0%}, converted: {comparison['converted']}")

    # ---------------------------------------------------------------- learning
    rule("LEARNING  -  did human overrides create or destroy value?")
    review = orchestrator.learning.review_overrides(df)
    if review.get("available"):
        naive = review["naive_comparison"]
        adjusted = review["adjusted_overall"]
        print(f"  Naive read         overrides look {naive['s2d_lift_pts']:+.1f} pts on conversion, "
              f"{money(naive['value_lift_inr'])} on value")
        print(f"  Adjusted read      {adjusted['s2d_lift_pts']:+.1f} pts, "
              f"{money(adjusted['value_lift_inr'])} +/- {money(adjusted['value_lift_se_inr'])} "
              f"-> {adjusted['verdict']}")
        print(f"  Confounding worth  {money(review['confounding_gap_inr'])} per file: humans discount "
              f"the files that were hardest to convert")

        print("\n  By reason (propensity-adjusted):")
        table = pd.DataFrame(review["reason_league_table"])
        for record in table.head(8).to_dict(orient="records"):
            print(f"    {record['label']:<30} n={record['n_override']:>5}  "
                  f"S2D {record['s2d_lift_pts']:+6.1f} pts  value {record['value_lift_inr']:+9,.0f}  "
                  f"{record['verdict']}")

        evidence = review.get("evidence_effect", [])
        if evidence:
            print("\n  Competitor claims, evidenced vs not:")
            for record in evidence:
                print(f"    {record['label']:<32} n={record['n_override']:>5}  "
                      f"value {record['value_lift_inr']:+9,.0f}  {record['verdict']}")

        for suggestion in review["suggested_adjustments"]:
            print(f"\n  PROPOSAL           {suggestion['reason']}: {suggestion['proposal']}")
            print(f"                     {suggestion['evidence']} (requires {suggestion['requires']})")

    print("\n  Standing review questions:")
    for answer in orchestrator.learning.questions(df):
        print(f"    Q: {answer['question']}")
        print(f"    A: {answer['answer']}")

    # --------------------------------------------------------------- federated
    if not args.skip_federated:
        rule("PHASE 2  -  FEDERATED REGIONAL PERSONALIZATION (simulated)")
        from aavas.federated import client as fed_client
        from aavas.federated import regional_personalization as personalization
        from aavas.federated import server as fed_server

        fed_df = df.assign(discount=df["policy_card_rate"] - df["sanctioned_rate"])
        features = ["ltv", "foir", "bureau_score", "competitor_gap_bps",
                    "sales_target_pct", "branch_target_pct", "sanctioned_loan_amount"]
        clients = fed_client.build_clients(fed_df, features, "discount")
        run = fed_server.run(clients)
        print(personalization.compare(run, clients).to_string(index=False))
        print("\n  Read this table before committing to federated infrastructure: if personalization")
        print("  does not beat the global model out of sample, one central model is sufficient.")

    # -------------------------------------------------------------- governance
    rule("GOVERNANCE  -  the audit trail for this file")
    history = audit.loan_history(result["loan_id"])
    print(f"  Decisions recorded {len(history['decisions'])}   "
          f"Overrides {len(history['overrides'])}   Outcome: {'yes' if history['outcome'] else 'no'}")
    for entry in reversed(history["audit_entries"]):
        print(f"    {entry['created_at']}  {entry['actor']:<16} {entry['action']}")
    print(f"\n  Versions           {result['versions']}")
    print(f"  Model versions     {result['model_versions']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
