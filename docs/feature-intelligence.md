# Agentic Feature Engineering

**Audience:** model risk, credit policy and the data science team.
**Status:** built and running. `GET /features`, `#/features` in both UIs.
**Last verified:** 23 September 2026, 435 tests passing.

---

## The one principle everything follows from

> **Feature discovery can be automated. Feature admission cannot.**

A discovered feature is a hypothesis. Admitting it changes what every future model
is fitted on, what appears in a customer-facing explanation, and what a regulator
will ask about. That is a decision a human takes through the approval path, not a
threshold a search clears.

This matches the design principle in the specification this was built against —
*the LLM should not directly decide that a feature is good* — and goes one step
further: **nothing** decides a feature is good. The agent produces evidence and
refusals; a person decides, and records the decision where it can be audited.

---

## What is built

| Role | Where | State |
|---|---|---|
| Data Profiler | `features/profiling.py` | numeric, categorical and temporal profiles; missingness, quantiles, skew, kurtosis, outliers, cardinality, rare-category and unseen-category risk |
| Statistical Scientist | `features/profiling.py` | IV + WoE by bin, mutual information, Mann-Whitney U, KS, Cramér's V, VIF, PSI, WoE monotonicity |
| Feature Discovery | `features/discovery.py` | four generator families, leakage-safe by construction |
| Feature Hypothesis | `features/cards.py` | recipe, inputs, expected direction, family |
| Research trigger | `features/discovery.py` | six conditions, computed from evidence, **no model call** |
| Feature Testing | `features/discovery.py` | signal, redundancy, stability, coverage |
| Feature Selection | `features/cards.py` | Keep / Review / Reject, veto-ordered |
| Feature Documentation | `features/cards.py` | the feature card |
| Feature Governance | `governance/feature_registry.py` + `governance/approvals.py` | proposals, four admission gates, two required roles |
| Lineage | `features/cards.py` | source → transformation → decision |
| Funnel | `features/funnel.py` | raw attributes → derived → screened → approved → in the model, measured per contract; see §8 |

**Not built: code generation and sandboxed execution.** See §6.

---

## 1. Leakage is prevented, not detected

Generators are handed only the columns already inside the stage's own contract. A
Stage-1 candidate therefore *cannot be constructed* from a post-assessment field —
there is no code path that builds it and then rejects it.

```python
allowed_columns("stage1")   # contract minus outcome, latent, decision and
                            # post-assessment columns
```

`tests/test_feature_discovery.py` asserts the intersection with every forbidden
set is empty, for both stages.

## 2. Four refusals, and one of them is not statistical

| Refusal | Meaning |
|---|---|
| `leakage` | built from a column the stage may not see |
| `redundant` | correlates above 0.92 with a feature already in the contract |
| `unstable` | PSI above 0.25 between the older and newer halves |
| `thin_coverage` | fewer than 90 % of rows carry a usable value |
| `degenerate` | almost no variation |
| **`no_reason_label`** | **no human-authored reason label** |

The last one is the design. `feature_config.reason_code_labels` is what turns an
attribution into a sentence a branch can read to a customer. A feature with no
label cannot be explained, so it cannot be admitted **however strong it is** — and
the agent will not write the label itself. A machine-generated explanation of a
machine-generated feature is a circle with a customer at the end of it.

On a stage-2 run: **358 candidates generated, 0 admissible, and the 10 most useful
are the ones refused *only* for want of a label.** Strong, stable, independent and
unexplainable — each is a question for a human, and the answer is often no.

## 3. The decision is veto-ordered, not scored

A single weighted score would let a strong statistic outvote a leakage finding.
That is the one trade a lender must never make, so the dimensions are ordered:

```
risk      → leakage, instability, bias, suspicious strength   VETO (final)
business  → explainability, availability at decision time     VETO (final)
statistics→ IV, significance, separation                      contributes
predictive→ MI, SHAP, model contribution                      contributes
```

**Three outcomes, not two.** *Review* exists because "the evidence is mixed" is the
honest answer more often than keep or reject, and collapsing it hides exactly the
cases a human should look at.

An IV above 0.5 is a **veto**, not a recommendation: at that strength the question
is whether the feature leaks, not whether to use it.

## 4. Scoring the contract found two real defects

The card engine was pointed at the features **already in production**, on the same
evidence a candidate would face.

**a. Thirteen contract features had no authored reason label.** They fell back to a
title-cased column name — `appraised_property_value` would have rendered as
"Appraised Property Value" where a sentence belongs. Labels are now authored for
all of them, and a test fails if a feature is ever admitted without one. The test
found one more than the manual scan did, because it walks every contract block
rather than the six model stages.

**b. Severe multicollinearity in the stage-2 contract.**

| Feature | VIF |
|---|---|
| `sanctioned_loan_amount` | **38.6** |
| `appraised_property_value` | **35.7** |
| `ltv` | 3.9 |

The three are algebraically related — `ltv = loan / value` — so this is expected
and real. It does not hurt predictive accuracy, and it does damage something this
system cares about more: with collinear inputs an attribution can swing between
near-identical files, and an unstable attribution becomes an unstable customer
explanation. Worth a decision; not worth a silent fix.

Two caveats the page states: the decision is **target-specific** (9 rejects against
`defaulted`, 11 against `disbursed`), and VIF is **not** — it is a statement about
the feature matrix whichever target is chosen.

## 5. Research is conditional, and the condition is the cheap half

Routing a known method on a known problem through a model is slower, dearer and
less predictable than executing it. The trigger decides when a question is
genuinely outside the library, and computes that **from the evidence, with no
model call**:

| Trigger | Fires when |
|---|---|
| `unusual_distribution` | \|skew\| > 3 or kurtosis > 20 |
| `non_monotonic_relationship` | WoE moves up and down and IV > 0.05 |
| `methods_disagree` | MI is high while rank AUC is ~0.5 — not a linear relationship |
| `unexpected_strength` | IV > 0.5 |
| `high_cardinality` | too many categories for one-hot, too few rows for safe target encoding |
| `unstable` | PSI > 0.25 with no obvious cause |

In practice it stays quiet: ordinary contract features trigger nothing, while
`realized_yield_pct` — a genuine leak — fires three at once.

## 6. What is deliberately not built

**Code generation and sandboxed execution.** Both are in the specification and
both are omitted, on purpose:

* **An in-process `exec` with a blocklist is not a sandbox.** It is a blocklist.
  Honest isolation means a separate container with no network, a read-only mount,
  a memory and CPU limit and a wall clock — infrastructure, not a Python function.
* **Generated code that reaches the feature contract is the highest-blast-radius
  thing in a pricing engine.** It would sit upstream of every model, every price
  and every customer explanation.
* Shipping a convincing-looking sandbox is worse than shipping none, because the
  next person assumes it holds.

What *is* built is everything around that seam: the **function registry** answers
"is this already covered?" before anything would be generated, the **research
trigger** decides whether the question is new, and the **admission gates** mean
generated code could never widen the contract on its own. If generation is added
later, it plugs in at one place and still cannot admit a feature.

`tests/test_feature_discovery.py` asserts that `exec`, `eval`, `compile` and
`subprocess` appear nowhere in the discovery agent, and that no route contains
`generate` or `execute` — so this stays a decision rather than becoming an
accident.

**SHAP contribution and model uplift on the card.** The card has slots for both
and they are empty. Filling them honestly means fitting a model with and without
the candidate and measuring the difference out of time — real work, not a lookup.
An empty slot is better than a number from a fit nobody ran.

## 7. Admission, end to end

```
run discovery  →  candidate tested, refusals recorded
      ↓
POST /features/propose        evidence stored against the governance store
      ↓
POST /governance/approvals    artefact_type = "feature"
      ↓
gates: feature_proposed · reason_label_authored · feature_no_leakage · feature_not_redundant
      ↓
model_risk AND credit_policy both sign
      ↓
apply → 501: the API does not write configuration
```

The last step is deliberate. Admission is an edit to `feature_config.yaml`; the
approval is the record that the edit was authorised, and the endpoint says so
rather than implying it wrote the file.

## 8. The funnel, measured

`features/funnel.py` runs the whole pipeline end to end for one model contract
and counts what survives each stage. Every count is a list of names in
`reports/features/funnel_<model>.json`, and every number next to a name is one a
test computed.

```bash
PYTHONPATH=src python3 scripts/run_feature_funnel.py            # stage2, risk, conversion, stage1
PYTHONPATH=src python3 scripts/run_feature_funnel.py --models all --sample 10000
curl 'localhost:8000/features/funnel?model=stage2'              # also the Funnel tab, #/features
```

| Stage | What it counts |
|---|---|
| **Raw attributes** | Every dataset column the model could see at decision time, before the contract is consulted. Each column gets a disposition; excluded ones keep a reason: `identifier`, `time_index`, `outcome`, `latent`, `decision`, `decision_derived` (computed from a decision, e.g. `offered_rate` = the sanctioned rate), `post_assessment` (Stage 1 only), `own_output` (`risk_grade` for the risk model), and `unclassified` — a column `ingestion.schemas` does not declare. The frame is the one the model's trainer builds, so the override model's `competitor_gap_bps` is the gap at the *recommended* rate it is actually fed |
| **Derived candidates** | Everything the five generator families build from the numeric raw attributes, uncapped: ratio, gap, log, interaction, and **trend** — `x − median(x)` over applications in the prior 90 days, strictly earlier dates only. There is no per-loan history, so this is the market's trend, not the customer's |
| **Passed screening** | Raw and derived items that clear coverage (≥ 0.90), degeneracy, stability (PSI ≤ 0.25, older half against newer half by date), signal (\|Spearman\| ≥ 0.02 with the target; bias-corrected Cramér's V for categoricals) and redundancy (\|Spearman\| ≤ 0.92 with any contract feature or any stronger feature already kept; raw before derived, so a log of a column never displaces the column) |
| **Approved in review** | Screened items with an authored entry in `reason_code_labels`. A governance approval is shown as evidence and, without the label, does not count. Nothing is approved by the funnel |
| **In the model** | The contract's columns, each reconciled: screened and approved, failed a screen (which, against what), or excluded at raw (why) |

The funnel generates from every raw attribute the stage may see, which is wider
than `POST /features/discover` (the contract's own columns, §1); both leave the
forbidden columns out by construction. `bureau_score = −1` is read as missing
first (`feature_config.sentinels`), so no candidate is arithmetic on the code. Signal is exact over every population row;
redundancy is Spearman on a 6,000-row sample, seed 42.

**The run of 20 October 2026** (20,000 loans):

| | Risk-adjusted pricing `stage2` | Risk (PD) `risk` | Acceptance `conversion` | Indicative pricing `stage1` |
|---|---|---|---|---|
| Target, rows | `sanctioned_rate`, 20,000 | `defaulted`, 13,181 disbursed | `disbursed`, 20,000 | `sanctioned_rate`, 20,000 |
| Raw attributes (numeric + categorical) | **38** (26 + 12) | **37** (26 + 11) | **40** (28 + 12) | **24** (13 + 11) |
| Columns excluded | 38 | 39 | 36 | 52 |
| Derived candidates | **1,096** | **1,096** | **1,277** | **273** |
| Passed screening (raw + derived) | **154** (16 + 138) | **143** (15 + 128) | **103** (13 + 90) | **37** (7 + 30) |
| Approved in review | **16** | **15** | **13** | **7** |
| In the model (of which screened and approved) | **24** (14) | **20** (12) | **15** (10) | **16** (6) |
| Strong, stable and unexplainable | 138 | 128 | 90 | 30 |

What the run says:

* **No derived candidate is approved**, in any model, because none has an
  authored label. The strongest stage-2 rejects are `competitor_rate −
  prior_negotiation_rounds` (signal 0.767, overlap 0.807, PSI 0.002),
  `deviation_count − competitor_rate` (−0.743, 0.798, 0.002) and `estimated_ltv ×
  competitor_rate` (0.741, 0.757, 0.002): strong, stable, and with no sentence a
  branch could say to a customer.
* **10 of the 24 stage-2 contract features fail a screen**, and the page says
  which. `sanctioned_loan_amount` is redundant with `appraised_property_value`
  (ρ = 0.954 — the VIF finding of §4b, from the other direction); `bureau_score`
  covers 0.856 of rows, missing exactly when `new_to_credit`, which is structural;
  eight fail the univariate signal floor against `sanctioned_rate`
  (`delinquency_count_24m`, `foir`, `tenure_years`, `applicant_age`,
  `sales_target_pct`, `branch_target_pct`, `city`, `channel`). Every screen is
  univariate, so this is a question for the fitted model's attributions, not a
  verdict.
* **Nothing is unstable.** The largest PSI on any stage-2 item is 0.088: the
  synthetic book is stationary from September 2024 to April 2026. The screen is
  real and has nothing to catch here.
* **Leakage.** For stage 2 the exclusions are 12 outcome, 9 decision, 6
  decision-derived, 5 latent, 3 identifier, 2 time-index and 1 unclassified
  (`state_competitive_intensity`, a per-state constant from the generator that
  no schema group declares). The latent columns are also used once, to
  *evaluate*: `risk_grade` tracks `true_pd` (η = 0.92), `property_type` tracks
  `true_lgd` (η = 0.85) and `competitor_rate` tracks `true_pd` (|ρ| = 0.80),
  because the generator builds the grade from the true PD and the competitor
  quote from the card rate. Excluding a latent column does not remove a near-copy
  of it; the page shows these rather than claiming a clean bill.

**Freshness.** `GET /features/funnel` serves the saved report only while every
input it was computed from is unchanged — the raw data file's bytes, the model's
contract block and the sentinel rules, the reason labels, the feature approvals
on record for that model, the funnel, builder and loader source, and the run
parameters. Age is not a criterion. Otherwise it computes the funnel (seconds)
and says why in `served.stale_because`. A GET never writes the report.

---

## Routes

| Route | Returns |
|---|---|
| `GET /features` | the agent's generators, refusals, thresholds and what it may not do |
| `GET /features/profile` | every column the stage may see, profiled |
| `GET /features/examine/{name}` | the full battery plus a card and a research verdict |
| `GET /features/cards` | a card per contract feature, with the lineage graph |
| `GET /features/funnel?model=stage2` | the measured funnel, its exclusions, reconciliation and rejected table (§8); `refresh=true` recomputes, `detail=true` adds every tested item |
| `POST /features/discover` | generate, test, refuse |
| `POST /features/propose` | record a tested candidate — **not** an admission |
| `GET /features/proposals` | what has been proposed, with refusals kept |

There is no route that widens the contract. The absence is a control, and a test
asserts it.
