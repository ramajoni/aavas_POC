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

---

## Routes

| Route | Returns |
|---|---|
| `GET /features` | the agent's generators, refusals, thresholds and what it may not do |
| `GET /features/profile` | every column the stage may see, profiled |
| `GET /features/examine/{name}` | the full battery plus a card and a research verdict |
| `GET /features/cards` | a card per contract feature, with the lineage graph |
| `POST /features/discover` | generate, test, refuse |
| `POST /features/propose` | record a tested candidate — **not** an admission |
| `GET /features/proposals` | what has been proposed, with refusals kept |

There is no route that widens the contract. The absence is a control, and a test
asserts it.
