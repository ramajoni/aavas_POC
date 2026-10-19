# Target architecture: gap analysis and build roadmap

Maps the proposed 2026 Agentic Credit Scoring architecture onto what exists in this
repository today, and sequences the work that is actually left.

Read with [architecture.md](architecture.md) (what is built) and the
[README](../README.md) (why it is built that way).

**Summary.** Most of the proposed architecture already exists here under different
names. Four things genuinely do not, and only one of them is urgent. The urgent one
is not in the proposal's list, because it is not a missing capability — it is a
control gap in code that already ships.

---


> The agent flow, the Configure → Train → Validate → Approve → Run → Observe →
> Learn lifecycle, and A2UI as a presentation layer are mapped separately in
> [agentic-os-flow-gap.md](agentic-os-flow-gap.md). This document covers the
> model-side architecture.

## 1. The finding that reorders the roadmap

`applicant_age` is a live pricing input.

```
config/feature_config.yaml:14    stage1     numeric: applicant_age
config/feature_config.yaml:40    stage2     numeric: applicant_age
config/feature_config.yaml:70    risk       numeric: applicant_age
config/feature_config.yaml:131   prepayment numeric: applicant_age
```

It is a feature in four of the six model contracts — including `risk`, whose PD
drives the grade, which builds the card rate. And it carries a customer-facing
reason-code label:

```
config/feature_config.yaml:189
  applicant_age: {label: "Applicant age", up: "Age loading", down: "Age benefit"}
```

So the engine can already produce an explanation that tells an applicant their rate
rose because of their age, and **nothing in the system measures whether age-based
disparity exists, or how large it is.**

This is not a future capability to design. It is an existing surface with no
measurement behind it, and the data to measure it has been in the schema since day
one. It costs no data collection, no consent change and no legal review to *measure*.
It is P0 for that reason alone.

Whether age *should* be a pricing feature is a separate decision, and a human one
(§6). The gap is that the decision has never been forced, because nothing surfaces
the consequence.

## 2. What can be audited today, with no new data

The proposal treats protected attributes as something to be acquired. Five usable
axes are already in `ingestion/schemas.py`:

| Axis | Column | In the model? | Notes |
|---|---|---|---|
| Age | `applicant_age` | **Yes** — 4 contracts | Directly priced, and reason-coded |
| Employment / income formality | `segment` | Yes, and in the card rate | 4 levels: salaried formal/informal, self-employed professional/non-professional |
| Region | `state`, `city` | **No** — enters only via competitor intensity | Identity columns, not features |
| Channel | `channel` | Yes | |
| Credit history | `new_to_credit` | Yes | |

`segment` is the one that matters most commercially. Aavas lends into exactly the
informal-income and self-employed population that formal-sector credit models
underserve, and `segment` is a deliberate, disclosed loading in
`constraints.card_rate()`.

**Which creates the trap.** A demographic-parity measurement on `segment` will show a
large gap *by construction*, because policy puts it there on purpose. If the fairness
report prints that as a red number, nobody will read the second report.

So the fairness layer has to decompose disparity into:

- **policy-attributable** — the part explained by the published card-rate grid, and
- **residual** — the part the models added on top.

Only the residual is a finding. This is the same decomposition the explanation agent
already performs for a different purpose: `commercial_share()` separates
commercial-pressure contribution from customer-risk contribution, and it is
computable precisely because the feature contract keeps sales context in its own
block. The fairness decomposition is that idea applied to a second axis, and it
should reuse the machinery rather than duplicate it.

## 3. Layer-by-layer map

| Proposed layer | In this repo | Verdict |
|---|---|---|
| Agentic Supervisor | `agents/orchestrator.py` — deterministic 10-agent pipeline | **Present.** Deterministic on purpose; see §5 |
| Data Agent | `ingestion/validator.py`, `assert_no_leakage()` | Present |
| Feature Agent (discovery) | `features/*_features.py` — hand-built, config-declared | Partial: validated, not *discovered* — see §8 |
| Policy Agent | `agents/policy_agent.py`, `optimization/constraints.py` | Present |
| Privacy Agent / DP / secure agg | — | **Missing.** Deprioritise, see §5 |
| Federated training | `federated/` — in-process, linear regional adjustment | Present as simulation; own result says gains < 1 bp |
| Agentic AutoML | Optuna on `stage2` only; fixed 6-model suite | **Missing.** Low priority, see §4 and §8 |
| Calibration | Isotonic on conversion, prepayment, risk | Present, and disciplined |
| Model Evaluation Agent | `models/*/evaluate.py`, `learning/champion_challenger.py` | Present |
| **Fairness Agent** | Only `fair_pricing.commercial_uplift_review_bps` | **Missing.** This is a commercial-conduct flag, not group fairness |
| Explainability | `xai/` — SHAP + ablation, reason codes, counterfactuals, OmniXAI | Present, and ahead of the proposal |
| Human governance | Approval routing, override capture, append-only audit | Present |
| PD × LGD × EAD | PD fitted; LGD a policy table; EAD implicit | Partial, defensibly — see §4 |
| MIN/MAX optimiser | `optimization/price_optimizer.py` | Present, applied to rate not PD cutoff |
| Pricing agent | `agents/pricing_agent.py`, `yield_calculator.py` | Present |
| Three-zone decisioning | Approval routing + `human_decision_required` | Partial: routed by authority, not by PD band |
| Monitoring / drift | `learning/drift_detection.py` — PSI + conversion drift | Present |
| Retraining / champion-challenger | `learning/retraining.py`, `champion_challenger.py` | Present, promotes on value not AUC |

Roughly three-quarters of the proposal is built. The diagram makes the system look
further away than it is, because it renames things the repo already has.

## 4. The real gaps, in build order

### P0 — Fairness measurement (2–3 days)

New `src/aavas/fairness/`, offline, called from `learning/` — **not** in the request
path. The decision path must stay deterministic and fast.

```
fairness/
  metrics.py          selection rate, TPR/FPR/FNR gaps, AUC/KS by group,
                      calibration-by-group, mean sanctioned rate, mean override bps
  decomposition.py    policy-attributable vs residual disparity  (§2)
  intersectional.py   pairwise axes, min-subgroup floor, confidence intervals
  report.py           one brief per axis, same shape as reports/omnixai/
```

Reuse, do not reinvent: `learning/override_learning.py` already implements the exact
statistical discipline this needs — stratification, standard errors, a sample-size
floor, and an explicit `insufficient evidence` verdict on thin slices. Copy that
contract.

**Primary metric must be calibration-by-group, not demographic parity.** They are
mathematically incompatible whenever base rates differ across groups, and the
proposal lists both as if they add up. For a lender, calibration is the defensible
primary: if predicted PD is 10%, realised default should be near 10% *in every
group*. Parity is monitored and explained, not optimised.

### P1 — Fairness gate on promotion (half a day)

`learning/champion_challenger.py:84-92` already has the gate structure — a `reasons`
list accumulating four independent gates (minimum shadow decisions, a value bar, a
conversion-drop tolerance, and a z-test for statistical distinguishability). Add a
fifth beside them:

```python
if fairness_gap > cfg["max_fairness_gap"]:
    reasons.append(f"residual disparity {fairness_gap:.3f} exceeds {cfg['max_fairness_gap']}")
```

`promote()` already refuses when any gate fails and requires human sign-off. The
proposal's "mandatory gates" idea is one config key and four lines away.

### P2 — Fairness in the governance record (1 day)

`governance/store.py` has five append-only tables. Add `fairness_evaluations` with
the same insert-only discipline, and a fairness block in the model card written by
`models/base.py`. A fairness result that is not in the audit trail is a report, not
a control — the repo already argues this about exceptions in JSON blobs.

### P3 — Fairness drift (half a day, once P0 exists)

`learning/drift_detection.py` already runs PSI and conversion drift on a cadence.
Fairness drift is the same loop over the P0 metrics. Nearly free after P0.

### P4 — EAD (low priority, possibly nothing to do)

The proposal wants `PD × LGD × EAD`. Here:

- **PD** — fitted (`models/risk/`), lifetime→annual conversion explicit.
- **LGD** — a policy table by design (`models/risk/train.py:3-4`, "LGD is a policy
  table, not a fitted model"), LTV-adjusted in `features/stage2_features.py:65`.
  Defensible: loss severity on secured lending is driven by collateral and recovery
  process, not borrower behaviour, and a thin book cannot fit it.
- **EAD** — for an amortising term home loan, exposure at default is essentially the
  outstanding balance, which `yield_calculator.py` already computes as average
  outstanding across expected life.

So EAD is present in effect. Adding an EAD *model* for structural symmetry with a
credit-card architecture would be work with no decision attached to it. **Recommend
documenting why, not building it.**

### P5 — Model portfolio (1–2 days for the one that matters)

Skip AutoGluon, FLAML and TabPFN. For a regulated lender the case for a tabular
foundation model is a research question, not a roadmap item, and the repo's own
philosophy — promote on value, not accuracy — already resists "more candidates is
better".

The single addition worth making is **EBM** (`interpret`): glass-box, directly
comparable to the existing HistGB, and a model a regulator can read. It slots into
the existing `models/base.py` card-and-artefact contract with no new machinery.

### P6 — Privacy / DP (defer)

Do not build. The repo's own federated evaluation shows sub-basis-point gains
concentrated in two contested markets, and the honest reading in
`architecture.md §12` is that federation is not yet worth its operational cost.
Adding differential privacy and secure aggregation to a federation that buys nothing
is negative-value work. Revisit only when a real multi-institution consortium exists
with a signed data agreement — at which point DP is a *requirement*, not an
enhancement.

## 5. What I would cut from the proposal, and why

**An LLM supervisor agent.** `agents/orchestrator.py` is deterministic, and that is
the feature, not the limitation. Every decision stores an exact application snapshot
plus policy, feature-config and model versions, so any decision can be replayed
identically years later. An LLM planner in the pricing path destroys that property
and cannot be defended to a model-risk function. Keep agentic behaviour in the
offline layers — feature proposal, analysis, report drafting — where a human reads
the output before it touches a price.

**Three fairness libraries.** Fairlearn + AIF360 + Aequitas have overlapping scope
and heavy dependency footprints. This repo's stated rule is that every optional
dependency has a working fallback, and `.venv-xai` exists precisely because one
library (`omnixai` → `dash<3` → `Flask<2.3`) could not coexist with FastAPI. Do not
repeat that three more times. The Level 1–3 metrics are arithmetic — selection rates,
rate gaps, calibration curves by group — and are ~200 lines written natively. Pull in
Fairlearn only if a specific reduction algorithm is needed for mitigation, behind the
same optional-with-fallback pattern.

**Group-specific thresholds.** The proposal already flags the legal exposure, and it
is right to. Under RBI fair-practice expectations, applying a different approval
threshold by protected group is not something to build speculatively because a
library exposes the API. Build measurement (P0) and keep pre-processing (reweighing)
and in-processing (constraints) as the mitigation options. Leave post-processing
documented and unimplemented, gated on legal sign-off.

**Demographic parity as a co-equal objective.** See P0 — pick calibration as primary.
Optimising both is not conservative, it is incoherent.

## 6. Decisions that need a human, not an engineer

These block or shape the work above and are not engineering calls:

1. **Is `applicant_age` an intended pricing feature?** It is one today, in four
   contracts, with an "Age loading" reason code. Confirm, remove, or constrain.
2. **Which axes are protected for Aavas?** Gender, religion and caste are not
   collected. Collecting them *in order to test fairness* is itself a consent and
   legal decision, and many lenders deliberately decline it. Measuring on age,
   segment and region requires no such decision.
3. **Primary fairness criterion** — calibration-by-group vs parity. A policy choice
   for the pricing committee, not a modelling choice.
4. **Tolerance thresholds** (`max_fairness_gap`) — risk appetite, same committee that
   owns `min_conversion_probability`.
5. **May "Age loading" appear in customer-facing text?**

## 7. Sequence

```
P0  fairness measurement          ██████        2–3 d   no dependencies
P1  promotion gate                ██            0.5 d   needs P0
P2  governance record             ████          1 d     needs P0
P3  fairness drift                ██            0.5 d   needs P0
P5  EBM in the model suite        ████          1–2 d   independent
—   tuning split fix (§8)       ·             1 h     independent, latent
P4  EAD                           ·             document only
P6  privacy / DP                  ·             defer
```

About a week of engineering closes the responsible-AI gap that matters. The rest of
the proposal is either already built, honestly not worth building yet, or a decision
waiting on a human rather than a commit.

---

## 8. Layers 3 and 4 in depth

The expanded Layer 3 / Layer 4 specification deserves a closer read than the one-line
rows in §3, because the honest answer is not "missing". It is that **parts of both
layers are in tension with guarantees this system already provides**, and those
guarantees are worth more than the capabilities that would displace them.

### 8.1 Layer 3 — what exists against each claimed function

| Claimed function | Here today | Reality |
|---|---|---|
| Discovery — relationships, interactions, temporal signals | — | Absent. Features are hand-designed |
| Generation — DTI | `foir` (obligations ÷ income) | Present under another name |
| Generation — credit utilisation, repayment velocity, cash-flow volatility, transaction patterns | — | **No transaction table exists.** Data gap, not a code gap |
| Generation — bureau-derived | `bureau_score`, `bureau_vintage_months`, `delinquency_count_24m` | Present, thin |
| Validation — IV, WOE, mutual information, redundancy, outlier sensitivity | — | Absent. Zero hits across `src/` |
| Validation — business-rule consistency | `ingestion/validator.py:66-100` | **Present and strong** — see below |
| Leakage — contract leakage | `assert_no_leakage()` | Present, but one-directional |
| Leakage — temporal, post-decision, target | Chronological split only | Partial |
| Stability — PSI | `learning/drift_detection.py:23` | Present, but as monitoring, not as an acceptance gate |
| Stability — CSI, institution-wise | — | Absent |
| Governance — business meaning | `reason_code_labels` per feature | **Present, and mandatory** |
| Governance — fairness / proxy check | — | Absent (P0 in §4) |
| Governance — human approval of features | — | Absent |
| Approved Feature Registry | `config/feature_config.yaml` | **Present, and better than proposed** |

Three of these need expanding.

**The registry already exists, as data.** `feature_config.yaml` carries
`feature_config_version: "FEAT-2026.09"`, is the only place a model's columns are
declared, is read by exactly one module (`features/feature_pipeline.py`), declares
monotone constraints as data beside the feature list, and its version is stamped onto
every decision record. A `features/feature_registry.py` would be a weaker version of
this — code that can drift from what was actually used, instead of a versioned
artefact that cannot. Do not build it.

**Validation is business-rule validation, not statistical validation.**
`validator.py` checks LTV, FOIR, tenure and amount against policy, rates against the
product envelope, that Stage-1 bands are not inverted, and that no row carries a
realised yield without a disbursal. That is genuinely good — it caught 958 impossible
rows during development — and it is a different thing from what Layer 3 means by
validation. There is no IV, no WOE, no mutual information, no redundancy or
outlier-sensitivity analysis anywhere in the repo. For a credit-scoring platform IV
and WOE are table stakes and their absence is a fair criticism.

**Leakage protection is narrower than it reads.** `assert_no_leakage()` guards exactly
one direction — post-assessment, outcome and latent columns entering **stage1**. It
returns immediately for every other stage. There is no temporal-leakage check, no
post-decision-variable check for stage2, and no guard at all on the risk, conversion,
prepayment or override contracts. The chronological split carries most of the real
protection. Widening this guard is cheap and worth doing.

### 8.2 The constraint the proposal does not account for

Layer 3 ends at "Approved Feature Registry" and Layer 4 begins with a model factory.
Both steps collide with commitments already made here.

**Explainability caps how autonomous feature discovery can be.** Every feature in
`feature_config.yaml` carries a hand-authored reason-code label with a direction:

```yaml
foir: {label: "Obligations to income", up: "High existing obligations", down: "Low existing obligations"}
```

That label is what a branch says to a customer, and what appears in an adverse-action
explanation. An auto-discovered interaction term has no such sentence, and no
human-meaningful direction. So a Feature Discovery Agent here cannot end at "accept" —
it can only ever end at *"propose to a human who writes the label"*. That is a
worthwhile tool, and it is not the autonomous loop the diagram draws.

**Monotone constraints cap what the model factory may contain.** Conversion is
constrained non-increasing in price and competitor gap; prepayment non-decreasing.
This is not a quality nicety — `price_optimizer` searches a grid over price, and the
guarantee that the demand curve cannot bend the wrong way on a thin slice is what
makes that search safe. Violations currently measure 0%.

An arbitrary model portfolio breaks this. TabPFN and FT-Transformer do not support
monotone constraints at all; an ensemble of a monotone and a non-monotone learner is
not monotone. So the factory is admissible only if **every member supports monotone
constraints and every blend preserves them** — which rules out most of the proposed
portfolio for the conversion and prepayment models specifically.

It does not rule them out for `stage1`, `stage2`, `risk` or `override`, which carry no
monotone requirement. The correct design is therefore a **portfolio scoped per model
contract**, not one factory feeding all six. The proposal treats the model suite as
homogeneous; it is not.

### 8.3 Layer 4 — what exists against each sub-agent

| Sub-agent | Here today |
|---|---|
| Model Strategy | Absent — suite is fixed in `config/model_config.yaml` |
| AutoML | Absent |
| HPO | `models/stage2/train.py:76-101` — Optuna, **stage2 only**, `enabled: false` by default, 2 hyperparameters |
| Ensemble | Absent (`models/base.py` hits are sklearn imports, not ensembling) |
| Calibration | **Present** — isotonic on conversion, prepayment, risk |
| Benchmark | `models/*/evaluate.py` + `learning/champion_challenger.py` |
| Champion / Challenger / Promotion | **Present** — promotes on replayed value, four gates, human sign-off |
| Human gate | **Present** — `promote()` refuses without it |

The right half of Layer 4 is built and is stronger than the proposal, which treats
promotion as a benchmarking output. Here it is a gated governance act.

### 8.4 One specific finding in the tuning path

`models/stage2/train.py:96` tunes against a **random** 3-fold split:

```python
return -cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=3).mean()
```

The repo is otherwise strict about this. `ingestion/loader.py:4` states the reason
plainly — "the default split is chronological. A random split lets the model peek at
the" future — and every reported metric comes from a chronological 80/20 split. The
tuning objective does not follow that rule, so hyperparameters would be selected
against a leaky estimate of generalisation.

**This is latent, not active:** `model_config.yaml:51` has `enabled: false`, so the
search does not run by default and no shipped artefact was tuned this way. But it is
a trap set for whoever turns it on. Replace `cv=3` with `TimeSeriesSplit(3)` on the
time-ordered frame — about an hour including a regression test, and it should happen
before any AutoML work expands the search surface.

### 8.5 What I would actually build from Layers 3 and 4

Keep, in priority order:

1. **Widen `assert_no_leakage()`** to cover all six contracts and post-decision
   variables, not just stage1. Cheap, and it protects the guarantee the system already
   claims. *(half a day)*
2. **Fix the tuning split** (§8.4). *(1 hour)*
3. **IV / WOE / correlation report** per feature contract, offline, as evidence
   attached to the feature registry rather than a gate. Standard credit-scoring
   practice, and the reviewers will ask for it. *(1–2 days)*
4. **Per-contract model portfolio**, monotone-constrained where required, starting
   with EBM. *(§4, P5)*
5. **Feature proposal tooling** that ends in a human writing the reason-code label —
   not autonomous acceptance. *(later; only worth it once transaction data exists)*

Drop: autonomous feature acceptance, a single homogeneous model factory, ensembling
across the conversion and prepayment contracts, and tabular foundation models in the
priced path.

The reframing the proposal offers — Feature Intelligence → Model Intelligence →
Decision Intelligence — is a good description of the *layers*. It is not an argument
for making either layer more autonomous than the explainability and monotonicity
contracts allow.

---

## 9. Algorithm shortlist, filtered for this system

The proposed catalogue is a fair survey of the field. Most of it should not be built
here, and the reasons are not preference — four commitments already made in this
repository do the filtering:

1. **Every feature carries a human-authored reason-code label.** Anything that
   produces a feature no one can write a sentence about is inadmissible in the priced
   path.
2. **Conversion and prepayment are monotone-constrained.** Learners that cannot carry
   monotone constraints are disqualified for those two contracts (§8.2).
3. **Calibration is load-bearing.** Probabilities are multiplied by rupees, so
   anything that distorts the base rate destroys the output's meaning.
4. **Optional dependencies must degrade gracefully**, and `.venv-xai` is the standing
   reminder of what a heavy dependency costs.

### 9.1 Layer 3

| Method | Verdict | Reason |
|---|---|---|
| **IV / WoE** | **Build** | Table stakes for credit scoring; monotone WoE binning aligns with the existing constraint philosophy |
| **Mutual information** | **Build** | Cheap, non-linear, and doubles as the proxy test against protected axes (§4 P0) |
| ANOVA F-test, Chi-square | Skip | Subsumed by MI, which handles non-linear dependence |
| Permutation importance | **Already built** | `xai/omnixai_analysis.py`, with Spearman rank agreement vs KernelSHAP |
| SHAP-based importance | **Already built** | `xai/shap_explainer.py` + the ablation fallback |
| Boruta | Skip | Wraps RF, slow, marginal gain over the permutation importance already computed |
| RFE / RFECV / SelectKBest / SFS | **Evidence only** | May rank features; must never *remove* one. See 9.3 |
| L1 / Lasso / Elastic Net (embedded) | Conditional | Only meaningful if the WoE scorecard is built; irrelevant to tree learners |
| **Pearson / Spearman / VIF / clustering** | **Build** | Cheap redundancy evidence; VIF matters mainly for the scorecard |
| **Adversarial validation** | **Build** | Best value-per-line here — train a classifier to separate train from test; AUC materially above 0.5 means the chronological split is drifting |
| Temporal / target-leakage checks | **Build** | Widening `assert_no_leakage()` is already §8.5 item 1 |
| PSI | **Already built** | `learning/drift_detection.py:23` |
| CSI | **Build** | Genuine gap beside the existing PSI |
| Jensen–Shannon divergence | Optional | Better behaved than PSI on empty bins; a refinement, not a gap |
| Polynomial / interaction features | Skip | No reason-code label is possible for them |
| Deep Feature Synthesis | **Inapplicable** | DFS needs related tables with keys; this repo has one flat lifecycle table |
| Genetic programming | Skip | Uninterpretable expressions; dead on arrival against adverse-action explanation |
| **PCA / Kernel PCA / SVD / Autoencoder** | **Hard skip** | See 9.3 |
| Isolation Forest / LOF / One-Class SVM | Skip | `validator.py` already quarantines on *known* business rules — 958 impossible rows during development. Rule-based filtering beats unsupervised outlier detection wherever the rules are known, and here they are |
| KNN / Iterative / MICE imputation | **Skip, actively** | See 9.3 |
| Fairness / proxy detection | **Build** | Already §4 P0 |

### 9.2 Layer 4

| Method | Verdict | Reason |
|---|---|---|
| **WoE Logistic Scorecard** | **Build** | The artefact a regulator reads. Defensible baseline, and it makes the IV/WoE work pay twice |
| **EBM / GAM** | **Build** | Glass-box, comparable to the incumbent HistGB (already §4 P5) |
| Logistic Regression | Present | Baseline |
| Ridge / Lasso / Elastic Net logistic | Marginal | Only as scorecard variants |
| Decision Tree / RF / Extra Trees | Skip | Nothing over HistGB |
| XGBoost | **Present** | Optional, with HistGB fallback |
| LightGBM / CatBoost | Marginal | CatBoost's categorical handling is real, but this repo ordinal-encodes deliberately to keep attributions readable per original feature |
| HistGradientBoosting | **Present** | The incumbent |
| SVM / Linear SVM / KNN / Naive Bayes | **Skip entirely** | No monotone support, poor calibration, weaker on tabular. KNN on mixed-type tabular is a non-starter |
| MLP / TabNet / FT-Transformer / SAINT | **Skip in the priced path** | No monotone constraints → disqualified for conversion and prepayment; and on 20k rows they do not beat a GBM elsewhere |
| TabPFN / TabICL | Research only | Same monotone problem, plus a sample-size ceiling |
| LSTM / GRU / TCN / Transformer | **Wrong tool** | See 9.4 — the real sequential gap wants a survival model |
| GCN / GraphSAGE / GAT | **Skip** | There is no graph. No guarantor links, no shared-collateral edges, no applicant-to-applicant relation in the schema |
| **SMOTE / ADASYN / Balanced RF / EasyEnsemble** | **Skip, with evidence** | See 9.3 |
| Cost-sensitive / class weights | Skip for now | Same calibration objection unless followed by recalibration on an unweighted holdout; no current need |
| Optuna / TPE | **Present, split is wrong** | §8.4 |
| Bayesian optimisation / Hyperband / ASHA | Later | Only earns its keep at portfolio scale |
| Voting / Bagging / Stacking / Blending | **Constrained per contract** | §8.2 — admissible only where monotonicity is not required |
| Platt / Isotonic | **Isotonic present** | Monotone, so the price constraint survives calibration |
| Beta calibration | Worth trying | Smooth, parametric, monotone; better than isotonic's step function on thin positive classes — `prepayment` at 15.3% is the candidate |
| Out-of-time validation | **Present** | Chronological 80/20 |
| Shadow deployment | **Present** | `shadow_compare()`, gated at `shadow_min_decisions: 300` |
| Rolling-window comparison | **Build** | Genuine addition to champion/challenger |

### 9.3 The three cuts that matter, and why

**Dimensionality reduction is incompatible with this product.** PCA, kernel PCA,
truncated SVD and autoencoders all produce components with no business meaning. You
cannot tell a declined applicant that their rate rose because principal component 4
was elevated, and you cannot write a `reason_code_label` for it. Dimensionality
reduction and adverse-action explanation are mutually exclusive. This is not a
tuning preference — it removes the system's core governance claim.

**Statistical imputation would destroy an informative signal.** The repo fills missing
values with domain sentinels, deliberately:

```
features/stage1_features.py:44   out["bureau_score"] = out["bureau_score"].fillna(-1)
features/stage1_features.py:34   estimated_ltv ... .fillna(0.70)
features/stage1_features.py:38   declared_obligations.fillna(0.0)
```

`-1` is not a missing bureau score; it is *"this applicant has no bureau file"*, which
is itself among the strongest signals in the file and which a tree can split on
cleanly. MICE would replace that with a plausible invented score — deleting the signal
and asserting a fact about a customer that is not true. Keep the sentinels.

**Resampling would break the thing the models exist to produce.** Base rates from the
shipped model cards:

| Model | Base rate |
|---|---|
| conversion | 65.9% |
| override | 36.3% |
| prepayment | 15.3% |
| risk | **9.4%** |

None of these is severe imbalance; 9.4% is comfortably within the range where a
calibrated classifier works without intervention. And resampling changes the base
rate, so predicted probabilities stop corresponding to real-world frequencies. In a
system whose value chain is literally `P(conversion) × lifetime value` in rupees,
that is not a modelling trade-off — it is discarding the output's meaning to improve a
metric nobody uses here. The whole imbalanced-learning row solves a problem this
dataset does not have.

### 9.4 The addition the catalogue misses

The list proposes LSTM / GRU / TCN / Transformer under "sequential credit behaviour".
There is no behavioural sequence data in this repo, so none of them is applicable
today.

But there *is* a documented sequential gap, and it is already named in the
limitations: expected life is a 24-month balance-transfer probability projected as a
**constant annual hazard**, and the phase-2 upgrade named in `architecture.md §16` is
a survival model.

The right method is therefore a **discrete-time hazard model, or Cox with
time-varying covariates** — not a recurrent network. It:

- needs no new data (`months_to_exit` and `realized_life_years` are already in the
  schema),
- feeds directly into `yield_calculator`, where expected life multiplies every rupee
  of lifetime value,
- stays interpretable enough to keep the reason-code contract, and
- fixes the single largest source of imprecision in the economics.

That is the highest-value modelling addition on this page, and it is not on the list.

### 9.5 Revised shortlists

Replacing the proposed ones:

> **Layer 3:** IV / WoE → Mutual Information (doubling as the proxy test) →
> correlation + VIF → adversarial validation → CSI beside the existing PSI.
> Permutation importance and SHAP are already built.

> **Layer 4:** WoE Logistic Scorecard → EBM → keep HistGB / XGBoost →
> **survival model for expected life**. Monotone-constrained members only on the
> conversion and prepayment contracts; ensembling only where monotonicity is not
> required.

Nine methods instead of thirty, and every one of them either answers a question a
reviewer will ask or fixes a gap the repo has already admitted to.

---

## 10. Stage-2 Loan Acceptance Confidence — BRD against the build

Unlike §§8–9, this specification is not about a future platform. It describes the
system in this repository, plus one genuinely new component. Roughly ten of the
fourteen functional requirements are already met.

### 10.1 Functional requirements

| FR | Requirement | Status | Where |
|---|---|---|---|
| 01 | Retrieve Stage-2 customer and loan information | **Built** | `POST /pricing/final-assessment`, `underwriting_agent` |
| 02 | Ingest CRM notes and interaction information | **Partial** | `knowledge/sales_note_retriever.py` + TF-IDF `vector_store.py`; no CRM/WhatsApp/transcript connector |
| 03 | LLM extraction of intent and pricing signals | **New** | — |
| 04 | Identify competitor information where mentioned | **Partial** | `SalesNoteRetriever.competitor_signals()` aggregates note commentary, caveated as unverified |
| 05 | Classify competitor information by source and reliability | **Built, coarser** | `competitor_evidence_verified` + enforcement at `constraints.py:216` |
| 06 | Standardised contextual features from unstructured text | **New** | Insertion point exists: `features/sales_features.py` |
| 07 | Calculate an acceptance probability | **Built** | The `conversion` model *is* this — P(disbursed \| rate) |
| 08 | Calculate an evidence confidence score | **Built at Stage 1 only** | `stage1_features.confidence()`; see 10.2 |
| 09 | Provide the primary drivers behind the score | **Built** | `explanation_agent` — reason codes, commercial-vs-risk split, caveats |
| 10 | Support price simulations across multiple rates | **Built** | `conversion_agent.profile()` — negotiation table, `rate_for_70pct_conversion`, `rate_for_50pct_conversion` |
| 11 | Provide results to the Price Uplift engine | **Built** | Conversion is step 7 of the orchestrator |
| 12 | Capture salesperson overrides and reasons | **Built** | `POST /pricing/override`; `overrides` table holds `ai_rate` **and** `human_rate`, reason, approver, authority |
| 13 | Capture the final customer decision | **Built** | `POST /pricing/outcome`, `outcomes` table |
| 14 | Retain scoring evidence for audit | **Built** | Decision snapshot + policy, feature-config and model versions; `GET /pricing/audit/{loan_id}` |

FR-07 deserves emphasis: **Loan Acceptance Probability already exists and is
calibrated.** The `conversion` model predicts P(disbursed | offered rate), isotonically
calibrated to a 0.002 gap, monotone non-increasing in price and competitor gap, on a
65.9% base rate. It is not a new model to build — it is the model the whole pricing
engine already turns on.

### 10.2 Evidence Confidence already exists, at the wrong stage

The BRD's headline second number is implemented — for Stage 1:

```python
features/stage1_features.py:104
def confidence(df, band_width_bps):
    """Two drivers: how wide the predicted interval is, and how much of the
    pre-assessment information is actually present on the file."""
    ...
    return (0.55 * width_score + 0.45 * known).clip(0.15, 0.95)
```

`known` is a weighted presence check over bureau score (0.30), competitor rate (0.15),
property value (0.25), declared obligations (0.10) and credit history (0.20). That is
exactly the BRD's "quantity, freshness, consistency and reliability of available
information", minus freshness.

And the Stage-2 raw material is already computed. `underwriting_agent.py:152-157`
builds a `completeness` block on every digital twin:

```python
completeness={
    "post_assessment_fields_present": len(present),
    "post_assessment_fields_expected": len(POST_ASSESSMENT_EVIDENCE),
    "completeness_ratio": round(ratio, 2),
    "missing": [...],
}
```

Nobody joins these two. **FR-08 is a port, not a build** — reuse the Stage-1 shape,
drive `known` from `completeness` plus competitor-evidence status and note recency,
and surface it beside the acceptance probability. Half a day, not a workstream.

### 10.3 Drop §9 of the BRD

The proposed scoring framework — historical 30%, conversation 20%, rate sensitivity
15%, competitor 15%, urgency 10%, affordability 5%, market 5% — contradicts §7 and §19
of the same document, which correctly say the LLM should produce contextual features
while calibrated models produce the probability.

A hand-weighted sum would be strictly worse than what ships today: uncalibrated, unable
to represent interactions, and with no monotone guarantee in price. The `conversion`
model already learns these weights from actual disbursal outcomes. The BRD's own note
that "actual weights should be determined through historical validation" is the right
instinct; the correct conclusion is to delete the table, not to tune it.

**Division of labour: LLM → features. Conversion model → probability.** That is already
the architecture.

### 10.4 Three controls the BRD needs and does not have

**Materialise LLM-extracted signals onto the decision record.** The system's core
governance claim is that any decision can be replayed exactly, because the snapshot
stores the application as priced plus every config and model version. An LLM call
breaks this: it is non-deterministic, and a hosted model changes silently beneath a
fixed version string. The BRD asks for prompt/version tracking as a *non-functional*
requirement — necessary, and not sufficient.

The requirement must be functional and explicit: **extracted signal values are
persisted as first-class fields on the decision snapshot, and audit replays from the
stored values — never by re-calling the model.** Without it the audit trail silently
stops being reproducible, which is the one property the whole governance design exists
to protect.

**Shrink toward the base rate when evidence is thin.** §4.2 says the engine "can make
more conservative decisions when evidence confidence is low", which is directionally
right and operationally undefined — two engineers will implement it two ways. The
principled rule: shrink the acceptance estimate toward the segment base rate in
proportion to evidence weight. One line, defensible, and it composes with the existing
isotonic calibration instead of fighting it.

**Present acceptance as a slope, not a point.** §11 shows 9.30% → 84%, 9.35% → 81%,
9.40% → 76%. The conversion model's calibration gap is 0.002, so the *average* is
trustworthy — but AUC is 0.700, so an individual file's point estimate is not precise
to one percentage point, and a three-point difference between adjacent rows is not a
real distinction. Render the table as a shape with a band. The defensible sentence for
this profile is the elasticity the repo already reports: **roughly −9.6 points per
+25 bps.**

### 10.5 The governance gap the BRD misses entirely

The document never mentions fair pricing, and a price-uplift capability is precisely
where it becomes load-bearing.

`pricing_policy.yaml` already carries `fair_pricing.commercial_uplift_review_bps: 10` —
when commercial context prices a customer **up** past that threshold, the decision is
flagged and written to the audit log as `fair_pricing_review_required`. The stated
reasoning is that commercial pressure normally prices a customer *down*, so pricing up
is a different kind of decision and a fair-lending reviewer should see it.

This feature makes pricing customers up the *deliberate objective*. That control
therefore moves from an occasional exception to a systematic one, and its threshold
needs revisiting before launch, not after.

More specifically: **pricing on urgency is pricing on vulnerability.** The BRD's own
worked example is a customer who will accept a higher rate because they need
disbursement before month-end. Sometimes that is a property deadline. Sometimes it is a
medical bill or a distress sale. The extracted `urgency` signal cannot tell the
difference, and a model that charges more to whoever needs the money most is the
clearest consumer-duty problem a lending platform can build.

Concrete controls, all cheap:

1. **Cap the uplift attributable to urgency** separately from total uplift — the
   machinery exists, `commercial_share()` already decomposes how much of a rate came
   from commercial context rather than customer risk.
2. **Route urgency-driven uplift above the cap to human review**, reusing the existing
   `fair_pricing_review_required` audit path.
3. **Include urgency in the fairness measurement** from §4 P0 — check whether
   urgency-driven uplift concentrates in any segment or age band.

None of this blocks the capability. It is the difference between a price-uplift engine
that survives a conduct review and one that does not.

### 10.6 What to build

| Item | Status | Note |
|---|---|---|
| Stage-2 evidence confidence (FR-08) | **Built** | `acceptance/evidence_agent.py`, same shape as `stage1_features.confidence()` |
| Competitor reliability levels (FR-05) | **Built** | Four tiers in `acceptance/signals.py`; stronger source wins, weaker claim retained |
| LLM extraction → signals (FR-03, FR-06) | **Built** | `acceptance/extraction_agent.py` — Claude, with a deterministic rule fallback |
| Signal persistence / replay (10.4) | **Built** | `AcceptanceSignals.from_dict()`; audit replays from the record, never re-calls the model |
| Urgency-uplift review route (10.5) | **Built** | `AcceptanceAgent._governance()` raises `urgency_driven_uplift_review_required` |
| Configurable data sources | **Built** | `acceptance/sources.py` + `config/acceptance.yaml`; runs on one, several or none |
| Wire into the orchestrator / Stage-2 API | **Not done** | Deliberate: changes the priced path, needs a decision |
| Feature-contract extension for urgency / intent | **Not done** | Needs retraining + a reason-code label each (§8.2) |
| CRM / transcript connectors (FR-02) | **Not done** | Source stubs exist and are disabled; scope depends on source systems |

Everything else in the BRD was already shipped. The honest summary for a steering
committee is that this is an **extension of an existing engine**, not a new capability —
provided the LLM stays on the feature side of the line the BRD itself draws in §19.

The subsystem now lives in `src/aavas/acceptance/` (4 agents, 25 tests) and is **not
wired into the priced path**. Connecting it is a deliberate, separate decision.
