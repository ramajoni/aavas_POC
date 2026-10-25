# Solution Architecture — Agentic Credit Pricing OS

**Status:** Proof of concept, running end to end on synthetic data.
**Version of record:** app `0.1.0` · policy `HL-POLICY-2026.09` · features `FEAT-2026.09` ·
model config `MODEL-2026.09` · agent set `ASET-2026.09-d076e1df` · connectors `CONN-2026.09`
**Last verified against the code:** 23 September 2026, `make test` → 393 passed, 3 skipped.
**Also versioned:** governance `GOV-2026.09`

---

## 0. About this document

**Audience.** Architecture review, model risk, credit policy and engineering leadership
at a lender evaluating or governing this system. It assumes familiarity with retail
credit and with ML systems, but not with this codebase.

**Scope.** The complete pricing decisioning platform: the agent runtime, the model
suite, the governance record, the observability layer and the external integration
surface. It covers what is built, what is deliberately not built, and what cannot yet
be answered.

**What this document is not.** It is not a developer guide — [architecture.md](architecture.md)
covers component internals and [flows.md](flows.md) covers step-by-step behaviour. It is
not a proposal: every figure quoted here was read from the running system or the test
suite, and every gap named is a gap that exists today.

**A note on the numbers.** This system is fitted on synthetic data. Metrics demonstrate
that the pipeline works; they are not expected production performance. Where that
distinction matters, it is stated rather than assumed.

| Companion document | Covers |
|---|---|
| [architecture.md](architecture.md) | Component internals, module responsibilities |
| [flows.md](flows.md) | Every path through the system, step by step |
| [data-model.md](data-model.md) | Datasets, tables, artefacts, configuration keys |
| [api-reference.md](api-reference.md) | Endpoint semantics and payloads |
| [operations.md](operations.md) | Running, testing, building, troubleshooting |
| [agentic-os-flow-gap.md](agentic-os-flow-gap.md) | The proposed agent flow mapped onto the code |
| [target-architecture-gap.md](target-architecture-gap.md) | The 2026 target architecture mapped onto the code |

---

## 1. Executive summary

The system prices home loans in two stages, explains every price, routes every
recommendation to a human authority, and records enough to defend the decision later.
It is organised as a pipeline of nine agent steps over a shared digital twin rather
than a model behind an endpoint, because the questions a lender must answer about a
price — *which grade, under which policy, against which competitor claim, approved by
whom* — are owned by different agents and have to be separable.

**Three properties distinguish it from a scoring service.**

1. **It recommends; it does not approve.** Every recommendation carries the authority
   level that must sign it off. Recording a human override is not approval, and the
   system says so in the payload rather than implying otherwise.
2. **A decision is reconstructible.** 12 of 14 provenance links resolve *as of the
   moment the decision was priced*, and **none is broken** — which agent set, which
   model version, which card, which dataset hash, which policy. The two that are
   partial say exactly what they cannot answer.
3. **What it cannot measure, it declines to report.** The observability layer returns a
   metric with no source as a null carrying the reason, rather than a plausible figure.

**Scale.** 17,103 lines of Python across 16 packages, 2,106 lines of frontend, 3,545
lines of tests (393 passing), 58 API routes, 8 configuration files holding 500 settings.

**Maturity.** Both halves of the lifecycle now exist. Sign-off over configuration, models
and the agent set is built, with per-role signatures, gates that block, and an apply step
separate from approval. Sign-in is now required by default, so a signature carries a verified (if
deployment-local) identity; it stays short of production-ready until the user table is
replaced by SSO.

---

## 2. Architectural drivers

### 2.1 Business context

A home-loan lender prices in two moments. At enquiry it must quote a credible indicative
range before any document exists. After assessment it must quote a firm rate that covers
credit risk, funds cost and servicing, wins the customer against a competitor who has
usually already quoted, and stays inside policy. The commercial pressure at the second
moment is what makes this a governance problem rather than a modelling one: the rate a
branch wants to offer and the rate the book can afford diverge under target pressure,
and both are defensible positions.

### 2.2 Constraints

| # | Constraint | Consequence in the architecture |
|---|---|---|
| C1 | A priced decision must be defensible to an auditor years later | Append-only decision store; full application snapshot; version stamps on every decision |
| C2 | The engine may not approve its own price | Authority routing is computed, never satisfied internally; override capture is separate from approval |
| C3 | Customer communications may not be read without consent | Consent gate blocks communication connectors regardless of their config switch |
| C4 | A model may not be fed a field it will not have at inference | Two feature contracts, stage-separated, with a leakage guard in the validator |
| C5 | The POC must run with nothing installed | SQLite store, in-memory vector store, optional dependencies degrade to declared fallbacks |
| C6 | Fitted on synthetic data | Every metric surface states it; drift boards state that both windows come from one generator |

### 2.3 Quality attributes, and how each is evidenced

| Attribute | Target | Measured | Where |
|---|---|---|---|
| Latency, indicative | interactive | p50 **22 ms**, p95 68 ms | `/control-tower` |
| Latency, final assessment | branch-tolerable | p50 **310 ms**, p95 2,948 ms, max **48.3 s** | `/control-tower` |
| Auditability | a decision reconstructs | **12 of 14** links, **0 broken** | `/system-of-record/reconstruct` |
| Reliability | success and failure are countable | request rows opened **before** the pipeline | `/control-tower` |
| Fairness | every champion is evaluated | measured on the training path; **2 of 4 classifiers fail** | `/governance/monitors` |
| Explainability | every price has reason codes in bps | 2 independent layers | `/xai` |
| Availability under degradation | no tool takes the engine down silently | 17 tools, each with a declared fallback | `/agents/tools` |
| Testability | behaviour asserted, not described | **393** tests | `make test` |

The latency tail is reported **by stage** rather than pooled. The pooled median of 284 ms
is a duration no request has taken: pre-assessment scores a band from a loaded model and
final assessment runs the whole pipeline. One number over two workloads describes
neither, and the 48-second outlier — one request a branch waited on — would have been
invisible in a mean.

---

## 3. Architecture principles

These are not aspirations. Each is enforced somewhere, and the enforcement point is
named.

| # | Principle | Enforced by |
|---|---|---|
| P1 | **The engine recommends; a human approves.** | `constraints.approval_authority()`; `within_authority` recorded and flagged, never used to block |
| P2 | **Nothing exists only as code.** | `agent_registry` content-hashes the agent/skill/tool declaration into the store and stamps it on every decision |
| P3 | **A tool can be unavailable; a skill cannot.** | `test_capabilities.py` rejects a skill module that imports a store, connector or model artefact |
| P4 | **A collected signal is never a model feature.** | `connectors.yaml` `consumes` layer; `McpSource` refuses any connector not declared for `acceptance_signals` |
| P5 | **A metric with no source is reported as having none.** | `/control-tower` returns null plus the reason; `test_control_tower.py` asserts it |
| P6 | **Degrade loudly, never silently.** | Every optional dependency catches `Exception` and reports the method actually used |
| P7 | **Declared, then asserted against the code.** | The roster, capability and pipeline-order tests fail on drift in both directions |
| P8 | **Absence is a finding, not an omission.** | Dormant agents, blocked connectors, absent registries and unsourced tiles are all rendered rather than hidden |

P6 deserves a note, because the failure mode is subtle. Optional dependencies are probed
with `except Exception`, not `except ImportError`: a library built against a different
version of its own dependency raises `AttributeError` on import, and a broken optional
dependency must degrade exactly like a missing one. When the SHAP explainer is
unavailable the payload reports the ablation method it actually used, so a reader never
mistakes one for the other.

---

## 4. System context

```text
                        ┌────────────────────────┐
     Branch RM ────────▶│                        │
     Credit officer ───▶│   AGENTIC CREDIT       │◀─── Model risk
     Pricing committee ▶│   PRICING OS           │◀─── Credit policy
     Customer (portal) ▶│                        │◀─── Internal audit
                        └───────────┬────────────┘
                                    │  all external reads over MCP,
                                    │  all currently blocked
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
        acceptance_signals   application_record     economics
        WhatsApp · Gmail ·   Salesforce LOS/LMS     Oracle Fusion
        Outlook                                     (proposed, not applied)
```

Five external systems are declared across **three layers**, and the layer is the
load-bearing field. Collapsing them into one source list is how a Salesforce column
quietly becomes an LLM input:

- **`acceptance_signals`** — a customer's own words, extracted into a closed signal
  vocabulary with a verbatim quote. Never a model feature.
- **`application_record`** — typed fields the engine already has columns for. A
  nine-field allow-list; anything else returned is dropped, because a connector must not
  be able to widen the feature contract.
- **`economics`** — funding, servicing, acquisition and capital charge. **Proposed for
  approval, never applied on read**, because a change here moves every price.

---

## 5. Container view

| Container | Technology | Responsibility | State |
|---|---|---|---|
| Pricing API | FastAPI + Uvicorn | 58 routes; the only write path to the governance store | stateless |
| Lending workspace | React 19 + Vite, `/` | RM and admin surface | browser only |
| OS console | React 19 + Vite, `/os.html` | Lifecycle and governance surface | browser only |
| Governance store | SQLite (Postgres-ready via `AAVAS_STORE_URL`) | Decisions, overrides, outcomes, audit, requests, approvals and signatures, model registry, agent manifests, prompts, knowledge indexes, federated rounds, monitor runs | **durable, append-only** |
| Model artefacts | joblib on disk | 6 fitted models + cards | versioned |
| Knowledge store | in-memory TF-IDF (pgvector under `phase2`) | Policy clauses, sales notes | rebuilt at start |
| MLflow | file store `mlruns/`, optional server | AutoML search history | read-only from the API |

The two frontends are **one build with two entry points** (`rollupOptions.input`) sharing
a single component library. That is deliberate: the explainability page a model-risk
reviewer opens in the console is the same component an RM opens in the workspace, so the
two audiences cannot be shown different numbers.

The API reads the MLflow **file store** directly rather than importing the library, so
tracking pages populate with no MLflow server running and the service keeps no dependency
on it.

---

## 6. Component view

```text
                              api/  (58 routes)
                                │
    ┌───────────────┬───────────┼────────────┬──────────────┬─────────────┐
    ▼               ▼           ▼            ▼              ▼             ▼
 agents/        models/      xai/       governance/    analytics/   connectors/
 1,622          1,237        1,608        665            522          289
 orchestrator   6 learners   2 layers   store · audit   perf/drift   3 gates
 roster         cards        SHAP+omni  registries      (control      MCP only
 capabilities                            manifests       tower)
 trace
    │                                        │
    ├── optimization/ 690   scenarios, yield, constraints
    ├── features/     419   two stage-separated contracts
    ├── acceptance/ 1,082   3 dormant agents, signal vocabulary
    ├── learning/     692   drift, outcomes, override learning, retraining
    ├── automl/     2,180   HPO, imbalance, locality, leaderboard
    ├── federated/    368   FedAvg/FedProx/uniform/trimmed-mean
    ├── knowledge/    341   policy + sales-note retrieval
    ├── ingestion/    382   schemas, loader, validator (leakage guard)
    └── synthetic/    537   20,000-application generator
```

---

## 7. The decision path

### 7.1 Two stages, two contracts

| | Stage 1 — pre-assessment | Stage 2 — final assessment |
|---|---|---|
| Trigger | Enquiry, before documents | After verification and valuation |
| Inputs | Declared income, requested amount, soft-pull bureau | Verified income, appraised value, bureau depth, banking behaviour, legal/technical outcome |
| Model | `stage1`, gradient-boosted **quantile** (16 features) | `stage2`, HistGB rate anchor (24 features) |
| Output | Indicative band `{min, max, width_bps}`, eligible amount with binding constraint, confidence | Recommended rate, economics, reason codes, authority, scenarios |
| Latency | p50 22 ms | p50 310 ms |

The contracts are **stage-separated and enforced**: `ingestion/validator.py` guards
against a stage-2-only field reaching a stage-1 model. A band quoted at enquiry using a
field that only exists after valuation is not an indicative band — it is a promise the
system cannot keep.

### 7.2 The nine-step pipeline

```text
 1 underwriting    build the digital twin
 2 risk            PD · lifetime PD · LGD (policy table) · grade · expected loss
 3 policy          card rate and the permissible envelope
 4 sales_context   read commercial pressure against the card rate
 5 pricing         search the envelope for best risk-adjusted yield
 6 policy          enforce — did the recommendation stay inside?
 7 sales_context   re-read against the recommendation
 8 conversion      price response and the negotiation table
 9 explanation     reason codes in bps, commercial-vs-risk split
```

Two agents appear twice because they genuinely run twice, against different reference
prices. `policy` proposes an envelope and then checks the result against it; a single
pass could not distinguish "priced inside the band" from "band computed to fit the
price". `tests/test_agents.py` asserts the **declared order equals the order the
orchestrator actually ran**.

Every agent writes onto one `CustomerDigitalTwin`, which is what makes this a pipeline
rather than six models called in sequence. `snapshot()` stores the file **exactly as
priced**, so a later audit re-runs against the real application rather than a
reconstruction.

### 7.3 Agents, tools and skills

Three registries, because the three fail differently and conflating them produces a
system that monitors its arithmetic and does not notice its database is gone.

| | Count | Question | Failure mode |
|---|---|---|---|
| **Agent** | 12 (9 wired, 3 dormant) | who decided | the decision does not happen |
| **Tool** | 17 across 8 kinds | what it called that lives outside this process | **unavailable** — each declares its fallback |
| **Skill** | 11 | what it worked out for itself | a bug; no health check will find it |

**The discriminator is that a tool can be unavailable and a skill cannot.** It is
enforced, not documented: a module declared as a skill that imports `governance.store`,
`connectors` or a model artefact fails the suite. Four modules written as skills were
reclassified by that check — `acceptance.sources` reaches a connector, and
`outcome_capture`, `override_learning` and `retraining` all read the store
(`retraining.assess()` counts decisions as one of its four triggers).

Tool kinds: 6 model artefacts, 1 store, 2 indexes, 2 optional libraries, 1 connector
transport, 1 source registry, 3 procedures, 1 LLM endpoint. **Tools are also the natural
MCP surface** — they already have a name, a stated failure mode and a fallback. Skills
are not exposed: there is nothing to call remotely, and a remote call would only add a
way for arithmetic to become unavailable.

Three agents are **built, tested and called by nothing**. They read a customer's own
words; wiring them in puts an LLM on the priced path, which is a decision for a human
rather than a default. The roster reports them as dormant rather than omitting them,
because a roster that hid them would describe a pipeline other than the one running.

---

## 8. Model architecture

| Model | Learner | Target | n_train | Features | Role in the price |
|---|---|---|---|---|---|
| `stage1` | GB quantile | sanctioned rate | 16,000 | 16 | Indicative band |
| `stage2` | HistGB | sanctioned rate | 16,000 | 24 | Rate anchor |
| `risk` | HistGB classifier | defaulted | 10,544 | 20 | PD → grade → expected loss |
| `conversion` | HistGB classifier | disbursed | 16,000 | 14 | Price response |
| `prepayment` | HistGB classifier | BT-out 24m | 10,544 | 11 | Expected life |
| `override` | HistGB | overridden + bps | 16,000 | 16 | Override expectation |

### 8.1 What is modelled and what is not

**PD is fitted. LGD is a policy table** (`model_config.risk.lgd_by_property_type`,
LTV-adjusted) because loss severity on a secured home loan is a collateral and recovery
question, not a borrower-behaviour one — the model card states this rather than leaving
it inferable. **EAD is not modelled and is not in the formula**: expected loss is
`PD × LGD × 100`, expressed as a percentage of exposure. For a disbursed home loan on a
known amortisation schedule, a model here would be predicting arithmetic.

### 8.2 Validation discipline

- **Chronological splits.** A random split on a book with a rate cycle leaks the future
  into the fit. `_clean_boundary()` pushes rows sharing the boundary date onto the
  validation side, because an index cut through a busy date puts the same day on both.
- **Out-of-time calibration.** The AutoML leaderboard reports out-of-time ECE, not the
  in-sample `calib_ece_after` — the latter is ~0 for isotonic by construction and would
  rank every calibrated candidate as perfect.
- **Age-band fairness runs on the training path**, not only inside an AutoML search.
  It previously gated candidates in a search that has never promoted anything, while the
  six champions actually pricing decisions were fitted by `scripts/train_all.py` and
  never evaluated at all. Running it exposed that **`risk` (calibration gap 0.0613) and
  `prepayment` (0.0587) fail the 0.05 threshold**, and `stage1`/`stage2` are regressors
  where it is not measurable — recorded as *not measured*, which is not *passed*. The
  models were not adjusted to pass; the gate now blocks their promotion and the fairness
  monitor alerts on them. Calibration by group is binding; demographic parity is
  reported, not gated. A parity gate on a credit model would force equal outcomes across
  groups with genuinely different observed default rates; a calibration gate requires the
  predicted probability to mean the same thing in each group, which is the property that
  can actually be defended.
- **Champion decisions sit inside split noise.** Two of four verdicts flipped between
  passes differing only in boundary handling, and the risk champion clears the fairness
  gate by 0.0016. This is reported rather than smoothed over: a leaderboard that presents
  a 0.0016 margin as a decision is misleading its reader.

### 8.3 A known defect, reported not fixed

`calibration.py` selects the calibrator by an **in-sample** ECE, which is ~0 for isotonic
regression whatever the data. Fixing it changes which calibrator future searches pick,
so it is recorded here and left for a deliberate decision rather than changed quietly.

---

## 9. Data architecture

### 9.1 Stores

| Store | Contents | Durability |
|---|---|---|
| `decisions` | Full payload, snapshot, versions, trace, elapsed_ms | **append-only** |
| `overrides` | Human rate, reason, approver, `within_authority` | append-only |
| `outcomes` | Disbursal, response, sanctioned rate, delinquency | keyed by `loan_id` |
| `audit_log` | Actor, action, entity, detail | append-only |
| `model_registry` | Full model card against `(name, version)` | versioned |
| `agent_manifests` | Agent/skill/tool declaration against a content hash | versioned |

Decisions and audit entries are append-only by design: a pricing decision that can be
quietly edited afterwards is not an audit trail.

### 9.2 Two joins that are structurally weak

Both were wrong on the first implementation and are now correct-but-limited, and both
sit on the feedback edge:

1. **Overrides join on `decision_id`, not `loan_id`.** A file priced more than once —
   which the demo flow does six times — would otherwise attach an override made against
   a different decision and report it as this one's. An override belongs to the price it
   moved.
2. **Outcomes are keyed by `loan_id` alone.** A file priced six times has **one** outcome
   and no way to say which decision it followed. This is the join the learning loop needs
   and does not have.

### 9.3 A gap in the outcome record

There is no **lost-to-competitor reason**. The outcome says `rejected`; it does not say
to whom or at what rate. The override table carries `competitor_lower_rate`, but that is
why a human moved the price, not why the customer left. Combined with §9.2, this is why
the learning loop is the weakest part of the record.

### 9.4 The `-1` sentinel

Bureau scores are 300–900, with `-1` reserved for new to credit. The distinction is
load-bearing and easy to lose: reading the low end of a partial-dependence curve as
"worst credit" actually reads the **new-to-credit cohort**. The credit-score band
excludes the sentinel from its deciles and reports that cohort separately — which
surfaced that new-to-credit files are priced **+0.52 % p.a. above** the weakest scored
decile while the risk model puts them **0.024 below** it in PD. That contradiction is now
computed on screen rather than buried in a curve.

---

## 10. Governance architecture

### 10.1 The system of record

Twelve registries, audited on a distinction that decides what each is worth: **a registry
is queryable as of a past moment.**

| Kind | Count | Registries | Answers |
|---|---|---|---|
| **Registry** | 7 | agent, skill, tool, model, knowledge, prompt, dataset | as of the decision |
| **Per-decision data** | 1 | trace | as of the decision |
| **Versioned config** | 3 | policy, feature, connector | only while that version of the file survives |
| **Absent** | 1 | memory | nothing |

Eight of the twelve answer a question about the past, up from two. The four that were
absent closed as follows:

* **Prompt** — content-hashed into `prompts`. Nothing is on the priced path today, which
  is precisely why it exists now: wiring the extraction agent in without it would put an
  unversioned prompt in front of a customer's own words.
* **Knowledge** — each built index records its corpus, backend, document count and a hash
  over the documents, so re-indexing an unchanged corpus reuses the id rather than
  producing noise.
* **Dataset** — a content hash over the frame each model was *actually fitted on*, not its
  parent. `risk` and `prepayment` share one id (disbursed loans only); the other four
  share another.
* **Memory** stays absent, and correctly: no agent carries state between decisions, so
  there is nothing to register. That is a description of the system, not a gap.

### 10.2 The agent manifest

The agent, skill and tool declarations are snapshotted into `agent_manifests` and the
snapshot id is stamped on every decision through `versions()`.

**The id is derived, not declared.** `ASET-2026.09-d076e1df` is a sha256 over the
declaration plus the pipeline order. A hand-bumped version number is a promise someone
has to keep on every edit, and the one edit where they forget is the one that matters:
two different rosters would then share a version and the audit would say they were the
same. A hash simply changes.

The manifest deliberately excludes config. "Which roster ran" and "under which rules" are
two questions; folding them together makes a policy edit look like an agent change.

`log_decision()` publishes the manifest **before** writing the decision that names it. A
stamped id with nothing behind it is a dangling reference — precisely the failure the
registry exists to prevent.

### 10.3 Reconstruction, tested rather than asserted

`GET /system-of-record/reconstruct/{decision_id}` walks the audit claim link by link
against a real decision. Current result: **11 resolved, 2 partial, 1 broken of 14.**

| Link | Status |
|---|---|
| Which decision · file as priced · agents that ran | resolved |
| Which agent set, and which skills they had | resolved — from the snapshot that ran |
| Which model versions, and which card as of that version | resolved |
| Which feature contract · which policy | resolved |
| Which human decision · which final rate | resolved |
| Which tools | **partial** — registered but not permissioned; only model artefacts carry a version |
| Which data the models were fitted on | **partial** — window and row count, no dataset identity or hash |
| Which branch personalization | **broken** — federated learning has an implementation and no persistence |

A decision priced *before* the registry existed still reports the agent-set link as
broken, which is correct: its roster genuinely cannot be recovered.

The test asserts the property, not the mechanism: it prices a decision, drops an agent
from the roster, republishes, and checks the old decision still reports the old manifest
with `is_current: false` and a diff naming what was dropped.

### 10.4 Authority, and what it is not

`constraints.approval_authority()` computes the lowest authority that may sign off a
given deviation from the card rate. Recording an override then captures the human rate,
the reason from a closed list, the evidence flag, the approver and `within_authority`.

**An out-of-authority override is stored and flagged, not blocked.** An engine that
silently refused one would push the negotiation off-system, where nothing can audit it.
One such override is on record, visible on the runtime page.

**Recording an override is not approval.** There is no sign-off path in this system for
a config, a model, an agent set or a policy version — see §14.

---

## 11. Explainability

Two independent layers, kept apart because they answer different questions and fail
differently.

1. **Per-decision reason codes**, in basis points against named drivers, with a
   commercial-versus-risk split and a fair-pricing review flag. On the priced path.
2. **Model-level analysis** — global importance, partial dependence, the credit-score
   band, SHAP against LIME rank agreement, counterfactuals. Offline, in a separate
   environment.

**Rank agreement is a trust gate.** KernelSHAP and permutation importance are compared by
Spearman correlation; a model whose two attribution methods disagree on ranking gets a
`weak` trust label on screen rather than a confident bar chart.

Attribution lists render the **contribution**, not the feature value, with the value
beneath and an explicit units note — an earlier version showed values, which read as
`NaN` for categoricals and silently misinformed.

---

## 12. Observability — the AI Control Tower

Six control zones over one workload.

| # | Zone | State | Gap |
|---|---|---|---|
| 1 | Business & policy control | partial | readable and versioned; **not editable** |
| 2 | Agent factory | partial | observable and versioned; not configurable |
| 3 | Model & data control | partial | AutoML is not the training path; 5 of 5 connectors blocked |
| 4 | Human approval centre | partial | built, with sign-in enforced; a config change is still applied by editing the file |
| 5 | Runtime operations centre | ready | a request row opens before the pipeline runs |
| 6 | Continuous learning centre | partial | monitors record having run; nothing schedules them |

**The reporting rule.** Every metric carries the table or artefact that produced it, and a
metric with no source is returned as a **null carrying the reason** — rendered at the
same size as a real tile and listed again at the foot of the page. Three remain in that
state, down from five, and each is correctly unanswerable: acceptance signals (the agents
are dormant), realised conversion (too few outcomes to describe the book rather than the
files someone followed up), and the effect of federation on a priced decision (rounds
exist; the adjustment is not applied).

*Success rate* was the sharpest of the five and is now answered — but only because the
evidence changed. A row is opened in `requests` **before** the pipeline runs and closed
however it ends, so a failed request leaves one. Computed over decisions it would still be
100 % by construction, and `tests/test_control_tower.py` asserts the *source*, not the
presence of a number.

Most of what the tower reports is not new analysis. `analytics/performance.py` computed
model health, drift, operational load and champion/challenger state for a governance
dashboard that was never wired up — **nothing in the repository imported it** until this
layer existed.

---

## 13. External integration

All five connectors reach the outside world over **MCP**: one client, one audit path, one
place to revoke. Three gates, reported separately because they are fixed in different
ways.

| Gate | Fixed by | Current state |
|---|---|---|
| The switch | `enabled: true` in versioned config, owned by the business | all off |
| The server | An MCP endpoint the bank runs | none configured |
| **Consent** | Recorded per customer and per purpose | **`backend: none`** |

**The consent gate actually blocks.** Enabling WhatsApp and Gmail in `acceptance.yaml`
and collecting returns nothing, and the sources are recorded as *unavailable* — proven by
a test. A pricing engine that read a customer's messages because a YAML flag was flipped
has a consent problem, not a configuration problem. `max_message_age_days: 180` keeps a
two-year-old thread out of scope even where consent exists.

The live MCP session is **not implemented**. `call()` returns `available: false` with a
reason rather than an empty list, because silence would let an acceptance score claim it
read a source it never touched.

---

## 14. Approval and the one thing that still limits it

Sign-off over the system itself is built, and `config/governance.yaml` declares it:
five roles, five artefact types, and the gates each change must clear.

**Four properties make it a control rather than a form.**

1. **Every required role must sign.** A configuration change needs credit policy *and*
   technology; a model promotion needs model risk. One signature on a promotion is not a
   control.
2. **Gates run when the request is raised, not when it is approved.** A model that fails
   the fairness gate cannot be promoted by approving it harder.
3. **A failed gate can be overridden, never silently.** It takes model risk, a written
   justification of at least 40 characters, and the text is stored against the approval
   permanently. The risk champion is currently in exactly this position.
4. **The requester may not sign their own request**, checked against the person rather
   than the role.

**Approval changes nothing on its own.** A separate apply step re-checks the
authorisation, which makes *approved-but-not-released* a state the system can be in —
the state a change process actually lives in.

### What still limits it

- **Identity is deployment-local.** Sign-in is enforced by default against the users in
  `config/access.yaml`, and every route is role-checked. Those users are a file, not a
  directory: production needs SSO, and `security.principal_from_request` is the single seam
  where that goes. `AAVAS_AUTH_MODE=disabled` still exists for scripts, and every surface
  reports it when it is on.
- **Configuration is edited directly, not through sign-off.** The signed-in administrator
  edits a file at `#/config`: the change must parse, keep every section and carry a new
  version, a sample application is priced with it and it is rolled back if that fails, and
  it is recorded in `config_changes` with an undo. The approvals flow is not in front of it:
  a config approval still only records that an edit was authorised, and its apply endpoint
  still returns 501.
- **`champion_challenger.py` is reachable but has no replay.** The
  `challenger_beats_champion` gate reports *not evaluated* rather than passing, because no
  shadow comparison has been recorded for any pair.

---

## 15. Security and privacy

| Concern | Position |
|---|---|
| Consent | Structural gate, not a checkbox; blocks regardless of config (§13) |
| Data minimisation | `minimise: true`; connectors ask only for what the signal vocabulary can use — a connector returning a whole mailbox has collected more than it can justify |
| Field allow-list | Salesforce may write 9 declared fields; anything else is dropped |
| Personal data flag | Declared per connector and surfaced in the UI |
| Secrets | `.env` is gitignored and untracked; `ANTHROPIC_API_KEY` is deliberately empty |
| Credentials | None shipped; MCP servers are the bank's to run |
| Branch as identity | `branch_id` is an identity column, **not a model feature** — pricing on branch identity is a fairness exposure, not a signal |
| Authentication | Two modes. `disabled` (default) for the POC — anonymous, every role, and **every governance surface says so**. `apikey` enforces roles on every write, and the UI prompts for a key on the two pages that need a role — `api.js` holds it in `localStorage`, which is right for a key a developer pastes into their own browser and wrong for a real credential. Production needs SSO; `security.principal_from_request` and `api.js` are the only two seams that change |
| Tool permissions | Declared per agent and operation; only the learning agent may write the governance store. Enforced at `McpSource.fetch`, the one path that reads a customer's own words |
| Request log | Opened before the pipeline runs, so a failed or abandoned request is countable rather than invisible |

Authentication remains the significant row. It is off by default and that is a real
limitation, not a formality — which is why it is the single reason the approval zone
reports partial rather than ready.

---

## 16. Deployment view

```text
docker compose up                      api          :8000   (healthcheck /health)
docker compose --profile phase2 up     + postgres    :5432   pgvector, shared store
docker compose --profile tracking up   + mlflow      :5000
```

The API container carries a volume for model artefacts and the governance store, so both
survive a restart. Postgres and MLflow are **off by default**: the POC must run with
nothing installed, and a profile that has to be switched on is an honest statement that
the default deployment does not include it.

`AAVAS_STORE_URL` switches the store to Postgres without a code change.
`AAVAS_VECTOR_BACKEND` switches the knowledge store to pgvector.

The Vite dev server proxies same-origin `/api/*` to the API. **The static build does not
provide that proxy** — hosting `dist/client` elsewhere needs an equivalent same-origin
reverse proxy, which is the kind of detail that silently breaks a first deployment.

---

## 17. Architecture decision records

Each records a decision taken in this system, the alternative rejected, and why.

**ADR-01 — Two-stage pricing with separate feature contracts.**
*Rejected:* one model with missing-value handling. *Why:* a band quoted at enquiry using a
post-valuation field is a promise the system cannot keep; the validator enforces the
separation rather than trusting it.

**ADR-02 — A declared agent roster, asserted against the code.**
*Rejected:* introspecting classes ending in `Agent`. *Why:* introspection is accurate
about what exists and silent about what matters — what an agent decides, which models it
needs, whether anything calls it. The cost of declaring is drift, so tests check both
directions.

**ADR-03 — Tools and skills split on availability.**
*Rejected:* one "capabilities" list. *Why:* they need different governance. The check that
a skill may not import a store, connector or artefact is what makes "a skill cannot be
unavailable" true rather than asserted; it reclassified four modules immediately.

**ADR-04 — Content-hashed agent manifest.**
*Rejected:* a hand-maintained `agent_set_version`. *Why:* a version someone maintains is a
promise kept on every edit but one, and that one edit makes two different rosters share a
version.

**ADR-05 — Recording an override is not approving it.**
*Rejected:* blocking out-of-authority overrides. *Why:* an engine that refuses pushes the
negotiation off-system, where nothing can audit it. Store it, flag it, surface it.

**ADR-06 — Optional dependencies catch `Exception`, not `ImportError`.**
*Why:* a library built against a different dependency raises `AttributeError` on import.
A broken optional dependency must degrade exactly like a missing one, and the payload must
name the method actually used.

**ADR-07 — All external reads over MCP, behind three gates.**
*Rejected:* five bespoke SDK integrations. *Why:* one transport gives one audit path and
one place to revoke. The gates are reported separately because a switch, an unreachable
server and a missing consent record are fixed by different people.

**ADR-08 — Communications produce signals, never features.**
*Why:* the feature contract is the leakage guard. A connector that could add a column
could add an unvalidated, unexplainable input to a priced model.

**ADR-09 — Report a metric's absence rather than omitting it.**
*Rejected:* dropping tiles with no data. *Why:* a dashboard that silently drops what it
cannot measure looks complete, and a control tower whose dashboard cannot be told apart
from its mock-up is worse than none.

**ADR-10 — LGD as a policy table; EAD not modelled.**
*Why:* loss severity on a secured loan is a collateral and recovery question; EAD on a
known amortisation schedule is arithmetic. Modelling either would present a policy choice
as a prediction.

**ADR-11 — Append-only decision and audit tables.**
*Why:* a decision that can be quietly edited afterwards is not an audit trail.

**ADR-12 — Two frontend entry points, one component library.**
*Why:* the RM and the model-risk reviewer must not be shown different numbers for the
same thing.

**ADR-13 — Fairness moved out of `automl/` into `models/`.**
*Rejected:* importing `automl.fairness` from `models`. *Why:* fairness is a property of a
model, not of the way it was selected, and `models` may not depend on `automl`. The move
is what let the training path evaluate it — and it immediately showed two champions
failing a gate that had only ever been applied to candidates nobody promoted.

**ADR-14 — Request rows are opened before the pipeline runs.**
*Rejected:* logging on completion. *Why:* a completion-only log cannot record the failure
that stopped completion happening. `abandoned` is a distinct outcome for exactly that case.

**ADR-15 — Approval and apply are separate calls.**
*Rejected:* applying on final signature. *Why:* *approved but not released* is a state a
change process lives in, and collapsing it removes the operator's control over when a
change lands.

**ADR-16 — Gates run at request time, and a failure needs a written override.**
*Rejected:* evaluating gates at approval. *Why:* a signer should see the gate result
before signing, not after. An unevaluable gate reports neither pass nor fail: blocking on
unmeasurable would make every new model unpromotable, and passing it silently is the
opposite mistake.

**ADR-17 — Outcomes are keyed by decision, not by file.**
*Rejected:* one outcome per `loan_id`. *Why:* a file priced six times had one outcome and
no way to say which decision it followed — the join the learning loop needs. Legacy rows
were carried across with `decision_id` NULL rather than guessed.

**ADR-18 — Identity is derived, never declared.**
Applied three times: the agent manifest, the dataset hash and the prompt id are all content
hashes. *Why:* a version someone maintains is a promise kept on every edit but one, and
that one edit is the one an audit will ask about.

It also earned its keep immediately. An end-to-end run regenerated the dataset and produced
**identical metrics with a different dataset hash**. The cause was `branch_id`, derived from
Python's builtin `hash(city)` — which is salted per process, so every regeneration remapped
every city to a different branch. Metrics never moved because `branch_id` is an identity
column, not a feature, so nothing else in the system could have caught it; AutoML's locality
search uses it as its unit, so branch-level results were never reproducible. Fixed to a
`blake2b` digest, with a test asserting the builtin `hash()` decides nothing that lands in a
dataset.

---

## 18. Risks

| # | Risk | Severity | Position |
|---|---|---|---|
| R1 | **Two champions fail the fairness gate** — `risk` 0.0613, `prepayment` 0.0587 against 0.05 | **High** | Found by running the gate on the training path. Models were *not* adjusted to pass; promotion is blocked and the monitor alerts |
| R2 | Authentication is disabled by default, so signatures are unverified | **High** | §14, §15. The single reason the approval zone is partial |
| R3 | Champion selection sits inside split noise | Medium | Reported on screen; needs repeated-split evidence before promotion |
| R4 | `calibration.py` selects on in-sample ECE | Medium | §8.3 — reported, not silently changed |
| R5 | Final-assessment p95 is 2.9 s with a 48 s outlier | Medium | Surfaced by stage; cold model load is the candidate, unconfirmed |
| R6 | Nothing schedules the monitors | Medium | Passes record having run; `scripts/run_monitors.py` is what a cron would call |
| R7 | No shadow replay behind `challenger_beats_champion` | Medium | The gate reports *not evaluated* rather than passing |
| R8 | The regional adjustment is not applied on the priced path | Low | Rounds persist; no decision has been moved by one |
| R9 | All figures come from synthetic data | Context | Stated on every metric surface |

**R1 is the finding of this iteration.** Fairness existed and gated AutoML candidates in a
search that has never promoted anything, while the six models actually pricing decisions
had never been evaluated. Running it on the real path is what exposed it — which is the
argument for putting a control where the work happens rather than where it is convenient.

---

## 19. What is deliberately not built

Each of these is a decision, not an omission.

- **Wiring the three acceptance agents.** Changes what the engine prices on. The prompt
  registry that was its prerequisite now exists.
- **Promoting an AutoML champion.** AutoML searches and explains; it is not the training
  path.
- **Adjusting `risk` or `prepayment` to pass the fairness gate.** Tuning a model until a
  fairness threshold clears is gaming the threshold. The failure is surfaced, promotion is
  blocked, and the remedy — recalibration or a documented, approved exception — is a
  decision for model risk.
- **Editable configuration through the API.** The approval path now exists; writing config
  files from a web request is a separate decision with its own blast radius.
- **A live MCP session.** No endpoint ships here; the stub says so rather than returning
  empty.
- **A Business Configuration agent.** Targets and rate boundaries are configuration a
  committee approves. Wrapping them in an agent adds a layer that can only get it wrong;
  what is missing is the approval path, not an agent.
- **A fully autonomous feature-admission agent.** Feature *discovery* can be automated;
  feature *admission* cannot, because a feature with no human-authored reason label
  cannot appear in a customer-facing explanation.

---

## 20. Roadmap, in dependency order

| # | Work | Unblocks |
|---|---|---|
| 1 | **Recalibrate `risk` and `prepayment` by age band**, or document an accepted exception through the approval path | R1; unblocks model promotion |
| 2 | **SSO** behind `security.principal_from_request`, with roles from the directory | R2; moves the approval zone from partial to ready |
| 3 | **Schedule the monitors** and alert on a gap in the run history | R6 |
| 4 | **Shadow replay** behind `challenger_beats_champion` | R7; makes the promotion gate real rather than declared |
| 5 | **Apply the regional adjustment** on the priced path and stamp the round id | R8; the last partial provenance link |
| 6 | **Stamp tool versions** — shap build, MCP server — onto the decision | The other partial link |
| 7 | **Editable configuration** behind the approval path | Closes zone 1 |

Item 1 is the only one that blocks a model decision today. Item 2 is the only one that
blocks real data.

---

*Every figure in this document was read from the running system or the test suite on
23 September 2026. Where a number could not be produced, the document says so.*
