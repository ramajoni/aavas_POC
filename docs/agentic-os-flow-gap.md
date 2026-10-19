# The agentic OS flow, and A2UI, against this repository

A map of the proposed 17-agent flow and the Configure → Train → Validate →
Approve → Run → Observe → Learn lifecycle onto what `aavas` actually contains,
followed by an assessment of A2UI as the presentation layer.

The short version: **the run-time half is largely built and the design-time half
barely exists.** Of the 17 agents, 9 run today, 3 are built and dormant, 2 exist
as modules rather than agents, and 3 are genuinely absent. Of the five proposed
UI modules, two are live and three have no interface at all. That asymmetry —
not any individual missing agent — is the finding.

Cross-reference: [target-architecture-gap.md](target-architecture-gap.md) maps
the model-side architecture. This document covers the agent flow and the UI.

---

## 1. The 17 agents against the code

| # | Proposed agent | In this repo | State |
|---|---|---|---|
| A | Business Configuration | `config/pricing_policy.yaml` (rate floors/ceilings, approval authority) | **Config, no agent.** Product, segment, geography, yield/growth/conversion targets and risk appetite are not modelled as configuration at all |
| B | Policy, Role & Governance | [policy_agent.py](../src/aavas/agents/policy_agent.py), [constraints.py](../src/aavas/optimization/constraints.py), [governance/audit.py](../src/aavas/governance/audit.py) | **Wired.** Policy and approval authority are real; *roles* are not — there is no user model, only an `approved_by` string parsed as `name:level` |
| C | Use Case Configuration | — | **Absent.** Stage-1/Stage-2/acceptance are switched by code path, not config |
| D | Data Intake & Data Quality | [validator.py](../src/aavas/ingestion/validator.py) — `validate`, `coerce_types`, `quarantine`, `assert_no_leakage`; [connectors/](../src/aavas/connectors/) — WhatsApp, Gmail, Outlook, Salesforce, Oracle Fusion over MCP | **Module, not an agent.** Every quality check named in the paste exists except freshness. Five connectors are now declared and gated; all blocked (no servers, no consent store). Still no AA or bureau pull |
| E | Stage 1 Assessment | [preassessment_agent.py](../src/aavas/agents/preassessment_agent.py) | **Wired** |
| F | Stage 2 Assessment | [underwriting_agent.py](../src/aavas/agents/underwriting_agent.py) + [pricing_agent.py](../src/aavas/agents/pricing_agent.py) | **Wired**, split differently: the twin is built once and the pricing decision is its own agent |
| G | Feature Intelligence | `config/feature_config.yaml`, [feature_pipeline.py](../src/aavas/features/feature_pipeline.py), [automl/fairness.py](../src/aavas/automl/fairness.py) | **Contracts, no agent.** Leakage and fairness checks exist; usefulness, stability and redundancy checks do not. See §4 for why this one cannot simply be automated |
| H | Model Intelligence | [automl/](../src/aavas/automl/) — 41 algorithms, HPO, imbalance, calibration, ensembles, Pareto selection, MLflow | **Built, not wired to training.** `make automl` searches and explains; `models/` still comes from `make train` |
| I | Risk Intelligence | [risk_agent.py](../src/aavas/agents/risk_agent.py), [models/risk/predict.py](../src/aavas/models/risk/predict.py) | **Wired.** PD is modelled; **LGD is a policy table** (`model_config.yaml: lgd_by_property_type`), **EAD is not modelled at all** |
| J | Human / Sales Context | [sales_context_agent.py](../src/aavas/agents/sales_context_agent.py) | **Wired.** Kept in a separate feature block on purpose: it moves price for reasons that are not risk |
| K | Loan Acceptance Score | [acceptance/](../src/aavas/acceptance/) — extraction, evidence, acceptance | **Built, dormant.** Nothing calls it |
| L | Confidence | `stage1_features.confidence()`, `twin.completeness`, `evidence_agent` | **Partial.** Stage-1 has a confidence score; there is no agent that reports confidence across risk + acceptance + pricing together |
| M | Decision & Pricing Optimizer | [price_optimizer.py](../src/aavas/optimization/price_optimizer.py) | **Wired.** Already maximises `P(convert) × risk-adjusted value` over the policy band — the paste's objective, implemented |
| N | Human Review / Override | `learning_agent.capture_override`, `constraints.approval_level` | **Wired.** Every field the paste lists is captured |
| O | Customer Outcome | [outcome_capture.py](../src/aavas/learning/outcome_capture.py) | **Built.** `compare_to_decision` closes the loop against what was recommended |
| P | Repayment Outcome | `defaulted` in the training table | **Partial.** The default label exists historically; nothing ingests live repayment, DPD, restructure or recovery |
| Q | Federated Calibration & CL | [federated/](../src/aavas/federated/), [drift_detection.py](../src/aavas/learning/drift_detection.py), [retraining.py](../src/aavas/learning/retraining.py), [champion_challenger.py](../src/aavas/learning/champion_challenger.py) | **Built (simulation).** PSI, feature/conversion/performance drift, shadow comparison and a retraining trigger all exist |

**Score: 9 wired · 3 dormant · 2 modules · 3 absent.**

The three absent ones (A, C, and a real Confidence agent) are exactly the ones
the lifecycle document needs, which is the next section.

---

## 2. Design-time against run-time

The second paste splits the OS into Design-Time (configure, train, validate,
approve) and Run-Time (run, observe, learn, revalidate). Scored against the repo:

| Lifecycle stage | Repo | Gap |
|---|---|---|
| 1. Pre-configuration | 6 YAML files, versioned (`*_config_version`), served read-only at `#/config` | No validation of a config change, no approval before it takes effect |
| 2. Data prep & training | `make data`, `make train`, `make automl` | AutoML is not the training path; the two disagree about what a champion is |
| 3. Model & agent validation | Model cards with limitations, `champion_challenger.shadow_compare`, `analytics/performance.py`, `make xai` | No scenario or stress testing; no simulation harness a credit committee could run |
| 4. Agent configuration validation | — | **Absent.** Agent permissions, allowed data, tools and output schemas are code, not configuration, so there is nothing to validate |
| 5. Ready-for-run gate | `model_registry.promote(name, version, actor)` | One actor, one call. No multi-party sign-off, no "published version" of the *agent set* — only of individual models |
| 6. Agent running | Orchestrator + [step trace](../src/aavas/agents/trace.py) | Live |
| 7. Request/response monitoring | `decision_logger` + the trace: every agent, its timing, its output, model versions, policy version, latency, override, final decision | Missing from the paste's list: **data sources used**, **prompt/instruction version**, **agent-set version** |
| 8. Outcome monitoring | `outcome_capture`, `performance.operations()` | Conversion and rate outcomes only; no repayment feed |
| 9. Continuous learning | Drift, retraining assessment, override learning, federated | Runs as scripts, not as a scheduled loop with a record of what it decided |
| 10. Change / re-approval | `champion_challenger.promote(..., approved_by, verdict)` | Models only. A policy change, a new feature or a new agent has no approval path at all |

**The pattern:** stages 6–9 are real and observable. Stages 1–5 and 10 are either
YAML edited by hand or nothing. An agent in this repo does not have a lifecycle —
it has a git history.

### The one that matters most

Stage 4 — *agent configuration validation* — is absent, and it is the stage that
makes the rest coherent. Today an agent's permissions ("may this agent read a
customer's call notes?"), its allowed data and its output schema are expressed
only as Python. That is why stage 5 cannot exist: there is nothing to sign off
on. [agents/roster.py](../src/aavas/agents/roster.py) is the beginning of the
answer — it declares what each agent decides, which models it needs and whether
anything calls it — but it is descriptive, not enforced.

Making it enforced is a small change with a large consequence: if the roster
declared each agent's **allowed inputs** and the orchestrator refused to pass
anything else, then "agent configuration" becomes a thing that can be reviewed,
versioned and approved.

---

## 3. The five UI modules

| Module | Today | |
|---|---|---|
| 1. Configure | `#/config` | **Visible, not editable.** All 395 settings, searchable, with env overrides and the file as written. No approval path, so no editing |
| 2. Train | — | **No interface.** `make train`, `make automl` |
| 3. Validate & Approve | `#/models`, `#/xai`, `#/credit-score`, `#/mlflow` | **Half.** Everything needed to *judge* a model is on screen; nothing to *approve* one |
| 4. Run | `#/application`, `#/offer` | **Live** |
| 5. Monitor & Improve | `#/dashboard`, `#/learning`, `#/audit`, `#/agents` | **Live** |

Modules 4 and 5 are built. Module 3 has the evidence and no verdict button.
Modules 1 and 2 do not exist as an interface at all — which is the strongest
argument in the A2UI proposal, and the reason to take the next section seriously.

---

## 4. A2UI as the presentation layer

The proposal is that agents declare what they need, configuration decides what is
allowed, and A2UI renders it — rather than hand-writing `Stage1Form.tsx`,
`Stage2Form.tsx` and so on. For the Configure and Train modules, which have no UI
at all, this is a genuinely good fit, and the repo is closer to it than it looks:

* **The "missing parameters" contract already exists.**
  `AcceptanceSignals.missing` ([signals.py](../src/aavas/acceptance/signals.py))
  lists exactly what a source could not supply, against a closed vocabulary
  (`SIGNAL_SPEC`) with per-field type and range. That is the agent-side half of
  the proposal, already written and tested.
* **So does source configuration.** [sources.py](../src/aavas/acceptance/sources.py)
  builds enabled sources from `config/acceptance.yaml`, and records which were
  unreachable. An agent that knows what it is missing *and* which source should
  have supplied it is precisely what a dynamic input surface needs.
* **And the completeness signal.** `twin.completeness` already drives what
  Stage-1 tells the user it could not verify.

Three constraints it has to respect, in order of how badly each one bites.

### 4.1 A collected field is not a feature

`config/feature_config.yaml` is the guard against Stage-1 leakage, and
`validator.assert_no_leakage()` enforces it per stage. A dynamic UI that can
collect arbitrary parameters must never be able to widen a model's feature
contract. Collection and the feature contract are different things and have to
stay different things: A2UI belongs on the collection side of that line, and a
new field reaches a model only through a deliberate contract change and a
retrain.

### 4.2 A field with no reason label cannot be explained

Every feature in `feature_config.yaml` carries a human-authored sentence for each
direction (`up: "Age loading", down: "Age benefit"`). That is what makes a reason
code readable on a decision. A field collected dynamically has no such sentence,
so if it reached the priced path the explanation would have a contribution it
could not name. This caps how autonomous the Feature Intelligence agent (G) can
be, for the same reason.

### 4.3 A parameter registry is a second source of truth

The proposed registry carries validation ranges, role permissions and visibility
rules per field. The repo already holds those, spread across `feature_config.yaml`
(what a model may see), `pricing_policy.yaml` (authority levels),
`acceptance.yaml` (signal vocabulary and reliability) and `api/contracts.py`
(types and ranges — `bureau_score` is 300–900 or −1, declared once, in a
validator). A registry authored *beside* those will drift from them within a
quarter. It should be **derived** from them: generated at import, with a test
asserting the generated registry and the configs agree — the same discipline
`tests/test_agents.py` applies to the agent roster.

### 4.4 What I would verify before committing

A2UI's version story (v0.9.1 production, v1.0 candidate) comes from the material
supplied; it is worth confirming against the specification directly before
building on v1.0, along with whether a renderer exists for React 19. The
trusted-catalog property is the one that matters for a BFSI deployment — an agent
selecting from approved components rather than emitting executable UI — and that
is the property to check first, because the whole argument rests on it.

---

## 5. Build order

Ordered by what unblocks the most, not by what is most interesting.

**P0 — Make agent configuration a thing that exists.**
Extend [roster.py](../src/aavas/agents/roster.py) from description to contract:
declared inputs, outputs and allowed data per agent, enforced by the
orchestrator. Without this there is nothing for stages 4, 5 and 10 to act on.
Small, self-contained, and every later item depends on it.

**P1 — A published agent-set version.**
Today each model is promoted individually. Add a version over the whole set —
models, policy version, feature-config version, agent roster — recorded on every
decision. This is one field on the decision payload and one registry table, and
it is what makes "which configuration priced this file?" answerable.

**P2 — Wire the acceptance agents in.**
They are built, tested and dormant. This is the single largest capability gain
available, and it is a human decision, not an engineering one: it puts an LLM on
the priced path and promotes competitor signals into pricing. See
[target-architecture-gap.md](target-architecture-gap.md) §10.

**P3 — A Confidence agent.**
The pieces exist separately (stage-1 confidence, twin completeness, evidence
reliability, model calibration). One agent that reports a single confidence with
its drivers, and raises the human-review flag, is mostly assembly.

**P4 — Configure and Train as interfaces**, via A2UI if §4 checks out and via
plain forms if it does not. Both modules are empty today, so either beats
nothing; the A2UI case is that the same renderer then serves the dynamic
input surfaces in P2.

**P5 — Approval workflow.** Multi-party sign-off on a published agent-set
version, replacing the single `actor` string. Needs P1 first.

---

## 6. What I would not build

* **A Business Configuration agent.** Targets, risk appetite and rate boundaries
  are configuration a human sets and a committee approves. Wrapping them in an
  agent adds a layer that can only get it wrong; what is missing is the *config
  and its approval path*, not an agent.
* **A Use Case Configuration agent.** Switching Stage-1 or acceptance scoring on
  and off is a feature flag. `config/acceptance.yaml` already does this for
  sources; extend the pattern rather than agentifying it.
* **A fully autonomous Feature Intelligence agent.** §4.2 is the reason: a
  feature with no human-authored reason label cannot appear in a customer-facing
  explanation. Feature *discovery* can be automated; feature *admission* cannot.
* **Modelling EAD.** For a disbursed home loan the exposure is the outstanding
  balance on a known amortisation schedule. A model would be predicting
  arithmetic.

---

## 7. The AI Control Tower as the layer above

The framing is right, and it is worth being precise about *why* it is right in
this repository specifically: the console built for §3 is already the control
tower, and this engine is already one workload under it. What the tower adds is
not a new place to look. It is the accounting of which zones have a surface at
all.

### 7.1 The six zones against the code

| # | Zone | Reached at | State | What is missing |
|---|---|---|---|---|
| 1 | Business & policy control | `/config`, `/pricing/policy/explain` | partial | Read-only. A policy change is a file edit and a restart — no validation, no approval in front of it |
| 2 | Agent factory | `/agents` | partial | Roster is declared and observable; agents are not configurable and the agent *set* has no version |
| 3 | Model & data control | `/tracking`, `/connectors`, `/control-tower/models` | partial | AutoML is not the training path; 5 of 5 connectors are blocked |
| 4 | Human approval centre | — | **absent** | Nothing can be signed off. Authority exists over a single priced decision; no config, model, agent set or policy version has an approval path |
| 5 | Runtime operations centre | `/control-tower/operations`, `/agents/trace/{id}` | ready | Only successful decisions leave a row |
| 6 | Continuous learning centre | `/control-tower/drift`, `/pricing/learning/*` | partial | Runs on demand, not on a schedule, and records nothing about having run. Federated learning has no surface |

Zone 4 is the same blocker §2 identified from the other direction. Everything in
the design-time half converges on it: there is no published, approved release to
run and therefore nothing to roll back to.

### 7.2 The finding that made this cheap to build

`src/aavas/analytics/performance.py` computes model health, input and conversion
drift, operational load and champion/challenger state. Its own docstring says it
is "for the governance dashboard". **Nothing imported it** — not the API, not a
script, not a test. Zones 5 and 6 were written and unreachable; wiring them was a
router and two pages, not an analysis.

`aavas/federated/` is in the same position with one exception: `scripts/run_demo.py`
calls it. Nothing persists a round, a participant or a global version, so the
tower reports federated learning as having no source rather than showing a zero.

### 7.3 The rule the dashboard enforces

A control-tower mock-up fills every tile. Real estates cannot. `GET /control-tower`
therefore returns each metric with the table or artefact that produced it, and
returns a metric with no source as a **null carrying its reason**:

```
Success rate      no source   only decisions that succeeded are written. A failed or
                              refused request leaves no row, so any success rate
                              computed here would be 100% by construction
```

Five tiles are in that state today. They are rendered at the same size as a
sourced tile and listed again at the foot of the page, because the absence is the
finding. `tests/test_control_tower.py` asserts the rule rather than the numbers.

### 7.4 What the real numbers said that the mock-up did not

Three figures came out of the wiring that no mock would have produced:

* **Latency is two workloads, not one.** Pooled p50 is 284 ms — a figure no
  request has taken. Pre-assessment runs p50 22 ms / max 68 ms; final assessment
  runs p50 310 ms / p95 2,948 ms / max **48.3 s**. The tower reports them
  separately for that reason. The 48-second decision is one request a branch
  waited on; cold model load after a restart is the obvious candidate and is
  worth confirming against the API log rather than assuming.
* **The engine prices above card, not below.** Mean recommended 11.43 % against
  a mean card rate of 11.13 % — **+29.8 bps**, not a discount. Whether that is
  correct is a business question; that it is the opposite of the usual assumption
  about a discount optimiser is why the tile is signed and labelled.
* **The override rate is 10.7 %, and one override was recorded outside the
  approver's authority.** It is stored and flagged rather than blocked: an engine
  that silently refused an override pushes the negotiation off-system, where
  nothing can audit it.

### 7.5 What a real control tower still needs

* **An approval path (zone 4).** Nothing else in the design-time half can be
  finished without it. `learning/champion_challenger.py` already declares the
  promotion gates and `model_config.yaml` sets them; no route calls it, so a
  promotion today happens by re-running training rather than by passing a gate.
* **A request log, not a decision log.** Success rate, timeout rate and failure
  rate all need a row written *before* the pipeline runs, not after it succeeds.
* **A schedule.** Drift computed when someone opens a page is a report, not a
  monitor. Nothing records that it ran or what it found.
