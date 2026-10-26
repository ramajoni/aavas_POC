# Aavas Home Loan Pricing, Yield Maximization and Sales Override Intelligence

A runnable proof of concept for the two-stage pricing problem: give the customer a
credible price early, determine the right permissible price after assessment,
capture every human override with its reason, and learn from what the book
actually earned.

Solution provider: [Ajoni Technologies](https://www.ajonitech.com/).

The core design decision is that **this is not a rate-prediction model**. A model
that predicts the sanctioned rate only reproduces historical commercial behaviour,
including its mistakes. Here, the Stage-2 model produces an *anchor*; the
recommendation comes from simulating every permissible price and choosing the one
that maximises risk-adjusted value, subject to policy and to conversion and
portfolio guard-rails.

```
Risk-Adjusted Value = P(conversion) x Lifetime Economics
                      - Credit Loss - Funding/Servicing - Prepayment/BT Impact
```

---

## Quick start

```bash
pip install -e ".[dev]"          # or: pip install -r requirements.txt
make data                        # generate the synthetic lifecycle dataset
make train                       # train and register all six models  (~1 min)
make demo                        # one customer, end to end, then the learning loop
make test                        # 88 tests
make api                         # http://localhost:8000/docs
make xai-install && make xai     # OmniXAI global + instance analysis, with dashboards
```

### Two surfaces

The same build serves two separate UIs against one API:

| Surface | URL | Who operates it |
|---|---|---|
| **Lending workspace** | `http://localhost:5173/` | Branch and credit staff: price a file, record an override, capture an outcome — plus an admin dashboard |
| **Agentic Credit Pricing OS** | `http://localhost:5173/os.html` | Model risk, credit policy and engineering: the control tower over this workload, and below it the lifecycle — Configure → Train → Validate & Approve → Run → Observe |

The OS console opens on a **lifecycle board** that reports what is in place at
each stage *and what is not* — a readiness view that only ever showed green
would be decoration. It shares the component library and the API client with the
lending app; only the shell differs.

Everything runs on numpy, pandas, scikit-learn, FastAPI and PyYAML. XGBoost,
SHAP, Optuna, MLflow, psycopg and Flower are all optional: each is detected at
runtime and has a working fallback, so nothing in this repo fails because a
library is missing (see [Optional dependencies](#optional-dependencies)).

---

## Documentation

This README is the business case and the results. The technical documentation lives in
[`docs/`](docs/):

| Document | Covers |
|---|---|
| [docs/feature-intelligence.md](docs/feature-intelligence.md) | Agentic feature engineering — discovery, the four refusals, the veto-ordered decision, and what is deliberately not built |
| [docs/solution-architecture.md](docs/solution-architecture.md) | **The formal architecture document** — drivers, principles, the four views, quality attributes with measured evidence, 12 decision records, risks and roadmap |
| [docs/architecture.md](docs/architecture.md) | Layers, components, the agent pipeline, the decision kernel, governance |
| [docs/agentic-os-flow-gap.md](docs/agentic-os-flow-gap.md) | The 17-agent flow and the design-time/run-time lifecycle against this repo, and A2UI as a presentation layer |
| [docs/flows.md](docs/flows.md) | Every path through the system, step by step, with sequence diagrams |
| [docs/api-reference.md](docs/api-reference.md) | Endpoints, contracts, response shapes, a copy-pasteable demo flow |
| [docs/data-model.md](docs/data-model.md) | Lifecycle table, feature contracts, store schema, every config knob |
| [docs/operations.md](docs/operations.md) | Running, testing, containers, troubleshooting, changing behaviour without code |

---

## The two stages

### Stage 1 - pre-assessment

Income is self-declared, the property may be unvisited, there may be no bureau
file. Quoting a single precise rate here is a promise the underwriter has to
break, so Stage 1 quotes an interval from a quantile trio (q10 / q50 / q90), plus
an eligibility range and a confidence level.

```
Indicative range   11.60% - 12.35%  (width 75 bps)
Confidence         0.78 (high)
Eligible amount    Rs 1,251,000  (capped by property value (LTV cap), so the rate does not move it)
Caveat             Income is self-declared. For this profile, verified income is
                   typically lower, which can move both the rate and the amount.
```

The band is clamped to the product envelope and to a credibility window
(`min_band_width_bps` / `max_band_width_bps`) — too tight is falsely precise, too
wide is not a quote. Confidence falls when the file is thin, so a new-to-credit
enquiry with no competitor quote gets a visibly lower number rather than a
confidently wrong band.

### Stage 2 - post-assessment

Verified income, valuation, LTV, bureau depth, banking behaviour, deviations,
legal and technical outcomes. The pipeline:

```
digital twin -> risk (PD, grade, expected loss)
             -> policy envelope (card rate, permissible band, affordability cap)
             -> sales context (competitor gap, target pressure, override risk)
             -> scenario simulation across every permissible price
             -> yield optimization
             -> policy validation and approval routing
             -> explanation (reason codes, value decomposition, counterfactuals)
             -> audited decision record
```

Note the order: **risk runs first**, because the grade it assigns is what the
policy card rate is built from. The agents are a pipeline, not six models in
parallel.

---

## Why price uplift is not yield

This is the heart of the brief, and it is visible in the scenario table the engine
produces for every file:

```
 rate   conversion   expected life   lifetime value %   expected value (Rs)
12.04        0.869            8.00              9.68               165,398
12.24        0.854            8.00             10.37               173,454
12.54        0.783            8.00             11.40               173,889
12.79        0.737            8.00             12.09               175,456   <- peak
13.39        0.573            6.89             11.44               146,506
13.59        0.371            5.14              9.24                82,947
```

Above ~13.4% the customer stops converting *and* the survivors leave early, so
expected life collapses from 8.0 years to 5.1 and lifetime value falls with it.
The contractual yield is still rising the whole way. That divergence is the
entire argument for optimizing yield rather than price.

Three guard-rails constrain the search (`config/pricing_policy.yaml`):

- a **minimum conversion probability** — do not win the price and lose the customer;
- a **maximum balance-transfer probability** — do not book a loan that leaves;
- an **EV tolerance tie-break** — where two prices are economically
  indistinguishable, the customer gets the lower one.

### The extrapolation guard

A monotone gradient-boosted model is piecewise constant outside its training
range: ask it about a rate no customer was ever offered and it confidently
returns the last value it saw. Because lifetime value keeps rising linearly while
that flat conversion line goes nowhere, expected value starts climbing again near
the ceiling — a pure artefact that would eventually make the optimizer recommend
the product ceiling.

The conversion model therefore records the rate and competitor-gap range it was
actually fitted on, and the scenario generator trims the search to it:

```
upper end trimmed from 16.50% to 13.73%: beyond that the conversion model
is extrapolating, not predicting
```

---

## Human overrides: the confounding problem

The brief asks whether overrides improved conversion and long-term yield. Answering
that naively gives the wrong answer, because **overrides are not randomly
assigned** — salespeople discount precisely the files that were going to be hard
to close.

On this repo's data:

| Read | Conversion effect | Value effect per file |
|---|---|---|
| Naive comparison | **−3.3 pts** | **−Rs 6,717** |
| Propensity-adjusted | **+4.9 pts** | +Rs 840 ± 1,587 (no measurable effect) |

The naive number says "stop overriding". Almost all of it is selection: roughly
**Rs 7,600 per file** of the apparent damage is the confound, not the decision.

The correction (`src/aavas/learning/override_learning.py`) scores every file with
the override model, buckets by predicted override propensity, and compares
overridden to non-overridden files *within* a bucket. Both readings are reported
side by side so the difference is visible rather than asserted.

This is stratification, not a randomised experiment. Verdicts carry a standard
error and a sample-size floor, thin slices return `insufficient evidence`, and the
output is a **proposal to the pricing committee**, capped at ±15 bps — never an
automatic change to anyone's rate. Properly answering "was the salesperson right?"
needs a price test, which is the natural next step for this POC.

The reason-level league table and the evidenced-vs-unevidenced competitor split
are produced the same way; specific verdicts depend on the data and seed, so read
them from your own run rather than from this README.

---

## Governance

Interest-rate decisions affect customers, so the engine recommends and explains —
it never approves.

- **Policy is data, not code.** Every spread, cap, cost and authority limit lives
  in `config/pricing_policy.yaml`. `optimization/constraints.py` is the single
  source of truth for the card rate; the synthetic generator calls the same
  function, so the history the models learn from and the rules enforced at runtime
  cannot drift apart.
- **Affordability is a real constraint.** Raising the price raises the EMI, which
  can push a thin file past the FOIR cap. The permissible band's upper end is
  solved for, and the decision record says when affordability rather than policy
  is what binds.
- **Unverified competitor claims are captured but capped.** A claimed competitor
  quote informs the recommendation and is recorded; without evidence it cannot
  justify more than `max_unevidenced_discount_bps`. The engine blocks the discount
  and says why:
  ```
  POLICY BLOCK  reason 'competitor_lower_rate' requires evidence for a discount
                above 15 bps (requested 20 bps)
  ```
- **Approval routing, not approval.** Every recommendation names the lowest
  authority that may sign it. Overrides record the approver and whether they had
  the authority.
- **Both numbers always survive.** The AI recommendation is never overwritten by
  the human decision; without both there is nothing to learn from.
- **Full audit trail.** Every decision stores the application exactly as priced,
  the policy version, the feature-config version and every model version.
  `GET /pricing/audit/{loan_id}` returns the file's whole history.
- **Model cards with limitations.** Every artefact carries its training window,
  metrics and a written list of what it cannot be trusted to do. `GET /model-cards`
  exposes them.

---

## Explainability

SHAP is used when it is installed and working. When it is not, a baseline
**ablation explainer** takes over: hold the file at the median customer, restore
one real feature at a time, and rescale the attributions to sum to the distance
from the baseline prediction — so the reason codes always reconcile with the
number on the screen. The method used is reported in the payload rather than
implied.

Raw attributions are translated into sentences a branch can use:

```
Reason codes (ablation):
  Weaker risk grade (+68 bps)                  Risk grade
  Competitive market pressure (-41 bps)        Competitor pricing
  No credit deviations (-40 bps)               Credit deviations
  Weaker bureau profile (+8 bps)               Bureau score
  commercial pressure accounts for 24% of the movement (-41 bps)
```

That last line answers a question a single SHAP plot cannot: **how much of this
customer's rate came from commercial pressure rather than their own risk
profile?** It is computable because the feature contract keeps sales context in a
separate block from customer risk.

Alongside it: a value decomposition (rate → funding → servicing → capital →
expected loss → margin → lifetime value), a negotiation table showing conversion
at each 10 bps step, the rate needed for 70% and 50% conversion, and actionable
counterfactuals for the customer (*"clearing 1 credit deviation would improve the
indicated rate by about 12 bps"*).

---

## Model-level explainability: OmniXAI

Reason codes explain a *decision*. Governance also has to answer what the model
does across the whole book, and whether one file's explanation survives being
recomputed by a different method. That is a separate pass, built on
[OmniXAI](https://github.com/salesforce/OmniXAI):

```bash
make xai-install                 # build .venv-xai (once)
make xai                         # analyse all six trained models, serve the dashboards
make xai-reports                 # same analysis, no server
make xai ARGS="--models stage2 conversion --instances 8"
make xai ARGS="--fast"           # smaller samples, no counterfactuals
.venv-xai/bin/python scripts/run_omnixai.py --retrain   # train first, then analyse
```

Budget ten seconds for `stage1` up to about five minutes for a calibrated
classifier with counterfactuals: KernelSHAP is model-agnostic, so its cost is
model evaluations - and a `CalibratedClassifierCV` costs three of those per call.
`--fast` trades sample size for a run that finishes in well under a minute.
`make xai` serves the first three models by default — one dashboard per process,
about 1.5 GB each — and `--max-dashboards` raises that. Every model still gets a
report either way.

Per model it produces `reports/omnixai/<model>/` (a markdown brief, the ranked
`global_importance.csv`, and `analysis.json`) plus a dashboard on its own port:

| Analysis | Explainer | Answers |
|---|---|---|
| Global importance | KernelSHAP over a holdout sample | what moves this model, in rank order |
| Global importance | Permutation importance | ...and does a second method agree |
| Global shape | PDP and ALE | *how* each feature moves it, not just how much |
| Credit score | deciles of `bureau_score` + PDP over scored files | what the model does at the worst and best credit score |
| Instance | KernelSHAP + LIME side by side | why this file, twice, independently |
| Instance | MACE counterfactuals | the smallest change that flips the outcome |
| Dataset | correlation, mutual information, imbalance, chi-square | what the holdout itself looks like |
| Prediction | ROC/PR, confusion, residuals | where the model is wrong |

Two numbers in every report exist to keep the rest honest. **Rank agreement**
(Spearman, KernelSHAP vs permutation) says whether the global ranking is a
property of the model or of the method: 0.96 on `stage2` and 0.91 on `override`
means read the ranking; 0.39 on `risk` means do not lean on it. **Top-3
agreement** does the same per file for SHAP vs LIME — `risk` scores 0-2 out of 3
there too, which is the same warning twice.

### The credit score, at both ends

`bureau_score` sits in all six feature contracts, so every report carries a
`credit_score` block answering the question a credit reviewer asks first: what
does this model do at the bottom and the top of the score? It is read two ways,
because they can disagree — **observed** (the files that actually sit in the
bottom and top deciles, which carries everything that travels with a score:
income, LTV, vintage) and **holding everything else fixed** (the PDP re-read over
scored files only). Both ends are also explained as individual files: the
lowest- and highest-scoring applications join the instance sample, so the band
links straight to the attributions behind it.

One thing this block exists to prevent: `-1` is the new-to-credit sentinel, not a
bad score. It is the smallest *value* in the book, so the raw PDP's low end is
the thin-file cohort — read as "worst credit" it describes a different
population. The band drops that point, says it dropped it, and reports the
new-to-credit files as their own cohort. In `stage1` they price *worse* than the
bottom decile of scored files, which is a policy answer, not a modelling one.

The band is served at `GET /xai/{model}/credit-score` for one model, and
`GET /xai/credit-score` for all of them at once. In the admin view it appears
twice, deliberately: beside the model's own brief under **Model Explainability →
Credit score**, and as its own page, **Credit Score XAI** (`#/credit-score`),
which puts every model's band side by side. One model's band says what the score
does to that prediction; six of them say what it does to the customer — the rate
they are quoted, the risk they are assessed at, whether the engine expects them
to convert or to leave. Those can point in different directions, and only the
cross-model view shows it.

### Both dashboards from inside the admin view

MLflow and the OmniXAI dashboards are separate servers with their own
lifecycles, so the admin view gives each of them its own page:

| Page | Route | What it holds |
|---|---|---|
| **MLflow Tracking** | `#/mlflow` | Every AutoML search with its candidates, plus the MLflow UI embedded |
| **XAI Dashboard** | `#/xai-dash` | The interactive OmniXAI dashboard, one process per model |
| **Model Explainability** | `#/xai` | The offline briefs — global importance, the credit-score band, SHAP against LIME, dashboard captures. Needs no running process |
| **Credit Score XAI** | `#/credit-score` | Every model's credit-score band side by side, with the response curves. Needs no running process |
| **AI Control Tower** | `#/control-tower` | The whole estate: six control zones, live numbers from the governance store and model cards, and — listed rather than hidden — the tiles nothing in this system can fill |
| **Feature Intelligence** | `#/features` | Candidate features generated from the stage contract, the four refusals, feature cards over what is already in production, and the lineage graph |
| **Approvals & Gates** | `#/governance` | Change requests, per-role signatures, the gates each change must clear, monitor run history and the prompt/knowledge/federated registries |
| **System of Record** | `#/record` | The twelve registries, the registered agent sets, and a walker that tries to rebuild a real decision link by link |
| **Runtime & Drift** | `#/runtime` | Latency by stage, approval routing, override reasons, input and conversion drift, champions and the promotion gates |
| **Agent Control Plane** | `#/agents` | Three tabs: the agent roster and step trace, the 17 **tools** with what makes each unavailable and what runs instead, and the 11 **skills** with the config that governs each |
| **Configuration** | `#/config` | All settings across the config files, searchable, with the environment overrides in force. Editable by the signed-in administrator: each change is validated, must carry a new version, is test-priced and rolled back if pricing fails, and is recorded with an undo; read-only for every other role |
| **Connectors** | `#/connectors` | The five external systems — WhatsApp, Gmail, Outlook, Salesforce, Oracle Fusion — over MCP, and which gate is holding each |

### Agents, tools and skills are three registries

They fail differently, so they are declared separately. **An agent** decides
something and is accountable for a field on the decision. **A tool** is what it
called that lives outside this process — a model artefact, a store, an MCP
server, an optional library, an LLM endpoint — and every one names what makes it
unavailable and what runs instead. **A skill** is deterministic logic with no
external dependency, so it has no availability and no fallback.

The discriminator is exactly that: **a tool can be unavailable; a skill cannot.**
`tests/test_capabilities.py` enforces it — a module that reaches the store, a
connector or a model artefact is rejected as a skill, and an agent must declare
every capability module it imports *and* not declare one it stopped importing.
Four modules were written as skills and reclassified because that check rejected
them. Tools are also the set an MCP surface would expose; skills are not.

### Sign-off, and the gate that is currently failing

Changes to the system itself — configuration, models, the agent set — go through
`/governance`: every required role must sign, gates run when the request is raised
rather than when it is approved, a failed gate needs a written justification from
model risk that is stored permanently, and the requester may not sign their own
request. **Approval changes nothing on its own**; a separate apply step re-checks
the authorisation.

Running the fairness gate on the *training* path — it previously ran only inside an
AutoML search that has never promoted anything — showed that **`risk` (calibration
gap 0.0613) and `prepayment` (0.0587) fail the 0.05 threshold**. The models were not
adjusted to pass. Promotion is blocked, the fairness monitor alerts, and the remedy
is recalibration or a documented exception through the approval path.

**Sign-in is required.** The lending workspace and the OS console open on a role picker
(borrower, relationship manager, area manager, risk and model validation), and every API
route is checked against the role rules in `config/access.yaml` before it runs; a route
with no rule is closed to everyone but the administrator. An override is recorded under
the person signed in, at their role's authority, never a name typed into the request.

| User | Password | Role | Lands on |
|---|---|---|---|
| `rahul` | `Rahul@2026` | Borrower | Customer portal: indicative offer, apply, upload documents |
| `rohit` | `Rohit@2026` | Relationship manager · Pune | My applications (submitted online, with documents), loan assessment, overrides |
| `meera` | `Meera@2026` | Area manager · Pune West | Dashboard, override review, audit |
| `arjun` | `Arjun@2026` | Risk and model validation | Models, XAI, governance; the OS console |
| `admin` | `Admin@2026` | Platform administrator | Everything ("Sign in with a username instead") |

**Borrowers can register themselves** with "New borrower? Create an account" on the sign-in
screen: name, mobile, optional email, a username and a password. The account is stored in the
database (`borrower_accounts`), is always a borrower whatever the request says, cannot take a staff
or reserved username, and is signed straight in. Turn it off with `registration.borrowers: false`
in `config/access.yaml`.

A borrower applies after seeing an indicative offer and uploads documents (PDF, JPEG or
PNG, judged by their contents; types and limits in `config/intake.yaml`). The relationship
manager's **My applications** lists every submitted application with the range we
recorded (never one the browser sent), downloads its documents, and opens it for
assessment. A borrower reaches only their own applications.

On the loan assessment page, **Customer behaviour** reads the customer on screen from the
database: what they have told us (competitor quote and its evidence, negotiation, price
sensitivity, timing), what they have done (offer, application, uploads, pricing, overrides,
outcome), and what similar customers in the loan book did at the rate on the scenario
slider. The 20,000-loan book is held in the `loan_book` table of the store
(`AAVAS_STORE_URL`, SQLite by default), loaded from `data/processed/training_data.csv` on
first use and reloaded when that file changes; the book-level behaviour on Competitor
Intelligence reads the same table.

These are synthetic personas; production replaces the user table with SSO. Set
`AAVAS_SESSION_SECRET` to keep sessions across API restarts. `AAVAS_AUTH_MODE=disabled`
switches enforcement off for scripts (every governance surface then says so), and
`apikey` keeps the older per-key scheme.

### Nothing exists only as code

The agent, skill and tool declarations are snapshotted into the governance store
and the snapshot id — `ASET-2026.09-d076e1df` — is stamped on every decision.
**The id is a content hash over the declaration, not a number someone maintains**,
so it cannot drift from the code it names. Asked which skills an agent had when a
decision was priced, the reconstruction reads the snapshot that ran rather than
today's roster; a test prices a decision, drops an agent, republishes, and checks
the old decision still reports the old set.

### The control tower's reporting rule

`GET /control-tower` returns every metric with the table or artefact that produced
it, and returns a metric with **no** source as a null carrying the reason there is
none — rendered at the same size as a real tile and listed again at the foot of
the page. Five are in that state today: success rate (only decisions that
*succeeded* are written, so any rate computed here would be 100 % by
construction), agent failures, realised conversion, acceptance signals and
federated learning. A control tower whose dashboard cannot be told apart from its
mock-up is worse than no dashboard, so `tests/test_control_tower.py` asserts the
rule rather than the numbers.

Most of what it reports is not new analysis. `aavas/analytics/performance.py` has
computed model health, drift, operational load and champion/challenger state
since long before the tower — and nothing imported it. See
[docs/agentic-os-flow-gap.md §7](docs/agentic-os-flow-gap.md).

The search history is read straight off the MLflow **file store** (`mlruns/`), so
`#/mlflow` is populated with no MLflow server running and the API keeps no
dependency on the library. Each page probes its own server, says whether it is
up and how to start it when it is not, and embeds it when it is. `GET /tracking`,
`/tracking/runs` and `/tracking/services` are the endpoints behind both.

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000   # or: docker compose --profile tracking up
make xai                                                  # OmniXAI dashboards, one per model
```

`make xai` writes `reports/omnixai/ui/dashboards.json` naming which model is on
which port — a Dash process does not advertise what it holds, so without the
manifest all an admin can see is that *something* answers on 8766. **XAI Dashboard** embeds whichever of those is
running; **Model Explainability** keeps the captures, which are the record that
survives the process being stopped.

Reachability is a TCP probe **from the API host**. It says a server is
listening, not that your browser can reach it — the two differ the moment the
API runs in a container, and the page says so.

Details that are easy to get wrong and are handled explicitly:

- **Units.** A classifier's KernelSHAP values are log-odds (logit link); PDP and
  ALE are in probability. The reports label each column rather than letting
  log-odds be read as percentage points.
- **Which class.** OmniXAI explains the *top predicted* label, so a file with
  `P(disbursed) = 0.02` is explained as an explanation of `lost`. The report says so.
- **Direction.** "Increases/decreases" is only printed for numerics; for an
  unordered categorical the peak and trough categories are reported instead.
- **Counterfactuals can be genuinely unavailable.** `risk` never predicts default
  for any row of the reference population, so no counterfactual of the same shape
  exists across the 0.5 boundary. The report states that instead of omitting the
  section.
- **The holdout is what gets explained**, split exactly the way training split it.

### Why a separate venv

`omnixai 1.3.1` needs `dash < 3` (its dashboard calls the removed
`app.run_server`), which pins `Flask < 2.3` — older than a current FastAPI
environment carries. `scripts/setup_xai_env.sh` therefore builds `.venv-xai`
with `--system-site-packages`: numpy, pandas and scikit-learn are *inherited*
from the environment the models were trained in, so every artefact loads
unchanged, and only the conflicting packages are shadowed. Nothing outside
`.venv-xai` is touched.

One compatibility shim is applied at runtime, in `xai/omnixai_analysis.py`:
`shap >= 0.45` returns a single `(rows, features, classes)` array where omnixai
indexes `shap_values[label][row]`. Left alone, that index silently returns a
*row* — every classification attribution would be wrong rather than broken.
Older shap is not an option, because only `>= 0.46` supports numpy 2.

---

## API

```
POST /pricing/pre-assessment     indicative band, eligibility, confidence
POST /pricing/final-assessment   risk-adjusted recommendation + reason codes
POST /pricing/override           capture the human decision
POST /pricing/outcome            capture what actually happened
```

Supporting surface: `GET /health`, `/model-cards`, `/pricing/decision/{id}`,
`/pricing/audit/{loan_id}`, `/pricing/policy/explain?question=...` (cites the
policy clause behind a limit), `/pricing/override/reasons`,
`/pricing/learning/override-review`, `/pricing/learning/questions`.

Interactive docs at `/docs`.

---

## Models

Six models, all trained by `make train`, all registered with a model card.
Metrics below are from a chronological 80/20 split of 20,000 synthetic
applications — **the data is synthetic, so treat these as evidence the pipeline
works, not as expected production performance.**

| Model | Purpose | Headline |
|---|---|---|
| `stage1` | Quantile band (q10/q50/q90) | 80.4% band coverage, 74 bps mean width, median MAE 23.1 bps |
| `stage2` | Post-assessment anchor | MAE 11.7 bps, 90.3% within 25 bps |
| `conversion` | P(disbursed \| offered rate) | AUC 0.700, calibration gap 0.002, −9.6 pts per +25 bps |
| `prepayment` | BT-out within 24 months | AUC 0.643, base rate 15.3% |
| `risk` | PD → grade → expected loss | AUC 0.697 |
| `override` | Will a human move this price | AUC 0.682, magnitude MAE 9.7 bps |

Assessment is worth roughly **11 bps of pricing accuracy** (Stage-1 anchor MAE
23.1 → Stage-2 11.7), which is the commercial case for collecting documents.

Two design choices worth calling out:

- **Monotone constraints.** Conversion is constrained non-increasing in price and
  in competitor gap; prepayment non-decreasing. The optimizer can never be handed
  a demand curve that bends the wrong way on a thin slice. Violations: 0%.
- **Calibrated probabilities.** Conversion, prepayment and risk probabilities are
  isotonically calibrated, because they get multiplied by rupees downstream.
  Isotonic calibration is monotone, so the price constraint survives it.

### Searching for a better one: `make automl`

The six models above are what `make train` fits from `config/model_config.yaml`.
The agentic AutoML search asks the other question — *is this the right learner at
all?* — across the algorithm catalogue in [automl/registry.py](src/aavas/automl/registry.py):

```bash
make automl                                  # all four classification contracts, into MLflow
make automl ARGS="--contracts risk"          # one
make automl ARGS="--fast"                    # small budget, no ensembles
make automl-status                           # where tracking points
```

Flow per contract: algorithm selection → HPO → imbalance → calibration →
ensembles → fairness → locality → champion selection, ranked on a blended score
rather than argmax AUC and admitted only through the gates in
`config/automl.yaml`. Every search writes one parent MLflow run with one nested
run per candidate, and a JSON report to `reports/automl/<contract>.json`.
Read it in the admin view under **MLflow Tracking → AutoML searches**.

Four things are structural rather than optional:

- **Only the four classification contracts are searched.** `stage1` and `stage2`
  predict a rate; every ranking metric here — AUC, KS, ECE, PSI — is defined for
  a probability.
- **Each contract is fed exactly what its own trainer feeds its model**: same row
  filter, same frame builder, same feature contract. Otherwise a candidate is
  being compared against a champion fitted on a different feature space.
- **Two chronological splits.** The trainers' holdout is carved off first and
  never handed to the search; what remains is split again, earlier rows to fit,
  later rows to validate, with the boundary date pushed wholly onto the
  validation side. A champion selected on the rows it will later be measured on
  has not been measured.
- **Nothing is promoted.** The search selects and explains a champion; `models/`
  still holds what `make train` trained. Promotion changes the priced path and is
  a separate, deliberate decision.

The fairness gate is computed over **age bands** by default
(`defaults.fairness_group` in `config/automl.yaml`), because `applicant_age` is a
numeric pricing feature in four of the six contracts — see
[docs/target-architecture-gap.md](docs/target-architecture-gap.md) §1. Calibration
by group is the binding criterion; selection-rate parity is reported but not
gated, because the two cannot both hold when base rates differ.

### Unit discipline

The risk model is fitted on a lifetime-to-date default flag, so its output is a
*lifetime* PD, while pricing subtracts an *annual* expected loss from an annual
spread and the grade cut-offs are annual. The conversion happens once, explicitly
(`risk.pd_horizon_years`). Getting this wrong silently inflates expected loss ~4x
and grades every file D or E — it did, during development, until the grade
distribution gave it away.

---

## Layout

```
config/          pricing_policy, model_config, feature_config, federated_config
data/synthetic/  lifecycle generator CLI
src/aavas/
  synthetic/     causal data generator (PD/LGD, elasticity, overrides, outcomes)
  ingestion/     schemas, validation, loading, chronological split
  features/      stage1 / stage2 / sales contracts + the transformer
  models/        stage1, stage2, conversion, prepayment, risk, override
  xai/           explainer (shap or ablation), reason codes, counterfactuals,
                 omnixai_analysis (global + instance analysis and its dashboards)
  optimization/  constraints, scenario generator, yield calculator, optimizer
  agents/        the ten agents and the orchestrator
  learning/      outcome capture, override learning, drift, retraining, champion/challenger
  knowledge/     vector store, policy indexer, sales-note retriever
  governance/    store, audit, decision logger, model registry
  federated/     phase-2 regional personalization (in-process Flower simulation)
  analytics/     portfolio and performance analytics (not yet wired into the API)
  api/           FastAPI app and the four endpoints
application/     React + Vite UI: customer portal, assessment workspace, ops screens
docs/            architecture, flows, API reference, data model, operations runbook
scripts/         train_all, run_demo, run_api_flow, run_omnixai, setup_xai_env
tests/           88 tests
notebooks/       01 EDA, 02 stage 1, 03 stage 2 + yield + override learning
reports/omnixai/ generated: one explainability brief per model
```

Two deliberate deviations from the original outline: the package sits under
`src/aavas/` rather than directly under `src/` (top-level names like `models`,
`features` and `api` collide with installed packages), and the risk model has its
own `models/risk/` directory because pricing needs expected loss in money terms.

---

## The synthetic data

No Aavas production data was available, so `src/aavas/synthetic/generator.py`
fabricates 20,000 applications with an explicit causal structure: latent PD/LGD,
**heterogeneous price elasticity**, competitor behaviour that varies by state,
overrides driven by competitor gap and target pressure, conversion sampled from
the true elasticity, balance transfer driven by the rate gap after disbursal, and
realized yield computed from the actual behaviour.

Two properties make the POC meaningful:

1. **Elasticity is known.** Each file carries a latent `price_elasticity` the
   conversion model has to recover from observables. It gets a rank correlation
   of 0.47 — real data could never be scored this way.
2. **Overrides are informative, not noise.** Some genuinely create value and some
   destroy it, so the override-learning module has to tell them apart.

Files that breach hard policy are dropped as "declined at credit", so every row is
policy-consistent — a rule the validator enforces and which caught 958 impossible
rows during development.

---

## Continuous learning

- **Outcome capture** links realized yield back to the decision that produced it
  and reports expected versus realized.
- **Drift detection** watches two different failures: input drift (PSI against the
  training distribution) and, more dangerously, **conversion drift** — the same
  files converting differently because competitors moved, which no
  input-distribution monitor would catch.
- **Champion/challenger** promotes on replayed *value*, not on AUC: the challenger
  must beat the champion's expected value by a configured margin, on a paired
  test, without dropping conversion beyond tolerance, and with human sign-off.
  `promote()` refuses if any gate fails.
- **Retraining** is triggered by evidence (drift, volume, model age), runs into a
  challenger slot, and reaches production only through that gate.

---

## Phase 2

**Federated learning** (`src/aavas/federated/`) is simulated in-process with
Flower-shaped seams (`get_parameters` / `fit` / `evaluate`). What is federated is
deliberately narrow — a linear regional rate adjustment, because gradient-boosted
trees do not average across clients meaningfully.

The module's real output is the decision table:

```
        region  n_rows  global_mae  local_mae  personalized_mae  alpha_global  vs_global
       Gujarat    3272       14.72      14.15             14.19          0.150      -0.53
Madhya Pradesh    2460       13.75      13.69             13.69          0.231      -0.06
   Maharashtra    2402       13.98      13.97             13.95          0.240      -0.03
     Rajasthan    5080       13.51      13.10             13.13          0.150      -0.38
 Uttar Pradesh    1784       13.97      14.02             14.00          0.332      +0.03
```

Evaluated on each region's **holdout** — on training rows a thin region's
overfitted local model always wins and the comparison is meaningless. Gains are
sub-basis-point and concentrated in the contested markets. Read this table before
committing to federated infrastructure: if personalization does not beat the
global model out of sample, one central model is sufficient and federation buys
nothing but operational complexity.

**Vector store** (`src/aavas/knowledge/`) holds only what belongs there: sales
negotiation notes, competitor observations, credit notes and the pricing policy
itself. Numeric features stay in the feature store where they can be validated,
versioned and monotone-constrained. The default backend is in-memory TF-IDF;
pgvector is a config swap behind the same interface.

---

## Proving continuous and federated learning

The **Learning Proof** page (risk and model validation, and the administrator)
lists each claim with the evidence on file, marks a claim "Not yet" when the
evidence does not exist rather than assuming it, and says how to check it.
`GET /learning/continuous` and `GET /learning/federated` serve the same evidence;
`tests/test_learning_proof.py` asserts it.

**Continuous learning**

| Claim | Evidence |
|---|---|
| Every recommendation is recorded with its versions | `decisions` rows, each with model and policy versions |
| Every override is captured with its reason | `overrides` rows |
| Every outcome is joined to the decision it judged | `outcomes.decision_id`; expected vs actual conversion per file |
| Drift is monitored and decides when to retrain | `monitor_runs`; "Run checks now" (or `make monitors`) records a run |
| More outcomes make the model better | learning-curve backtest, below |
| A retrained model is promoted only through the gate | `model_registry` champion/challenger rows and the gate in `model_config.yaml` — **not yet**: no challenger has been shadow-compared |

The backtest refits the conversion model on a growing window of history and
scores every fit on the same held-out months (1 Jan – 22 Apr 2026, 3,759 files),
which none of them saw:

```
history  files   AUC    log loss
 3 mo     2,995  0.671  0.598
 6 mo     6,054  0.679  0.596
 9 mo     9,104  0.689  0.593
12 mo    12,125  0.691  0.589
16 mo    16,241  0.699  0.584      AUC +2.8 pts, log loss -2.4%, better at every step
```

**Federated learning**

| Claim | Evidence |
|---|---|
| Files stay in their state; only parameters move | 30 rounds × 5 states × 8 numbers = 1,200 numbers sent, against 14,998 files never sent; a fingerprint (hash) of each state's files, computed in place |
| The shared model learns | error 14.9 bps before any round → 13.9 bps after 30 |
| It matches pooling all the data centrally | federated 13.94 bps = a model fitted on every state's files in one place, 13.94 bps |
| Each state can personalise | 4 of 5 states do better with their personalised blend (table above) |
| Runs are recorded and reproducible | `federated_rounds`; the run id is a hash of the trained parameters, so "Run and record" twice gives the same id |

Viewing the page computes a run without recording it; "Run and record"
(`POST /learning/federated`) files the round and an audit entry. The rounds are
an in-process simulation; with Flower installed the same client contract runs
over the network.

---

## Optional dependencies

| Library | Used for | Without it |
|---|---|---|
| `xgboost` | Stage-2 learner | falls back to sklearn `HistGradientBoostingRegressor` |
| `shap` | Attributions | falls back to the ablation explainer |
| `optuna` | Stage-2 tuning | config hyper-parameters stand |
| `mlflow` | Experiment tracking | tracking is a no-op |
| `psycopg` + `pgvector` | Knowledge store | in-memory TF-IDF |
| `flwr` | Federated transport | in-process simulation |
| `omnixai` + extras | Global/instance analysis and its dashboards | `make xai` is unavailable; everything else runs (see [Model-level explainability](#model-level-explainability-omnixai)) |

The shap probe catches `Exception`, not `ImportError`: a shap built against a
different numpy ABI raises `AttributeError`, and a broken optional dependency
should degrade exactly like a missing one rather than take a pricing screen down.
That is not hypothetical — it is what this repo's development machine has.

---

## Known limitations

- **The data is synthetic.** Every metric here demonstrates that the pipeline
  works, not that it will perform this way on Aavas data.
- **Observational price response.** The rate was never randomised, so the fitted
  demand curve is a correlation the monotone constraint makes safe to *use*, not a
  causal effect. A price experiment is the highest-value next step.
- **Selection bias.** Stage-1 and Stage-2 are fitted on sanctioned files, and
  prepayment and risk on disbursed ones. The models describe files that got
  through, not all applicants.
- **Censoring.** Recent cohorts have not had 24 months to transfer out, so the
  BT-out base rate is understated for the newest months.
- **Override model drift by construction.** It predicts overrides against the
  *historical* recommendation. Once this engine replaces that baseline, the
  relationship shifts and it must be refit.
- **Expected life is crude.** A 24-month BT probability is projected as a constant
  annual hazard. Transparent and monotone, but a survival model is the phase-2
  upgrade.
- **The propensity correction is not causal.** It removes the first-order
  selection effect in the override analysis. It is not a randomised test.
