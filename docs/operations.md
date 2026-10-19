# Operations runbook

Running, testing, building and troubleshooting the system. Architecture is in
[architecture.md](architecture.md); endpoint semantics in
[api-reference.md](api-reference.md).

---

## 1. Prerequisites

* Python ≥ 3.10 (numpy, pandas, scikit-learn, FastAPI, pydantic, PyYAML, joblib, uvicorn)
* Node ≥ 18 and npm — only for the `application/` frontend
* Optional: Docker (+ compose), Playwright Chromium for the UI test

Everything optional has a working fallback (see [§7](#7-optional-dependencies)), so a bare
install runs the whole POC.

---

## 2. Cold start

```bash
pip install -e ".[dev]"     # or: pip install -r requirements.txt
make data                   # synthetic lifecycle dataset  -> data/raw/
make train                  # six models + cards + registry -> models/   (~1 min)
make test                   # 88 tests
make demo                   # one customer end to end, then the learning loop
make api                     # http://localhost:8000/docs
```

`make all` = `data → train → test → demo`.

The Makefile exports `PYTHONPATH=$(CURDIR)/src`, so every target works straight after a clone
without installing the package.

### Make targets

| Target | Does |
|---|---|
| `install` | `pip install -e ".[dev]"` |
| `data` | `data/synthetic/generate_dataset.py` |
| `train` | `scripts/train_all.py` — trains, evaluates, writes cards, registers champions |
| `demo` | `scripts/run_demo.py --fresh-store` — full walkthrough, resets the governance store first |
| `test` | `pytest -q` |
| `AAVAS_AUTH_MODE` | `disabled` (default) or `apikey`. With `apikey`, set `AAVAS_API_KEYS` as `key:name:role|role,...` and the UI prompts for a key on the governance and feature pages |
| `verify-algorithms` | `scripts/verify_algorithms.py` — fits every registered learner and runs every strategy against real data, reporting works / broken / unavailable |

### Algorithm availability

`make verify-algorithms` fits every registered learner and runs every strategy
against real data. Current result: **68 working, 0 broken, 12 unavailable of 80.**

The 16 are unavailable for three genuinely different reasons, and the registry now
says which:

| Kind | Count | Fixable by |
|---|---|---|
| Missing package | 1 — `tabpfn` | an install (a foundation-model download; research only, and capped at 10k rows) |
| No adapter | 4 — `tabnet`, `ft_transformer`, `saint`, `tabicl` | writing one, plus `pytorch_tabnet` / `rtdl` |
| No such data shape | 7 — 4 sequential, 3 graph | **nothing.** See below |

The last group is not an oversight. A loan carries one application date and scalar
outcomes, not a repayment path, so there is no sequence to model; manufacturing
one from scalars would be inventing the data. And the only edges available for a
graph are branch and sales officer, which are identity columns deliberately held
out of the risk features — a graph over them would price a customer on who served
them. `tests/test_automl.py` asserts the reasons say this rather than reading like
a missing install.

`pip install -e ".[learners]"` covers the boosting and interpretable families.
All of `xgboost`, `catboost`, `lightgbm`, `imbalanced-learn` and `interpret-core`
are installed, which took the catalogue from 19 of 41 learners to **29 of 41**.

**One thing the install changed in behaviour, not just availability:** without
`imbalanced-learn`, `smote` and `adasyn` fell back to random oversampling. That
was reported honestly — the strategy renamed itself `smote->random_oversample` —
but the leaderboard was comparing a fallback. They now run as real SMOTE and
ADASYN.

| `monitors` | `scripts/run_monitors.py` — drift, fairness and retraining passes that record having run |
| `api` | `uvicorn aavas.api.main:app --reload --port 8000` |
| `notebooks` | Jupyter Lab on `notebooks/` (01 EDA, 02 stage 1, 03 stage 2 + yield + override learning) |
| `xai-install` | Builds `.venv-xai` via `scripts/setup_xai_env.sh` |
| `xai` | OmniXAI analysis + dashboards |
| `xai-reports` | Same analysis, `--no-ui` |
| `docker-up` / `docker-phase2` | Compose, default / phase-2 profile |
| `clean` | Removes artefacts, generated CSVs, `*.db` and `__pycache__` |

Useful script flags:

```bash
python scripts/train_all.py --regenerate --rows 50000
python scripts/train_all.py --only stage2 conversion
python scripts/run_demo.py --loan-id AAV0000123 --skip-federated
python scripts/run_api_flow.py             # exercises a running API end to end
```

---

## 3. Frontend

```bash
cd application
npm install
npm run dev                          # :5173 by default
npm run dev -- --port 4173 --strictPort   # or pin a port
```

Two entry points are served from one build, against one API.

**Lending workspace** — `http://localhost:5173/`

| Screen | URL |
|---|---|
| Dashboard | `#/dashboard` |
| AI Control Tower | `#/control-tower` |
| Assessment workspace | `#/application` |
| Customer journey | `#/offer` |
| Model registry | `#/models` |
| Model explainability | `#/xai` |
| Credit score XAI | `#/credit-score` |
| XAI dashboard | `#/xai-dash` |
| MLflow tracking | `#/mlflow` |
| Agent control plane | `#/agents` |
| Configuration | `#/config` |
| Connectors | `#/connectors` |
| Runtime & drift | `#/runtime` |
| Approvals & gates | `#/governance` |
| Override learning | `#/learning` |
| Audit history | `#/audit` |
| Policy questions | `#/policy` |

**Agentic Credit Pricing OS** — `http://localhost:5173/os.html`

| Module | URL |
|---|---|
| Control tower | `#/overview` (estate dashboard), `#/overview/lifecycle` (lifecycle board) |
| 1 Configure | `#/configure`, `#/configure/connectors` |
| 2 Train | `#/train`, `#/train/models` |
| 3 Validate & Approve | `#/validate`, `#/validate/credit-score`, `#/validate/xai-dash` |
| 4 Run | `#/run` |
| 5 Observe & Improve | `#/observe` (runtime & drift), `#/observe/learning`, `#/observe/audit` |

The console is the control tower over one workload — this pricing engine. Below the tower it is
organised by the lifecycle rather than the lending workflow, and it reuses the same page
components as the lending admin view — one implementation, two places to reach it from.

### The control tower's reporting rule

`GET /control-tower` returns every metric with the table or artefact that produced it, and returns
a metric with **no** source as a null carrying the reason there is none. Five tiles are currently
in that state — success rate, agent failures, realised conversion, acceptance signals and
federated learning — and they are listed on the page rather than omitted. `tests/test_control_tower.py`
asserts the rule: a sourced metric must name its source, an unsourced one must give a reason, and
the two sets must agree with the page's own summary. A control tower whose dashboard cannot be
told apart from its mock-up is worse than none.

The Vite dev server proxies same-origin `/api/*` to `http://127.0.0.1:8000` — set
`AAVAS_API_URL` to target another backend. **The static build does not provide that proxy**:
hosting `dist/client` anywhere else needs an equivalent same-origin reverse proxy to the API.

```bash
npm run build        # vite build + scripts/prepare-sites-build.mjs
npm run test:sites   # worker/build contract test
npm run test:ui      # Playwright flow test (needs the dev server, :8000, chromium)
```

`npm run test:ui` creates real decision, override and outcome records while checking the
complete flow, invalid bureau-score rejection, mobile navigation, overflow and browser errors.
Evidence lands in `application/qa/` and `application/design-qa.md`.

---

## 3b. AutoML searches

```bash
make automl                                  # all four classification contracts, tracked
make automl ARGS="--contracts risk override" # a subset
make automl ARGS="--fast"                    # small HPO budget, no ensembles
make automl ARGS="--fairness-group segment"  # gate on a different axis
make automl-status                           # tracking uri / experiment / mlflow version
```

Tracking is off by default across the repo; `run_automl.py` is the one place that turns it on,
because a search nobody can audit later is a search that did not happen. It sets
`MLFLOW_TRACKING_URI` to the **absolute** file store at `mlruns/`, so a run lands in the store
the admin view and `mlflow ui` read whatever the working directory. Reports go to
`reports/automl/<contract>.json`; MLflow gets one parent run per search and one nested run per
candidate. Nothing is promoted — `models/` still comes from `make train`.

---

## 4. Explainability runs

```bash
make xai-install                                   # one-time, builds .venv-xai
make xai                                           # all six models + dashboards
make xai-reports                                   # reports only, nothing to Ctrl-C
make xai ARGS="--models stage2 conversion --instances 8"
make xai ARGS="--fast"                             # smaller samples, no counterfactuals
.venv-xai/bin/python scripts/run_omnixai.py --retrain
```

Budget ~10 seconds for `stage1` up to ~5 minutes for a calibrated classifier with
counterfactuals — KernelSHAP is model-agnostic, so its cost is model evaluations, and a
`CalibratedClassifierCV` costs three per call. `make xai` serves the first three models by
default (~1.5 GB per dashboard process); `--max-dashboards` raises that. Every model still
gets a report either way.

`.venv-xai` is built with `--system-site-packages` so numpy/pandas/scikit-learn are inherited
and artefacts load unchanged; only the packages omnixai pins differently (`dash`, `Flask`,
`shap`) are shadowed inside it, and omnixai itself is installed `--no-deps` because its own
pins (`numpy < 1.24`, `scikit-learn < 1.2`) predate this repo's stack. Nothing outside
`.venv-xai` is touched — see the header comments in `requirements-xai.txt` and
`scripts/setup_xai_env.sh`.

### Dashboards in the admin view

`#/mlflow` ("MLflow Tracking") lists every AutoML search with its candidates and embeds the MLflow
UI; `#/xai-dash` ("XAI Dashboard") embeds the OmniXAI dashboard for each served model. Both probe
their server first and say how to start it when it is down. `#/credit-score` ("Credit Score XAI")
needs no server at all: it reads the credit-score band out of every report and puts the six models
side by side.

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000   # or: docker compose --profile tracking up
make xai                                                  # OmniXAI, one dashboard per model
```

The run history is read from the `mlruns/` file store directly, so the page is populated whether
or not an MLflow server is up. `make xai` writes `reports/omnixai/ui/dashboards.json` naming the
model on each port, and removes it on a clean exit; the API probes every entry, so a manifest left
by a crashed run reads as "not running". Status is a TCP connect **from the API host** — behind a
container boundary a server can be up and still unreachable from the browser.

---

## 5. Containers

```bash
docker compose up --build                      # API on :8000, healthcheck on /health
docker compose --profile phase2 up --build     # + postgres/pgvector on :5432
docker compose --profile tracking up           # + MLflow on :5000
```

`models/` is a named volume, so artefacts and the governance store survive restarts.
Set `AAVAS_STORE_URL` to the Postgres DSN to share state across replicas.

---

## 6. Environment variables

See [data-model.md §5](data-model.md#environment-variables). Copy `.env.example` to `.env` to
change any of them; the POC runs with no `.env` at all.

---

## 7. Optional dependencies

| Library | Used for | Without it |
|---|---|---|
| `xgboost` | Stage-2 learner | sklearn `HistGradientBoostingRegressor` |
| `shap` | Attributions | Ablation explainer (method reported in the payload) |
| `optuna` | Stage-2 tuning | Config hyper-parameters stand |
| `mlflow` | Experiment tracking | No-op |
| `psycopg` + `pgvector` | Knowledge store | In-memory TF-IDF |
| `flwr` | Federated transport | In-process simulation |
| `omnixai` + extras | Model-level analysis | `make xai` unavailable; everything else runs |

`requirements-xai.txt` pins the overlay stack. The shap probe catches `Exception`, not
`ImportError`: a shap built against a different numpy ABI raises `AttributeError`, and a
broken optional dependency should degrade exactly like a missing one rather than take a
pricing screen down.

---

## 8. Testing

```bash
make test                       # 88 tests
pytest tests/test_optimizer.py -q
pytest -k override -q
```

| File | Covers |
|---|---|
| `test_stage1.py` | Band discipline, clamping, confidence, eligibility |
| `test_stage2.py` | Anchor inside the envelope, transformer contract, unseen categories |
| `test_policy.py` | Card rate, permissible band, affordability, approval levels, evidence rules |
| `test_optimizer.py` | Scenario generation, yield maths, guard-rails, tie-break, extrapolation trim |
| `test_learning.py` | Override capture, propensity adjustment, drift, champion/challenger |
| `test_api.py` | All endpoints, validation, error codes, audit retrieval |
| `test_omnixai.py` | XAI analysis plumbing and report contracts |
| `test_connectors.py` | The three connector gates, the layer split, and that a blocked channel yields nothing |
| `test_governance_controls.py` | Identity, per-role signatures, separation of duties, gates that block, and monitors that record having run |
| `test_provenance.py` | Dataset identity, fairness on the training path, the prompt/knowledge/federated registries, the PSI defect and tool permissions |
| `test_system_of_record.py` | The twelve registries are audited honestly, and a real priced decision is walked link by link rather than assumed |
| `test_algorithm_catalogue.py` | Every unavailable algorithm gives a reason; every HPO strategy approaches a known optimum; a calibrator actually transforms; federated strategies are not one implementation under five names |
| `test_agent_boundaries.py` | The four reusable packages import only what they declare, expose what they promise, and never reach into each other; the kernel imports nothing but logging |
| `test_capabilities.py` | Agents, tools and skills have not drifted from the code, in both directions; a skill that reaches a store, connector or artefact is rejected |
| `test_control_tower.py` | Every metric names a source or gives a reason; the unsourced set is listed, not hidden |

`tests/conftest.py` isolates the governance store (`AAVAS_STORE_URL` is redirected to a temp
file) so a test run never writes the developer's database, and tests that need artefacts or a
dataset **skip cleanly** rather than fail — `pytest` is useful immediately after a clone.
`scripts/run_api_flow.py [base_url]` replays the Swagger examples against a running server.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/health` returns `degraded`, `models_missing` non-empty | Artefacts not trained | `make train` |
| `503` on `/pricing/pre-assessment` | `stage1` artefact missing | `make train` |
| `503` on `/pricing/learning/*` | `data/processed/training_data.csv` absent | `make data && make train` |
| `422` with `bureau_score` error | Value outside 300–900 and not −1 (Swagger's placeholder `0` triggers this) | Send a real score or `-1` |
| `404` on `/pricing/override` | No post-assessment decision for that loan id | Run the final assessment first, same `loan_id` |
| Override returns `within_authority: false` | Approver level below `required_approval` | Route to the named authority; do not release the price |
| `POLICY BLOCK … requires evidence` | Unevidenced competitor claim above 15 bps | Verify the competitor quote or reduce the discount |
| Recommendation differs from the anchor | By design — the anchor is a comparison point, the decision comes from the scenario search | Read `comparisons` and `binding_constraint` |
| Frontend shows "pricing service unavailable" | API not on :8000, or the page is not served through the proxy | Start `make api`; use the Vite dev server or an equivalent `/api` proxy |
| `make xai` fails to import dash/omnixai | `.venv-xai` not built | `make xai-install` |
| Classification attributions look wrong under `shap ≥ 0.45` | omnixai indexes `shap_values[label][row]` | Already shimmed in `xai/omnixai_analysis.py`; do not downgrade shap below 0.46 (numpy 2) |
| Grades collapse to D/E, expected loss ~4× too high | Lifetime vs annual PD confusion | `risk.pd_horizon_years` in `model_config.yaml` |
| Governance store locked / stale demo data | Concurrent writers or leftover state | `python scripts/run_demo.py --fresh-store`, or delete `aavas_governance.db` |

---

## 10. Changing behaviour without changing code

| To change | Edit | Takes effect |
|---|---|---|
| Card rate, spreads, loadings, caps | `config/pricing_policy.yaml` | Next request (config is cached per process; restart or `config.reload()`) |
| Who may discount how much | `approval_authority` | Next request |
| Allowed override reasons / evidence rules | `override` block | Next request |
| Optimizer guard-rails, step size, search space | `optimizer` block | Next request |
| Stage-1 band width, rounding, eligibility cap | `stage1` block | Next request |
| Fair-pricing review threshold, quote validity | `fair_pricing` block | Next request |
| Reason-code wording | `feature_config.yaml → reason_code_labels` | Next request |
| Model features, monotone constraints | `feature_config.yaml` | Next `make train` |
| Learners, hyper-parameters, calibration, drift thresholds, promotion gates | `model_config.yaml` | Next `make train` |

Bump `policy_version` / `feature_config_version` / `model_config_version` when you change one —
they are stamped on every decision, and an audit that cannot tell which rules applied is not
an audit.
