# Changelog

Every change to the Aavas agentic pricing application, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is below
1.0.0, a change to pricing behaviour, the API or the data contract is a minor release.

The version lives in `src/aavas/__init__.py` and is stamped on every priced decision.
`python scripts/release.py check` confirms pyproject.toml and the web application agree
with it; `python scripts/release.py minor|patch` cuts a release from [Unreleased].
Detailed notes for each release from 0.6.0 on are in [docs/releases](docs/releases/).

Releases 0.1.0 to 0.5.0 were reconstructed from the git history on 2026-11-19; the code
still said 0.1.0 then, so decisions priced before 0.6.0 carry `app_version` 0.1.0.

## [Unreleased]


## [0.7.0] - 2026-11-19

### Added
- A reusable lending policy kit (`policy_kit/`): the common structure of a pricing
  policy that any lender can fill in and run in this engine. It holds 44 sections in six
  parts (governance; pricing; credit and eligibility; authority and overrides; controls;
  decision record), each marked as required by the engine, optional or
  jurisdiction-specific, with an owner and a plain purpose. It ships a blank template, a
  JSON Schema and an adoption guide.
- `scripts/policy_kit.py`: `build` regenerates the schema, the template and the README
  table from the live policy, so the kit keeps step as the policy changes; `validate`
  checks any company's policy against the schema and the engine's own consistency
  checks, naming every value not yet filled in.
- `jsonschema` in `requirements.txt`, for the validator.

## [0.6.4] - 2026-11-19

### Changed
- `requirements-optional.txt` notes that LightGBM needs the OpenMP runtime on Linux
  (`apt install libgomp1`). Without it LightGBM cannot load and model selection quietly
  trains other learners, so a server's models differ from the ones tested elsewhere.

### Fixed
- The demo data scripts (`scripts/demo_scenarios.py`, `scripts/seed_demo_cases.py`)
  still built files for HL-POLICY-2026.13: 15-25 year tenures, competitor offers around
  9%, incomes sized for 20-year EMIs and one bureau score under the new 675 minimum.
  Under 2026.14 every rate request was refused. They now use tenures of at most ten
  years, competitor offers in the 13-17% market and incomes that pass the stressed
  affordability test, and each case still shows the decision it was built for.
- `scripts/seed_demo_cases.py` sent files to the area manager without the customer
  interaction the readiness check requires (existing customer, conversation notes,
  negotiation rounds), so its rate requests were refused. Each seeded file now answers
  them.

## [0.6.3] - 2026-11-19

### Fixed
- `requirements.txt` lacked `python-multipart`, which FastAPI needs for the borrower
  document uploads: an environment built from the requirements alone could not start
  the API. Found by running the test suite in a clean environment on the server.

### Added
- `requirements-optional.txt`: the optional packages the deployed server runs with
  (shap, xgboost, lightgbm, catboost, anthropic). They were installed by hand on the
  server and recorded nowhere, so a rebuild from the requirements would have dropped the
  AI explanation, SHAP attributions and three learner families without an error.

## [0.6.2] - 2026-11-19

### Security
- The AutoGluon Assistant engine is refused unless
  `training.autogluon_assistant.allow_unsandboxed` is `true` (default `false`,
  `config/agentic_os.yaml` AOS-2026.12). It runs code it writes itself, unsandboxed, as
  the platform user, so it must not start on a shared or production host until it runs
  in an isolated container.

### Fixed
- An AutoGluon Assistant run repointed `AAVAS_RAW_PATH` and `AAVAS_PROCESSED_PATH` in
  the process environment while it read its dataset, so a pricing request served at
  that moment could read the run's data. The loader now takes the paths explicitly.
- Two promotion requests for the same run at once could both pass the "already
  promoted" check, and the second replaced the champions' backup with the models just
  promoted. Promotion now runs one at a time, re-reads the run from disk, and never
  overwrites a backup.

## [0.6.1] - 2026-11-19

### Changed
- `scripts/setup_xai_env.sh` builds the OmniXAI overlay on Windows (Git Bash) as well, and
  inherits the repo's own training environment (`.venv`) through a `.pth` file when it
  exists, so the explained models load in exactly the packages they were trained in.
  On Windows it installs `chroma-hnswlib` (the same library, prebuilt) in place of
  `hnswlib`, which has no Windows wheel and needs the MSVC build tools.
- `requirements-xai.txt` pins `setuptools<81`: omnixai imports `pkg_resources`, which
  setuptools 81 removed.

### Fixed
- The OmniXAI analysis failed on every model under pandas 3: omnixai 1.3.1's ALE wrote
  fractions into integer tables, and its imbalance analysis wrote into a read-only
  array. Both methods are replaced at run time with the one-line fix
  (`_patch_omnixai_for_pandas3` in `src/aavas/xai/omnixai_analysis.py`).
- Two explainability tests depended on local artefacts: one named the risk model's
  rank-agreement score from an earlier training run, the other a dashboard screenshot
  taken by hand. They now check the trust thresholds and every model's reading
  against them, and serve a capture the test places itself.

## [0.6.0] - 2026-11-19

Agentic retraining and serving on AWS, drift monitoring with OpenObserve, and the RBI
Fair Practices Code, model risk and FREE-AI controls.
[Release notes](docs/releases/0.6.0.md).

### Added
- Agentic OS: rebuild the models as a workflow of data, policy checks, training and
  serving, each run in its own workspace, with promotion to champion as a separate
  approved step (`/agentic/*`, `config/agentic_os.yaml` AOS-2026.11) and its screen
  in the Train module, with a Resume button for a failed run.
- Training on an Amazon SageMaker training job (scikit-learn image, all 8 models), and
  serving from a SageMaker Serverless endpoint that is checked to answer exactly as
  training recorded before the step passes.
- AutoGluon Assistant (MLZero) as a challenger-only training engine (inactive until
  Bedrock model access is granted).
- Drift detection: data drift (PSI, KS, chi-square), model drift on the champions'
  predictions and concept drift on outcomes with the date the change began
  (`scripts/run_drift.py`, `config/monitoring.yaml` MON-2026.11), with simulated
  `data_shift`, `price_war` and `default_surge` scenarios to prove the alerts.
- OpenObserve: drift findings shipped to three streams, three alerts and a dashboard;
  alerts post back to `/monitoring/alert-webhook`. `/monitoring/drift`,
  `/monitoring/drift/run` and `/monitoring/alerts` for the model validation role.
- Key Facts Statement for an offer (RBC Directions para 29): APR on the net amount
  disbursed, repayment schedule, benchmark-rise impact, penal, prepayment and switching
  charges, document release and grievance commitments, AI-use disclosure, three working
  days' validity and the borrower's acknowledgement (`/pricing/offers/{id}/kfs`).
- Written reasons for a rejection in plain words (HFC Directions para 135,
  `/pricing/rejection-note/{loan_id}`).
- Model risk management (RBI draft MRM guidance 2026, FREE-AI): an inventory of 13
  models and AI components with owners, validators, risk tiers, dependencies, kill
  switches and fallbacks; independent validation records; RMCB reporting and inventory
  review (`/governance/model-risk/*`, `config/model_risk.yaml` MRM-2026.11).
- `config/fair_practices.yaml` FPC-2026.11: charges, benchmark, KFS rules, grievance and
  document-release commitments, rejection wording.
- Store tables `kfs_statements` and `model_validations` (created on start).
- Release tooling: this changelog, `docs/releases/`, `scripts/release.py` and a test that
  fails when the versions drift apart.

### Changed
- RBI register RBI-ARTEFACTS-2026.11: 52 requirements, each citing its paragraph, with
  live evidence for the new controls; KFS, penal-charge and prepayment rows now cite the
  consolidated Responsible Business Conduct Directions 2025.
- Access rules ACCESS-2026.11: `/agentic` and `/monitoring` for model validation, the
  alert webhook public (token-checked), `/pricing/rejection-note` for RMs and area managers.
- A simulated drift scenario no longer replaces the live book's latest drift report.
- Data and model paths can be pointed at a run's workspace (`AAVAS_RAW_PATH`,
  `AAVAS_NOTES_PATH`, `AAVAS_PROCESSED_PATH`, `AAVAS_MODEL_DIR`); `scripts/train_all.py
  --role champion|challenger`.
- Requirements: `boto3`, `botocore[crt]`, `httpx` (the FastAPI test client).
- The application version is maintained: 0.6.0 in the Python package, pyproject.toml and
  the web application (they said 0.1.0 and 0.0.0).

### Fixed
- The SageMaker serving container could not import the handler; it now imports from the
  unpacked model artefact.
- OpenObserve dashboard creation was rejected (server-managed fields sent).
- More than 900 API tests had not been running: the test client's dependency was missing.
- Tests written before HL-LAP-POLICY-2026.14 now use its rules: tenures of at most ten
  years, incomes that pass the stressed affordability test, rates inside the band, and
  figures worked out from the policy rather than copied from the old one. The approval
  gate tests pick whichever champion fails its fairness test instead of naming one.

### Security
- The AutoGluon Assistant's generated code gets an allow-listed environment (no platform,
  OpenObserve or webhook secrets) and runs in its own folder. It is still unsandboxed.
- A relationship manager gets rejection notes only for files they logged; the note
  carries no internal policy wording.
- The alert webhook refuses calls without the shared token; OpenObserve may call only
  localhost.
- The SageMaker execution role and bucket carry the AWS account number and the
  repository is public, so they are not committed: they come from
  `AAVAS_SAGEMAKER_ROLE_ARN` and `AAVAS_SAGEMAKER_BUCKET` (the environment or the
  git-ignored `.env`), and a test fails if a config file carries an account number.

## [0.5.0] - 2026-11-18

### Changed
- Pricing policy HL-LAP-POLICY-2026.14 replaces HL-POLICY-2026.13: rates 13-17% from a
  grade matrix (A+ to D) plus a tenure adjustment, tenure 36-120 months; grade D, a
  bureau score below 675 or a grid rate above 17% gets no offer; no premium above the
  matrix rate; affordability tested at the quoted rate + 2% against the grade's FOIR cap;
  new-to-credit files referred; open items closed as decisions D-01 to D-09.
- A declined file is logged and refused (422) instead of priced; risk grades relabelled
  A+ to D on the same PD bands.
- One byte-identical policy file for every engine (LangGraph rule engine included).

### Added
- Policy consistency checks (`scripts/check_policy.py`, 83 checks) and 48 sample policy
  cases (`tests/data/policy_cases.csv`).

### Fixed
- A file priced at the 13% floor passed when it was unaffordable even there.
- Policy versions ordered by number, not by prefix.

## [0.4.1] - 2026-11-18

### Fixed
- Approval briefing under pandas 3 copy-on-write.

### Added
- `scripts/seed_demo_cases.py`: five demo files for the RM and area manager walkthrough.

## [0.4.0] - 2026-11-18

### Added
- Area manager review of lower-rate requests: suggested decision, checklist, rate
  scenarios against the branch's history, branch history tabs.
- File readiness and step-by-step status for the RM and area manager.
- RBI regulatory register (Fair Practices Code, model risk draft, FREE-AI) with the
  platform's evidence, and readable pricing-policy versions.
- Claude token usage per call; demo reset scripts.

## [0.3.0] - 2026-10-26

### Added
- RM worklist with stages, targets, suggested offers and nudges; customers accept or
  decline in their portal.
- Area manager approvals with headroom, deep dive and disbursements against target.
- Business model cards: risk to price, segments, acceptance curves, Stage 1 weights.

## [0.2.0] - 2026-10-25

### Added
- Sign-in with role-based access (borrower, RM, area manager, model validation,
  administrator) from `config/access.yaml`.
- Online applications with document upload; the RM's application list.
- Customer behaviour and competitor intelligence views; versioned, validated
  configuration editing with rollback.

## [0.1.0] - 2026-10-19

### Added
- Two-stage home-loan pricing engine (FastAPI) with the lending workspace and the
  Agentic Credit Pricing OS frontends (React + Vite).
- Per-loan SHAP explanations of the rate, risk and conversion models and an AI-written
  explanation whose figures are checked against the page.
