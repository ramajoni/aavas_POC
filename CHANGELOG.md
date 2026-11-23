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


## [0.11.0] - 2026-11-23

### Added
- **Release notes in the application.** Staff (relationship manager, area manager, model
  validation, administrator; never the borrower) open them from the footer, which now shows
  "App v<version>" beside the policy and model versions, with a small "New" dot after an
  upgrade until they have looked. The page lists every release newest first with its date
  and a one-line summary, opens the running release at its highlights, shows any release's
  full notes (tables and nested lists included) and filters by text.
- `GET /releases` and `GET /releases/{version}`, read from `CHANGELOG.md` and
  `docs/releases/`. The version is checked strictly (ASCII digits only) and used only as a
  lookup key; notes are rendered without raw HTML.

### Changed
- `config/access.yaml` ACCESS-2026.12: the Release notes page and route for the staff roles.

## [0.10.1] - 2026-11-23

### Fixed
- An RM could ask the area manager for a rate outside the policy band (13.00-17.00%): on AWS a
  request at 12.73% reached the approval list, although no rate below 13% can ever be
  approved. The request is now refused with the band in the message
  (`reference.card_rate_floor` to `card_rate_ceiling`). Requests already raised are untouched.

## [0.10.0] - 2026-11-23

### Added
- **Best offer** on the area manager's Pricing intelligence tab
  (`GET /pricing/rate-requests/{id}/best-offer`). Pricing intelligence looked only at the
  interest rate; best offer weighs all four levers of an offer: the rate, the loan amount
  sanctioned, the tenure and the processing fee.
  - It tries a few hundred offers per file. Each gets its own grade, card rate and floor
    (a longer tenure or a different amount can move both), the policy checks (stressed FOIR,
    LTV, age at maturity, loan cap, tenure, the 13-17% band) and an EMI comfort limit (never
    more than 10% above what the customer asked for).
  - It picks three: **most value** (highest risk-adjusted value, at least as likely to accept
    as the recommendation), **most likely to close** (highest acceptance, earning at least what
    the recommendation earns) and **the cheapest way to match the ask** (reaches the acceptance
    the RM's cut would buy, at the highest value). Ties go to the offer that changes fewer
    levers.
  - "Compare the offers": the RM's ask, the recommendation and the three best offers side by
    side, likely to accept and risk-adjusted value, each row naming what changed. A table
    gives every lever, EMI, APR, grade and default risk; "what each lever buys on its own"
    shows one change at a time; every eligible offer can be opened as a chart.
  - Offers that need a smaller loan or a different tenure are marked: the acceptance model
    has not seen customers offered less, or a different tenure, than they asked for. The fee
    is in no model; its effect on acceptance is scored through the all-in cost (APR).
  - Decision support only: the area manager's decision still changes only the rate.
- **Risk-aware pricing** on the same tab (`GET /pricing/rate-requests/{id}/risk-pricing`):
  - the file's risk score (grade, annual and lifetime PD, LGD, expected loss a year and in
    rupees, break-even rate) and where the PD sits inside its grade band, flagged near a
    boundary, with what one grade up or down does to the card rate;
  - the five factors that move this borrower's default risk most against a typical borrower;
  - every candidate rate with EMI, FOIR, stressed FOIR against the grade cap, expected loss,
    margin over risk cost, acceptance and risk-adjusted value;
  - peers by grade from the loan book (typical rates, realised defaults, where the ask and the
    recommendation fall among same-grade loans) and a plain verdict.
- A four-lever offer engine (`src/aavas/analytics/offer_evaluator.py`) shared by both, and
  `config/best_offer.yaml` (OFFER-2026.11) for the ranges it searches and how "best" is
  chosen; a bad edit to it through the configuration screen is rejected.

### Changed
- "How the prediction is made" and the table of every rate now include the RM's ask when it
  is below the floor (marked outside policy); before, both showed a dash for most requests.
  The decision suggestion now states the ask's acceptance and value, and still never picks
  an option outside policy.
- `config/model_risk.yaml` MRM-2026.13: the risk, conversion and prepayment models' intended
  use covers this analysis.
- The deep dive's one-line grade sentence is replaced by the risk score tiles.

### Security
- Loan ids are checked for their form (`^[A-Za-z0-9-]{3,40}$`) on every pricing, offer and
  rate-request input and query, not only on new applications, so an id can never carry a
  path.

### Notes
- Every figure on the tab uses the pipeline's own FOIR for the models, so the new sections
  agree with the decision and with each other. That FOIR is worked out at a fixed rate
  before pricing, so default risk does not move with the rate; 0.11.0 replaces it with a
  policy assessment rate.

## [0.9.6] - 2026-11-21

### Fixed
- `trim` could leave a special situation out of the 10 (on AWS it missed the new-to-credit
  case) because the greedy pick spent a slot on a grade-and-band pair. A swap pass now
  exchanges a kept request for one left out whenever that covers more single values (every
  grade, band, ask position, evidence type, reason, segment and special situation), and only
  then more weighted pairs.

### Security
- `trim` removed each file's upload folder at `upload_dir()/<loan_id>` with the id read from
  the store. An id such as `../x` stored through an endpoint that only checks its length could
  have pointed outside the upload root. The folder is now removed only when the id is a plain
  loan id (`^[A-Za-z0-9-]{3,40}$`) that resolves directly under the upload root; anything
  else is skipped and reported.

## [0.9.5] - 2026-11-21

### Added
- `scripts/demo_catalogue.py trim --keep N`: cut the area manager's approval list to N
  pending requests that between them still show every kind of situation. A deterministic
  coverage search picks them from the requests a store already holds: every grade, tenure
  band, ask position, evidence type, request reason and segment at least once, then the
  special situations (new to credit, balance-transfer threat, month-end target, large
  ticket, LTV, FOIR and age near their limits, late EMIs, a score near 675); the
  hand-written scenarios win ties. A dry run is the default. `--apply` backs up the store,
  then removes every other file that reached the area manager (its rows in every loan table,
  its uploads and its decision-memory entries). The API's request log, the loan book and the
  audit trail are never trimmed, and files that never reached the area manager are left alone.

## [0.9.4] - 2026-11-21

### Added
- `scripts/demo_catalogue.py`: every area-manager case the policy and the engine allow,
  found by probing the engine rather than tuned by hand. `probe` prices several hundred
  applicant profiles on a throwaway copy of the store (it refuses the live port) and picks one
  per cell of grade (A+ to C) x tenure band x ask position (within room, at the floor, below it)
  x competitor evidence (five kinds); `create` posts them to a server and checks each lands in
  its cell, labelling it by what it observes if not; `report` writes the coverage page
  `docs/demo/area_manager_cases.md`. The catalogue holds 92 cases for the area manager to
  decide (82 cells and 10 special situations: new to credit, a score near 675, LTV, ticket,
  FOIR and age near their limits, late EMIs, a balance-transfer threat, a month-end target
  with no evidence, a customer not bargaining on price) and, with `--decide-history`, 9 cases
  the area manager has already decided: approve as asked, approve at the floor, counter,
  decline and uplift. Every rate is checked against the 13-17% band before it is sent.
- Every cell without a case says why: 20 are impossible under the policy (for example A+ up
  to 60 months, whose card rate is already the 13% floor); 78 were not found by the probe,
  most because the engine always recommends the floor for grade C and for B over 84 months.
- `scripts/scratch_api.py`: start and stop a throwaway API over a copy of the store, for
  probing on any machine, including the AWS server (whose models differ from a laptop's).

## [0.9.3] - 2026-11-21

### Removed
- The "Further upsell room" tile on the area manager's request summary (Approvals). It
  showed "No room" with "Fewer than 70% of customers like this accept even at the system
  recommendation" on most files under HL-LAP-POLICY-2026.14, which told the manager nothing
  they could act on. The summary now has three tiles: ask against recommendation, the upsell
  by holding the recommendation, and the extra interest over the loan. The price-uplift line
  in the decision panel still appears where customers like this accept a higher rate.

## [0.9.2] - 2026-11-21

### Fixed
- Two extended demo cases asked for rates below 13%, which the policy forbids
  (`card_rate_floor`: no concession below 13%): E1 asked 12.75% and E2 12.70%. Both were A+
  files recommended at exactly 13.00%, where no valid ask can exist (the RM may only ask more
  than 25 bps under the recommendation). E1 is now a grade-B four-year file (ask 14.15%) and E2
  a grade-A seven-year file (ask 13.62%); they keep their tenure bands and the decision each
  shows. The 0.9.1 notes said every ask was inside the band; that was wrong for these two.

### Added
- `scripts/demo_scenarios.py` refuses any ask outside the policy band (`card_rate_floor` to
  `card_rate_ceiling`, 13.00-17.00%) before it is sent, naming the scenario.

## [0.9.1] - 2026-11-21

### Added
- `scripts/demo_scenarios.py --set core|extended|all`. The extended set adds twelve
  area-manager cases that cover the decisions HL-LAP-POLICY-2026.14 allows: grades A+, A, B
  and C; tenures in all three bands (36-60, 61-84, 85-120 months); all four segments;
  verified, unverified and no competitor evidence; a bureau score near the 675 minimum; a new
  to credit applicant sent to manual review; a month-end-target request with no customer
  evidence; a price-insensitive existing customer; tickets near the Rs 75 lakh and 90% LTV
  caps; and asks within the room, exactly at the floor and below it. `core` (the original ten)
  stays the default.
- An `at_floor` ask kind (asks exactly the policy floor). Each created request now prints its
  card, floor, recommendation, ask and where the ask sits (within room, at floor, below floor).

### Fixed
- `scripts/demo_scenarios.py` always sent `new_to_credit: false`. A scenario without a bureau
  score (a score below zero) is now sent as new to credit.

## [0.9.0] - 2026-11-21

### Added
- The model tools work behind a proxy. MLflow and each OmniXAI dashboard now have a probe
  address, which the API checks from the server, and a public address, which the browser opens
  (`AAVAS_MLFLOW_PUBLIC_URL`; `run_omnixai.py --public-base`, which serves each Dash dashboard
  under `<base>/<port>/`). The MLflow and XAI Dashboard pages link and embed the public address.
  On AWS they are served at `/api/tools/mlflow/` and `/api/tools/omnixai/<port>/`, behind a
  password.
- `AAVAS_MLZERO_CLI` and a single resolver for the AutoGluon Assistant command: the configured
  Windows path (`Scripts/mlzero.exe`) also finds the Linux one (`bin/mlzero`). Readiness reports
  whether it is installed and, separately, whether it is allowed on this host; the Agentic OS
  screen says "installed, but not allowed on this host" instead of "not found".

## [0.8.2] - 2026-11-20

### Fixed
- After a restart, Milvus hands back existing collections "released", and refuses to read
  or search them until they are loaded. The Milvus store never loaded them, so precedent
  search failed on every start after the first ("Collection 'sales_notes__meta' is in state
  'released'"). It now loads each collection once per process before reading it. Found while
  building the index on the AWS server; the fake Milvus client in the tests now releases
  collections on a simulated restart, and a real Milvus Lite test releases them too.

## [0.8.1] - 2026-11-20

### Fixed
- `scripts/setup_xai_env.sh` compiled `hnswlib` from source on Linux, which needs a compiler
  toolchain the production server does not have (and should not need). It now installs
  `hnswlib` only as a prebuilt wheel, falling back to `chroma-hnswlib` (the same library,
  prebuilt) on every platform, as it already did on Windows.

## [0.8.0] - 2026-11-20

### Added
- **Decision memory for the area manager.** Every decision on a lower-rate request is
  remembered as an episode: the fifteen factors with their signals as the manager saw them,
  the suggestion shown, the decision, the rate and the note, and which past cases were
  recalled at the time. The learning agent writes it (still the only agent that writes the
  governance store); the episode is never edited, and what the customer did is joined when
  memory is read.
- **Manager memory agent** (on request, outside the priced path). When an area manager opens a
  request it recalls their own most similar past decisions and other managers', each with
  why it is alike and what the customer did; how this manager tends to decide (cut, hold or
  price higher, how often they take the suggestion, which factors move them, as counts); and
  whether today's suggestion differs from their usual decision. It never moves the suggestion.
- `GET /pricing/rate-requests/{request_id}/memory`; `POST .../decide` returns `memory`.
- A "Your decision memory" panel on the decision-support screen. The screen's factor signals
  now come from the server (`analytics/decision_factors.py`), so the screen and memory read one
  set of rules.
- The `decision_memory` table. The memory registry in the system of record is now a registry,
  not absent.
- **Milvus for the area manager's precedents.** A Milvus vector store (Milvus Lite, embedded,
  one file on the server) with local BAAI bge-small-en-v1.5 embeddings via fastembed. When a
  manager reviews a lower-rate request, "Most similar past negotiations" lists the eight past
  notes closest in meaning to this file's conversation, competitor and reason, same city
  first, with what happened in each (took the loan or not, at what rate, priced up or down).
  Context for a person only: it never changes a price.
- `GET /pricing/rate-requests/{request_id}/precedents`; `/history` returns `similar_negotiations`.
- `AAVAS_VECTOR_BACKEND=milvus` (with `AAVAS_MILVUS_URI`, `AAVAS_EMBED_CACHE`) turns it on; the
  index is rebuilt only when the notes change. Milvus Lite has no Windows build, so there, or if
  Milvus fails, the app falls back to the in-memory TF-IDF search and says which one answered.
- The embedding model is in the model-risk inventory (`embedding_bge_small`, MRM-2026.12) as a
  third-party model with a kill switch and the TF-IDF fallback.
- `pymilvus`, `milvus-lite` (Linux and macOS only; pymilvus 3.x no longer installs it) and
  `fastembed` in `requirements-optional.txt`.

## [0.7.1] - 2026-11-19

### Fixed
- On Windows, with `core.autocrlf` on, git checked out the generated policy kit files
  with CRLF line endings, and two of the kit's drift tests failed on a fresh clone.
  `.gitattributes` now keeps `policy_kit/*.json`, `*.yaml` and `*.md` as LF in every
  checkout.

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
