"""The reusable agent packages, declared.

Four of the agents in this system are whole subsystems rather than pipeline
steps, and each is independently useful: feature engineering, AutoML, acceptance
scoring and price optimisation. This module is their manifest - what each one is,
what it exposes, what it depends on, which config it owns, and what it would take
to lift it into another codebase.

**Why a manifest rather than four new directories.** Each package already has a
folder; what was missing was a boundary. Measuring the real dependency graph found
the four were *nearly* independent already, and that every cross-package edge was
a single pure function living wherever it was first written - column coercion in
the feature package, PSI in the learning package, fairness in the model package.
Those are now kernel primitives, and the peer coupling is gone. Moving the
directories would have renamed the problem; extracting the kernel removed it.

``tests/test_agent_boundaries.py`` enforces every claim below: the public surface
exists, the declared dependencies are the actual ones, and no package reaches into
another package's internals.
"""

from __future__ import annotations

from typing import Any

# What every package is allowed to depend on. Anything here is infrastructure:
# config, logging, schemas and the pure primitives in the kernel.
SHARED_KERNEL = {"kernel", "config", "logging_utils", "paths", "ingestion"}

PACKAGES: list[dict[str, Any]] = [
    {
        "name": "feature_engineering",
        "title": "Agentic Feature Engineering",
        "package": "aavas.features",
        "version": "1.0",
        "purpose": "Generates candidate features from a stage's own contract, tests each for "
                   "signal, redundancy, stability and coverage, and refuses to admit any of "
                   "them.",
        "public": [
            "aavas.features.discovery:discover",
            "aavas.features.discovery:FeatureDiscoveryAgent",
            "aavas.features.discovery:allowed_columns",
            "aavas.features.discovery:research_needed",
            "aavas.features.profiling:profile",
            "aavas.features.profiling:examine",
            "aavas.features.cards:build_card",
            "aavas.features.cards:decide",
            "aavas.features.cards:lineage",
        ],
        "depends_on": ["kernel", "config", "ingestion", "logging_utils"],
        "owns_config": ["config/feature_config.yaml"],
        "entry_points": ["GET /features", "POST /features/discover", "GET /features/cards"],
        "emits": "A discovery report: candidates with tests and refusals. Never a feature.",
        "reuse": "Takes a DataFrame and a stage name. The stage contract comes from "
                 "feature_config.yaml; point that at another contract and it works on another "
                 "dataset unchanged. Nothing in it touches the pricing domain.",
        "standalone": True,
    },
    {
        "name": "automl",
        "title": "Agentic AutoML",
        "package": "aavas.automl",
        "version": "1.0",
        "purpose": "Algorithm selection, hyper-parameter search, imbalance handling, "
                   "calibration, ensembling and champion selection under hard gates.",
        "public": [
            "aavas.automl:AutoMLSearch",
            "aavas.automl:SearchConfig",
            "aavas.automl:SearchResult",
            "aavas.automl:candidates",
            "aavas.automl:inventory",
            "aavas.automl.selection:Candidate",
            "aavas.automl.selection:Selection",
            "aavas.automl.hpo:optimize",
            "aavas.automl.imbalance:apply",
            "aavas.automl.imbalance:diagnose",
            "aavas.automl.calibration:fit",
            "aavas.automl.calibration:expected_calibration_error",
        ],
        "depends_on": ["kernel", "config", "logging_utils"],
        "owns_config": ["config/automl.yaml"],
        "entry_points": ["make automl", "scripts/run_automl.py", "GET /tracking/runs"],
        "emits": "A leaderboard and a champion recommendation, with the gates each candidate "
                 "cleared or failed.",
        "reuse": "The most portable of the four: it takes X, y and a contract, and knows "
                 "nothing about lending. The fairness gate needs a group column, and reports "
                 "'not measurable' rather than passing when there is none.",
        "standalone": True,
    },
    {
        "name": "acceptance",
        "title": "Acceptance Scoring Agent",
        "package": "aavas.acceptance",
        "version": "1.0",
        "purpose": "Turns a customer's own words into a closed vocabulary of acceptance "
                   "signals, weighs how far each can be trusted, and scores the likelihood "
                   "that this customer accepts.",
        "public": [
            "aavas.acceptance:AcceptanceAgent",
            "aavas.acceptance:ExtractionAgent",
            "aavas.acceptance:EvidenceAgent",
            "aavas.acceptance:AcceptanceSignals",
            "aavas.acceptance:Signal",
            "aavas.acceptance:SourceRegistry",
            "aavas.acceptance:SourceDocument",
        ],
        "depends_on": ["kernel", "config", "logging_utils", "connectors", "agents"],
        "owns_config": ["config/acceptance.yaml"],
        "entry_points": ["dormant - not called by the orchestrator"],
        "emits": "A Loan Acceptance Confidence Score with a verbatim quote behind every signal.",
        "reuse": "Portable wherever free text carries intent. Two things travel with it: the "
                 "signal vocabulary, which is the leakage guard, and the permission check at "
                 "the connector boundary - the dependency on `agents` is that check, and "
                 "removing it would remove the control.",
        "standalone": False,
        "not_standalone_because": "It calls aavas.agents.capabilities to check it may reach a "
                                  "connector. That edge is a control, not an accident: lifting "
                                  "this package means taking the permission model with it.",
    },
    {
        "name": "pricing",
        "title": "Price Optimization Agent",
        "package": "aavas.optimization",
        "version": "1.0",
        "purpose": "Searches the policy-permissible band for the price with the best "
                   "risk-adjusted expected value, and reports what it rejected and why.",
        "public": [
            "aavas.optimization.price_optimizer:optimize",
            "aavas.optimization.price_optimizer:evaluate_scenarios",
            "aavas.optimization.scenario_generator:generate",
            "aavas.optimization.yield_calculator:compute",
            "aavas.optimization.yield_calculator:decompose",
            "aavas.optimization.yield_calculator:breakeven_rate",
            "aavas.optimization.constraints:card_rate",
            "aavas.optimization.constraints:permissible_band",
            "aavas.optimization.constraints:approval_level",
            "aavas.optimization.constraints:check",
        ],
        "depends_on": ["kernel", "config", "ingestion", "logging_utils"],
        "owns_config": ["config/pricing_policy.yaml"],
        "entry_points": ["POST /pricing/final-assessment"],
        "emits": "A recommended rate with its economics, the scenarios rejected, and the "
                 "binding constraint.",
        "reuse": "The least portable, and honestly so: it encodes a lending pricing policy - "
                 "card rates, LTV and FOIR limits, approval authority. The *shape* transfers "
                 "to any priced product; the policy does not.",
        "standalone": True,
    },
]

BY_NAME = {p["name"]: p for p in PACKAGES}


def summary() -> dict[str, Any]:
    return {
        "packages": len(PACKAGES),
        "standalone": [p["name"] for p in PACKAGES if p["standalone"]],
        "coupled": [p["name"] for p in PACKAGES if not p["standalone"]],
        "shared_kernel": sorted(SHARED_KERNEL),
        "note": ("A package is 'standalone' when it depends on nothing but the shared kernel "
                 "and its own config. Acceptance is not, and the reason is recorded rather "
                 "than removed - the edge it carries is a permission check."),
    }


def listing() -> list[dict[str, Any]]:
    return [dict(p) for p in PACKAGES]
