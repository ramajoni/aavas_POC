"""Who the agents are, what each decides, and which of them are wired in.

Declared rather than introspected. A roster built by scanning for classes whose
name ends in ``Agent`` would be accurate about what exists and silent about what
matters: what an agent decides, which models it needs, and whether the pipeline
actually calls it. ``tests/test_agents.py`` asserts the declaration and the code
have not drifted apart, which is the check introspection would have given us
without the meaning.

``wired`` is the load-bearing field. Three agents in ``aavas.acceptance`` are
implemented, tested and **not** called by the orchestrator: wiring them changes
what the engine prices on, which is a decision for a human, not a default. An
agent that exists but never runs is exactly the thing a control plane has to
show rather than hide.
"""

from __future__ import annotations

from typing import Any

# Order is the order the orchestrator runs them in; two agents appear twice
# because they genuinely run twice, against different reference prices.
PIPELINE: list[dict[str, Any]] = [
    {"agent": "underwriting", "action": "build digital twin"},
    {"agent": "risk", "action": "assess"},
    {"agent": "policy", "action": "envelope"},
    {"agent": "sales_context", "action": "read against the card rate"},
    {"agent": "pricing", "action": "optimise yield"},
    {"agent": "policy", "action": "enforce"},
    {"agent": "sales_context", "action": "re-read against the recommendation"},
    {"agent": "conversion", "action": "price response"},
    {"agent": "explanation", "action": "reason codes"},
]

AGENTS: list[dict[str, Any]] = [
    {"name": "underwriting", "cls": "UnderwritingAgent",
     "module": "aavas.agents.underwriting_agent", "wired": True,
     "stages": ["pre_assessment", "final_assessment"], "models": [],
     "tools": [],
     "skills": ["feature_build_sales", "feature_build_stage2"],
     "decides": "The digital twin: every derived field the rest of the pipeline reads, "
                "plus the notes that say what could not be verified."},
    {"name": "preassessment", "cls": "PreAssessmentAgent",
     "module": "aavas.agents.preassessment_agent", "wired": True,
     "stages": ["pre_assessment"], "models": ["stage1"],
     "tools": ["model.stage1"],
     "skills": ["eligibility_rules"],
     "decides": "The indicative band and the confidence attached to it, before any "
                "document has been collected."},
    {"name": "risk", "cls": "RiskAgent", "module": "aavas.agents.risk_agent",
     "wired": True, "stages": ["final_assessment"], "models": ["risk", "lgd"],
     "tools": ["model.risk", "model.lgd"],
     "skills": ["yield_calculation"],
     "decides": "PD, the risk grade derived from it, and the annual expected loss the "
                "price has to cover."},
    {"name": "policy", "cls": "PolicyAgent", "module": "aavas.agents.policy_agent",
     "wired": True, "stages": ["final_assessment"], "models": [],
     "tools": [],
     "skills": ["eligibility_rules", "approval_authority"],
     "decides": "The card rate and the floor/ceiling the recommendation must sit inside - "
                "and, on the second pass, whether it did."},
    {"name": "sales_context", "cls": "SalesContextAgent",
     "module": "aavas.agents.sales_context_agent", "wired": True,
     "stages": ["final_assessment"], "models": ["override"],
     "tools": ["model.override"],
     "skills": ["feature_build_sales"],
     "decides": "Commercial pressure: competitor gap, targets, negotiation state. Kept in "
                "its own block because it moves price for reasons that are not risk."},
    {"name": "pricing", "cls": "PricingAgent", "module": "aavas.agents.pricing_agent",
     "wired": True, "stages": ["final_assessment"],
     "models": ["stage2", "conversion", "prepayment", "override"],
     "tools": ["model.stage2", "model.conversion", "model.prepayment", "model.override"],
     "skills": ["scenario_generation", "price_optimization"],
     "decides": "The recommended rate, by searching the policy band for the price with the "
                "best risk-adjusted yield rather than the best margin."},
    {"name": "conversion", "cls": "ConversionAgent",
     "module": "aavas.agents.conversion_agent", "wired": True,
     "stages": ["final_assessment"], "models": ["conversion"],
     "tools": ["model.conversion"],
     "skills": ["counterfactual"],
     "decides": "What the customer does at that price: conversion at the recommendation and "
                "the negotiation table around it."},
    {"name": "explanation", "cls": "ExplanationAgent",
     "module": "aavas.agents.explanation_agent", "wired": True,
     "stages": ["final_assessment"], "models": ["stage2", "conversion"],
     "tools": ["model.stage2", "model.conversion", "explainer.shap"],
     "skills": ["feature_build_stage2", "yield_calculation", "counterfactual", "reason_codes"],
     "decides": "Reason codes in basis points, the commercial-versus-risk split, and whether "
                "the file needs a fair-pricing review."},
    {"name": "learning", "cls": "LearningAgent", "module": "aavas.agents.learning_agent",
     "wired": True, "stages": ["override", "outcome"], "models": ["uplift"],
     "tools": ["store.governance", "procedure.outcome_capture",
               "procedure.override_learning", "procedure.retraining_assessment",
               "model.uplift"],
     "skills": ["drift_detection"],
     "decides": "What happens after the decision: the human override and the eventual "
                "outcome, captured so the next model can be fitted on them."},

    # ------------------------------------------------------------- on request
    # Runs, but not in the pricing pipeline: a person's screen calls it.
    {"name": "market_research", "cls": "MarketResearchAgent",
     "module": "aavas.agents.market_research_agent", "wired": False, "on_demand": True,
     "stages": ["competitor_check"], "models": [],
     "tools": ["llm.web_research"],
     "skills": ["claim_scoring"],
     "decides": "Whether a competitor's quoted rate is believable: what that lender advertises "
                "online today, set against the quote, whether a sanction letter was seen and "
                "the customer's profile, as a score out of 100.",
     "runs": "When a relationship manager or area manager opens a file with a competitor "
             "quote; findings are reused per lender for a week. Outside the priced path: an "
             "internet search never moves a recommendation, it says how far a quote can "
             "justify a discount."},

    # ---------------------------------------------------------------- dormant
    {"name": "feature_discovery", "cls": "FeatureDiscoveryAgent",
     "module": "aavas.features.discovery", "wired": False,
     "stages": [], "models": [],
     "tools": [],
     # It used to reach into the learning package for one pure function. That
     # function is a kernel primitive now, and the kernel is not a skill: a skill
     # is a declared capability an agent uses, not every utility it calls.
     "skills": [],
     "decides": "Nothing. It proposes: candidate features generated from the stage's own "
                "contract, each with its signal, redundancy, stability and coverage, and "
                "the reasons it may not be admitted.",
     "dormant_because": "Design-time by construction. Feature discovery can be automated; "
                        "feature admission cannot - a feature with no human-authored reason "
                        "label cannot appear in a customer-facing explanation, so admission "
                        "is an approved change to feature_config.yaml rather than a "
                        "threshold this agent clears."},
    {"name": "extraction", "cls": "ExtractionAgent",
     "module": "aavas.acceptance.extraction_agent", "wired": False,
     "stages": [], "models": [],
     "tools": ["llm.claude", "connector.mcp", "source.registry"],
     "skills": ["signal_vocabulary"],
     "decides": "Pulls acceptance signals out of the configured sources, with a verbatim "
                "quote for each. Claude when enabled, deterministic rules otherwise.",
     "dormant_because": "Reads free text, including a customer's own words. Wiring it in "
                        "puts an LLM on the priced path."},
    {"name": "evidence", "cls": "EvidenceAgent",
     "module": "aavas.acceptance.evidence_agent", "wired": False,
     "stages": [], "models": [],
     "tools": [],
     "skills": ["signal_vocabulary"],
     "decides": "How far each signal can be trusted: verified, customer-reported, "
                "salesperson estimate or market benchmark, and how old it is.",
     "dormant_because": "Only meaningful once extraction runs."},
    {"name": "acceptance", "cls": "AcceptanceAgent",
     "module": "aavas.acceptance.acceptance_agent", "wired": False,
     "stages": [], "models": [],
     "tools": ["source.registry"],
     "skills": ["signal_vocabulary"],
     "decides": "The Loan Acceptance Confidence Score - how likely this customer is to "
                "accept, and on what evidence.",
     "dormant_because": "Its competitor signals would feed the priced path. Promoting them "
                        "changes what the engine prices on, which is a human decision."},
]

BY_NAME = {a["name"]: a for a in AGENTS}


def roster(loaded_models: set[str] | None = None) -> list[dict[str, Any]]:
    """The agents, each annotated with where it sits in the pipeline.

    ``loaded_models`` is what the bundle actually has. An agent whose model is
    missing is reported as degraded rather than absent: the orchestrator still
    calls it, and the engine falls back rather than failing.
    """
    positions: dict[str, list[int]] = {}
    for index, step in enumerate(PIPELINE, start=1):
        positions.setdefault(step["agent"], []).append(index)

    out = []
    for agent in AGENTS:
        missing = ([m for m in agent["models"] if m not in loaded_models]
                   if loaded_models is not None else [])
        out.append({**agent, "pipeline_positions": positions.get(agent["name"], []),
                    "missing_models": missing,
                    "status": ("on_demand" if agent.get("on_demand")
                               else "dormant" if not agent["wired"]
                               else "degraded" if missing else "ready")})
    return out


def summary(loaded_models: set[str] | None = None) -> dict[str, Any]:
    agents = roster(loaded_models)
    return {"agents": len(agents),
            "wired": sum(1 for a in agents if a["wired"]),
            "on_demand": sum(1 for a in agents if a["status"] == "on_demand"),
            "dormant": sum(1 for a in agents if a["status"] == "dormant"),
            "degraded": sum(1 for a in agents if a["status"] == "degraded"),
            "pipeline_steps": len(PIPELINE)}
