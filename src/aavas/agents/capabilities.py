"""Tools and skills, kept apart from agents and from each other.

Three registries, because the three things are governed differently and fail
differently. The split is not taxonomy for its own sake - each answers a
different question when a decision goes wrong.

* **Agent** (``roster.py``) - *who decided*. Accountable for a field on the
  decision, appears in the step trace, can be dormant. Failure means the
  decision does not happen.
* **Tool** (here) - *what it called that lives outside this process*. A model
  artefact on disk, a database, an MCP server, an optional library, an LLM
  endpoint. Failure means **unavailable**, and every one of them names the
  fallback that runs instead. This is the set that needs availability
  monitoring, and the set an MCP surface would expose.
* **Skill** (here) - *what it worked out for itself*. Deterministic logic with
  no external dependency: eligibility rules, yield arithmetic, reason codes.
  Nothing can make a skill unavailable, so a skill has no fallback. Failure
  means a bug, and the remedy is a test, not a health check.

The discriminator is exactly that: **a tool can be unavailable; a skill cannot.**
It is worth stating because the two are constantly conflated, and conflating them
produces a system that monitors its arithmetic and does not notice its database
is gone.

Declared rather than introspected, for the reason the roster is:
``tests/test_capabilities.py`` asserts that what an agent declares matches what
its module actually imports, in both directions. A skill an agent quietly started
using, or one it stopped using, fails the suite.

Config reads and type schemas (``aavas.config``, ``aavas.ingestion.schemas``) are
infrastructure rather than capabilities and are deliberately not listed: every
module uses them, so listing them per agent would say nothing.
"""

from __future__ import annotations

from typing import Any

# Packages whose modules must be declared as a tool or a skill when an agent
# imports one. Anything outside this set is infrastructure.
CAPABILITY_PACKAGES = {"optimization", "features", "xai", "learning",
                       "knowledge", "governance", "connectors", "acceptance"}

# --------------------------------------------------------------------- tools
# Each one names what makes it unavailable and what runs instead. A tool with no
# stated fallback is a tool that takes the engine down when it is missing, and
# that is a decision to make on purpose rather than discover in production.

TOOLS: list[dict[str, Any]] = [
    # --- model artefacts -----------------------------------------------------
    {"name": "model.stage1", "kind": "model", "module": "aavas.models.stage1.predict",
     "provides": "Indicative rate band from a quantile model.",
     "unavailable_when": "the stage1 artefact is missing from models/",
     "fallback": "the policy band alone, with the agent reported as degraded"},
    {"name": "model.stage2", "kind": "model", "module": "aavas.models.stage2.predict",
     "provides": "The rate anchor the optimiser searches around.",
     "unavailable_when": "the stage2 artefact is missing from models/",
     "fallback": "the policy card rate as the anchor"},
    {"name": "model.risk", "kind": "model", "module": "aavas.models.risk.predict",
     "provides": "PD and lifetime PD, graded; LGD from model.lgd, with the policy table as "
                 "the fallback when that artefact is missing.",
     "unavailable_when": "the risk artefact is missing from models/",
     "fallback": "the grade supplied on the file and zero expected loss, stated in the notes "
                 "rather than applied silently"},
    {"name": "model.conversion", "kind": "model", "module": "aavas.models.conversion.predict",
     "provides": "Probability the customer accepts at a given rate.",
     "unavailable_when": "the conversion artefact is missing from models/",
     "fallback": "no price-response curve; the optimiser loses its objective"},
    {"name": "model.prepayment", "kind": "model", "module": "aavas.models.prepayment.predict",
     "provides": "Balance-transfer-out probability, which sets expected life.",
     "unavailable_when": "the prepayment artefact is missing from models/",
     "fallback": "the policy expected-life assumption"},
    {"name": "model.override", "kind": "model", "module": "aavas.models.override.predict",
     "provides": "Whether a human is likely to override, and by how much.",
     "unavailable_when": "the override artefact is missing from models/",
     "fallback": "sales context without an override expectation"},
    {"name": "model.lgd", "kind": "model", "module": "aavas.models.lgd.predict",
     "provides": "Expected loss severity on default, fitted on observed recoveries and floored "
                 "by policy.",
     "unavailable_when": "the lgd artefact is missing from models/",
     "fallback": "the policy LGD table by property type and LTV, reported as lgd_source"},
    {"name": "model.uplift", "kind": "model", "module": "aavas.models.uplift.predict",
     "provides": "Acceptance points a proposed discount buys on this file, adjusted for who "
                 "historically got discounted, with the value it costs.",
     "unavailable_when": "the uplift artefact is missing from models/",
     "fallback": "no discount assessment; POST /pricing/uplift answers 503"},

    # --- stores and indexes --------------------------------------------------
    {"name": "store.governance", "kind": "store", "module": "aavas.governance.store",
     "provides": "Append-only decisions, overrides, outcomes and the audit log.",
     "unavailable_when": "the SQLite file (or AAVAS_STORE_URL target) cannot be opened",
     "fallback": "none - a decision that cannot be recorded must not be served, and "
                 "/health reports decisions_recorded as -1"},
    {"name": "index.policy", "kind": "index", "module": "aavas.knowledge.policy_indexer",
     "provides": "Clause retrieval behind /pricing/policy/explain.",
     "unavailable_when": "pgvector is not configured",
     "fallback": "in-memory TF-IDF over the policy file"},
    {"name": "index.sales_notes", "kind": "index", "module": "aavas.knowledge.sales_note_retriever",
     "provides": "Comparable past negotiations for the acceptance sources.",
     "unavailable_when": "pgvector is not configured",
     "fallback": "in-memory TF-IDF over the training notes"},

    # --- optional libraries --------------------------------------------------
    {"name": "explainer.shap", "kind": "library", "module": "aavas.xai.shap_explainer",
     "provides": "Attributions behind the reason codes.",
     "unavailable_when": "shap is absent or built against a different dependency",
     "fallback": "an ablation explainer; the method used is reported in the payload rather "
                 "than the result being passed off as SHAP"},
    {"name": "explainer.omnixai", "kind": "library", "module": "aavas.xai.omnixai_analysis",
     "provides": "Global importance, credit-score bands and the interactive dashboards.",
     "unavailable_when": "omnixai is absent from the analysis environment",
     "fallback": "make xai is unavailable; nothing on the priced path changes"},

    # --- external systems ----------------------------------------------------
    {"name": "connector.mcp", "kind": "connector", "module": "aavas.connectors.mcp_client",
     "provides": "WhatsApp, Gmail, Outlook, Salesforce and Oracle Fusion over one transport.",
     "unavailable_when": "the switch is off, the MCP server is unreachable, or - for any "
                         "channel carrying a customer's own words - no consent store exists",
     "fallback": "the source is recorded as unavailable with its reason; it is never "
                 "recorded as read-and-empty"},
    {"name": "source.registry", "kind": "source", "module": "aavas.acceptance.sources",
     "provides": "The declared sources a signal may come from, and the collection itself.",
     "unavailable_when": "a source is blocked, unreachable, or refuses the connector - it "
                         "rejects any connector not declared for the acceptance_signals layer",
     "fallback": "the source is reported unavailable with its reason and contributes no signal; "
                 "it started as a skill and was reclassified because it imports "
                 "aavas.connectors, and a skill that reaches a connector is a tool"},
    # These three read the governance store, so they can be unavailable and are
    # tools rather than skills. They were written as skills and the determinism
    # check in tests/test_capabilities.py rejected them - which is the check
    # earning its place rather than a taxonomy argument.
    {"name": "procedure.outcome_capture", "kind": "procedure",
     "module": "aavas.learning.outcome_capture",
     "provides": "Records what happened to a file and compares it with what was decided.",
     "unavailable_when": "the governance store cannot be read",
     "fallback": "none - an outcome that cannot be recorded is an outcome the next model "
                 "will not be fitted on"},
    {"name": "procedure.override_learning", "kind": "procedure",
     "module": "aavas.learning.override_learning",
     "provides": "Whether human overrides created or destroyed value, by reason.",
     "unavailable_when": "the governance store cannot be read, or too few overrides exist",
     "fallback": "the review reports insufficient evidence rather than a verdict"},
    {"name": "procedure.retraining_assessment", "kind": "procedure",
     "module": "aavas.learning.retraining",
     "provides": "Whether the evidence justifies retraining: drift alerts, a shifted "
                 "conversion curve, decisions accumulated, champion age.",
     "unavailable_when": "the governance store cannot be read - the decision count is one "
                         "of the four triggers",
     "fallback": "the assessment runs on the drift triggers alone and says so"},
    {"name": "llm.web_research", "kind": "llm", "module": "aavas.agents.market_research_agent",
     "provides": "The home-loan rates a lender advertises online today, each tied to a page the "
                 "search actually returned.",
     "unavailable_when": "ANTHROPIC_API_KEY is unset, web search is off for the organisation, "
                         "or the search fails",
     "fallback": "the competitor quote is scored on its evidence and the customer's profile "
                 "alone, and the check says the fact check did not run"},
    {"name": "llm.claude", "kind": "llm", "module": "aavas.acceptance.extraction_agent",
     "provides": "Signal extraction from free text with a verbatim quote.",
     "unavailable_when": "ANTHROPIC_API_KEY is unset or the call fails",
     "fallback": "deterministic rule extraction over the same signal vocabulary"},
]

# -------------------------------------------------------------------- skills
# Deterministic. No artefact, no socket, no key. Each names the config that
# governs it, because a skill whose thresholds can move is only deterministic
# with respect to a version of that file.

SKILLS: list[dict[str, Any]] = [
    {"name": "eligibility_rules", "module": "aavas.optimization.constraints",
     "does": "LTV, FOIR, tenure and age limits, and the eligible-amount envelope with its "
             "binding constraint.",
     "governed_by": "config/pricing_policy.yaml"},
    {"name": "approval_authority", "module": "aavas.optimization.constraints",
     "does": "The lowest authority that may sign off a given deviation from the card rate.",
     "governed_by": "config/pricing_policy.yaml"},
    {"name": "yield_calculation", "module": "aavas.optimization.yield_calculator",
     "does": "Gross spread, net margin, expected loss, lifetime value and the breakeven rate.",
     "governed_by": "config/pricing_policy.yaml (economics block)"},
    {"name": "scenario_generation", "module": "aavas.optimization.scenario_generator",
     "does": "Candidate prices across the permissible band, trimmed where the conversion "
             "model would be extrapolating.",
     "governed_by": "config/model_config.yaml"},
    {"name": "price_optimization", "module": "aavas.optimization.price_optimizer",
     "does": "Picks the scenario with the best risk-adjusted expected value, not the best "
             "margin, and reports what was rejected and why.",
     "governed_by": "config/pricing_policy.yaml"},
    {"name": "feature_build_stage2", "module": "aavas.features.stage2_features",
     "does": "Verified income, valuation, bureau depth, banking behaviour, deviations, the "
             "legal/technical outcome and the policy LGD table.",
     "governed_by": "config/feature_config.yaml"},
    {"name": "feature_build_sales", "module": "aavas.features.sales_features",
     "does": "Competitor gap, target pressure, negotiation state and the pressure index.",
     "governed_by": "config/feature_config.yaml"},
    {"name": "reason_codes", "module": "aavas.xai.reason_codes",
     "does": "Turns attributions into basis points against named drivers, and splits "
             "commercial from risk.",
     "governed_by": "config/feature_config.yaml (reason_code_labels)"},
    {"name": "counterfactual", "module": "aavas.xai.counterfactual",
     "does": "What would have to change, and by how much, to move the rate.",
     "governed_by": "config/pricing_policy.yaml"},
    {"name": "drift_detection", "module": "aavas.learning.drift_detection",
     "does": "PSI per feature and the conversion curve compared between two windows.",
     "governed_by": "config/model_config.yaml (drift)"},
                {"name": "signal_vocabulary", "module": "aavas.acceptance.signals",
     "does": "The closed set of acceptance signals and their coercion rules. An extractor "
             "may not invent a signal outside it.",
     "governed_by": "config/acceptance.yaml"},
    {"name": "claim_scoring", "module": "aavas.intelligence.claim_score",
     "does": "How believable a competitor's quoted rate is, out of 100: the evidence seen, "
             "the quote against what the lender advertises, and whether the customer could "
             "get its best rate; and how far the quote can justify a discount.",
     "governed_by": "config/market_research.yaml (claim_scoring)"},
]

# ---------------------------------------------------------------- permissions
# Registering a tool says an agent *can* call it. This says whether it *may*, and
# what the call is allowed to do. The distinction is not pedantic: the pricing
# agent reads six model artefacts and writes nothing, while the learning agent is
# the only thing in the system permitted to write to the governance store. A
# registry without this cannot answer "should that agent have been able to do
# that", which is the question an incident starts with.
#
# Enforcement, stated precisely rather than implied:
#
# * ``tests/test_capabilities.py`` asserts the grant and the roster agree in both
#   directions, so a permission cannot drift from the tool an agent declares.
# * ``check()`` is called at the boundary where it has a security consequence -
#   ``McpSource.fetch``, the only path that reads a customer's own words.
#
# It is not wrapped around every call site. Doing that across nine agents on the
# priced path would be a large change to working code for a smaller gain than it
# looks, and claiming blanket enforcement while shipping a declaration would be
# worse than either.

READ, WRITE, PROPOSE = "read", "write", "propose"

PERMISSIONS: dict[str, dict[str, Any]] = {
    # agent -> tool -> allowed operations
    "preassessment": {"model.stage1": [READ]},
    "risk": {"model.risk": [READ], "model.lgd": [READ]},
    "sales_context": {"model.override": [READ]},
    "pricing": {"model.stage2": [READ], "model.conversion": [READ],
                "model.prepayment": [READ], "model.override": [READ]},
    "conversion": {"model.conversion": [READ]},
    "explanation": {"model.stage2": [READ], "model.conversion": [READ],
                    "explainer.shap": [READ]},
    # The only agent permitted to write the governance store. Everything else
    # reads it; a pricing agent that could write a decision could write its own.
    "learning": {"store.governance": [READ, WRITE],
                 "procedure.outcome_capture": [READ, WRITE],
                 "procedure.override_learning": [READ],
                 "procedure.retraining_assessment": [READ],
                 # Read at the override stage: what the proposed discount buys.
                 "model.uplift": [READ]},
    # Dormant. Declared so that wiring them in is a permission decision taken in
    # daylight rather than a side effect of flipping `wired`.
    "extraction": {"llm.claude": [READ], "connector.mcp": [READ], "source.registry": [READ]},
    # On request, outside the priced path: reads the web, writes nothing itself
    # (intelligence/market_rates.py keeps what it found).
    "market_research": {"llm.web_research": [READ]},
    "acceptance": {"source.registry": [READ]},
}


def permitted(agent: str, tool: str, operation: str = READ) -> bool:
    return operation in (PERMISSIONS.get(agent) or {}).get(tool, [])


def check(agent: str, tool: str, operation: str = READ) -> None:
    """Raise unless this agent may perform this operation on this tool."""
    if not permitted(agent, tool, operation):
        allowed = (PERMISSIONS.get(agent) or {}).get(tool)
        raise PermissionError(
            f"agent {agent!r} may not {operation} {tool!r}"
            + (f"; it is permitted {allowed}" if allowed else " at all"))


def permissions_for(agent: str) -> dict[str, list[str]]:
    return dict(PERMISSIONS.get(agent) or {})


def writers() -> dict[str, list[str]]:
    """Which agents may write to which tools. Short by design."""
    return {agent: sorted(tool for tool, ops in tools.items() if WRITE in ops)
            for agent, tools in PERMISSIONS.items()
            if any(WRITE in ops for ops in tools.values())}


TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}
SKILLS_BY_NAME = {s["name"]: s for s in SKILLS}

# Module -> the capability that owns it, for the drift check. One module may back
# more than one skill (constraints backs both eligibility and authority), so this
# maps to a set.
BY_MODULE: dict[str, set[str]] = {}
for _item in (*TOOLS, *SKILLS):
    BY_MODULE.setdefault(_item["module"], set()).add(_item["name"])


def summary() -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for tool in TOOLS:
        kinds[tool["kind"]] = kinds.get(tool["kind"], 0) + 1
    return {"tools": len(TOOLS), "skills": len(SKILLS), "tool_kinds": kinds}


def tools(loaded_models: set[str] | None = None,
          connector_status: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The tools, each annotated with whether it is available right now.

    Only model artefacts and connectors can be answered from here; the rest
    report ``unknown`` rather than a guess, because a tool this module has not
    probed is not a tool this module can call healthy.
    """
    out = []
    for tool in TOOLS:
        available: bool | None = None
        if tool["kind"] == "model" and loaded_models is not None:
            available = tool["name"].split(".", 1)[1] in loaded_models
        elif tool["kind"] == "connector" and connector_status is not None:
            available = bool(connector_status.get("ready"))
        out.append({**tool, "available": available,
                    "status": "ready" if available else
                              "unavailable" if available is False else "unknown"})
    return out


def tool_permissions() -> list[dict[str, Any]]:
    """Flattened agent x tool x operations, for the control plane."""
    out = []
    for agent, granted in sorted(PERMISSIONS.items()):
        for tool, operations in sorted(granted.items()):
            out.append({"agent": agent, "tool": tool, "operations": operations,
                        "kind": (TOOLS_BY_NAME.get(tool) or {}).get("kind")})
    return out


def skills() -> list[dict[str, Any]]:
    """Skills have no availability. Returned as declared, on purpose."""
    return [dict(skill) for skill in SKILLS]
