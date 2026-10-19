"""Agents, tools and skills are three registries, and they must not drift apart.

The declarations are hand-written because what matters about a capability - what
makes it unavailable, what runs instead, which config governs it - cannot be read
off an import statement. The price of declaring is drift, so these tests are the
check introspection would have given for free, in both directions:

* an agent that quietly starts using a skill fails here until it declares it;
* an agent that declares one it no longer imports fails here too.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from aavas.agents import capabilities, roster

ROOT = Path(__file__).resolve().parents[1]


def _module_path(dotted: str) -> Path:
    return ROOT / "src" / Path(dotted.replace(".", "/") + ".py")


def _first_party_imports(dotted: str) -> set[str]:
    """Absolute dotted names this module imports from within aavas."""
    tree = ast.parse(_module_path(dotted).read_text(encoding="utf-8"))
    package = dotted.rsplit(".", 1)[0].split(".")
    found: set[str] = set()

    def resolve(node: ast.ImportFrom) -> str | None:
        if node.level == 0:
            return node.module if (node.module or "").startswith("aavas") else None
        base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
        return ".".join([*base, node.module]) if node.module else ".".join(base)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = resolve(node)
            if resolved and resolved.startswith("aavas"):
                found.add(resolved)
                for alias in node.names:                    # `from ..xai import counterfactual`
                    found.add(f"{resolved}.{alias.name}")
    return found


AGENT_IDS = [a["name"] for a in roster.AGENTS]

# An agent importing another agent is composition, and the roster already records
# it. Only capability modules are in scope for the drift check below.
AGENT_MODULES = {a["module"] for a in roster.AGENTS}


# --------------------------------------------------------------- the registries

def test_a_tool_names_what_breaks_it_and_what_runs_instead():
    """A tool with no stated fallback is a tool that takes the engine down when it
    is missing. That is a decision to take deliberately, not to discover."""
    for tool in capabilities.TOOLS:
        assert tool["unavailable_when"], f"{tool['name']} never becomes unavailable?"
        assert tool["fallback"], f"{tool['name']} declares no fallback"


def test_a_skill_has_no_fallback_because_it_cannot_be_unavailable():
    """The discriminator between the two registries. If a skill ever needs a
    fallback it is a tool and belongs in the other list."""
    for skill in capabilities.SKILLS:
        assert "fallback" not in skill and "unavailable_when" not in skill
        assert skill["governed_by"], f"{skill['name']} names no governing config"


def test_no_name_is_both_a_tool_and_a_skill():
    assert not set(capabilities.TOOLS_BY_NAME) & set(capabilities.SKILLS_BY_NAME)


@pytest.mark.parametrize("item", [*capabilities.TOOLS, *capabilities.SKILLS],
                         ids=[i["name"] for i in (*capabilities.TOOLS, *capabilities.SKILLS)])
def test_every_declared_capability_resolves_to_real_code(item):
    importlib.import_module(item["module"])


def test_skills_are_deterministic_by_construction():
    """A skill module must not reach for a model artefact, a socket or a store.
    This is the property that makes 'a skill cannot be unavailable' true rather
    than merely asserted."""
    forbidden = ("aavas.governance.store", "aavas.connectors", "aavas.models.")
    for skill in capabilities.SKILLS:
        imports = _first_party_imports(skill["module"])
        for dotted in imports:
            assert not dotted.startswith(forbidden), (
                f"skill {skill['name']} imports {dotted}, which can be unavailable - "
                f"it is a tool, not a skill")


# ------------------------------------------------------- agents against the code

@pytest.mark.parametrize("agent", roster.AGENTS, ids=AGENT_IDS)
def test_declared_tools_and_skills_exist(agent):
    for name in agent["tools"]:
        assert name in capabilities.TOOLS_BY_NAME, f"{agent['name']} claims unknown tool {name}"
    for name in agent["skills"]:
        assert name in capabilities.SKILLS_BY_NAME, f"{agent['name']} claims unknown skill {name}"


@pytest.mark.parametrize("agent", roster.AGENTS, ids=AGENT_IDS)
def test_an_agent_declares_every_capability_module_it_imports(agent):
    """Drift direction one: a capability picked up and not declared."""
    declared = set(agent["tools"]) | set(agent["skills"])
    for dotted in _first_party_imports(agent["module"]):
        owners = capabilities.BY_MODULE.get(dotted)
        if not owners or dotted in AGENT_MODULES:
            continue
        assert owners & declared, (
            f"{agent['name']} imports {dotted} but declares none of {sorted(owners)}")


@pytest.mark.parametrize("agent", roster.AGENTS, ids=AGENT_IDS)
def test_an_agent_does_not_declare_a_skill_it_never_imports(agent):
    """Drift direction two: a capability dropped and still advertised. A control
    plane that lists a skill the agent stopped using describes a system that is
    not the one running."""
    imports = _first_party_imports(agent["module"])
    for name in agent["skills"]:
        module = capabilities.SKILLS_BY_NAME[name]["module"]
        assert module in imports, (
            f"{agent['name']} declares skill {name} but no longer imports {module}")


@pytest.mark.parametrize("agent", roster.AGENTS, ids=AGENT_IDS)
def test_model_tools_agree_with_the_models_field(agent):
    """The roster's models list and the model tools are two spellings of one fact."""
    from_tools = {t.split(".", 1)[1] for t in agent["tools"]
                  if capabilities.TOOLS_BY_NAME[t]["kind"] == "model"}
    assert from_tools == set(agent["models"]), (
        f"{agent['name']}: models={sorted(agent['models'])} but model tools={sorted(from_tools)}")


def test_a_tool_living_inside_an_agent_is_declared_by_that_agent():
    """The import check above skips agent-to-agent imports, so a tool whose code
    sits inside an agent's own module would otherwise go unchecked. The LLM in
    the extraction agent is the case that matters."""
    for tool in capabilities.TOOLS:
        if tool["module"] not in AGENT_MODULES:
            continue
        owner = next(a for a in roster.AGENTS if a["module"] == tool["module"])
        assert tool["name"] in owner["tools"], (
            f"{tool['name']} lives in {owner['name']} and is not declared by it")


def test_every_model_artefact_is_reachable_through_some_agent():
    """A model nothing calls is either dead weight or a wiring bug."""
    model_tools = {t["name"] for t in capabilities.TOOLS if t["kind"] == "model"}
    used = {t for a in roster.AGENTS for t in a["tools"]}
    assert model_tools - used == set(), f"no agent uses: {sorted(model_tools - used)}"
