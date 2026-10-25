"""The agent roster's contract with the code it describes.

The roster is declared, not introspected, because what matters about an agent -
what it decides, which models it needs, whether anything calls it - cannot be
read off a class name. The price of declaring it is drift, so these tests are
the check introspection would have given us for free.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from aavas.agents import roster

ROOT = Path(__file__).resolve().parents[1]


def test_every_agent_class_in_the_repo_is_on_the_roster():
    """A new agent that nobody added to the roster is invisible in the control
    plane, which is the one place that is supposed to show all of them."""
    declared = {(a["module"], a["cls"]) for a in roster.AGENTS}
    found = set()
    for package in ("agents", "acceptance", "features"):
        for path in sorted((ROOT / "src" / "aavas" / package).glob("*.py")):
            if path.name.startswith("_"):
                continue
            module = f"aavas.{package}.{path.stem}"
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("class ") and line.split("class ")[1].split("(")[0].strip(":").endswith("Agent"):
                    found.add((module, line.split("class ")[1].split("(")[0].strip(":")))
    assert found - declared == set(), f"agent classes missing from the roster: {found - declared}"


def test_every_declared_agent_actually_exists():
    for agent in roster.AGENTS:
        module = importlib.import_module(agent["module"])
        assert hasattr(module, agent["cls"]), f"{agent['module']}.{agent['cls']} is gone"


def test_the_pipeline_only_names_agents_that_are_wired_in():
    wired = {a["name"] for a in roster.AGENTS if a["wired"]}
    for step in roster.PIPELINE:
        assert step["agent"] in wired, f"{step['agent']} is in the pipeline but not wired"


def test_the_pipeline_matches_the_order_the_orchestrator_runs(training_data):
    """The declared order and the executed order must be the same list, or the
    control plane is drawing a pipeline that is not the one that ran."""
    from aavas.api import examples
    from aavas.api.deps import get_orchestrator

    result = get_orchestrator().process(examples.FINAL_ASSESSMENT, persist=False)
    ran = [(s["agent"], s["action"]) for s in result["trace"]["steps"]]
    declared = [(s["agent"], s["action"]) for s in roster.PIPELINE]
    assert ran == declared


def test_dormant_agents_say_why():
    """"Not wired in" is a decision. A roster that reported it without the
    reason would leave the next reader to guess whether it was deliberate."""
    for agent in roster.AGENTS:
        if not agent["wired"]:
            assert agent.get("dormant_because"), f"{agent['name']} is dormant with no reason given"
            assert not agent["pipeline_positions"] if "pipeline_positions" in agent else True


def test_status_reflects_missing_artefacts():
    full = {a["name"]: a for a in roster.roster({"stage1", "stage2", "risk", "lgd", "conversion",
                                                 "prepayment", "override", "uplift"})}
    assert full["risk"]["status"] == "ready"
    assert full["learning"]["status"] == "ready"

    # Risk without its LGD model still prices - on the policy table - and says so.
    no_lgd = {a["name"]: a for a in roster.roster({"stage1", "stage2", "risk", "conversion",
                                                   "prepayment", "override", "uplift"})}
    assert no_lgd["risk"]["status"] == "degraded"
    assert no_lgd["risk"]["missing_models"] == ["lgd"]
    assert full["acceptance"]["status"] == "dormant"

    degraded = {a["name"]: a for a in roster.roster({"stage1", "stage2", "conversion"})}
    # The agent still runs; the engine falls back. That is not the same as absent.
    assert degraded["risk"]["status"] == "degraded"
    assert degraded["risk"]["missing_models"] == ["risk", "lgd"]
    assert degraded["policy"]["status"] == "ready", "an agent with no model cannot degrade"


def test_summary_counts_add_up():
    summary = roster.summary(set())
    assert summary["agents"] == len(roster.AGENTS)
    assert summary["wired"] + summary["dormant"] == summary["agents"]
    assert summary["pipeline_steps"] == len(roster.PIPELINE)


# ------------------------------------------------------------------- tracing


def test_a_failing_agent_is_recorded_and_re_raised():
    """A trace that swallowed the exception would show a pipeline that finished
    when it did not."""
    from aavas.agents.trace import Trace

    trace = Trace()
    with pytest.raises(ValueError):
        with trace.step("risk", "assess") as step:
            step.summary = "never reached"
            raise ValueError("model unavailable")

    recorded = trace.as_dict()
    assert recorded["failed"] == ["risk"]
    assert recorded["steps"][0]["status"] == "failed"
    assert "model unavailable" in recorded["steps"][0]["error"]


def test_trace_accounts_for_time_spent_outside_the_agents():
    from aavas.agents.trace import Trace

    trace = Trace()
    with trace.step("policy", "envelope") as step:
        step.summary = "card 9.7%"
    recorded = trace.as_dict()
    assert recorded["unattributed_ms"] >= 0
    assert recorded["total_ms"] >= recorded["steps"][0]["ms"]


def test_a_step_that_says_nothing_says_so():
    from aavas.agents.trace import Trace

    trace = Trace()
    with trace.step("policy", "envelope"):
        pass
    assert trace.as_dict()["steps"][0]["summary"] == "no output worth summarising"
