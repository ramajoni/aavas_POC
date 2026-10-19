"""The system of record, and the claim that rests on it.

The registries are easy to list. The test that matters is whether a decision can
actually be rebuilt from them, so most of this exercises the walker against a
real decision rather than asserting the shape of a catalogue.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.api.main import app

EXPECTED = ["agent", "skill", "tool", "model", "connector", "knowledge",
            "prompt", "policy", "memory", "feature", "trace", "dataset"]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def index(client: TestClient) -> dict:
    response = client.get("/system-of-record")
    assert response.status_code == 200
    return response.json()


def test_all_twelve_registries_are_reported(index: dict) -> None:
    assert [r["key"] for r in index["registries"]] == EXPECTED


def test_an_absent_registry_is_named_rather_than_omitted(index: dict) -> None:
    """A catalogue that lists only what exists describes a more complete system
    than the one running."""
    absent = [r for r in index["registries"] if r["kind"] == "absent"]
    assert absent, "nothing is absent? check the audit is still honest"
    for registry in absent:
        assert not registry["versioned"] and not registry["queryable_as_of"]
        assert len(registry["note"]) > 40, f"{registry['key']} gives no reason"


def test_only_a_real_registry_claims_to_answer_about_the_past(index: dict) -> None:
    """Versioned config and code declarations must not claim as-of queryability.
    This is the distinction the whole audit turns on."""
    for registry in index["registries"]:
        if registry["queryable_as_of"]:
            assert registry["kind"] in {"registry", "per-decision data"}, (
                f"{registry['key']} is a {registry['kind']} and claims to answer as of a "
                f"past moment")


def test_the_model_registry_is_the_one_that_is_queryable_as_of(index: dict) -> None:
    model = next(r for r in index["registries"] if r["key"] == "model")
    assert model["kind"] == "registry" and model["queryable_as_of"]


def test_code_declarations_do_not_claim_a_version(index: dict) -> None:
    for registry in index["registries"]:
        if registry["kind"] == "code declaration":
            assert not registry["versioned"], (
                f"{registry['key']} is declared in code; it has no version to record")


# ------------------------------------------------------------ reconstruction

@pytest.fixture(scope="module")
def chain(client: TestClient) -> dict:
    """Price a real decision and walk the chain over it.

    The test store starts empty, so the decision and the registered cards are
    created here rather than skipped for - a reconstruction test that skips when
    there is nothing to reconstruct tests nothing.
    """
    from aavas.api import examples
    from aavas.governance import model_registry

    for card in model_registry.load_bundle().cards().values():
        existing = model_registry.champion(card["name"])
        if not existing or existing["version"] != card["version"]:
            from aavas.models import base as model_base
            model_registry.register(model_base.ModelCard(**{
                k: v for k, v in card.items()
                if k in model_base.ModelCard.__dataclass_fields__}))

    priced = client.post("/pricing/final-assessment", json=examples.FINAL_ASSESSMENT)
    assert priced.status_code == 200, priced.text

    response = client.get("/system-of-record/reconstruct/latest")
    assert response.status_code == 200, response.text
    return response.json()


def test_the_chain_covers_every_link_the_claim_names(chain: dict) -> None:
    questions = " ".join(link["question"].lower() for link in chain["chain"])
    for term in ("agent", "skill", "tool", "model", "feature", "data", "policy",
                 "personalization", "human", "final rate"):
        assert term in questions, f"the chain never asks about {term}"


def test_every_link_reports_one_of_three_states(chain: dict) -> None:
    for link in chain["chain"]:
        assert link["status"] in {"resolved", "partial", "broken"}
        if link["status"] != "resolved":
            assert link["why"], f"link {link['step']} is not resolved and gives no reason"


def test_the_agent_set_link_resolves_from_the_registered_manifest(chain: dict) -> None:
    """This was broken until the agent set became a registered object. It reads
    the snapshot stamped on the decision, not today's roster."""
    link = next(l for l in chain["chain"] if l["question"] == "Which version of the agent set?")
    assert link["status"] == "resolved"
    assert link["value"]["agent_manifest"].startswith("ASET-")
    assert link["value"]["agents"] > 0 and link["value"]["skills"] > 0


def test_skills_resolve_against_the_snapshot_not_todays_roster(chain: dict) -> None:
    link = next(l for l in chain["chain"] if "skills" in l["question"])
    assert link["status"] == "resolved"
    assert "manifest recorded on this decision" in link["why"]


def test_no_link_has_nothing_behind_it(chain: dict) -> None:
    """Every link now resolves at least partially. Branch personalization was the
    last broken one and closed when federated rounds began to persist."""
    broken = {link["question"] for link in chain["chain"] if link["status"] == "broken"}
    assert broken == set(), f"links with no evidence at all: {sorted(broken)}"


def test_the_remaining_partials_are_the_ones_that_are_honestly_partial(chain: dict) -> None:
    """Two links resolve against the system as it is now rather than as it was.
    Naming them keeps the difference visible: 'partial' is a finding, not a pass."""
    partial = {link["question"] for link in chain["chain"] if link["status"] == "partial"}
    assert partial <= {
        "Which tools did they call, and under what permission?",
        "Which branch personalization?",
        "Which human decision?",
        "Which final rate?",
        "Which data were the models fitted on?",
    }, f"an unexpected link became partial: {sorted(partial)}"


def test_the_dataset_link_resolves_to_a_content_hash(chain: dict) -> None:
    link = next(l for l in chain["chain"] if "data were the models fitted on" in l["question"])
    assert link["status"] in {"resolved", "partial"}
    datasets = (link["value"] or {}).get("datasets") or {}
    assert datasets, "no model reported a dataset identity"
    assert all(d.get("dataset_id", "").startswith("DS-") for d in datasets.values())


def test_the_verdict_matches_the_counts(chain: dict) -> None:
    summary = chain["summary"]
    assert summary["resolved"] + summary["partial"] + summary["broken"] == summary["links"]
    assert summary["reconstructible"] is (summary["broken"] == 0 and summary["partial"] == 0)
    assert summary["reconstructible"] is False, (
        "the chain claims a decision reconstructs exactly - verify that against the "
        "registries before accepting it")


def test_model_cards_resolve_as_of_the_decisions_versions(chain: dict) -> None:
    """The one link that genuinely answers a question about the past."""
    card_link = next(l for l in chain["chain"] if "model card" in l["question"])
    assert card_link["status"] == "resolved"
    for name, card in (card_link["value"] or {}).items():
        assert card["version"], name
        assert card["features"] > 0, f"{name} card records no features"


def test_an_unknown_decision_is_a_404(client: TestClient) -> None:
    assert client.get("/system-of-record/reconstruct/DEC-NOPE").status_code == 404


# ---------------------------------------------------------- the agent manifest

def test_the_manifest_id_is_derived_from_the_declaration_not_declared() -> None:
    """A hand-bumped version is a promise someone has to keep on every edit. The
    id is a content hash, so two different rosters cannot share one."""
    from aavas.governance import agent_registry

    first = agent_registry.manifest()["content_hash"]
    agent_registry.clear_cache()
    assert agent_registry.manifest()["content_hash"] == first, "the hash is not stable"


def test_changing_the_roster_changes_the_manifest_id(monkeypatch) -> None:
    from aavas.agents import roster
    from aavas.governance import agent_registry

    before = agent_registry.current_id()
    agent_registry.clear_cache()
    monkeypatch.setattr(roster, "AGENTS", roster.AGENTS[:-1])
    after = agent_registry.current_id()
    agent_registry.clear_cache()
    assert after != before, "dropping an agent did not change the manifest id"


def test_an_old_decision_still_resolves_to_the_roster_that_ran(client: TestClient,
                                                               monkeypatch) -> None:
    """The property the whole registry exists for: after the roster changes, a
    decision priced under the old one must still report the old one."""
    from aavas.agents import roster
    from aavas.governance import agent_registry

    priced = client.post("/pricing/final-assessment", json=_example()).json()
    original = priced["versions"]["agent_manifest"]

    # Change the roster and publish the new set, as a deployment would.
    agent_registry.clear_cache()
    monkeypatch.setattr(roster, "AGENTS", roster.AGENTS[:-1])
    changed = agent_registry.publish(actor="test")
    assert changed != original

    chain = client.get(f"/system-of-record/reconstruct/{priced['decision_id']}").json()
    link = next(l for l in chain["chain"] if l["question"] == "Which version of the agent set?")
    assert link["value"]["agent_manifest"] == original, "the decision reports today's roster"
    assert link["value"]["is_current"] is False
    assert "changed since this decision was priced" in link["why"]

    diff = client.get("/system-of-record/manifest-diff",
                      params={"left": original, "right": changed}).json()
    assert diff["identical"] is False
    assert diff["changed"]["agents"]["removed"], "the diff does not name what was dropped"

    agent_registry.clear_cache()


def test_a_manifest_can_be_read_back_exactly_as_recorded(client: TestClient) -> None:
    current = client.get("/system-of-record/manifest/current").json()
    assert current["is_current"] is True
    declaration = current["manifest"]
    assert {"agents", "skills", "tools", "pipeline"} <= set(declaration)
    assert all(a["name"] and a["module"] for a in declaration["agents"])


def test_an_unknown_manifest_is_a_404(client: TestClient) -> None:
    assert client.get("/system-of-record/manifest/ASET-nope").status_code == 404


def _example() -> dict:
    from aavas.api import examples
    return {**examples.FINAL_ASSESSMENT, "loan_id": "SOR-MANIFEST-001"}
