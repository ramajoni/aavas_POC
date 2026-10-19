"""The control tower.

The tests that matter here are not about numbers - those change with every
decision priced. They are about the rule the page exists to enforce: a metric
carries the source that produced it, and a metric with no source is returned
with its reason rather than dropped or filled in.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def tower(client: TestClient) -> dict:
    response = client.get("/control-tower")
    assert response.status_code == 200
    return response.json()


def test_every_metric_declares_a_source_or_a_reason(tower: dict) -> None:
    for panel in tower["panels"]:
        for metric in panel["metrics"]:
            if metric["sourced"]:
                assert metric["source"], f"{metric['label']} has a value and no source"
            else:
                assert metric["value"] is None, f"{metric['label']} is unsourced with a value"
                assert len(metric["detail"]) > 20, f"{metric['label']} gives no reason"


def test_unsourced_metrics_are_listed_not_hidden(tower: dict) -> None:
    listed = {(u["panel"], u["label"]) for u in tower["unsourced"]}
    found = {(p["title"], m["label"]) for p in tower["panels"]
             for m in p["metrics"] if not m["sourced"]}
    assert listed == found
    assert tower["counts"]["unsourced"] == len(found)
    # If this ever reaches zero, either the estate grew a lot of new telemetry or
    # somebody started filling tiles in. Both are worth a human look.
    assert found, "no unsourced tile at all - check nothing is inventing numbers"


def test_success_rate_comes_from_requests_not_decisions(tower: dict) -> None:
    """This tile was refused for as long as the only evidence was the decisions
    table, where a success rate is 100% by construction. It is answerable now
    because a request row is opened before the pipeline runs - so the *source* is
    what this test checks, not the presence of a number."""
    requests = next(p for p in tower["panels"] if p["key"] == "requests")
    rate = next(m for m in requests["metrics"] if m["label"] == "Success rate")
    assert rate["sourced"], "the success rate should now be answerable"
    assert "requests" in rate["source"], (
        f"a success rate sourced from {rate['source']!r} would be 100% by construction")
    assert {"Failure rate", "Client errors", "Abandoned"} <= {
        m["label"] for m in requests["metrics"]}


def test_the_federated_panel_reports_rounds_or_says_there_are_none(tower: dict) -> None:
    """Federated rounds persist now, so this panel fills from the store. What it
    must never do is show a zero: no round recorded and a round that moved nothing
    are different claims, and the tile says which.

    Whether a round exists depends on what else has run, so the behaviour is
    asserted rather than the outcome.
    """
    from aavas.governance import store

    federated = next(p for p in tower["panels"] if p["key"] == "federated")
    recorded = store.count("federated_rounds")

    if not recorded:
        assert all(not m["sourced"] for m in federated["metrics"])
        assert any("no round has been recorded" in m["detail"] for m in federated["metrics"])
        return

    sourced = [m for m in federated["metrics"] if m["sourced"]]
    assert sourced, "rounds are on file and the panel reports none of them"
    assert {"Participants", "Rounds", "Strategy"} <= {m["label"] for m in sourced}
    # The effect on a priced decision stays unsourced: the regional adjustment is
    # not applied on the priced path, and an effect of zero would be a lie.
    effect = next(m for m in federated["metrics"] if m["label"] == "Effect on a priced decision")
    assert not effect["sourced"]
    assert "not applied on the priced path" in effect["detail"]


def test_the_approval_zone_reports_what_it_can_and_cannot_enforce(tower: dict) -> None:
    """Sign-off exists now. It stays partial while authentication is disabled,
    because a signature from an unverified caller is a name in a text field - and
    a zone that read 'ready' on that basis would be the dangerous version."""
    approval = next(z for z in tower["zones"] if z["key"] == "approval")
    assert approval["status"] != "absent"
    assert approval["surfaces"], "the approval zone should now name its surfaces"

    governance = next(p for p in tower["panels"] if p["key"] == "governance")
    auth = next(m for m in governance["metrics"] if m["label"] == "Authentication")
    if auth["value"] == "disabled":
        assert approval["status"] == "partial"
        assert "signature" in approval["gap"].lower()


def test_zones_cover_the_six_control_areas(tower: dict) -> None:
    keys = [z["key"] for z in tower["zones"]]
    assert keys == ["policy", "agent_factory", "model_data", "approval", "runtime", "learning"]
    assert [z["n"] for z in tower["zones"]] == [1, 2, 3, 4, 5, 6]
    for zone in tower["zones"]:
        assert zone["status"] in {"ready", "partial", "absent"}
        assert zone["gap"], f"zone {zone['key']} claims no gap at all"


def test_agent_counts_match_the_roster(client: TestClient, tower: dict) -> None:
    agents = client.get("/agents").json()["summary"]
    panel = next(p for p in tower["panels"] if p["key"] == "agents")
    by_label = {m["label"]: m["value"] for m in panel["metrics"]}
    assert by_label["Declared"] == agents["agents"]
    assert by_label["Wired into the pipeline"] == agents["wired"]
    assert by_label["Dormant"] == agents["dormant"]


def test_operations_drift_models_governance_are_reachable(client: TestClient) -> None:
    """These four are the whole point: analytics/performance.py was written for a
    governance dashboard and, until the control tower, nothing imported it."""
    for path in ("/control-tower/operations", "/control-tower/models",
                 "/control-tower/governance", "/control-tower/drift"):
        assert client.get(path).status_code == 200, path


def test_operations_totals_agree_with_the_store(client: TestClient) -> None:
    from aavas.governance import store
    ops = client.get("/control-tower/operations").json()
    assert ops["decisions_total"] == store.count("decisions")
    assert ops["overrides_total"] == store.count("overrides")


def test_drift_reports_both_kinds(client: TestClient) -> None:
    drift = client.get("/control-tower/drift").json()
    assert drift["input_drift"], "no feature was checked for input drift"
    assert "conversion_drift" in drift
    assert drift["overall_status"] in {"ok", "warn", "alert"}
