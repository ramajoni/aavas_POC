"""The RBI artefacts register: each ask beside the platform's live evidence.

What must hold: every artefact names its source, status and requirements; every
evidence key the YAML names exists; evidence that decides its own coverage does
so from the facts (an LTV cap above RBI's tier is a gap, a failing fairness test
is flagged); only the model validation role (and the administrator) can read it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import yaml
from fastapi.testclient import TestClient

from aavas.api import access
from aavas.api.main import app
from aavas.governance import monitors, regulatory

PASSWORDS = {"rohit": "Rohit@2026", "meera": "Meera@2026", "arjun": "Arjun@2026"}
LAKH = 100_000


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def who(client, name) -> dict[str, str]:
    token = client.post("/auth/login", json={"username": name, "password": PASSWORDS[name]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def facts(**over):
    base = {"policy": {"constraints": {"max_ltv": 0.90, "max_loan_amount": 75 * LAKH}, "economics": {}},
            "book": [], "priced": [], "cards": {}, "registered": [], "latest_decomposition": {},
            "lives": [], "decisions": [], "explanation": []}
    return {**base, **over}


def test_every_artefact_names_its_source_status_and_asks() -> None:
    out = regulatory.register()
    assert out["artefacts"], "the register is empty"
    labels, statuses = out["coverage_labels"], out["status_labels"]
    for art in out["artefacts"]:
        assert art["source"].startswith("https://"), art["id"]
        assert art["status"] in statuses, art["id"]
        assert art["requirements"], art["id"]
        for req in art["requirements"]:
            assert req["coverage"] in labels, (art["id"], req["ask"])
            assert req["ask"]
    assert sum(out["summary"]["by_coverage"].values()) == out["summary"]["requirements"]
    assert out["policy_version"]


def test_every_evidence_key_in_the_file_is_known() -> None:
    spec = yaml.safe_load(regulatory.ARTEFACTS.read_text(encoding="utf-8"))
    keys = {r["evidence"] for a in spec["artefacts"] for r in a["requirements"] if r.get("evidence")}
    assert keys <= set(regulatory.EVIDENCE), keys - set(regulatory.EVIDENCE)


def test_a_policy_cap_above_the_rbi_tier_is_a_gap() -> None:
    mid = regulatory.EVIDENCE["ltv_30l_to_75l"]
    assert mid(facts())["coverage"] == "gap"
    tight = facts(policy={"constraints": {"max_ltv": 0.80, "max_loan_amount": 75 * LAKH}})
    assert mid(tight)["coverage"] == "in_place"
    # A new file over the tier is a gap whatever the policy says.
    assert mid({**tight, "priced": [(50 * LAKH, 0.85)]})["coverage"] == "gap"
    # Loans already booked over it are flagged, not counted as the control failing.
    booked = mid({**tight, "book": [(50 * LAKH, 0.85), (20 * LAKH, 0.88)]})
    assert booked["coverage"] == "in_place" and booked["attention"] and booked["figures"]["booked_over"] == 1


def test_the_small_and_large_tiers() -> None:
    assert regulatory.EVIDENCE["ltv_up_to_30l"](facts())["coverage"] == "in_place"
    # The product does not lend above ₹75 lakh, so the 75% tier cannot be breached.
    large = regulatory.EVIDENCE["ltv_above_75l"](facts())
    assert large["coverage"] == "in_place" and "largest loan" in large["text"]
    bigger = facts(policy={"constraints": {"max_ltv": 0.90, "max_loan_amount": 100 * LAKH}})
    assert regulatory.EVIDENCE["ltv_above_75l"](bigger)["coverage"] == "gap"


def test_a_failing_fairness_test_is_flagged() -> None:
    cards = {"risk": {"fairness": {"passed": False}}, "conversion": {"fairness": {"passed": True}},
             "lgd": {"fairness": None}}
    out = regulatory.EVIDENCE["model_fairness"](facts(cards=cards))
    assert out["coverage"] == "partial" and out["attention"]
    assert "fail: risk" in out["text"] and "not measured: lgd" in out["text"]
    clean = regulatory.EVIDENCE["model_fairness"](facts(cards={"conversion": {"fairness": {"passed": True}}}))
    assert clean["coverage"] == "in_place" and not clean["attention"]


def test_models_missing_from_the_registry_are_named() -> None:
    out = regulatory.EVIDENCE["model_inventory"](facts(cards={"stage1": {}, "risk": {}}, registered=["stage1"]))
    assert out["attention"] and out["figures"]["not_registered"] == ["risk"]


def test_monitors_that_never_ran_are_a_gap() -> None:
    never = {"drift": {"last_run": None, "last_status": None}, "fairness": {"last_run": None, "last_status": None}}
    assert regulatory.EVIDENCE["monitors"](facts(monitors=never))["coverage"] == "gap"
    now = datetime.now(timezone.utc).isoformat()
    fresh = {"drift": {"last_run": now, "last_status": "ok"}, "fairness": {"last_run": now, "last_status": "ok"}}
    assert regulatory.EVIDENCE["monitors"](facts(monitors=fresh))["coverage"] == "in_place"
    assert regulatory.EVIDENCE["monitors"](facts(monitors={**fresh, **{"drift": never["drift"]}}))["coverage"] == "partial"


def test_a_penal_element_in_the_rate_is_a_gap() -> None:
    clean = regulatory.EVIDENCE["rate_components"](facts(latest_decomposition={"customer_rate": 10, "less_funding_cost": -7.9}))
    assert clean["coverage"] is None and "no penal element" in clean["text"]
    penal = regulatory.EVIDENCE["rate_components"](facts(latest_decomposition={"plus_penal_interest": 2}))
    assert penal["coverage"] == "gap"


def test_a_fairness_monitor_that_checked_nothing_does_not_pass(monkeypatch) -> None:
    monkeypatch.setattr("aavas.governance.model_registry.registered", lambda: [])
    assert monitors.run_fairness(actor="test")["status"] == "warn"


def test_model_validation_reads_the_register_and_others_do_not(client) -> None:
    res = client.get("/governance/regulatory", headers=who(client, "arjun"))
    assert res.status_code == 200, res.text
    assert {a["id"] for a in res.json()["artefacts"]} >= {"fpc-rate-model", "ltv-ceilings", "model-risk"}
    for name in ("rohit", "meera"):
        assert client.get("/governance/regulatory", headers=who(client, name)).status_code == 403
