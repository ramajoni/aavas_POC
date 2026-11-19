"""Model risk management (RBI draft MRM guidance 2026, FREE-AI).

What must hold: every model in use is in the inventory, the rate grid and the Claude
calls included; a material model's tier is never pulled down by a low score elsewhere;
whoever builds or owns a model cannot validate it; a validation covers the version it
was done on and no later one; an overdue validation or RMCB report is flagged; and only
the model validation role reads and writes this.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from aavas.api import access
from aavas.api.main import app
from aavas.governance import model_risk, store

LEVELS = ["Low", "Medium", "High"]
PASSWORDS = {"rohit": "Rohit@2026", "arjun": "Arjun@2026"}


def _e(m, c, a):
    return {"materiality": m, "complexity": c, "autonomy": a}


def test_a_low_score_never_lowers_a_material_models_tier() -> None:
    assert model_risk.tier(_e("High", "Low", "Low"), LEVELS) == "High"
    assert model_risk.tier(_e("Medium", "Low", "Low"), LEVELS) == "Medium"
    assert model_risk.tier(_e("Low", "High", "Low"), LEVELS) == "Medium"     # complexity raises it one level
    assert model_risk.tier(_e("Medium", "Medium", "High"), LEVELS) == "High"


def test_every_registered_model_is_in_the_inventory() -> None:
    out = model_risk.assess()
    assert not [f for f in out["findings"] if f["ref"] == "MRM 21"], out["findings"]
    names = {m["name"] for m in out["models"]}
    assert {"policy_grid", "llm_note_extraction", "llm_market_research", "llm_narrative"} <= names
    assert all(m["enhanced_oversight"] for m in out["models"] if m["third_party"])


def test_whoever_builds_or_owns_a_model_cannot_validate_it() -> None:
    with pytest.raises(model_risk.ModelRiskError) as err:
        model_risk.record_validation("stage2", outcome="approved", findings="ok", validator="dev",
                                     roles=["data_science"])
    assert err.value.status == 403
    with pytest.raises(model_risk.ModelRiskError) as err:            # not a validator role at all
        model_risk.record_validation("stage2", outcome="approved", findings="ok", validator="admin",
                                     roles=["administrator"])
    assert err.value.status == 403


def test_an_exception_needs_the_exception_and_a_date() -> None:
    with pytest.raises(model_risk.ModelRiskError):
        model_risk.record_validation("lgd", outcome="approved_with_exceptions", findings="ok",
                                     validator="arjun", roles=["model_validation"])


def test_a_validation_covers_its_version_only(monkeypatch) -> None:
    model_risk.record_validation("conversion", outcome="approved", findings="Back-tested on 2026 H1.",
                                 validator="arjun", roles=["model_validation"])
    state = {m["name"]: m["validation"]["state"] for m in model_risk.assess()["models"]}
    assert state["conversion"] == "approved"
    real = model_risk.version
    monkeypatch.setattr(model_risk, "version",
                        lambda name, e, c: "v2 retrained" if name == "conversion" else real(name, e, c))
    out = model_risk.assess()
    assert {m["name"]: m["validation"]["state"] for m in out["models"]}["conversion"] == "changed since validation"
    assert any(f["model"] == "conversion" and f["ref"] == "MRM 30" for f in out["findings"])


def test_an_old_validation_and_an_unreported_one_are_flagged() -> None:
    old = (date.today() - timedelta(days=400)).isoformat() + "T00:00:00+00:00"
    current = model_risk.version("prepayment", model_risk.inventory()["prepayment"], model_risk._cards())
    store.insert("model_validations", {
        "validation_id": "VAL-OLD", "model": "prepayment", "model_version": current, "outcome": "approved",
        "findings": "ok", "exceptions": None, "remediation_due": None, "validator": "arjun",
        "validator_roles": ["model_validation"], "validated_at": old, "reported_to_rmcb_on": None})
    found = [f for f in model_risk.assess(today=date.today() + timedelta(days=400))["findings"]
             if f["model"] == "prepayment"]
    refs = {f["ref"] for f in found}
    assert "MRM 30" in refs and "MRM 33" in refs
    model_risk.report_to_rmcb("VAL-OLD", reported_on=date.today().isoformat(), recorded_by="arjun")
    with pytest.raises(model_risk.ModelRiskError):
        model_risk.report_to_rmcb("VAL-OLD", reported_on="2000-01-01", recorded_by="arjun")


def test_a_review_clears_the_never_reviewed_gap() -> None:
    model_risk.record_review(reviewer="arjun", notes="Tiers confirmed; LLM components kept High.")
    out = model_risk.assess()
    assert out["inventory_review"]["by"] == "arjun"
    assert not [f for f in out["findings"] if "never been reviewed" in f["text"]]


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def _who(client, name):
    token = client.post("/auth/login", json={"username": name, "password": PASSWORDS[name]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_only_model_validation_reads_and_records(client) -> None:
    mv, rm = _who(client, "arjun"), _who(client, "rohit")
    assert client.get("/governance/model-risk", headers=mv).status_code == 200
    assert client.get("/governance/model-risk", headers=rm).status_code == 403
    body = {"model": "uplift", "outcome": "approved", "findings": "Uplift curve stable across branches."}
    assert client.post("/governance/model-risk/validations", json=body, headers=rm).status_code == 403
    done = client.post("/governance/model-risk/validations", json=body, headers=mv)
    assert done.status_code == 200, done.text
    assert done.json()["validator"] == "arjun" and "model_validation" in done.json()["validator_roles"]
