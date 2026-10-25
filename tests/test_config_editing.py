"""Editing configuration from the admin console.

Every test works on a copy of config/, never the repository's files. What must
hold: only a signed-in administrator can write; a change must parse, keep its
sections and carry a new version; a change that breaks pricing is rolled back;
every change is recorded and can be undone.
"""

from __future__ import annotations

import shutil

import pytest
import yaml
from fastapi.testclient import TestClient

from aavas import config
from aavas.api import access, configuration
from aavas.api.main import app
from aavas.governance import store

ADMIN, ARJUN, ROHIT = ("admin", "Admin@2026"), ("arjun", "Arjun@2026"), ("rohit", "Rohit@2026")


@pytest.fixture
def config_copy(tmp_path, monkeypatch, bundle):
    original_dir = config.CONFIG_DIR
    copy = tmp_path / "config"
    shutil.copytree(original_dir, copy)
    monkeypatch.setenv("AAVAS_ACCESS_FILE", str(original_dir / "access.yaml"))
    monkeypatch.setattr(config, "CONFIG_DIR", copy)
    configuration._reload()
    yield copy
    monkeypatch.setattr(config, "CONFIG_DIR", original_dir)
    configuration._reload()


@pytest.fixture
def client(config_copy, monkeypatch) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def sign_in(client, who) -> dict[str, str]:
    token = client.post("/auth/login", json={"username": who[0], "password": who[1]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def policy_text(copy) -> str:
    return (copy / "pricing_policy.yaml").read_text()


def raise_unevidenced_cap(raw: str, to: int = 20) -> str:
    assert "max_unevidenced_discount_bps: 15" in raw
    return raw.replace("max_unevidenced_discount_bps: 15", f"max_unevidenced_discount_bps: {to}")


# ------------------------------------------------------------------- who

def test_only_the_administrator_may_edit(client, config_copy) -> None:
    edit = {"raw": raise_unevidenced_cap(policy_text(config_copy)), "version": "HL-POLICY-TEST-1"}
    for who in (ARJUN, ROHIT):
        assert client.put("/config/policy", json=edit, headers=sign_in(client, who)).status_code == 403
    assert client.get("/config", headers=sign_in(client, ARJUN)).json()["editable"] is False
    assert client.get("/config", headers=sign_in(client, ADMIN)).json()["editable"] is True
    assert "max_unevidenced_discount_bps: 15" in policy_text(config_copy)


def test_nobody_edits_with_authentication_disabled(config_copy, monkeypatch) -> None:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "disabled")
    anonymous = TestClient(app)
    assert anonymous.get("/config").json()["editable"] is False
    edit = {"raw": raise_unevidenced_cap(policy_text(config_copy)), "version": "HL-POLICY-TEST-1"}
    assert anonymous.put("/config/policy", json=edit).status_code == 403


# ---------------------------------------------------------------- applying

def test_an_edit_applies_to_the_next_decision_and_is_recorded(client, config_copy) -> None:
    admin = sign_in(client, ADMIN)
    edit = {"raw": raise_unevidenced_cap(policy_text(config_copy)), "version": "HL-POLICY-TEST-2"}

    preview = client.post("/config/policy/preview", json=edit, headers=admin).json()
    assert {c["path"] for c in preview["changes"]} == {
        "override.max_unevidenced_discount_bps", "policy_version"}
    assert "max_unevidenced_discount_bps: 15" in policy_text(config_copy), "preview wrote the file"

    applied = client.put("/config/policy", json=edit, headers=admin)
    assert applied.status_code == 200, applied.text
    assert applied.json()["after_version"] == "HL-POLICY-TEST-2"
    assert config.policy()["override"]["max_unevidenced_discount_bps"] == 20
    assert client.get("/health").json()["versions"]["policy_version"] == "HL-POLICY-TEST-2"
    assert "# ---" in policy_text(config_copy), "the comments must survive the edit"

    history = client.get("/config/policy/history", headers=sign_in(client, ARJUN)).json()["changes"]
    assert history[0]["changed_by"] == "admin" and history[0]["after_version"] == "HL-POLICY-TEST-2"
    assert store.query("SELECT * FROM audit_log WHERE action = 'edited_config' AND entity_id = 'policy'")


def test_a_change_must_carry_a_new_version(client, config_copy) -> None:
    response = client.put("/config/policy", headers=sign_in(client, ADMIN),
                          json={"raw": raise_unevidenced_cap(policy_text(config_copy))})
    assert response.status_code == 422
    assert "new policy_version" in response.json()["detail"]


def test_a_version_cannot_name_two_different_contents(client, config_copy) -> None:
    admin = sign_in(client, ADMIN)
    first = client.put("/config/policy", headers=admin, json={
        "raw": raise_unevidenced_cap(policy_text(config_copy), 20), "version": "HL-POLICY-TEST-3"})
    assert first.status_code == 200
    second = client.put("/config/policy", headers=admin, json={
        "raw": policy_text(config_copy).replace("max_unevidenced_discount_bps: 20",
                                                "max_unevidenced_discount_bps: 25"),
        "version": "HL-POLICY-2026.09"})
    assert second.status_code == 422 and "already named different content" in second.json()["detail"]


@pytest.mark.parametrize("mangle, message", [
    (lambda raw: raw + "\n  broken: [unclosed\n", "not valid YAML"),
    (lambda raw: "\n".join(l for l in raw.splitlines()
                           if not l.startswith("approval_authority") and not l.startswith("  - {level")),
     "sections cannot be removed"),
    (lambda raw: "- just\n- a list\n", "mapping"),
])
def test_a_broken_file_is_refused_and_nothing_is_written(client, config_copy, mangle, message) -> None:
    before = policy_text(config_copy)
    response = client.put("/config/policy", headers=sign_in(client, ADMIN),
                          json={"raw": mangle(before), "version": "HL-POLICY-TEST-4"})
    assert response.status_code == 422 and message in response.json()["detail"], response.text
    assert policy_text(config_copy) == before


def test_a_change_that_breaks_pricing_is_rolled_back(client, config_copy) -> None:
    before = policy_text(config_copy)
    broken = yaml.safe_load(before)
    broken["approval_authority"] = "not a list of levels"
    response = client.put("/config/policy", headers=sign_in(client, ADMIN), json={
        "raw": yaml.safe_dump(broken, sort_keys=False), "version": "HL-POLICY-TEST-5"})
    assert response.status_code == 422 and "rolled back" in response.json()["detail"], response.text
    assert policy_text(config_copy) == before
    assert config.policy()["policy_version"] == "HL-POLICY-2026.09"


@pytest.mark.parametrize("name", ["features", "governance"])
def test_the_feature_contract_and_the_approval_rules_are_never_edited_here(client, config_copy, name):
    file = config.config_path(name)
    before = file.read_text()
    response = client.put(f"/config/{name}", headers=sign_in(client, ADMIN),
                          json={"raw": before + "\n# edited\n", "version": "X-1"})
    assert response.status_code == 403
    assert file.read_text() == before
    meta = next(f for f in client.get("/config", headers=sign_in(client, ADMIN)).json()["files"]
                if f["name"] == name)
    assert meta["locked"]


def test_any_change_can_be_undone(client, config_copy) -> None:
    admin = sign_in(client, ADMIN)
    before = policy_text(config_copy)
    change = client.put("/config/policy", headers=admin, json={
        "raw": raise_unevidenced_cap(before), "version": "HL-POLICY-TEST-6"}).json()
    undone = client.post(f"/config/policy/history/{change['change_id']}/restore", headers=admin)
    assert undone.status_code == 200, undone.text
    assert undone.json()["kind"] == "restore"
    assert policy_text(config_copy) == before
    assert config.policy()["override"]["max_unevidenced_discount_bps"] == 15
