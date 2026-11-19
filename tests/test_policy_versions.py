"""Every policy version a decision was priced under can be read, with what changed.

What must hold: one policy is in force, and it is listed and readable; when earlier
versions are kept in the archive they are listed oldest first, each with what
changed from the one before; staff may read them, a borrower may not.
"""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from aavas.api import access
from aavas.api.main import app
from aavas.config import config_path, policy
from aavas.governance import policy_versions

PASSWORDS = {"rohit": "Rohit@2026", "meera": "Meera@2026", "arjun": "Arjun@2026", "rahul": "Rahul@2026"}


@pytest.fixture(autouse=True)
def only_the_files(monkeypatch):
    """The archive and the live file only: edits other tests make through the
    configuration screen land in the shared change history."""
    monkeypatch.setattr(policy_versions.store, "query", lambda *a, **k: [])


@pytest.fixture
def archive(monkeypatch, tmp_path):
    """An earlier version beside the live one: the live file with a different cap."""
    older = yaml.safe_load(config_path("policy").read_text(encoding="utf-8"))
    older["policy_version"] = "HL-POLICY-2000.01"
    # Differs from the live cap whatever it is set to, so the change is always detected.
    older["override"]["max_unevidenced_discount_bps"] = policy()["override"]["max_unevidenced_discount_bps"] + 10
    (tmp_path / "HL-POLICY-2000.01.yaml").write_text(yaml.safe_dump(older, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(policy_versions, "ARCHIVE", tmp_path)
    return older


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def who(client, name) -> dict[str, str]:
    token = client.post("/auth/login", json={"username": name, "password": PASSWORDS[name]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_one_policy_is_in_force() -> None:
    listed = policy_versions.versions()
    assert [v["version"] for v in listed] == [policy()["policy_version"]] and listed[0]["current"]
    live = policy_versions.get(policy()["policy_version"])
    assert live["current"] and live["previous"] is None and live["changes_from_previous"] == []


def test_versions_order_numerically() -> None:
    assert policy_versions._order("HL-POLICY-2026.9") < policy_versions._order("HL-POLICY-2026.10")
    assert policy_versions._order("HL-POLICY-2026.10") < policy_versions._order("HL-POLICY-2026.10-draft")


def test_an_archived_version_shows_what_changed(archive) -> None:
    names = [v["version"] for v in policy_versions.versions()]
    assert names == ["HL-POLICY-2000.01", policy()["policy_version"]]
    live = policy_versions.get(policy()["policy_version"])
    assert live["previous"] == "HL-POLICY-2000.01"
    change = {c["path"]: c for c in live["changes_from_previous"]}
    assert change["override.max_unevidenced_discount_bps"]["before"] == archive["override"]["max_unevidenced_discount_bps"]
    assert change["override.max_unevidenced_discount_bps"]["after"] == policy()["override"]["max_unevidenced_discount_bps"]
    old = policy_versions.get("HL-POLICY-2000.01")
    assert not old["current"] and old["next"] == policy()["policy_version"]
    assert policy_versions.get("HL-POLICY-1999.01") is None


def test_staff_read_the_policy_and_a_borrower_does_not(client) -> None:
    live = policy()["policy_version"]
    for name in ("rohit", "meera", "arjun"):
        assert client.get(f"/pricing/policy/versions/{live}", headers=who(client, name)).status_code == 200
    assert client.get("/pricing/policy/versions", headers=who(client, "rahul")).status_code == 403
    assert client.get("/pricing/policy/versions/HL-POLICY-1999.01", headers=who(client, "meera")).status_code == 404
