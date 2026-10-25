"""The design-time half: identity, sign-off, gates and recorded monitoring.

These test the controls, not the happy path. A gate that can be talked past, a
signature the requester can supply themselves, or an approval that silently
applies itself would each pass a naive test and defeat the control.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.api.main import app
from aavas.governance import approvals, monitors, request_log


@pytest.fixture(scope="module", autouse=True)
def registered_champions():
    """The test store starts empty. The gates read model cards, so the champions
    are registered here - a gate test against an empty registry would only prove
    the registry is empty."""
    from aavas.governance import model_registry
    from aavas.models import base as model_base

    for card in model_registry.load_bundle().cards().values():
        model_registry.register(model_base.ModelCard(**{
            k: v for k, v in card.items() if k in model_base.ModelCard.__dataclass_fields__}))


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("AAVAS_AUTH_MODE", "apikey")
    monkeypatch.setenv("AAVAS_API_KEYS",
                       "ka:alice:model_risk,kb:bob:technology|model_risk,"
                       "kc:carol:credit_policy,kd:dave:auditor")
    return {"alice": {"X-API-Key": "ka"}, "bob": {"X-API-Key": "kb"},
            "carol": {"X-API-Key": "kc"}, "dave": {"X-API-Key": "kd"}}


# ------------------------------------------------------------------ identity

def test_disabled_auth_says_so(client: TestClient, monkeypatch) -> None:
    """Sign-in is the default now (test_access_control.py), but scripts can still
    switch it off. What must never happen is that it looks secured while it is not."""
    monkeypatch.setenv("AAVAS_AUTH_MODE", "disabled")
    body = client.get("/governance").json()
    assert body["auth"]["mode"] == "disabled"
    assert body["auth"]["enforced"] is False
    assert body["principal"]["authenticated"] is False
    assert body["principal"]["identity_verified"] is False
    assert "disabled" in body["auth"]["warning"].lower()


def test_enabling_auth_rejects_an_unkeyed_caller(client: TestClient, keys) -> None:
    assert client.get("/governance/approvals").status_code == 401
    assert client.get("/governance/approvals", headers=keys["alice"]).status_code == 200


def test_a_bad_key_is_indistinguishable_from_no_key(client: TestClient, keys) -> None:
    """Distinguishing them tells a caller whether a key exists."""
    missing = client.get("/governance/approvals")
    wrong = client.get("/governance/approvals", headers={"X-API-Key": "not-a-key"})
    assert missing.status_code == wrong.status_code == 401
    assert missing.json()["detail"] == wrong.json()["detail"]


def test_an_unknown_role_on_a_key_is_dropped_not_granted(monkeypatch) -> None:
    from aavas.api import security

    monkeypatch.setenv("AAVAS_AUTH_MODE", "apikey")
    monkeypatch.setenv("AAVAS_API_KEYS", "kx:mallory:wizard|model_risk")
    assert security._key_table()["kx"].roles == ["model_risk"]


# --------------------------------------------------------------------- gates

def test_a_failed_gate_blocks_the_request(client: TestClient, keys) -> None:
    """The risk champion fails the fairness gate. It must not be promotable by
    approving it harder."""
    response = client.post("/governance/approvals", headers=keys["alice"], json={
        "artefact_type": "model", "artefact_id": "risk:v1", "action": "promote"})
    assert response.status_code == 422
    assert "fairness_pass" in response.json()["detail"]


def test_overriding_a_gate_needs_a_real_justification(client: TestClient, keys) -> None:
    short = client.post("/governance/approvals", headers=keys["alice"], json={
        "artefact_type": "model", "artefact_id": "risk:v1", "action": "promote",
        "override_justification": "fine"})
    assert short.status_code == 422


def test_an_override_is_recorded_against_the_approval_permanently(
        client: TestClient, keys) -> None:
    justification = ("Calibration gap accepted pending the age-band recalibration scheduled "
                     "for Q4; monitored weekly by model risk.")
    created = client.post("/governance/approvals", headers=keys["alice"], json={
        "artefact_type": "model", "artefact_id": "risk:v1", "action": "promote",
        "override_justification": justification}).json()["approval"]
    assert created["gates_passed"] is False
    assert created["override_justification"] == justification
    assert any(g["gate"] == "fairness_pass" and g["passed"] is False
               for g in created["gate_results"])


def test_a_gate_that_cannot_be_evaluated_does_not_count_as_passed() -> None:
    """Blocking on unmeasurable would make every new model unpromotable; passing
    it silently is the opposite mistake. It is reported as neither."""
    results = approvals.run_gates("model", "stage1:v1")
    fairness = next(r for r in results if r["gate"] == "fairness_pass")
    assert fairness["passed"] is None
    assert "not measured" in fairness["detail"].lower()


# ---------------------------------------------------------------- signatures

def _raise_clean(client: TestClient, keys) -> str:
    body = client.post("/governance/approvals", headers=keys["carol"], json={
        "artefact_type": "config", "artefact_id": "policy", "action": "publish",
        "summary": "raise the floor for the affordable segment"}).json()
    return body["approval"]["approval_id"]


def test_the_requester_may_not_sign_their_own_request(client: TestClient, keys) -> None:
    approval_id = _raise_clean(client, keys)
    denied = client.post(f"/governance/approvals/{approval_id}/sign",
                         headers=keys["carol"], json={"role": "credit_policy"})
    assert denied.status_code == 403
    assert "separation of duties" in denied.json()["detail"].lower()


def test_every_required_role_must_sign_before_it_is_approved(
        client: TestClient, keys) -> None:
    """A config change needs credit policy AND technology. One signature is not
    a control."""
    approval_id = _raise_clean(client, keys)
    from aavas.api import security
    import os
    os.environ["AAVAS_API_KEYS"] += ",ke:erin:credit_policy"
    security._key_table()

    first = client.post(f"/governance/approvals/{approval_id}/sign",
                        headers={"X-API-Key": "ke"}, json={"role": "credit_policy"})
    assert first.status_code == 200
    assert first.json()["approval"]["state"] == approvals.PENDING
    assert first.json()["approval"]["awaiting_roles"] == ["technology"]

    second = client.post(f"/governance/approvals/{approval_id}/sign",
                         headers=keys["bob"], json={"role": "technology"})
    assert second.json()["approval"]["state"] == approvals.APPROVED


def test_a_role_you_do_not_hold_cannot_be_signed_with(client: TestClient, keys) -> None:
    approval_id = _raise_clean(client, keys)
    denied = client.post(f"/governance/approvals/{approval_id}/sign",
                         headers=keys["dave"], json={"role": "technology"})
    assert denied.status_code == 403


def test_one_rejection_rejects_the_request(client: TestClient, keys) -> None:
    approval_id = _raise_clean(client, keys)
    rejected = client.post(f"/governance/approvals/{approval_id}/sign", headers=keys["bob"],
                           json={"role": "technology", "verdict": "reject",
                                 "comment": "not before the quarter closes"})
    assert rejected.json()["approval"]["state"] == approvals.REJECTED


# -------------------------------------------------------------------- apply

def test_approval_alone_changes_nothing(client: TestClient, keys) -> None:
    """Approved-but-not-released is a state a change process lives in."""
    approval_id = _raise_clean(client, keys)
    early = client.post(f"/governance/approvals/{approval_id}/apply", headers=keys["bob"])
    assert early.status_code == 409
    assert "awaiting" in early.json()["detail"]


def test_a_config_apply_says_it_did_not_write_the_file(client: TestClient, keys) -> None:
    """The API does not write configuration, and must not imply that it did."""
    approval_id = _raise_clean(client, keys)
    import os
    os.environ["AAVAS_API_KEYS"] += ",kf:frank:credit_policy"
    client.post(f"/governance/approvals/{approval_id}/sign",
                headers={"X-API-Key": "kf"}, json={"role": "credit_policy"})
    client.post(f"/governance/approvals/{approval_id}/sign",
                headers=keys["bob"], json={"role": "technology"})
    applied = client.post(f"/governance/approvals/{approval_id}/apply", headers=keys["bob"])
    assert applied.status_code == 501
    assert "does not write configuration" in applied.json()["detail"]


# ------------------------------------------------------------- request log

def test_a_request_row_is_opened_before_the_pipeline_runs(client: TestClient) -> None:
    """The property the whole log exists for: a request that fails still leaves a
    row, so a success rate is not 100% by construction."""
    from aavas.api import examples

    before = request_log.summary()["window"]
    client.post("/pricing/final-assessment",
                json={**examples.FINAL_ASSESSMENT, "loan_id": "REQLOG-OK"})
    client.post("/pricing/final-assessment", json={"loan_id": "REQLOG-BAD"})
    after = request_log.summary()
    assert after["window"] >= before + 2
    assert after["by_outcome"].get(request_log.CLIENT_ERROR, 0) >= 1
    assert after["success_rate_pct"] < 100


def test_dashboard_polling_is_not_logged(client: TestClient) -> None:
    before = request_log.summary()["window"]
    client.get("/health")
    client.get("/control-tower")
    assert request_log.summary()["window"] == before


# ---------------------------------------------------------------- monitors

def test_a_monitor_records_having_run(client: TestClient) -> None:
    result = monitors.run_fairness(actor="test")
    history = monitors.history("fairness", limit=1)
    assert history and history[0]["run_id"] == result["run_id"]
    assert history[0]["status"] == result["status"]


def test_the_fairness_monitor_alerts_on_a_champion_that_fails() -> None:
    """The finding this monitor exists for: the models in production had never
    been fairness-evaluated at all."""
    result = monitors.run_fairness(actor="test")
    assert result["status"] in {"ok", "warn", "alert"}
    if result["findings"]["failing"]:
        assert result["triggered"] is True
        assert result["status"] == "alert"


def test_a_failing_monitor_still_closes_its_row(monkeypatch) -> None:
    """A monitor that only writes on success cannot record the run that failed."""
    with pytest.raises(RuntimeError):
        with monitors.recorded("drift", actor="test"):
            raise RuntimeError("boom")
    last = monitors.history("drift", limit=1)[0]
    assert last["status"] == "failed"
    assert "boom" in str(last["findings"])


def test_a_monitor_that_never_ran_is_reported_as_such() -> None:
    status = monitors.status()
    assert "override_review" in status["monitors"]
    assert set(status["never_run"]) <= set(monitors.MONITORS)
