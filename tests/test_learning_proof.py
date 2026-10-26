"""Proof that the engine learns.

What must hold: the learning-curve backtest never scores a model on files it was
trained on, and more outcomes make it better; federated learning sends only
parameters, learns, and lands where pooling every state's data would; looking at
the evidence does not record a round, running one does; and only model
validation reaches it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.analytics import learning_proof as lp
from aavas.api import access
from aavas.api.main import app
from aavas.governance import store

PASSWORDS = {"arjun": "Arjun@2026", "rohit": "Rohit@2026"}


def test_the_backtest_scores_every_fit_on_the_same_unseen_months() -> None:
    curve = lp.learning_curve()
    rows = curve["rows"]
    assert len(rows) >= 3 and [r["months_of_history"] for r in rows] == sorted(r["months_of_history"] for r in rows)
    assert all(r["trained_to"] < curve["holdout_from"] for r in rows), "no fit may see the held-out months"
    assert rows[-1]["auc"] > rows[0]["auc"] and rows[-1]["log_loss"] < rows[0]["log_loss"]
    assert curve["gain"]["auc_pts"] > 0


def test_federated_learning_matches_pooling_without_pooling() -> None:
    proof = lp.federated()
    assert proof["mae_after_bps"] < proof["mae_before_bps"]
    assert abs(proof["mae_after_bps"] - proof["mae_centralised_bps"]) <= 0.5
    states = len(proof["states"])
    sent = proof["rounds"] * states * proof["parameters_per_state"]
    assert f"{sent:,} numbers sent" in proof["claims"][0]["evidence"]
    assert proof["history"][0]["round"] == 0 and len(proof["history"]) == proof["rounds"] + 1
    assert len({s["fingerprint"] for s in proof["states"]}) == states


def test_looking_does_not_record_a_round_running_does() -> None:
    with store.connect() as conn:
        conn.execute("DELETE FROM federated_rounds")
    looked = lp.federated()
    assert store.count("federated_rounds") == 0 and looked["recorded"] is False
    ran = lp.federated(record=True)
    assert store.count("federated_rounds") == 1 and ran["recorded"] is True
    assert ran["run_id"] == looked["run_id"], "same data and settings, same model id"


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def who(client, name) -> dict[str, str]:
    token = client.post("/auth/login", json={"username": name, "password": PASSWORDS[name]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_model_validation_sees_the_proof_and_the_rm_does_not(client) -> None:
    arjun, rohit = who(client, "arjun"), who(client, "rohit")
    body = client.get("/learning/continuous", headers=arjun)
    assert body.status_code == 200, body.text
    assert len(body.json()["claims"]) == 6 and body.json()["curve"]["rows"]
    assert client.get("/learning/federated?strategy=fedprox", headers=arjun).json()["strategy"] == "fedprox"
    assert client.get("/learning/federated?strategy=magic", headers=arjun).status_code == 422
    assert client.post("/learning/federated", json={"strategy": "fedavg"}, headers=arjun).json()["recorded"]
    assert client.get("/learning/continuous", headers=rohit).status_code == 403
    assert client.post("/learning/federated", json={}, headers=rohit).status_code == 403
