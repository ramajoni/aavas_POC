"""Drift monitoring: detection on the book as it is versus simulated drift, and the
OpenObserve alert webhook, which answers without a session but never without its token."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aavas.api import access, monitoring
from aavas.api.main import app
from aavas.monitoring import drift, openobserve

TOKEN = "test-webhook-token-0123456789"


@pytest.fixture(scope="module")
def clean(bundle, training_data):
    return drift.report("none", bundle=bundle, table=training_data)


@pytest.fixture(scope="module")
def price_war(bundle, training_data):
    return drift.report("price_war", bundle=bundle, table=training_data)


def test_the_book_as_it_is_raises_no_drift_alert(clean) -> None:
    assert clean["status"]["overall"] != "alert", clean["status"]
    assert clean["windows"]["reference"]["to"] <= clean["windows"]["current"]["from"]


def test_a_price_war_alerts_on_data_model_and_concept(price_war) -> None:
    assert price_war["status"] == {"data": "alert", "model": "alert", "concept": "alert", "overall": "alert"}
    flagged = {f["name"] for f in price_war["data"] if f["status"] == "alert"}
    assert "competitor_rate" in flagged
    # the change point is dated inside the current window, not before it
    dated = [f["change_detected_at"] for f in price_war["concept"] if f.get("change_detected_at")]
    assert dated and min(dated) >= price_war["windows"]["current"]["from"]


def test_default_surge_is_concept_drift_only(bundle, training_data) -> None:
    status = drift.report("default_surge", bundle=bundle, table=training_data)["status"]
    assert status["concept"] == "alert" and status["data"] != "alert"


# ------------------------------------------------------------- the webhook

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    monkeypatch.setattr(openobserve, "load_env_file", lambda path=None: None)
    monkeypatch.setattr(drift, "reports_dir", lambda: tmp_path)
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def test_the_webhook_refuses_alerts_without_the_token(client, monkeypatch) -> None:
    monkeypatch.setenv("AAVAS_O2_WEBHOOK_TOKEN", TOKEN)
    body = {"alert": "aavas_data_drift_alert", "status": "firing"}
    assert client.post("/monitoring/alert-webhook", json=body).status_code == 401
    assert client.post("/monitoring/alert-webhook", json=body,
                       headers={"X-Aavas-Webhook-Token": "wrong"}).status_code == 401
    assert not monitoring._alerts_file().exists()


def test_an_unconfigured_webhook_stays_closed(client, monkeypatch) -> None:
    monkeypatch.delenv("AAVAS_O2_WEBHOOK_TOKEN", raising=False)
    response = client.post("/monitoring/alert-webhook", json={"alert": "x"},
                           headers={"X-Aavas-Webhook-Token": ""})
    assert response.status_code == 503


def test_a_tokened_alert_is_recorded(client, monkeypatch) -> None:
    monkeypatch.setenv("AAVAS_O2_WEBHOOK_TOKEN", TOKEN)
    response = client.post("/monitoring/alert-webhook", headers={"X-Aavas-Webhook-Token": TOKEN},
                           content=b'{"alert": "aavas_concept_drift_alert", "rows": "not {json"}')
    assert response.status_code == 200, response.text
    # rows substituted as raw text can break the JSON; the alert name is still kept
    broken = b'{"alert": "aavas_model_drift_alert", "rows": "[{"name": "x"}]"}'
    assert client.post("/monitoring/alert-webhook", headers={"X-Aavas-Webhook-Token": TOKEN},
                       content=broken).status_code == 200
    entries = [json.loads(x) for x in monitoring._alerts_file().read_text(encoding="utf-8").splitlines()]
    assert [e["alert"] for e in entries] == ["aavas_concept_drift_alert", "aavas_model_drift_alert"]


def test_the_rest_of_monitoring_needs_a_session(client) -> None:
    assert client.get("/monitoring/drift").status_code == 401
    assert client.get("/monitoring/alerts").status_code == 401
    assert client.post("/monitoring/drift/run", json={}).status_code == 401
    assert not access.decide("GET", "/monitoring/drift", ["relationship_manager"]).allowed
    assert access.decide("GET", "/monitoring/drift", ["model_validation"]).allowed
