"""The relationship manager's limited view, offers, and the area manager's approvals.

What must hold: the RM never receives the analysis behind a price (not in the
pricing response, not in the audit trail, not from the explanation, uplift or
behaviour endpoints); the RM can send at or above the recommendation, can go a
little below it only with a reason, and must ask the area manager below that;
the area manager sees the cost and approves only within her own limit.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.api import access
from aavas.api.main import app
from aavas.governance import store

PASSWORDS = {"rohit": "Rohit@2026", "meera": "Meera@2026", "arjun": "Arjun@2026",
             "rahul": "Rahul@2026", "admin": "Admin@2026"}
FILE = {"segment": "salaried_formal", "city": "Jaipur", "channel": "branch_walkin",
        "property_type": "resale_flat", "declared_monthly_income": 55000,
        "requested_loan_amount": 2000000, "requested_tenure_years": 20, "applicant_age": 35,
        "declared_obligations": 5000, "estimated_property_value": 2900000, "bureau_score": 790,
        "sales_context": {"competitor_rate": 11.5, "competitor_evidence_verified": False},
        "verified_monthly_income": 53000, "appraised_property_value": 2900000,
        "sanctioned_loan_amount": 2000000, "tenure_years": 20, "bureau_vintage_months": 96,
        "delinquency_count_24m": 0, "banking_behaviour_score": 75, "deviation_count": 0,
        "legal_flag": 0, "technical_flag": 0}


@pytest.fixture
def client(monkeypatch, bundle) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def who(client, name) -> dict[str, str]:
    token = client.post("/auth/login", json={"username": name, "password": PASSWORDS[name]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def price(client, headers, loan_id) -> dict:
    response = client.post("/pricing/final-assessment?include_scenarios=true",
                           json={**FILE, "loan_id": loan_id}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


# ------------------------------------------------------------ what the RM sees

def test_the_rm_gets_the_price_not_the_analysis(client) -> None:
    rohit = who(client, "rohit")
    body = price(client, rohit, "OFR-VIEW-1")
    assert body["view"] == "relationship_manager"
    assert body["recommendation"]["recommended_rate"] and body["recommendation"]["conversion_probability"]
    assert "scenario_table" not in body["recommendation"] and "economics" not in body["recommendation"]
    assert "explanation" not in body and "price_sensitivity" not in body and "trace" not in body
    assert "headroom" not in body, "the pricing headroom is the area manager's"
    assert "why" not in body and "reason_codes" not in str(body), "why the model priced it is the area manager's"

    full = price(client, who(client, "meera"), "OFR-VIEW-2")
    assert "scenario_table" in full["recommendation"] and "reason_codes" in full["explanation"]


def test_the_rm_cannot_reach_the_analysis_any_other_way(client) -> None:
    rohit = who(client, "rohit")
    decision_id = price(client, rohit, "OFR-VIEW-3")["decision_id"]
    assert client.get(f"/pricing/explain/{decision_id}", headers=rohit).status_code == 403
    assert client.get(f"/pricing/decision/{decision_id}", headers=rohit).status_code == 403
    assert client.post("/pricing/uplift", json={"decision_id": decision_id, "proposed_rate": 11.0},
                       headers=rohit).status_code == 403
    assert client.get("/customers/OFR-VIEW-3/behaviour", headers=rohit).status_code == 403
    assert client.get("/pricing/audit/OFR-VIEW-3", headers=rohit).status_code == 403, \
        "the governance and audit trail is not the relationship manager's"

    meera = who(client, "meera")
    assert client.get(f"/pricing/explain/{decision_id}", headers=meera).status_code == 200
    assert client.get("/customers/OFR-VIEW-3/behaviour", headers=meera).status_code == 200


# ----------------------------------------------------------------- offers

def test_the_rm_sends_at_or_above_the_recommendation_directly(client) -> None:
    rohit = who(client, "rohit")
    rec = price(client, rohit, "OFR-SEND-1")["recommendation"]["recommended_rate"]
    sent = client.post("/pricing/offers", headers=rohit, json={"loan_id": "OFR-SEND-1", "rate": rec})
    assert sent.status_code == 200, sent.text
    assert sent.json()["route"] == "direct" and sent.json()["status"] == "sent"


def test_a_little_below_needs_a_reason_and_is_an_override(client) -> None:
    rohit = who(client, "rohit")
    rec = price(client, rohit, "OFR-SEND-2")["recommendation"]["recommended_rate"]
    lower = round(rec - 0.10, 2)
    assert client.post("/pricing/offers", headers=rohit,
                       json={"loan_id": "OFR-SEND-2", "rate": lower}).status_code == 422
    sent = client.post("/pricing/offers", headers=rohit, json={
        "loan_id": "OFR-SEND-2", "rate": lower, "reason": "relationship_value", "note": "salary account"})
    assert sent.status_code == 200, sent.text
    assert sent.json()["route"] == "with_reason"
    override = store.fetch_one("SELECT * FROM overrides WHERE loan_id = 'OFR-SEND-2'")
    assert override["approved_by"] == "rohit:sales_officer" and override["human_rate"] == lower


def test_below_the_rm_limit_goes_to_the_area_manager(client) -> None:
    rohit = who(client, "rohit")
    rec = price(client, rohit, "OFR-MGR-1")["recommendation"]["recommended_rate"]
    asked = round(rec - 0.25, 2)
    refused = client.post("/pricing/offers", headers=rohit,
                          json={"loan_id": "OFR-MGR-1", "rate": asked, "reason": "strong_negotiation"})
    assert refused.status_code == 409 and "area manager" in refused.json()["detail"]

    made = client.post("/pricing/rate-requests", headers=rohit, json={
        "loan_id": "OFR-MGR-1", "rate": asked, "reason": "strong_negotiation", "note": "will walk away"})
    assert made.status_code == 200, made.text
    request_id = made.json()["request_id"]
    assert client.post("/pricing/rate-requests", headers=rohit, json={
        "loan_id": "OFR-MGR-1", "rate": asked, "reason": "strong_negotiation"}).status_code == 409

    mine = client.get("/pricing/rate-requests", headers=rohit).json()["requests"]
    assert any(r["request_id"] == request_id for r in mine) and "analysis" not in mine[0]

    meera = who(client, "meera")
    queue = client.get("/pricing/rate-requests?status=pending", headers=meera).json()["requests"]
    item = next(r for r in queue if r["request_id"] == request_id)
    assert item["analysis"]["cost_per_year"] == round(0.25 / 100 * FILE["sanctioned_loan_amount"])
    assert item["analysis"]["acceptance_at_requested"] >= item["analysis"]["acceptance_at_recommended"]

    assert client.post(f"/pricing/rate-requests/{request_id}/decide", headers=rohit,
                       json={"approve": True, "note": "ok"}).status_code == 403
    approved = client.post(f"/pricing/rate-requests/{request_id}/decide", headers=meera,
                           json={"approve": True, "reason": "strong_negotiation", "note": "hold at this"})
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "approved" and approved.json()["offer"]["route"] == "manager_approved"
    override = store.query("SELECT * FROM overrides WHERE loan_id = 'OFR-MGR-1'")[-1]
    assert override["approved_by"] == "meera:regional_manager"


def test_the_area_manager_cannot_go_below_her_own_limit(client) -> None:
    rohit, meera = who(client, "rohit"), who(client, "meera")
    rec = price(client, rohit, "OFR-MGR-2")["recommendation"]["recommended_rate"]
    request_id = client.post("/pricing/rate-requests", headers=rohit, json={
        "loan_id": "OFR-MGR-2", "rate": round(rec - 0.30, 2), "reason": "strong_negotiation"}).json()["request_id"]
    too_low = client.post(f"/pricing/rate-requests/{request_id}/decide", headers=meera, json={
        "approve": True, "rate": round(rec - 0.50, 2), "reason": "strong_negotiation", "note": "x"})
    assert too_low.status_code == 403 and "credit head" in too_low.json()["detail"]
    declined = client.post(f"/pricing/rate-requests/{request_id}/decide", headers=meera,
                           json={"approve": False, "note": "hold the recommended rate"})
    assert declined.json()["status"] == "declined" and declined.json()["offer"] is None


def test_the_customer_decision_closes_the_offer(client) -> None:
    rohit = who(client, "rohit")
    rec = price(client, rohit, "OFR-DONE-1")["recommendation"]["recommended_rate"]
    offer_id = client.post("/pricing/offers", headers=rohit,
                           json={"loan_id": "OFR-DONE-1", "rate": rec}).json()["offer_id"]
    done = client.post(f"/pricing/offers/{offer_id}/customer-decision", headers=rohit,
                       json={"accepted": False, "loss_reason": "competitor_lower_rate",
                             "lost_to": "Other HFC", "lost_to_rate": 10.9})
    assert done.status_code == 200, done.text
    assert done.json()["status"] == "not_accepted"
    assert client.post(f"/pricing/offers/{offer_id}/customer-decision", headers=rohit,
                       json={"accepted": True}).status_code == 409
    assert store.fetch_one("SELECT * FROM outcomes WHERE loan_id = 'OFR-DONE-1'")["lost_to"] == "Other HFC"
