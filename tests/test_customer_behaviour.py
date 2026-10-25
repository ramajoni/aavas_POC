"""The loan book in the database, and one customer's behaviour against it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.api.main import app
from aavas.customers import behaviour, loan_book
from aavas.governance import store

APPLICANT = {
    "segment": "salaried_formal", "city": "Jaipur", "channel": "branch_walkin",
    "property_type": "resale_flat", "declared_monthly_income": 55000,
    "requested_loan_amount": 2000000, "requested_tenure_years": 20, "applicant_age": 35,
    "declared_obligations": 5000, "estimated_property_value": 2900000, "bureau_score": 750,
    "sales_context": {"competitor_rate": 11.5, "competitor_name": "HDFC",
                      "competitor_evidence_verified": False, "customer_negotiating": True,
                      "prior_negotiation_rounds": 1, "days_to_month_end": 15,
                      "sales_target_pct": 85},
}
VERIFIED = {"verified_monthly_income": 53000, "appraised_property_value": 2900000,
            "sanctioned_loan_amount": 2000000, "tenure_years": 20, "bureau_vintage_months": 96,
            "delinquency_count_24m": 0, "banking_behaviour_score": 75, "deviation_count": 0,
            "legal_flag": 0, "technical_flag": 0}


@pytest.fixture(scope="module")
def client(bundle) -> TestClient:
    return TestClient(app)


def price(client: TestClient, loan_id: str) -> None:
    a = {**APPLICANT, "loan_id": loan_id}
    assert client.post("/pricing/pre-assessment", json=a).status_code == 200
    assert client.post("/pricing/final-assessment", json={**a, **VERIFIED}).status_code == 200


# ----------------------------------------------------------------- the database

def test_the_whole_book_is_loaded_into_the_database(training_data) -> None:
    meta = loan_book.ensure_loaded()
    assert meta["rows"] == len(training_data)
    assert store.count("loan_book") == len(training_data)
    assert loan_book.ensure_loaded()["loaded_at"] == meta["loaded_at"], "reloaded for no reason"


def test_a_changed_source_is_reloaded(monkeypatch) -> None:
    before = loan_book.ensure_loaded()
    monkeypatch.setattr(loan_book, "fingerprint", lambda: "changed")
    after = loan_book.ensure_loaded()
    assert after["fingerprint"] == "changed"
    assert after["rows"] == before["rows"]


def test_book_level_behaviour_reads_the_database(client, training_data) -> None:
    assert client.get("/competitors/behaviour").json()["headline"]["files"] == len(training_data)


# ------------------------------------------------------------------ one customer

def test_a_priced_customer_is_analysed_from_what_was_recorded(client) -> None:
    price(client, "CB-TEST-1")
    body = client.get("/customers/CB-TEST-1/behaviour").json()
    s = body["signals"]
    assert (s["competitor_rate"], s["prior_negotiation_rounds"]) == (11.5, 1)
    assert s["customer_negotiating"] is True and s["competitor_evidence_verified"] is False
    assert s["recommended_rate"] and s["priced_at"]
    assert [e["kind"] for e in body["timeline"]] == ["indicative_offer", "final_assessment"]
    assert any("verbal" in r for r in body["readouts"])


def test_the_cohort_matches_the_customer_and_bands_add_up(client) -> None:
    price(client, "CB-TEST-2")
    cohort = client.get("/customers/CB-TEST-2/behaviour").json()["cohort"]
    assert cohort["matched_on"][0] == "salaried (formal)"
    assert sum(b["n"] for b in cohort["by_gap"]) == cohort["n"]
    rows = loan_book.query("SELECT COUNT(*) AS n FROM loan_book WHERE segment = ? AND "
                           "customer_negotiating = 1 AND competitor_evidence_verified = 0 AND "
                           "channel = ? AND existing_relationship = 0",
                           ("salaried_formal", "branch_walkin"))
    if rows[0]["n"] >= behaviour.MIN_COHORT:
        assert cohort["n"] == rows[0]["n"] and not cohort["relaxed"]


def test_a_thin_cohort_gives_up_the_least_important_traits_first(monkeypatch) -> None:
    monkeypatch.setattr(behaviour, "MIN_COHORT", 10**9)
    profile = {"segment": "salaried_formal", "customer_negotiating": 1,
               "competitor_evidence_verified": 0, "channel": "digital", "existing_relationship": 1}
    cohort = behaviour.cohort(profile)
    assert cohort["matched_on"] == []
    assert cohort["relaxed"][0] == "segment" and cohort["relaxed"][-1] == "existing relationship"
    assert cohort["n"] == store.count("loan_book")


def test_small_bands_are_suppressed(monkeypatch) -> None:
    monkeypatch.setattr(behaviour, "MIN_CELL", 10**9)
    cohort = behaviour.cohort({"segment": "salaried_formal"})
    assert all(b["suppressed"] and "win_rate" not in b for b in cohort["by_gap"])


def test_an_unknown_file_is_a_404(client) -> None:
    assert client.get("/customers/NEVER-PRICED/behaviour").status_code == 404


def test_repeated_events_are_counted_not_repeated(client) -> None:
    price(client, "CB-TEST-3")
    price(client, "CB-TEST-3")
    events = client.get("/customers/CB-TEST-3/behaviour").json()["timeline"]
    assert [(e["kind"], e["count"]) for e in events] == [("indicative_offer", 2), ("final_assessment", 2)]
