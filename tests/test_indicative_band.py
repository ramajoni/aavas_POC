"""The final rate stays inside the range the customer was quoted at Stage 1.

What must hold: once a loan has an indicative range, the Stage-2 search, the
recommendation and any offer stay inside it; policy still wins where the quoted
range lies outside the permissible band; and a file never quoted is priced as
before. The regression case is a real file that was quoted 10.50-11.20% and
then recommended 14.40%, because a flat demand curve let the optimiser walk to
the top of the policy band.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.api import access
from aavas.api.main import app
from aavas.ingestion.schemas import RateBand
from aavas.optimization import constraints, scenario_generator

PASSWORDS = {"rohit": "Rohit@2026"}
# The file from the report, numbers only.
FILE = {"segment": "self_employed_professional", "city": "Lucknow", "channel": "branch_walkin",
        "property_type": "builder_flat", "declared_monthly_income": 200000,
        "requested_loan_amount": 4000000, "requested_tenure_years": 20, "applicant_age": 31,
        "declared_obligations": 30000, "estimated_property_value": 6000000, "bureau_score": 700,
        "new_to_credit": False,
        "sales_context": {"competitor_rate": None, "competitor_name": None,
                          "competitor_evidence_verified": False, "sales_target_pct": 85,
                          "branch_target_pct": 90, "customer_negotiating": False,
                          "prior_negotiation_rounds": 1, "existing_relationship": False,
                          "month_end": False, "days_to_month_end": 15}}
VERIFIED = {"verified_monthly_income": 200000, "appraised_property_value": 6000000,
            "sanctioned_loan_amount": 4000000, "tenure_years": 20, "bureau_vintage_months": 96,
            "delinquency_count_24m": 0, "banking_behaviour_score": 75, "deviation_count": 0,
            "legal_flag": 0, "technical_flag": 0}


def test_the_search_is_held_to_the_quoted_range(training_data) -> None:
    application = training_data.head(1)
    envelope = constraints.permissible_band(application).iloc[0]
    low, high = float(envelope["min_rate"]), float(envelope["max_rate"])
    quoted = RateBand(round(low + 0.3, 2), round(low + 0.8, 2))
    held = scenario_generator.generate(application, quoted_band=quoted)
    assert held.rates.min() >= quoted.min_rate - 1e-9 and held.rates.max() <= quoted.max_rate + 1e-9
    assert any("indicative range quoted at Stage 1" in n for n in held.notes)
    assert {"quoted_min", "quoted_max"} <= set(held.anchors)


def test_policy_wins_when_the_quoted_range_is_outside_it(training_data) -> None:
    application = training_data.head(1)
    high = float(constraints.permissible_band(application).iloc[0]["max_rate"])
    outside = RateBand(high + 1.0, high + 2.0)
    priced = scenario_generator.generate(application, quoted_band=outside)
    assert priced.rates.max() <= high + 1e-9
    assert any("lies outside the policy band" in n for n in priced.notes)


def test_a_quote_below_policy_is_priced_at_the_policy_floor_not_the_ceiling(training_data) -> None:
    """The hosted case: quoted 9.60-10.15%, then verified at grade C with a policy
    floor of 10.40%. It was recommended 16.45% - the top of the band - because
    with the quote out of reach the whole band was searched."""
    application = training_data.head(1)
    low = float(constraints.permissible_band(application).iloc[0]["min_rate"])
    below = RateBand(round(low - 0.8, 2), round(low - 0.25, 2))
    priced = scenario_generator.generate(application, quoted_band=below)
    assert priced.rates.min() == pytest.approx(low) and priced.rates.max() == pytest.approx(low)
    assert any("nearest permissible rate" in n for n in priced.notes)


@pytest.fixture
def client(monkeypatch, bundle) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def who(client, name) -> dict[str, str]:
    token = client.post("/auth/login", json={"username": name, "password": PASSWORDS[name]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_the_reported_file_is_recommended_inside_its_quote_and_cannot_be_offered_above_it(client) -> None:
    rohit = who(client, "rohit")
    quote = client.post("/pricing/pre-assessment", json={"loan_id": "IND-BAND-1", **FILE}, headers=rohit)
    assert quote.status_code == 200, quote.text
    band = quote.json()["band"]

    final = client.post("/pricing/final-assessment", json={"loan_id": "IND-BAND-1", **FILE, **VERIFIED},
                        headers=rohit)
    assert final.status_code == 200, final.text
    body = final.json()
    rec = body["recommendation"]["recommended_rate"]
    assert band["min_rate"] - 1e-9 <= rec <= band["max_rate"] + 1e-9, (rec, band)
    assert body["indicative_band"]["within"] is True
    assert body["indicative_band"]["max_rate"] == band["max_rate"]

    above = client.post("/pricing/offers", headers=rohit,
                        json={"loan_id": "IND-BAND-1", "rate": round(band["max_rate"] + 0.25, 2)})
    assert above.status_code == 422 and "range quoted to the customer" in above.json()["detail"]
    assert client.post("/pricing/offers", headers=rohit,
                       json={"loan_id": "IND-BAND-1", "rate": rec}).status_code == 200


def test_a_file_never_quoted_is_priced_as_before(client) -> None:
    body = client.post("/pricing/final-assessment", json={"loan_id": "IND-BAND-NONE", **FILE, **VERIFIED},
                       headers=who(client, "rohit")).json()
    assert body["indicative_band"] is None, "nothing was quoted, so nothing holds the search"
