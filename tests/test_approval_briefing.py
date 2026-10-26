"""What the area manager reads before approving a lower rate.

What must hold: the briefing is the area manager's, not the relationship
manager's; it describes the file's own branch against every branch; a discount's
payoff is the propensity-adjusted read, not the raw one; small samples are
withheld rather than shown as rates; and a city with no history says so.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aavas.analytics import approval_briefing as ab
from aavas.analytics import portfolio
from aavas.api import access
from aavas.api.main import app

PASSWORDS = {"rohit": "Rohit@2026", "meera": "Meera@2026"}
FILE = {"segment": "salaried_formal", "city": "Jaipur", "channel": "branch_walkin",
        "property_type": "resale_flat", "declared_monthly_income": 55000,
        "requested_loan_amount": 2000000, "requested_tenure_years": 20, "applicant_age": 35,
        "declared_obligations": 5000, "estimated_property_value": 2900000, "bureau_score": 790,
        "sales_context": {"competitor_rate": 11.0, "competitor_name": "HDFC",
                          "competitor_evidence_verified": False},
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


def requested(client, loan_id: str) -> str:
    rohit = who(client, "rohit")
    priced = client.post("/pricing/final-assessment", json={"loan_id": loan_id, **FILE}, headers=rohit)
    assert priced.status_code == 200, priced.text
    rec = priced.json()["recommendation"]["recommended_rate"]
    asked = client.post("/pricing/rate-requests", headers=rohit,
                        json={"loan_id": loan_id, "rate": round(rec - 0.30, 2),
                              "reason": "competitor_lower_rate", "note": "HDFC at 11.0"})
    assert asked.status_code == 200, asked.text
    return asked.json()["request_id"]


def test_the_area_manager_gets_a_briefing_on_the_files_branch(client) -> None:
    request_id = requested(client, "BRF-1")
    body = client.get(f"/pricing/rate-requests/{request_id}/briefing", headers=who(client, "meera"))
    assert body.status_code == 200, body.text
    b = body.json()
    assert b["where"]["level"] == "branch" and b["where"]["city"] == "Jaipur"
    assert b["requested_discount_bps"] == pytest.approx(30, abs=0.6)
    assert "league" not in b["overrides"], "only the applied branch is shown, not every branch"
    assert b["overrides"]["effect_this_reason"]["label"] == "competitor_lower_rate"
    assert b["performance"]["branch"]["files"] < b["performance"]["all_branches"]["files"]
    assert b["performance"]["rm"]["requests"] >= 1 and b["performance"]["rm"]["pending"] >= 1
    assert b["signals"], "the briefing leads with what to read first"
    mine = b["competitors"]["this_file"]
    assert mine["competitor"] == "HDFC" and mine["benchmark"]["n"] > 0
    assert any(r["competitor"] == "HDFC" for r in b["competitors"]["landscape"])


def test_the_relationship_manager_does_not_get_it(client) -> None:
    request_id = requested(client, "BRF-2")
    assert client.get(f"/pricing/rate-requests/{request_id}/briefing",
                      headers=who(client, "rohit")).status_code == 403
    assert client.get("/pricing/rate-requests/RRQ-NOPE/briefing",
                      headers=who(client, "meera")).status_code == 404


def test_a_city_with_no_history_falls_back_to_every_branch() -> None:
    history = portfolio.frame()
    rows, where = ab.scope(history, "Udaipur")
    assert where["level"] == "all" and len(rows) == len(history) and "Udaipur" in where["note"]


def test_small_samples_are_withheld_not_shown_as_rates() -> None:
    assert ab._share(pd.Series([True] * 5))["rate"] is None
    assert ab._share(pd.Series([True] * ab.MIN_RATE_N))["rate"] == 1.0


def test_similar_decisions_match_reason_and_depth() -> None:
    history = portfolio.frame()
    jaipur = history[history["city"] == "Jaipur"]
    similar = ab.similar_decisions(jaipur, "competitor_lower_rate", 20.0)
    disc = jaipur[(jaipur["override_direction"] == "discount")
                  & (jaipur["override_reason"] == "competitor_lower_rate")]
    expected = int(((-disc["override_bps"]) - 20).abs().le(ab.SIMILAR_DEPTH_BPS).sum())
    assert similar["files"] == expected and similar["window_bps"] == [10.0, 30.0]


def test_the_rm_record_counts_what_happened() -> None:
    record = ab.rm_record(
        [{"status": "approved", "recommended_rate": 11.3, "requested_rate": 11.1},
         {"status": "declined", "recommended_rate": 11.3, "requested_rate": 10.9},
         {"status": "pending", "recommended_rate": 11.0, "requested_rate": 10.9}],
        [{"status": "accepted"}, {"status": "not_accepted"}, {"status": "sent"}])
    assert (record["approved"], record["declined"], record["pending"]) == (1, 1, 1)
    assert record["approval_rate"] == 0.5 and record["acceptance_rate"] == 0.5
    assert record["avg_requested_discount_bps"] == pytest.approx(23.3, abs=0.1)


def test_a_discount_deeper_than_any_before_is_called_out() -> None:
    history = portfolio.frame()
    jaipur = history[history["city"] == "Jaipur"]
    request = {"request_id": "R", "loan_id": "L", "reason": "competitor_lower_rate",
               "recommended_rate": 14.4, "requested_rate": 10.0}
    b = ab.briefing(request, {"city": "Jaipur"}, history=history)
    assert b["overrides"]["similar_here"]["files"] == 0
    assert any("went this deep" in s and "440 bps asked" in s for s in b["signals"])
    assert b["overrides"]["similar_here"]["deepest_bps"] < 440 and len(jaipur)
