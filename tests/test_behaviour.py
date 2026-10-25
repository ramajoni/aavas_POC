"""Customer behaviour around competitor offers: the arithmetic and the suppression."""

from __future__ import annotations

import pandas as pd
import pytest

from aavas.intelligence.behaviour import Behaviour


def book(n: int, **values) -> pd.DataFrame:
    row = {"disbursed": True, "ai_recommended_rate": 11.0, "sales_offered_rate": 10.9,
           "offered_rate": 10.9, "competitor_rate": 11.2, "competitor_gap_bps": -30.0,
           "customer_response": "accepted", "bt_out_24m": False,
           "prior_negotiation_rounds": 0, "competitor_evidence_verified": False,
           "days_to_month_end": 15, "sales_target_pct": 80.0, "channel": "branch_walkin",
           "existing_relationship": False}
    return pd.DataFrame([{**row, **values}] * n)


def test_discount_is_the_recommendation_minus_the_offer_in_bps() -> None:
    cell = Behaviour(book(40)).cell("all", Behaviour(book(40)).df)
    assert cell["avg_discount_bps"] == pytest.approx(10.0)
    assert cell["discounted_share"] == 1.0
    assert cell["win_rate"] == 1.0


def test_a_small_cell_is_suppressed_not_shown() -> None:
    b = Behaviour(pd.concat([book(40), book(5, competitor_evidence_verified=True)]))
    documented, verbal = b.summary()["by_evidence"]
    assert documented == {"label": "Documented quote", "n": 5, "suppressed": True}
    assert verbal["suppressed"] is False


def test_negotiation_rounds_three_and_above_share_a_bucket() -> None:
    b = Behaviour(pd.concat([book(30, prior_negotiation_rounds=3), book(30, prior_negotiation_rounds=6)]))
    rounds = {c["label"]: c["n"] for c in b.summary()["by_negotiation"]}
    assert rounds["3+ rounds"] == 60


def test_bt_out_is_measured_on_won_files_only() -> None:
    lost_and_left = book(40, disbursed=False, bt_out_24m=True)
    won_and_stayed = book(40)
    cell = Behaviour(pd.concat([lost_and_left, won_and_stayed])).summary()["headline"]
    assert cell["bt_out_24m"] == 0.0


def test_month_end_is_the_last_three_days(training_data) -> None:
    summary = Behaviour(training_data).summary()
    early, late = summary["by_month_end"]
    assert late["n"] == int((training_data["days_to_month_end"] <= 3).sum())
    assert early["n"] + late["n"] == len(training_data)
    assert summary["findings"], "the book is large enough for every finding"


def test_the_endpoint_is_not_read_as_a_lender_name(training_data) -> None:
    """/competitors/{name} would swallow 'behaviour' if it were declared first."""
    from fastapi.testclient import TestClient

    from aavas.api.main import app

    body = TestClient(app).get("/competitors/behaviour").json()
    assert body["headline"]["files"] == len(training_data)
    assert [c["label"] for c in body["by_month_end"]] == ["Rest of month", "Last 3 days of month"]
