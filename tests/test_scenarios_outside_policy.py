"""The scenario table and the summary always score the ask, even when it is below the floor.

What must hold: an ask under the risk-based floor still gets the model's and the branch's
acceptance (it used to come out empty, and the top card showed dashes), its table row is
flagged as outside policy, nothing else in the table moves, and the suggestion never treats
that reference row as a rate it could approve.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.analytics import decision_support
from aavas.api import access
from aavas.api.main import app
from test_risk_pricing import FILE, who


@pytest.fixture
def client(monkeypatch, bundle) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def test_an_ask_below_the_floor_is_scored_and_flagged(client) -> None:
    rohit, meera = who(client, "rohit"), who(client, "meera")
    priced = client.post("/pricing/final-assessment", json={"loan_id": "OUT-1", **FILE}, headers=rohit)
    assert priced.status_code == 200, priced.text
    env = priced.json()["envelope"]
    ask = round(float(env["min_rate"]) - 0.20, 2)
    if ask < 13.0:
        pytest.skip("this file's floor is the product floor: no rate below it can be asked")
    made = client.post("/pricing/rate-requests", headers=rohit,
                       json={"loan_id": "OUT-1", "rate": ask, "reason": "competitor_lower_rate", "note": "HDFC at 11.0"})
    assert made.status_code == 200, made.text
    body = client.get(f"/pricing/rate-requests/{made.json()['request_id']}/scenarios", headers=meera).json()
    s = body["summary"]
    assert s["model_at_requested"] is not None and s["branch_at_requested"] is not None
    assert s["discount_buys_model_pts"] is not None and s["value_change_model_inr"] is not None
    row = next(r for r in body["table"] if r["mark"] == "requested")
    assert row["rate"] == ask and row["outside_policy"] is True and row["acceptance_model"] is not None
    assert not any(r["outside_policy"] for r in body["table"] if r["mark"] != "requested")
    rec = next(r for r in body["table"] if r["mark"] == "recommended")
    assert rec["outside_policy"] is False
    assert [r["rate"] for r in body["table"]] == sorted(r["rate"] for r in body["table"])


def _row(rate, outside=False, v=100000, acc=0.8):
    return {"rate": rate, "acceptance_model": acc, "acceptance_branch": acc, "value_branch_inr": v, "value_model_inr": v,
            "outside_policy": outside, "mark": None}


def test_the_suggestion_does_not_treat_the_reference_row_as_an_option() -> None:
    scen = {"recommended_rate": 14.0, "table": [_row(13.6, True, v=900000, acc=0.95), _row(13.9), _row(14.0, v=95000)]}
    out = decision_support.suggest({"requested_rate": 13.6, "reason": "competitor_lower_rate"}, scen, floor_rate=13.9,
                                   similar=None, evidence_verified=True)
    assert out["rate"] == 13.9 and out["action"] == "counter"
