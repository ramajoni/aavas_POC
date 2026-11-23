"""A loan id has one form at every API boundary: letters, digits and hyphens, 3 to 40 characters,
the pattern the application endpoints already use. Anything else is refused with a 422 before
it reaches the pricing code, the store or a file path."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.api.examples import FINAL_ASSESSMENT, OUTCOME, OVERRIDE, PRE_ASSESSMENT
from aavas.api.main import app

BAD = ["../x", "a b", "A" * 41, "ab", "x/y", "a;b", ""]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("loan_id", BAD)
def test_a_malformed_id_is_refused_on_every_input(client, loan_id) -> None:
    bodies = {
        "/pricing/offers": {"loan_id": loan_id, "rate": 14.0},
        "/pricing/rate-requests": {"loan_id": loan_id, "rate": 14.0, "reason": "competitor_lower_rate"},
        "/pricing/pre-assessment": {**PRE_ASSESSMENT, "loan_id": loan_id},
        "/pricing/final-assessment": {**FINAL_ASSESSMENT, "loan_id": loan_id},
        "/pricing/override": {**OVERRIDE, "loan_id": loan_id},
        "/pricing/outcome": {**OUTCOME, "loan_id": loan_id},
    }
    for path, body in bodies.items():
        got = client.post(path, json=body)
        assert got.status_code == 422, (path, loan_id, got.status_code)
        assert any("loan_id" in str(e.get("loc")) for e in got.json()["detail"]), path
    for path, params in (("/pricing/offers", {"loan_id": loan_id}), ("/pricing/rate-requests", {"loan_id": loan_id}),
                         ("/pricing/offers/acceptance", {"loan_id": loan_id, "rates": "14.0"})):
        if loan_id == "":
            continue                       # an empty query value is the same as no filter on the lists
        assert client.get(path, params=params).status_code == 422, (path, loan_id)


def test_a_normal_application_id_still_works(client) -> None:
    ok = client.post("/pricing/pre-assessment", json={**PRE_ASSESSMENT, "loan_id": "APP1795203087796"})
    assert ok.status_code != 422, ok.text
    assert client.get("/pricing/offers", params={"loan_id": "APP-1795203087796"}).status_code == 200
    assert client.get("/pricing/rate-requests", params={"loan_id": "APP1795203087796"}).status_code == 200
    assert client.post("/pricing/rate-requests", json={"loan_id": "APP1795203087796", "rate": 14.0,
                                                       "reason": "competitor_lower_rate"}).status_code != 422
