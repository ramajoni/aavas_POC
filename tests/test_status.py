"""Where an application stands, for the RM and the area manager.

What must hold: one timeline per file, read from the store, moving step by step -
documents due, verified figures to fill, ready, with the area manager, offer sent -
with the indicative range and the risk-adjusted rate beside it; the board lists
every file with the same status; a borrower reads neither.
"""

from __future__ import annotations

import test_applications as _base
from test_applications import client, sign_in, upload  # noqa: F401

# The live policy allows 36-120 months and stresses affordability at rate + 2% against the grade's
# FOIR cap, so a Rs 20 lakh loan needs a 10-year tenure and an income that carries it. Policy lets a
# discount run at most 60 bps under the card rate, and against a rival quoting below the card the
# engine recommends at that floor; a rival at 15.5% sits above the card, which leaves room above the
# lowest rate policy allows for the manager to approve.
TENURE_YEARS = 10          # asked for and quoted at Stage 1
SANCTIONED_YEARS = 7       # what credit sanctions
INCOME = 120_000
SALES = {**_base.COMPLETE["sales_context"], "competitor_rate": 15.5}
COMPLETE = {**_base.COMPLETE, "tenure_years": SANCTIONED_YEARS, "verified_monthly_income": INCOME - 2_000,
            "sales_context": SALES}


def applicant(loan_id: str, name: str = "Rahul Verma") -> dict:
    return {**_base.applicant(loan_id, name), "requested_tenure_years": TENURE_YEARS,
            "declared_monthly_income": INCOME, "sales_context": {**_base.applicant(loan_id)["sales_context"],
                                                                 "competitor_rate": 15.5}}


def quote_and_apply(client, headers, loan_id, name="Rahul Verma", **extra) -> dict:
    return _base.quote_and_apply(client, headers, loan_id, name, **{
        k: v for k, v in applicant(loan_id, name).items()
        if k in ("requested_tenure_years", "declared_monthly_income", "sales_context")}, **extra)

REQUIRED = ("income_proof", "bank_statement", "identity_address")


def status(client, headers, loan_id) -> dict:
    response = client.get(f"/applications/{loan_id}/status", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def step(body, key) -> dict:
    return next(s for s in body["steps"] if s["key"] == key)


def test_a_file_moves_through_its_status(client) -> None:
    rahul, rohit, meera = sign_in(client, "rahul"), sign_in(client, "rohit"), sign_in(client, "meera")
    quote_and_apply(client, rahul, "ST-1")
    first = status(client, rohit, "ST-1")
    assert first["status"]["label"].startswith("Documents due")
    assert step(first, "indicative")["state"] == "done" and first["indicative"]["min_rate"] > 5
    assert step(first, "documents")["state"] == "current" and first["risk_adjusted"] is None
    assert first["next"]["kind"] == "documents" and first["next"]["owner"] == "Customer"
    assert first["aging"]["days_open"] == 0 and first["aging"]["step"] == "waiting for documents"
    assert first["upsell"]["kind"] == "quote" and first["upsell"]["rate"] == first["indicative"]["max_rate"]

    for kind in REQUIRED:
        assert upload(client, rahul, "ST-1", kind=kind).status_code == 200
    verify = status(client, rohit, "ST-1")
    assert verify["status"]["label"] == "Loan risk to compute"
    assert verify["next"]["kind"] == "verify" and verify["next"]["owner"] == "RM"

    assert client.put("/applications/ST-1", headers=rohit, json={"applicant": COMPLETE}).status_code == 200
    priced = client.post("/pricing/final-assessment", headers=rohit, json={
        k: v for k, v in {**applicant("ST-1"), **COMPLETE}.items() if k not in ("name", "conversation_notes")})
    rec = priced.json()["recommendation"]["recommended_rate"]
    ready = status(client, meera, "ST-1")
    assert ready["status"]["label"] == "Ready for the final offer"
    assert step(ready, "risk")["state"] == "done" and ready["risk_adjusted"]["rate"] == rec
    assert step(ready, "offer")["state"] == "current" and step(ready, "manager")["state"] == "skipped"
    assert ready["next"]["kind"] == "send_offer" and f"{rec:.2f}%" in ready["next"]["text"]

    assert client.post("/pricing/rate-requests", headers=rohit, json={
        "loan_id": "ST-1", "rate": round(rec - 0.4, 2), "reason": "strong_negotiation"}).status_code == 200
    asked = status(client, meera, "ST-1")
    assert asked["status"]["label"] == "With the area manager" and step(asked, "manager")["state"] == "current"
    assert asked["next"]["kind"] == "manager_review" and asked["next"]["owner"] == "Area manager"
    assert asked["next"]["request_id"].startswith("RRQ-")
    assert asked["aging"]["step"] == "with the area manager" and asked["aging"]["days_in_step"] == 0
    assert asked["upsell"]["kind"] == "hold" and asked["upsell"]["bps"] == round((rec - round(rec - 0.4, 2)) * 100)

    board = client.get("/applications/status-board", headers=meera).json()["applications"]
    row = next(r for r in board if r["loan_id"] == "ST-1")
    assert row["status"]["label"] == "With the area manager" and row["indicative"] and row["risk_adjusted"]
    assert row["next"]["kind"] == "manager_review"


def test_a_borrower_reads_neither(client) -> None:
    rahul = sign_in(client, "rahul")
    quote_and_apply(client, rahul, "ST-2")
    assert client.get("/applications/ST-2/status", headers=rahul).status_code == 403
    assert client.get("/applications/status-board", headers=rahul).status_code == 403
