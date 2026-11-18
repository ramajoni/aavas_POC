"""Where an application stands, for the RM and the area manager.

What must hold: one timeline per file, read from the store, moving step by step -
documents due, verified figures to fill, ready, with the area manager, offer sent -
with the indicative range and the risk-adjusted rate beside it; the board lists
every file with the same status; a borrower reads neither.
"""

from __future__ import annotations

from test_applications import COMPLETE, applicant, client, quote_and_apply, sign_in, upload  # noqa: F401

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
