"""The RM's worklist and the area manager's view of the area.

What must hold: a file sits in exactly one stage, read from the store - in
progress until the documents verification needs are in, then ready with the
engine's suggested offer, then pending rate approval once the RM asks the area
manager, then offer sent, then the customer's own answer; a stalled customer can
be nudged (recorded, not delivered); the area manager's deep dive frames the
decision with the floor, the recommendation and the ceiling, and disbursements
count accepted offers against each RM's target.
"""

from __future__ import annotations

from aavas.governance import store
from aavas.intake import worklist

from test_applications import client, quote_and_apply, sign_in, upload  # noqa: F401

REQUIRED = ("income_proof", "bank_statement", "identity_address")


def listed(client, headers, loan_id) -> dict:
    rows = client.get("/applications", headers=headers).json()["applications"]
    return next(r for r in rows if r["loan_id"] == loan_id)


def priced_count(loan_id: str) -> int:
    return store.fetch_one("SELECT COUNT(*) AS n FROM decisions WHERE loan_id = ? "
                           "AND stage = 'post_assessment'", (loan_id,))["n"]


def verify(client, headers, loan_id) -> None:
    for kind in REQUIRED:
        assert upload(client, headers, loan_id, kind=kind).status_code == 200


# ------------------------------------------------------------------- stages

def test_a_file_moves_through_the_worklist_one_stage_at_a_time(client) -> None:
    rahul, rohit, meera = sign_in(client, "rahul"), sign_in(client, "rohit"), sign_in(client, "meera")
    quote_and_apply(client, rahul, "WL-STAGE-1")

    row = listed(client, rohit, "WL-STAGE-1")
    assert row["stage"] == "in_progress" and row["suggested"] is None
    assert set(row["missing_documents"]) == set(REQUIRED)
    upload(client, rahul, "WL-STAGE-1", kind="income_proof")
    assert listed(client, rohit, "WL-STAGE-1")["missing_documents"] == ["bank_statement", "identity_address"]

    verify(client, rahul, "WL-STAGE-1")
    row = listed(client, rohit, "WL-STAGE-1")
    assert row["stage"] == "ready" and row["verified_at"]
    suggested = row["suggested"]
    assert 6 < suggested["rate"] < 18 and 0 < suggested["acceptance"] <= 1
    assert suggested["sanctioned_amount"] == 2000000 and suggested["amount_verified"] is False
    # Priced once on verification; looking again reuses that decision.
    before = priced_count("WL-STAGE-1")
    assert listed(client, rohit, "WL-STAGE-1")["suggested"]["decision_id"] == suggested["decision_id"]
    assert priced_count("WL-STAGE-1") == before

    asked = round(suggested["rate"] - 0.25, 2)
    made = client.post("/pricing/rate-requests", headers=rohit, json={
        "loan_id": "WL-STAGE-1", "rate": asked, "reason": "strong_negotiation", "note": "will walk"})
    assert made.status_code == 200, made.text
    assert listed(client, rohit, "WL-STAGE-1")["stage"] == "pending_rate_approval"
    # The manager sees the same stage: one place for everyone who looks.
    assert listed(client, meera, "WL-STAGE-1")["stage"] == "pending_rate_approval"

    approved = client.post(f"/pricing/rate-requests/{made.json()['request_id']}/decide", headers=meera,
                           json={"approve": True, "reason": "strong_negotiation", "note": "hold here"})
    assert approved.status_code == 200, approved.text
    row = listed(client, rohit, "WL-STAGE-1")
    assert row["stage"] == "offer_sent" and row["manager_approved_rate"] == asked

    # The customer answers on their side of the portal; staff cannot answer for them there.
    assert client.post("/applications/WL-STAGE-1/offer-decision", headers=rohit,
                       json={"accepted": True}).status_code == 403
    answered = client.post("/applications/WL-STAGE-1/offer-decision", headers=rahul, json={"accepted": True})
    assert answered.status_code == 200, answered.text
    assert answered.json()["offer"]["status"] == "accepted"
    assert listed(client, rohit, "WL-STAGE-1")["stage"] == "accepted"
    assert client.post("/applications/WL-STAGE-1/offer-decision", headers=rahul,
                       json={"accepted": True}).status_code == 409


def test_a_customer_who_declines_says_why(client) -> None:
    rahul, rohit = sign_in(client, "rahul"), sign_in(client, "rohit")
    quote_and_apply(client, rahul, "WL-DECLINE-1")
    verify(client, rahul, "WL-DECLINE-1")
    rate = listed(client, rohit, "WL-DECLINE-1")["suggested"]["rate"]
    assert client.post("/pricing/offers", headers=rohit,
                       json={"loan_id": "WL-DECLINE-1", "rate": rate}).status_code == 200
    declined = client.post("/applications/WL-DECLINE-1/offer-decision", headers=rahul,
                           json={"accepted": False, "loss_reason": "customer_deferred"})
    assert declined.status_code == 200, declined.text
    assert listed(client, rohit, "WL-DECLINE-1")["stage"] == "not_accepted"
    outcome = store.fetch_one("SELECT * FROM outcomes WHERE loan_id = 'WL-DECLINE-1'")
    assert outcome["loss_reason"] == "customer_deferred"


def test_stage_rules() -> None:
    sent = {"status": "sent"}
    assert worklist.stage("x", False, [], []) == "in_progress"
    assert worklist.stage("x", True, [], []) == "ready"
    assert worklist.stage("x", True, [], [{"status": "declined"}]) == "ready"
    assert worklist.stage("x", False, [], [{"status": "pending"}]) == "pending_rate_approval"
    assert worklist.stage("x", True, [sent], [{"status": "approved"}]) == "offer_sent"
    assert worklist.stage("x", True, [{"status": "superseded"}], []) == "ready"
    assert worklist.stage("x", True, [{"status": "accepted"}, sent], []) == "accepted"


def test_the_final_assessment_row_falls_back_to_declared_figures() -> None:
    row = worklist.stage2_row({"loan_id": "A-1", "declared_monthly_income": 50000,
                               "verified_monthly_income": "", "requested_loan_amount": 1500000,
                               "requested_tenure_years": 15, "name": "not sent",
                               "sales_context": {"competitor_rate": None, "competitor_name": "X"}})
    assert row["verified_monthly_income"] == 50000 and row["sanctioned_loan_amount"] == 1500000
    assert row["tenure_years"] == 15 and "name" not in row
    assert row["sales_context"] == {"competitor_name": "X"}


# ---------------------------------------------------------------- in progress

def test_in_progress_lists_what_is_due_and_sample_leads(client) -> None:
    rahul, rohit = sign_in(client, "rahul"), sign_in(client, "rohit")
    quote_and_apply(client, rahul, "WL-PROG-1")
    body = client.get("/applications/in-progress", headers=rohit).json()
    mine = next(i for i in body["items"] if i["ref"] == "WL-PROG-1")
    assert mine["sample"] is False and mine["detail"].startswith("Waiting for salary slips")
    samples = [i for i in body["items"] if i["sample"]]
    assert {s["stage"] for s in samples} >= {"quote_not_applied", "documents_pending", "dropped_off"}
    assert client.get("/applications/in-progress", headers=rahul).status_code == 403

    verify(client, rahul, "WL-PROG-1")
    body = client.get("/applications/in-progress", headers=rohit).json()
    assert not any(i["ref"] == "WL-PROG-1" for i in body["items"])


def test_the_rm_nudges_a_stalled_customer(client) -> None:
    rahul, rohit = sign_in(client, "rahul"), sign_in(client, "rohit")
    quote_and_apply(client, rahul, "WL-NUDGE-1")
    sent = client.post("/applications/nudges", headers=rohit, json={"ref": "WL-NUDGE-1", "channel": "whatsapp"})
    assert sent.status_code == 200, sent.text
    message = sent.json()["message"]
    assert message.startswith("Hi Rahul") and "salary slips" in message and "Rohit" in message

    lead = client.post("/applications/nudges", headers=rohit, json={"ref": "LEAD-S01", "channel": "sms"})
    assert lead.status_code == 200 and "9.65%–10.40%" in lead.json()["message"]

    item = next(i for i in client.get("/applications/in-progress", headers=rohit).json()["items"]
                if i["ref"] == "WL-NUDGE-1")
    assert item["nudges"] == 1 and item["last_nudge"]["channel"] == "whatsapp"
    assert client.post("/applications/nudges", headers=rahul, json={"ref": "WL-NUDGE-1"}).status_code == 403
    assert client.post("/applications/nudges", headers=rohit, json={"ref": "NOPE-1"}).status_code == 404
    assert client.post("/applications/nudges", headers=rohit,
                       json={"ref": "WL-NUDGE-1", "channel": "pigeon"}).status_code == 422


# ------------------------------------------------------------ area manager

def test_the_deep_dive_frames_the_decision(client) -> None:
    rahul, rohit, meera = sign_in(client, "rahul"), sign_in(client, "rohit"), sign_in(client, "meera")
    quote_and_apply(client, rahul, "WL-DIVE-1")
    verify(client, rahul, "WL-DIVE-1")
    rate = listed(client, rohit, "WL-DIVE-1")["suggested"]["rate"]
    request_id = client.post("/pricing/rate-requests", headers=rohit, json={
        "loan_id": "WL-DIVE-1", "rate": round(rate - 0.25, 2), "reason": "strong_negotiation"}).json()["request_id"]

    assert client.get(f"/pricing/rate-requests/{request_id}/deep-dive", headers=rohit).status_code == 403
    dive = client.get(f"/pricing/rate-requests/{request_id}/deep-dive", headers=meera)
    assert dive.status_code == 200, dive.text
    d = dive.json()
    assert d["breakeven_rate"] < d["requested_rate"] < d["recommended_rate"]
    assert d["ceiling_rate"] >= d["recommended_rate"]
    if d["ceiling_capped_by"] == "quoted_range":
        assert d["ceiling_rate"] <= d["quoted_max"] + 1e-9
    rates = [p["rate"] for p in d["curve"]]
    assert rates == sorted(rates) and rates[0] <= d["breakeven_rate"] and len(rates) > 20
    accept = [p["acceptance"] for p in d["curve"]]
    assert accept[0] >= accept[-1]
    parts = {c["label"] for c in d["build_up"]["components"]}
    assert {"Cost of funds", "Expected credit loss"} <= parts
    assert abs(sum(c["pct"] for c in d["build_up"]["components"]) - d["build_up"]["breakeven_rate"]) < 0.01


def test_disbursements_count_accepted_offers_against_target(client) -> None:
    rahul, rohit, meera = sign_in(client, "rahul"), sign_in(client, "rohit"), sign_in(client, "meera")
    before = client.get("/pricing/my/disbursements", headers=rohit).json()["rms"][0]
    quote_and_apply(client, rahul, "WL-DISB-1")
    verify(client, rahul, "WL-DISB-1")
    rate = listed(client, rohit, "WL-DISB-1")["suggested"]["rate"]
    client.post("/pricing/offers", headers=rohit, json={"loan_id": "WL-DISB-1", "rate": rate})
    client.post("/applications/WL-DISB-1/offer-decision", headers=rahul, json={"accepted": True})

    mine = client.get("/pricing/my/disbursements", headers=rohit).json()
    me = mine["rms"][0]
    assert me["username"] == "rohit" and me["target_inr"] > 0
    assert me["disbursed_inr"] == before["disbursed_inr"] + 2000000 and me["files"] == before["files"] + 1
    area = client.get("/pricing/area/disbursements", headers=meera).json()
    assert any(r["username"] == "rohit" for r in area["rms"]) and area["area"]["target_inr"] > 0
    assert client.get("/pricing/area/disbursements", headers=rohit).status_code == 403
    assert client.get("/pricing/my/disbursements", headers=meera).status_code == 403


# ------------------------------------------------------------- model cards

def test_the_model_cards_speak_in_business_terms(bundle) -> None:
    from aavas.analytics import segment_view

    segments = segment_view.segments()
    assert {s["segment"] for s in segments} >= {"salaried_formal", "self_employed_non_professional"}
    for s in segments:
        assert 0 < s["default_rate"] < 1 and abs(s["default_rate_annual"] - s["default_rate"] / 4) < 1e-3
        assert abs(s["expected_loss_pct"] - s["default_rate_annual"] * s["lgd"] * 100) < 0.01
    rates = [s["starting_rate"] for s in segments]
    assert rates == sorted(rates)

    curves = segment_view.acceptance_curves()
    for series in curves["series"]:
        accept = [p["acceptance"] for p in series["points"]]
        assert accept[0] > accept[-1], f"{series['label']}: a dearer loan cannot be likelier taken"
        assert series["discount_25bps_pts"] >= 0
    effect = segment_view.discount_effect()
    assert all(r["measured_low_pts"] <= r["measured_pts"] <= r["measured_high_pts"] for r in effect["rows"])
    floor = segment_view.floor_components()
    assert floor["cost_of_funds_pct"] > 0 and floor["operating_cost_pct"] > 0
