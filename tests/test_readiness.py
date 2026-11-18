"""A file is filled in step by step before it is priced and sent.

What must hold: the indicative step needs the customer's own figures and refuses
ones that cannot be right (a loan of 1250% of the property's value); the risk
step needs every verified figure and both checks, and a flagged check stops it;
the behaviour step needs each question answered either way (a competitor's quote
is optional); a later step is never
done while an earlier one is open; and nothing is sent, to the customer or the
area manager, from a stored file that is not complete.
"""

from __future__ import annotations

from aavas.intake import readiness

from test_applications import COMPLETE, applicant, client, quote_and_apply, sign_in  # noqa: F401


def full(**over) -> dict:
    a = {**applicant("RD-1"), **COMPLETE}
    sales = {**a["sales_context"], **over.pop("sales_context", {})}
    return {**a, **over, "sales_context": sales}


def step(result, key) -> dict:
    return next(s for s in result["steps"] if s["key"] == key)


def test_a_complete_file_is_ready_to_send() -> None:
    result = readiness.check(full())
    assert result["complete"] and result["next"] is None
    assert readiness.refusal(full(), "sending an offer") is None


def test_figures_that_cannot_be_right_stop_the_indicative_step() -> None:
    """The file that prompted this: ₹50 lakh asked against a ₹4 lakh property."""
    result = readiness.check(full(requested_loan_amount=5_000_000, estimated_property_value=400_000,
                                  declared_monthly_income=5_000_000, declared_obligations=20))
    first = step(result, "indicative")
    assert not first["done"] and any("1250%" in p and "₹50,00,000" in p for p in first["problems"])
    assert any("monthly figure" in w for w in first["warnings"])
    assert step(result, "risk")["locked"] and not step(result, "risk")["done"]
    assert result["next"] == "indicative"


def test_risk_pricing_waits_for_every_verified_figure() -> None:
    a = applicant("RD-2")
    result = readiness.check(a)
    assert step(result, "indicative")["done"]
    risk = step(result, "risk")
    assert not risk["done"] and len(risk["missing"]) == 10
    assert {"Legal check", "Banking behaviour", "Verified monthly income"} <= {m["label"] for m in risk["missing"]}
    assert result["next"] == "risk" and result["due"] == 10


def test_a_flagged_check_or_an_impossible_loan_stops_risk_pricing() -> None:
    assert any("legal check flagged" in p for p in step(readiness.check(full(legal_flag=1)), "risk")["problems"])
    over = step(readiness.check(full(sanctioned_loan_amount=2_800_000, appraised_property_value=2_900_000)), "risk")
    assert not over["done"] and any("97%" in p for p in over["problems"])


def test_behaviour_questions_need_an_answer_either_way() -> None:
    blank = full(sales_context={"customer_negotiating": None, "existing_relationship": None}, conversation_notes="")
    labels = {m["label"] for m in step(readiness.check(blank), "behaviour")["missing"]}
    assert labels == {"Is the customer negotiating?", "Existing Aavas customer?", "Notes from the conversation"}
    answered_no = full(sales_context={"customer_negotiating": False, "prior_negotiation_rounds": None,
                                      "existing_relationship": False})
    assert step(readiness.check(answered_no), "behaviour")["done"], "a no is an answer"
    rounds = step(readiness.check(full(sales_context={"prior_negotiation_rounds": 0})), "behaviour")
    assert [m["field"] for m in rounds["missing"]] == ["prior_negotiation_rounds"]


def test_a_competitor_quote_is_optional() -> None:
    none = full(sales_context={"competitor_rate": None, "competitor_name": None, "competitor_quote": None,
                               "competitor_evidence_verified": False})
    assert readiness.check(none)["complete"]
    rate_only = full(sales_context={"competitor_name": None})
    assert readiness.check(rate_only)["complete"], "a quoted rate without the lender's name is still fine"


def test_staff_read_the_steps_and_a_borrower_does_not(client) -> None:
    body = client.post("/applications/readiness", headers=sign_in(client, "rohit"),
                       json={"applicant": applicant("RD-3")})
    assert body.status_code == 200 and body.json()["next"] == "risk"
    assert client.post("/applications/readiness", headers=sign_in(client, "rahul"),
                       json={"applicant": applicant("RD-3")}).status_code == 403


def test_an_incomplete_file_does_not_go_to_the_area_manager(client) -> None:
    rahul, rohit = sign_in(client, "rahul"), sign_in(client, "rohit")
    quote_and_apply(client, rahul, "RD-AM-1")
    priced = client.post("/pricing/final-assessment", headers=rohit, json={
        k: v for k, v in {**applicant("RD-AM-1"), **COMPLETE}.items() if k not in ("name", "conversation_notes")})
    rate = round(priced.json()["recommendation"]["recommended_rate"] - 0.3, 2)
    ask = {"loan_id": "RD-AM-1", "rate": rate, "reason": "competitor_lower_rate"}
    refused = client.post("/pricing/rate-requests", headers=rohit, json=ask)
    assert refused.status_code == 422 and "Compute loan risk still needs" in refused.json()["detail"]
    assert client.put("/applications/RD-AM-1", headers=rohit, json={"applicant": COMPLETE}).status_code == 200
    assert client.post("/pricing/rate-requests", headers=rohit, json=ask).status_code == 200


def test_the_area_manager_sees_the_indicative_range_beside_the_risk_adjusted_rate(client) -> None:
    rahul, rohit = sign_in(client, "rahul"), sign_in(client, "rohit")
    quote_and_apply(client, rahul, "RD-AM-2")
    assert client.put("/applications/RD-AM-2", headers=rohit, json={"applicant": COMPLETE}).status_code == 200
    priced = client.post("/pricing/final-assessment", headers=rohit, json={
        k: v for k, v in {**applicant("RD-AM-2"), **COMPLETE}.items() if k not in ("name", "conversation_notes")})
    rec = priced.json()["recommendation"]["recommended_rate"]
    made = client.post("/pricing/rate-requests", headers=rohit, json={
        "loan_id": "RD-AM-2", "rate": round(rec - 0.4, 2), "reason": "strong_negotiation"})
    assert made.status_code == 200, made.text
    queue = client.get("/pricing/rate-requests?status=pending", headers=sign_in(client, "meera")).json()["requests"]
    band = next(r for r in queue if r["loan_id"] == "RD-AM-2")["analysis"]["indicative_band"]
    assert band and 5 < band["min_rate"] <= band["max_rate"] < 20
