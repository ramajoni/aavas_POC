"""Fair Practices Code: the Key Facts Statement, the written reasons for a rejection,
and the register evidence that decides its own coverage.

What must hold: the APR is the all-in cost on the amount actually received (RBI's own
illustration is reproduced); a KFS binds for three working days and is not replaced
while it does; no pre-payment charge on a floating-rate loan to an individual for
non-business purposes whatever the configuration says; the EMI never falls below the
interest on a benchmark rise; a declined applicant gets reasons in plain words; and
only the people who handle the offer can give its KFS.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from aavas.api import access
from aavas.api.main import app
from aavas.governance import decision_logger, regulatory, store
from aavas.intake import fair_practices, kfs

PASSWORDS = {"rohit": "Rohit@2026", "meera": "Meera@2026", "arjun": "Arjun@2026"}


def _offer(loan_id: str, *, rate: float = 14.0, amount: float = 25_00_000, years: float = 10,
           sent_by: str = "rohit", status: str = "sent") -> dict:
    payload = {"loan_id": loan_id, "stage": "post_assessment", "decision": "price",
               "application_snapshot": {"sanctioned_loan_amount": amount, "tenure_years": years},
               "recommendation": {"recommended_rate": rate}}
    decision_id = decision_logger.log_decision(loan_id, "post_assessment", payload)
    row = {"offer_id": f"OFR-T{loan_id}", "loan_id": loan_id, "decision_id": decision_id, "rate": rate,
           "recommended_rate": rate, "route": "within_authority", "reason": None, "note": None,
           "request_id": None, "sent_by": sent_by, "sent_at": store.now(), "status": status,
           "customer_decided_at": None}
    store.insert("offers", row)
    return row


# ------------------------------------------------------------- the arithmetic

def test_the_apr_reproduces_rbis_illustration() -> None:
    # RBC Directions para 29: ₹20,000 over 24 EMIs of ₹970 at an APR of 17.07%.
    assert kfs.rupees(kfs.emi(20_000, 15, 24)) == 970
    out = kfs.apr(20_000, 15, 24, upfront=400)
    assert abs(out["apr_pct"] - 17.07) <= 0.02, out
    assert out["effective_annual_pct"] > out["apr_pct"]          # compounding shown, not hidden


def test_charges_raise_the_apr_and_none_leave_it_at_the_rate() -> None:
    bare = kfs.apr(25_00_000, 14, 120, upfront=0)["apr_pct"]
    assert abs(bare - 14) < 0.02
    assert kfs.apr(25_00_000, 14, 120, upfront=12_500)["apr_pct"] > bare


def test_the_schedule_repays_exactly_the_loan() -> None:
    rows = kfs.schedule(25_00_000, 14, 120)
    assert len(rows) == 120 and rows[-1]["closing"] == 0
    assert sum(r["principal"] for r in rows) == 25_00_000
    assert all(r["emi"] == r["interest"] + r["principal"] for r in rows)


def test_rupees_round_half_up() -> None:                          # RBC para 103
    assert kfs.rupees(10.5) == 11 and kfs.rupees(10.49) == 10


def test_a_benchmark_rise_never_leaves_the_emi_below_the_interest() -> None:
    ten_years = kfs.benchmark_impact(10_00_000, 14, 120, [25, 100])
    assert 120 < ten_years[0]["months_if_emi_unchanged"] < ten_years[1]["months_if_emi_unchanged"]
    assert ten_years[0]["note"] is None
    # Over thirty years the EMI is nearly all interest: even +25 bps would leave it short.
    thirty = kfs.benchmark_impact(10_00_000, 14, 360, [25])
    assert thirty[0]["months_if_emi_unchanged"] is None and thirty[0]["note"]
    assert thirty[0]["emi_if_tenure_unchanged"] > kfs.rupees(kfs.emi(10_00_000, 14, 360))


def test_validity_counts_working_days() -> None:
    assert kfs.working_days_after(date(2026, 11, 20), 3) == date(2026, 11, 25)   # Fri -> Wed


def test_no_prepayment_charge_on_a_floating_individual_non_business_loan() -> None:
    conf = {"charges": {"prepayment": {"floating_individual_non_business_pct": 2, "other_pct": 2}}}
    assert kfs.prepayment_pct(conf, rate_type="floating", borrower="individual", purpose="non_business") == 0
    assert kfs.prepayment_pct(conf, rate_type="fixed", borrower="individual", purpose="non_business") == 2


# ------------------------------------------------------------ issuing a KFS

def test_a_kfs_binds_for_its_validity_and_is_not_replaced() -> None:
    offer = _offer("KFS-1")
    first = kfs.issue(offer["offer_id"], issued_by="rohit")
    body = first["payload"]
    assert body["part_1"]["apr"]["apr_pct"] > body["part_1"]["interest"]["annualised_rate_pct"]
    assert body["part_1"]["contingent_charges"]["prepayment_pct"] == 0
    assert body["illustrative"] and body["banner"]               # charges not yet board-approved
    assert "models helped" in body["part_2"]["use_of_models"]
    assert len(body["repayment_schedule"]) == 120
    again = kfs.issue(offer["offer_id"], issued_by="rohit")
    assert again["kfs_id"] == first["kfs_id"]
    acked = kfs.acknowledge(first["kfs_id"], recorded_by="rohit")
    assert acked["acknowledged_at"]


def test_a_kfs_is_only_given_for_a_live_offer() -> None:
    offer = _offer("KFS-2", status="superseded")
    with pytest.raises(kfs.KFSError) as err:
        kfs.issue(offer["offer_id"], issued_by="rohit")
    assert err.value.status == 409


def test_a_kfs_in_another_language_is_marked_for_translation() -> None:
    offer = _offer("KFS-3")
    hindi = kfs.issue(offer["offer_id"], issued_by="rohit", language="hi")
    assert hindi["payload"]["translation_required"]
    with pytest.raises(kfs.KFSError):
        kfs.issue(offer["offer_id"], issued_by="rohit", language="xx")


# ------------------------------------------------------------- rejection

def test_a_declined_applicant_gets_reasons_in_plain_words() -> None:
    reason = "bureau score 640 is below the policy minimum of 675: declined under standard policy"
    decision_logger.log_decision("REJ-1", "post_assessment", {
        "loan_id": "REJ-1", "stage": "post_assessment", "decision": "decline",
        "policy": {"allowed": False, "violations": [reason], "policy_version": "HL-LAP-POLICY-2026.14"}})
    note = fair_practices.rejection_note("REJ-1", requested_by="rohit")
    assert note["in_writing"] and note["reasons"][0]["category"] == "bureau"
    assert "675" not in note["reasons"][0]["text"]               # words for the applicant, not policy internals
    assert "grievance" in note["what_next"]


def test_a_priced_file_has_no_rejection_to_explain() -> None:
    _offer("REJ-2")
    with pytest.raises(fair_practices.NoDecline):
        fair_practices.rejection_note("REJ-2", requested_by="rohit")


# ------------------------------------------------------- register evidence

def test_a_protected_ground_among_the_inputs_is_a_gap() -> None:
    assert regulatory.EVIDENCE["protected_attributes"]({"protected_hits": []})["coverage"] == "in_place"
    hit = regulatory.EVIDENCE["protected_attributes"]({"protected_hits": ["risk: applicant_gender"]})
    assert hit["coverage"] == "gap" and hit["attention"]


def test_a_decline_without_a_reason_is_a_gap() -> None:
    f = {"declined": [{"policy": {"violations": ["x"]}}, {"policy": {"violations": []}}], "rejection_notes": 0}
    assert regulatory.EVIDENCE["rejection_reasons"](f)["coverage"] == "gap"


def test_offers_without_any_kfs_need_attention() -> None:
    f = {"kfs": {"issued": 0, "offers_with_kfs": 0, "acknowledged": 0}, "offers_sent": 3,
         "fair_practices": {"approval": {"approved": True}}}
    out = regulatory.EVIDENCE["kfs"](f)
    assert out["attention"] and out["coverage"] == "in_place"


def test_the_register_carries_the_fair_practices_rows() -> None:
    ids = {a["id"] for a in regulatory.register()["artefacts"]}
    assert {"kfs", "penal-charges", "prepayment", "floating-reset", "rejection-and-fairness",
            "borrower-commitments", "model-risk", "free-ai"} <= ids


# ------------------------------------------------------------- who may

@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def _who(client, name) -> dict[str, str]:
    token = client.post("/auth/login", json={"username": name, "password": PASSWORDS[name]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_the_kfs_is_given_by_the_offers_rm_or_the_area_manager(client) -> None:
    mine, theirs = _offer("API-1", sent_by="rohit"), _offer("API-2", sent_by="someone_else")
    rm, am, mv = _who(client, "rohit"), _who(client, "meera"), _who(client, "arjun")
    assert client.post(f"/pricing/offers/{mine['offer_id']}/kfs", json={}, headers=rm).status_code == 200
    assert client.post(f"/pricing/offers/{theirs['offer_id']}/kfs", json={}, headers=rm).status_code == 403
    assert client.post(f"/pricing/offers/{theirs['offer_id']}/kfs", json={}, headers=am).status_code == 200
    assert client.post(f"/pricing/offers/{mine['offer_id']}/kfs", json={}, headers=mv).status_code == 403


def test_a_rejection_note_is_for_the_rm_who_logged_the_file(client) -> None:
    decision_logger.log_decision("REJ-API", "post_assessment", {
        "loan_id": "REJ-API", "stage": "post_assessment", "decision": "decline",
        "policy": {"allowed": False, "violations": ["risk grade D has no offer under standard policy"]}})
    rm, am = _who(client, "rohit"), _who(client, "meera")
    assert client.get("/pricing/rejection-note/REJ-API", headers=rm).status_code == 403
    note = client.get("/pricing/rejection-note/REJ-API", headers=am)
    assert note.status_code == 200, note.text
    assert "internal_reasons" not in note.json() and "risk grade D" not in note.text
