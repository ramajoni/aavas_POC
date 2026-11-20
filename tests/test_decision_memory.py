"""Decision memory: the area manager's decisions, remembered and recalled.

What must hold:

* a decision is remembered once, with the factors the manager saw, the suggestion
  they were shown and the past cases recalled for them;
* the next like request recalls it, with what the customer then did;
* memory never moves the suggestion;
* only the learning agent writes memory, and a memory failure never blocks a decision;
* memory keeps files, not people - no name, no age;
* the screen and memory read one set of factor signals.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aavas.agents import capabilities
from aavas.analytics import decision_factors, decision_recall
from aavas.api import access
from aavas.api.main import app
from aavas.governance import store
from aavas.learning import decision_memory

ROOT = Path(__file__).resolve().parents[1]
PASSWORDS = {"rohit": "Rohit@2026", "meera": "Meera@2026", "admin": "Admin@2026"}
# The approval tests' file (tests/test_offers.py): a rival above the card leaves room
# between the recommendation and the floor for a cut the area manager may approve.
FILE = {"segment": "salaried_formal", "city": "Jaipur", "channel": "branch_walkin",
        "property_type": "resale_flat", "declared_monthly_income": 120000,
        "requested_loan_amount": 2000000, "requested_tenure_years": 10, "applicant_age": 35,
        "declared_obligations": 5000, "estimated_property_value": 2900000, "bureau_score": 790,
        "sales_context": {"competitor_rate": 15.5, "competitor_evidence_verified": False},
        "verified_monthly_income": 118000, "appraised_property_value": 2900000,
        "sanctioned_loan_amount": 2000000, "tenure_years": 10, "bureau_vintage_months": 96,
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


def ask_lower(client, loan_id: str, below: float = 0.40, reason: str = "strong_negotiation") -> str:
    """Price the file as the RM and ask the area manager for a lower rate."""
    rohit = who(client, "rohit")
    priced = client.post("/pricing/final-assessment?include_scenarios=true",
                         json={**FILE, "loan_id": loan_id}, headers=rohit)
    assert priced.status_code == 200, priced.text
    rec = priced.json()["recommendation"]["recommended_rate"]
    made = client.post("/pricing/rate-requests", headers=rohit, json={
        "loan_id": loan_id, "rate": round(rec - below, 2), "reason": reason, "note": "will walk away"})
    assert made.status_code == 200, made.text
    return made.json()["request_id"]


def episode(request_id: str) -> dict:
    rows = [r for r in decision_memory.episodes() if r["request_id"] == request_id]
    assert len(rows) == 1, f"expected one episode for {request_id}, found {len(rows)}"
    return rows[0]


# --------------------------------------------------------------- remembering

def test_a_decision_is_remembered_with_what_the_manager_saw(client) -> None:
    request_id = ask_lower(client, "MEM-SEEN-1")
    meera = who(client, "meera")
    before = client.get(f"/pricing/rate-requests/{request_id}/memory", headers=meera)
    assert before.status_code == 200, before.text
    seen = before.json()
    assert [f["key"] for f in seen["factors"]] == [k for _, k, _ in decision_factors.FACTORS]
    assert all(f["signal"] in decision_factors.SIGNALS for f in seen["factors"])
    assert seen["memory"]["available"] and seen["memory"]["moves_suggestion"] is False

    decided = client.post(f"/pricing/rate-requests/{request_id}/decide", headers=meera,
                          json={"approve": True, "reason": "strong_negotiation", "note": "close it at this"})
    assert decided.status_code == 200, decided.text
    assert decided.json()["memory"]["recorded"] is True

    ep = episode(request_id)
    assert ep["manager"] == "meera" and ep["action"] == "approve" and ep["note"] == "close it at this"
    assert [f["signal"] for f in ep["factors"]] == [f["signal"] for f in seen["factors"]]
    assert ep["suggested_action"] == (seen["suggestion"] or {}).get("action")
    assert ep["followed_suggestion"] == decision_memory.followed(seen["suggestion"], "approve", ep["decided_rate"])
    assert ep["format"] == decision_memory.FORMAT


def test_memory_keeps_files_not_people() -> None:
    """Nothing in a case lets memory sort people: no name, no age, no identity."""
    case = decision_factors.case(
        {"requested_rate": 15.0, "recommended_rate": 15.4, "reason": "strong_negotiation", "analysis": {}},
        {"risk": {"risk_grade": "B"}, "application_snapshot": {**FILE, "applicant_name": "A Person"}},
        {**FILE, "applicant_name": "A Person"})
    assert not {"applicant_age", "applicant_name", "loan_id", "gender"} & set(case)
    assert not {"applicant_age", "applicant_name"} & set(decision_recall.WEIGHTS)


def test_an_episode_is_never_replaced() -> None:
    request = {"request_id": "REQ-MEM-ONCE", "loan_id": "MEM-ONCE", "decision_id": None,
               "requested_rate": 14.6, "recommended_rate": 15.0, "reason": "strong_negotiation"}
    decided = {"status": "declined", "decided_by": "meera", "decided_at": store.now(),
               "decision_reason": "hold_recommended_rate", "decision_note": "first"}
    first = decision_memory.remember(request=request, decided=decided, observation=None, unavailable="test")
    again = decision_memory.remember(request=request, decided={**decided, "decision_note": "second"},
                                     observation=None)
    assert first["recorded"] and not again["recorded"] and again["episode_id"] == first["episode_id"]
    assert episode("REQ-MEM-ONCE")["note"] == "first"
    assert episode("REQ-MEM-ONCE")["factors_unavailable"] == "test"


def test_a_decision_stands_when_memory_cannot_be_written(client, monkeypatch) -> None:
    request_id = ask_lower(client, "MEM-FAIL-1")

    def broken(**_):
        raise RuntimeError("disk full")
    monkeypatch.setattr(decision_memory, "remember", broken)
    decided = client.post(f"/pricing/rate-requests/{request_id}/decide", headers=who(client, "meera"),
                          json={"approve": False, "note": "hold the recommended rate"})
    assert decided.status_code == 200, decided.text
    assert decided.json()["status"] == "declined"
    assert decided.json()["memory"]["recorded"] is False


# ------------------------------------------------------------------ recalling

def test_the_next_like_request_recalls_it_and_what_the_customer_did(client) -> None:
    rohit, meera = who(client, "rohit"), who(client, "meera")
    first = ask_lower(client, "MEM-LIKE-1")
    decided = client.post(f"/pricing/rate-requests/{first}/decide", headers=meera,
                          json={"approve": True, "reason": "strong_negotiation", "note": "worth keeping"}).json()
    answered = client.post(f"/pricing/offers/{decided['offer']['offer_id']}/customer-decision", headers=rohit,
                           json={"accepted": True})
    assert answered.status_code == 200, answered.text

    second = ask_lower(client, "MEM-LIKE-2")
    memory = client.get(f"/pricing/rate-requests/{second}/memory", headers=meera).json()["memory"]
    recalled = {c["request_id"]: c for c in memory["yours"]["cases"]}
    assert first in recalled, f"{first} not recalled: {list(recalled)}"
    card = recalled[first]
    assert card["outcome"]["state"] == "accepted" and "same reason" in card["matched"]
    assert card["note"] == "worth keeping" and card["decided"]["action"] == "approve"
    assert memory["profile"]["decisions"] >= 1 and memory["profile"]["lines"]
    assert second not in recalled, "a request must not recall itself"

    # The decision on the second request records which past cases were in front of it.
    client.post(f"/pricing/rate-requests/{second}/decide", headers=meera,
                json={"approve": False, "note": "not this time"})
    assert recalled[first]["episode_id"] in episode(second)["recalled"]


def test_another_managers_decisions_are_recalled_as_theirs(client) -> None:
    mine = ask_lower(client, "MEM-OTHER-1")
    client.post(f"/pricing/rate-requests/{mine}/decide", headers=who(client, "admin"),
                json={"approve": False, "note": "held by the administrator"})
    later = ask_lower(client, "MEM-OTHER-2")
    memory = client.get(f"/pricing/rate-requests/{later}/memory", headers=who(client, "meera")).json()["memory"]
    assert mine in {c["request_id"] for c in memory["others"]["cases"]}
    assert mine not in {c["request_id"] for c in memory["yours"]["cases"]}


def test_memory_never_moves_the_suggestion(client) -> None:
    meera = who(client, "meera")
    first, second = ask_lower(client, "MEM-STILL-1"), ask_lower(client, "MEM-STILL-2")
    before = client.get(f"/pricing/rate-requests/{second}/suggestion", headers=meera).json()
    client.post(f"/pricing/rate-requests/{first}/decide", headers=meera,
                json={"approve": False, "note": "hold"})
    after = client.get(f"/pricing/rate-requests/{second}/suggestion", headers=meera).json()
    assert {k: before[k] for k in ("action", "rate", "headline")} == {k: after[k] for k in ("action", "rate", "headline")}


def test_the_rm_cannot_read_memory(client) -> None:
    request_id = ask_lower(client, "MEM-RM-1")
    assert client.get(f"/pricing/rate-requests/{request_id}/memory", headers=who(client, "rohit")).status_code == 403


# ---------------------------------------------------------------- permissions

def test_only_the_learning_agent_writes_memory() -> None:
    assert set(capabilities.writers()) == {"learning"}
    assert "procedure.decision_memory" in capabilities.writers()["learning"]
    assert capabilities.permissions_for("manager_memory") == {"procedure.decision_memory": [capabilities.READ]}


def test_the_screen_reads_the_same_factors_memory_records() -> None:
    """Every signal row on the decision-support screen is one memory records, and the
    screen takes its signal from the server. The two tool rows carry no signal."""
    jsx = (ROOT / "application" / "src" / "DecisionSupport.jsx").read_text(encoding="utf-8")
    # A factor row is [group, key, status, ...]; a link target is the two-element [section, tab].
    on_screen = set(re.findall(r"\[\s*'(?:need|cost|branch)',\s*'(\w+)',", jsx)) - {"likeforlike", "alternatives"}
    assert on_screen == {k for _, k, _ in decision_factors.FACTORS}
    assert "/memory" in jsx and "server[key]" in jsx


# --------------------------------------------------------------- pure readings

def _ep(n: int, action: str, signal: str, outcome: str = "accepted", suggested: str | None = "hold") -> dict:
    return {"episode_id": f"E{n}", "request_id": f"R{n}", "loan_id": f"L{n}", "manager": "meera",
            "decided_at": f"2026-11-{n:02d}", "action": action, "requested_rate": 14.6, "recommended_rate": 15.0,
            "decided_rate": 14.6 if action != "hold" else 15.0, "suggested_action": suggested, "suggested_rate": 15.0,
            "followed_suggestion": suggested == action, "case_features": {"reason": "strong_negotiation"},
            "factors": [{"key": "competitor", "signal": signal}], "outcome": {"state": outcome}}


def test_a_case_is_most_like_itself_and_less_like_a_different_reason() -> None:
    case = {"reason": "strong_negotiation", "discount_bps": 40, "risk_grade": "B", "segment": "salaried_formal"}
    same, matched = decision_recall.similarity(case, dict(case))
    other, _ = decision_recall.similarity(case, {**case, "reason": "competitor_lower_rate"})
    assert same == 1.0 and "same reason" in matched and other < same


def test_the_factor_that_moves_a_manager_is_named_with_its_counts() -> None:
    eps = [_ep(1, "approve", "for"), _ep(2, "approve", "for"), _ep(3, "counter", "for"),
           _ep(4, "hold", "verify"), _ep(5, "hold", "verify")]
    lean = decision_recall.tendencies(eps)
    assert lean and lean[0]["factor"] == "competitor"
    assert lean[0]["cut_when_supports"] == [3, 3] and lean[0]["cut_when_argues"] == [0, 2]
    assert "3 of 3" in lean[0]["text"] and "0 of 2" in lean[0]["text"]


def test_too_little_evidence_names_no_habit() -> None:
    assert decision_recall.tendencies([_ep(1, "approve", "for"), _ep(2, "hold", "verify")]) == []
    assert decision_recall.consistency([decision_recall.card(_ep(1, "approve", "for"), 1.0, [])],
                                       {"action": "hold"}) is None


def test_a_suggestion_against_the_usual_decision_is_named() -> None:
    cards = [decision_recall.card(_ep(n, "approve", "for"), 0.9, []) for n in (1, 2, 3)]
    against = decision_recall.consistency(cards, {"action": "hold", "rate": 15.0})
    assert against["tone"] == "warn" and against["usual"] == "cut" and "3 of 3" in against["text"]
    agrees = decision_recall.consistency(cards, {"action": "counter", "rate": 14.8})
    assert agrees["tone"] == "flat"


def test_the_profile_counts_rather_than_rates() -> None:
    eps = [_ep(1, "approve", "for", suggested="approve"), _ep(2, "hold", "verify", "not_accepted"),
           _ep(3, "approve", "for", suggested="hold")]
    prof = decision_recall.profile(eps)
    assert prof["actions"] == {"cut": 2, "hold": 1, "uplift": 0}
    assert prof["suggestion_taken"] == [2, 3] and prof["departed_lower"] == 1
    assert prof["cuts_accepted"] == [2, 2] and prof["holds_accepted"] == [0, 1]
    assert not any("%" in line for line in prof["lines"])


def test_factor_signals_follow_the_screen_thresholds() -> None:
    req = {"requested_rate": 15.0, "recommended_rate": 15.4, "evidence_verified": False,
           "analysis": {"acceptance_at_requested": 0.62, "acceptance_at_recommended": 0.55,
                        "cost_full_term": 600000, "breakeven_rate": 14.6}}
    rows = {f["key"]: f for f in decision_factors.factors(
        req, {"risk": {"risk_grade": "A"}}, {**FILE}, None, None)}
    assert rows["response"]["value"] == 7 and rows["response"]["signal"] == "for"
    assert rows["interest"]["signal"] == "against"
    assert rows["margin"]["value"] == 0.4 and rows["margin"]["signal"] == "against"
    assert rows["competitor"]["signal"] == "verify"           # a quote with no letter seen
    assert rows["credit"]["signal"] == "clear"
    assert rows["track"]["signal"] == "neutral"               # no briefing: absent, not omitted
