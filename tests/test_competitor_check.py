"""The online competitor check: the market research agent and the claim score.

What must hold: a quote is scored differently with and without a sanction letter,
and against what the lender advertises, in stated points; a rate the agent cannot
tie to a page its search returned is never used; a lender is searched once and
reused; nothing works without signing in as someone who prices a file; and with
no search at all the claim still gets a score on its evidence alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aavas.agents.market_research_agent import (MARKET, MarketResearchAgent, ResearchFailed,
                                                summarise)
from aavas.api import access
from aavas.api.main import app
from aavas.config import policy
from aavas.governance import store
from aavas.intelligence import claim_score, market_rates

PASSWORDS = {"rohit": "Rohit@2026", "meera": "Meera@2026", "rahul": "Rahul@2026"}
OFFICIAL = {"published_min": 8.00, "published_max": 9.50, "source_quality": "lender_website",
            "best_rate_min_credit_score": None}
UNEVIDENCED = int(policy()["override"]["max_unevidenced_discount_bps"])


def points(result: dict) -> dict[str, int]:
    return {c["label"]: c["points"] for c in result["components"]}


# ------------------------------------------------------------------ the score

def test_a_letter_seen_and_a_rate_the_lender_advertises_is_credible() -> None:
    r = claim_score.score(8.6, True, {"bureau_score": 790}, OFFICIAL, lender_label="HDFC")
    assert (r["points"], r["band"], r["market_position"]) == (85, "credible", "within")
    assert r["max_discount_bps"] is None, "a believable, evidenced quote has no claim-specific cap"


def test_the_same_quote_without_the_letter_scores_lower_and_is_capped() -> None:
    r = claim_score.score(8.6, False, {"bureau_score": 790}, OFFICIAL, lender_label="HDFC")
    assert (r["points"], r["band"]) == (60, "plausible")
    assert r["max_discount_bps"] == UNEVIDENCED
    assert "Ask for it" in r["guidance"]


def test_a_quote_under_the_advertised_floor_loses_points_by_how_far() -> None:
    seen = claim_score.score(7.5, True, {"bureau_score": 790}, OFFICIAL, lender_label="HDFC")
    assert seen["gap_to_floor_bps"] == 50 and seen["band"] == "unconfirmed"
    assert "Confirm it is for this loan" in seen["guidance"]
    unseen = claim_score.score(7.6, False, {"bureau_score": 790}, OFFICIAL, lender_label="HDFC")
    assert unseen["band"] == "doubtful" and unseen["max_discount_bps"] == 0
    assert "cannot justify a lower rate" in unseen["guidance"]


def test_with_nothing_found_by_the_fact_check_the_claim_rests_on_its_evidence() -> None:
    """No online data must leave today's behaviour unchanged: an unverified quote
    keeps the policy allowance, an evidenced one keeps no claim-specific cap."""
    unseen = claim_score.score(8.6, False, {"bureau_score": 790}, None)
    assert (unseen["points"], unseen["max_discount_bps"]) == (35, UNEVIDENCED)
    seen = claim_score.score(8.6, True, {"bureau_score": 790}, None)
    assert (seen["points"], seen["max_discount_bps"]) == (70, None)


def test_the_lenders_best_rate_needs_a_profile_that_could_get_it() -> None:
    strong = claim_score.score(8.05, False, {"bureau_score": 800}, OFFICIAL, lender_label="HDFC")
    weak = claim_score.score(8.05, False, {"bureau_score": 690}, OFFICIAL, lender_label="HDFC")
    assert points(weak)["Customer profile"] == -15 and "690" in weak["components"][-1]["detail"]
    assert weak["points"] == strong["points"] - 15
    away_from_floor = claim_score.score(9.0, False, {"bureau_score": 690}, OFFICIAL)
    assert "Customer profile" not in points(away_from_floor), "only a quote at the floor is the best rate"


def test_rates_from_comparison_sites_count_for_less() -> None:
    secondary = {**OFFICIAL, "source_quality": "secondary"}
    r = claim_score.score(8.6, True, {"bureau_score": 790}, secondary)
    assert points(r)["Fact check against advertised rates"] == round(15 * 0.6)


# ------------------------------------------------------------------ the agent

def block(kind: str, **fields):
    return SimpleNamespace(type=kind, model_dump=lambda: {"type": kind, **fields}, **fields)


def findings(*rates, found=True):
    return {"found": found, "rates": list(rates), "best_rate_min_credit_score": 750, "notes": "n"}


def rate(url, low, high=None, kind="lender_website"):
    return {"lender": "HDFC", "min_rate": low, "max_rate": high, "applies_to": "salaried",
            "as_of": "", "source_url": url, "source_type": kind}


class FakeClient:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def response(stop, *content):
    return SimpleNamespace(stop_reason=stop, content=list(content), model="claude-opus-5-5",
                           usage=SimpleNamespace(server_tool_use=SimpleNamespace(web_search_requests=2)))


def test_the_agent_keeps_only_rates_tied_to_a_page_its_search_returned() -> None:
    searched = block("web_search_tool_result",
                     content=[{"type": "web_search_result", "url": "https://www.hdfc.bank.in/rates/"}])
    recorded = block("tool_use", name="record_findings", input=findings(
        rate("https://hdfc.bank.in/rates", 8.0, 9.5),
        rate("https://made-up.example/rates", 6.1),          # never returned by a search
        rate("https://hdfc.bank.in/rates", 1.5)))            # not a home-loan rate
    client = FakeClient(response("pause_turn", searched), response("tool_use", recorded))
    out = MarketResearchAgent(client=client).research("HDFC", ["HDFC Bank"])

    assert len(client.calls) == 2, "a paused turn is resumed, not abandoned"
    assert client.calls[1]["messages"][-1]["role"] == "assistant"
    assert client.calls[0]["fallbacks"] == "default"
    assert out["status"] == "ok" and (out["published_min"], out["published_max"]) == (8.0, 9.5)
    assert [r["source_seen"] for r in out["rates"]] == [True, False]
    assert out["source_quality"] == "lender_website" and out["searches"] == 4


def test_a_declined_search_fails_rather_than_guessing() -> None:
    client = FakeClient(response("refusal"))
    with pytest.raises(ResearchFailed):
        MarketResearchAgent(client=client).research("HDFC")


def test_without_a_key_the_online_check_is_unavailable(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = MarketResearchAgent(env_file=tmp_path / "missing.env")
    assert agent.status()["available"] is False


def test_nothing_found_is_reported_as_not_found() -> None:
    out = summarise("HDFC", findings(found=False), set(), 3, "m")
    assert out["status"] == "not_found" and out["published_min"] is None


# --------------------------------------------------------------------- the API

class StandIn(MarketResearchAgent):
    """The real scoring, a canned search."""

    def __init__(self, result=None, error=None, available=True):
        super().__init__(client=object())
        self.result, self.error, self.available, self.calls = result, error, available, []

    def status(self):
        return {"available": self.available, "reason": None if self.available else "not set up",
                "model": "m"}

    def research(self, lender, aliases=None, today=None):
        self.calls.append(lender)
        if self.error:
            raise ResearchFailed(self.error)
        return {**self.result, "lender": lender}


FOUND = {"status": "ok", "published_min": 8.0, "published_max": 9.5,
         "source_quality": "lender_website", "best_rate_min_credit_score": None, "notes": "",
         "rates": [{**rate("https://hdfc.bank.in/rates", 8.0, 9.5), "source_seen": True}],
         "searches": 2}


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    monkeypatch.setattr(market_rates, "_submit", lambda job, *args: job(*args))   # inline
    with store.connect() as conn:
        conn.execute(f"DELETE FROM {market_rates.TABLE}")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()
    market_rates.set_agent(None)


def who(client, name) -> dict[str, str]:
    token = client.post("/auth/login", json={"username": name, "password": PASSWORDS[name]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def ask(client, headers, **extra):
    body = {"loan_id": "CHK-1", "competitor_name": "HDFC Bank", "competitor_rate": 8.6,
            "evidence_seen": False, "bureau_score": 790, **extra}
    return client.post("/competitors/check", json=body, headers=headers)


def test_a_lender_is_searched_once_and_reused(client) -> None:
    agent = StandIn(FOUND)
    market_rates.set_agent(agent)
    rohit = who(client, "rohit")
    first = ask(client, rohit)
    assert first.status_code == 200, first.text
    assert first.json()["state"] == "researching" and first.json()["lender"] == "HDFC"
    assert first.json()["score"]["points"] == 35, "while it searches, the quote rests on its evidence"

    ready = ask(client, rohit).json()
    assert ready["state"] == "ready" and ready["score"]["points"] == 60
    assert ready["research"]["rates"][0]["source_url"] == "https://hdfc.bank.in/rates"
    seen = ask(client, rohit, evidence_seen=True).json()
    assert seen["score"]["points"] == 85
    assert agent.calls == ["HDFC"], "the same lender is not searched again while fresh"

    ask(client, rohit, refresh=True)
    assert agent.calls == ["HDFC", "HDFC"]
    actions = {r["action"] for r in store.query("SELECT action FROM audit_log WHERE entity_id IN ('HDFC', 'CHK-1')")}
    assert {"researched_competitor_rates", "requested_competitor_check"} <= actions


def test_an_unnamed_lender_is_compared_with_the_market(client) -> None:
    agent = StandIn(FOUND)
    market_rates.set_agent(agent)
    body = ask(client, who(client, "rohit"), competitor_name="Other housing finance company").json()
    assert body["lender"] == MARKET and agent.calls == [MARKET]


def test_a_failed_search_is_not_retried_at_once(client) -> None:
    agent = StandIn(error="search limit")
    market_rates.set_agent(agent)
    rohit = who(client, "rohit")
    ask(client, rohit)
    again = ask(client, rohit).json()
    assert again["state"] == "failed" and agent.calls == ["HDFC"]
    assert again["score"]["points"] == 35


def test_with_no_search_set_up_the_quote_is_still_scored(client) -> None:
    market_rates.set_agent(StandIn(available=False))
    body = ask(client, who(client, "rohit")).json()
    assert body["state"] == "unavailable" and body["score"]["max_discount_bps"] == UNEVIDENCED


def test_only_those_who_price_a_file_run_the_check(client) -> None:
    market_rates.set_agent(StandIn(FOUND))
    assert ask(client, who(client, "meera")).status_code == 200
    assert ask(client, who(client, "rahul")).status_code == 403
    assert ask(client, {}).status_code == 401
    lenders = client.get("/competitors/lenders", headers=who(client, "rohit"))
    assert lenders.status_code == 200
    names = [r["name"] for r in lenders.json()["lenders"]]
    assert "HDFC" in names and names[-1] == "Local Bank"


def test_a_billing_failure_is_reported_in_plain_words() -> None:
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    refused = anthropic.BadRequestError(
        "Your credit balance is too low to access the Anthropic API.",
        response=httpx2.Response(400, request=request), body=None)

    class Broke(FakeClient):
        def create(self, **kwargs):
            raise refused

    with pytest.raises(ResearchFailed, match="run out of credit"):
        MarketResearchAgent(client=Broke()).research("HDFC")
