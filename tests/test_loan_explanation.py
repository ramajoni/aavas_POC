"""Per-loan explainability: every feature's SHAP weight, the rate bridge, and the
AI-written explanation that may only repeat figures the page already shows."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aavas.api import deps
from aavas.api.main import app
from aavas.features.feature_pipeline import spec
from aavas.xai import narrative


@pytest.fixture(scope="module")
def client(bundle):
    return TestClient(app)


@pytest.fixture(scope="module")
def detail(client, decision):
    response = client.get(f"/pricing/explain/{decision['decision_id']}")
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------- attribution
def test_rate_model_weights_every_feature_not_just_the_top_codes(detail):
    features = detail["models"]["stage2"]["features"]
    assert len(features) == len(spec("stage2").columns)
    assert len(features) > len(detail["context"]["reason_codes"])


def test_rate_attributions_reconcile_to_the_model_rate(detail):
    model = detail["models"]["stage2"]
    total = model["base_value"] + sum(f["contribution"] for f in model["features"])
    assert total == pytest.approx(model["prediction"], abs=1e-6)


def test_rate_attributions_are_shap_when_shap_is_usable(detail, bundle):
    from aavas.xai.shap_explainer import PricingExplainer

    if PricingExplainer(bundle.stage2.model, bundle.stage2.transformer).backend != "shap":
        pytest.skip("shap is not usable in this environment")
    assert detail["models"]["stage2"]["method"] == "shap"


def test_weight_shares_add_up_to_the_whole(detail):
    for name in ("stage2", "risk", "conversion"):
        shares = [f["share_pct"] for f in detail["models"][name]["features"]]
        assert sum(shares) == pytest.approx(100, abs=0.5), name


def test_features_are_ranked_by_size_of_effect(detail):
    sizes = [abs(f["contribution"]) for f in detail["models"]["stage2"]["features"]]
    assert sizes == sorted(sizes, reverse=True)
    assert [f["rank"] for f in detail["models"]["stage2"]["features"]] == list(range(1, len(sizes) + 1))


def test_features_are_split_into_customer_risk_and_commercial_pressure(detail):
    groups = {f["feature"]: f["group"] for f in detail["models"]["stage2"]["features"]}
    assert groups["competitor_rate"] == "commercial"
    assert groups["bureau_score"] == "customer_risk"
    summary = detail["models"]["stage2"]["groups"]
    assert set(summary) == {"customer_risk", "commercial"}
    assert summary["customer_risk"]["share_pct"] + summary["commercial"]["share_pct"] == pytest.approx(100, abs=0.2)


def test_risk_and_conversion_models_are_explained_for_the_same_loan(detail):
    risk, conversion = detail["models"]["risk"], detail["models"]["conversion"]
    assert len(risk["features"]) == len(spec("risk").columns)
    assert len(conversion["features"]) == len(spec("conversion").columns)
    assert risk["positive_is_adverse"] is True
    assert conversion["positive_is_adverse"] is False


# --------------------------------------------------------------------- bridge
def test_bridge_walks_from_the_typical_customer_to_the_recommended_rate(detail, decision):
    bridge = detail["bridge"]
    walked = bridge["baseline_rate"] + sum(s["delta_bps"] for s in bridge["steps"]) / 100
    assert walked == pytest.approx(bridge["model_rate"], abs=0.001)
    assert bridge["model_rate"] + bridge["optimisation_bps"] / 100 == pytest.approx(
        bridge["recommended_rate"], abs=0.001)
    assert bridge["recommended_rate"] == decision["recommendation"]["recommended_rate"]


def test_recomputed_anchor_matches_the_recorded_decision(detail):
    assert not [w for w in detail["warnings"] if "anchor" in w]


def test_a_decision_explained_with_another_method_says_its_figures_may_differ(bundle, decision):
    from aavas.governance import decision_logger
    from aavas.xai.loan_explanation import LoanExplainer

    stored = decision_logger.get_decision(decision["decision_id"])
    explainer = LoanExplainer(bundle)
    current = explainer.explain(stored)["models"]["stage2"]["method"]
    stored["payload"]["explanation"]["attribution_method"] = "some_older_method"
    warnings = explainer.explain(stored)["warnings"]
    assert any("some_older_method" in w and current in w for w in warnings)


def test_feature_values_are_rounded_for_display(detail):
    floats = [f["value"] for m in detail["models"].values() for f in m["features"]
              if isinstance(f["value"], float)]
    assert floats and all(round(v, 4) == v for v in floats)


# ------------------------------------------------------------------ endpoints
def test_unknown_decision_is_not_found(client):
    assert client.get("/pricing/explain/DEC-DOES-NOT-EXIST").status_code == 404


def test_pre_assessment_decisions_have_no_final_rate_to_explain(client, orchestrator, sample_application):
    quote = orchestrator.pre_assessment({**sample_application, "loan_id": "XAI-PRE-ONLY"})
    response = client.get(f"/pricing/explain/{quote['decision_id']}")
    assert response.status_code == 422
    assert "final assessment" in response.json()["detail"]


# ------------------------------------------------------------------ narrative
FACTS = {
    "rate": {"recommended_rate_pct": 11.19, "card_rate_pct": 9.7, "model_rate_pct": 10.17},
    "loan": {"ltv_pct": 68.97, "expected_value_inr": 169118.0},
    "customer_options": ["A loan amount of Rs 1,800,000 instead of Rs 2,000,000 would improve the rate"],
}


def test_only_figures_absent_from_the_facts_are_flagged():
    # 11.19, 9.7, 69 (68.97 rounded), ₹1,69,118 and Rs 1,800,000 are all in the
    # facts; 149 bps is arithmetic the model did itself, so it is flagged.
    text = ("The recommended 11.19% is 149 bps above the 9.7% card rate. LTV is 69%; "
            "expected value is ₹1,69,118 and Rs 1,800,000 would help.")
    assert narrative.unsupported_figures([text], FACTS) == ["149"]


def test_an_invented_rate_is_flagged():
    assert narrative.unsupported_figures(["A rate of 12.75% would be fairer."], FACTS) == ["12.75"]


def test_the_facts_carry_the_negotiation_step_and_whole_rupees(detail):
    material = narrative.facts(detail)
    assert material["conversion"]["price_step_bps"] == 5
    assert isinstance(material["applicant"]["loan_amount_inr"], int)


def test_small_counts_are_not_treated_as_figures():
    assert narrative.unsupported_figures(["The top 3 drivers, in 2 groups."], FACTS) == []


class _FakeMessages:
    def __init__(self, reply: dict | None, stop_reason: str = "end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = [] if self.reply is None else [SimpleNamespace(type="text", text=json.dumps(self.reply))]
        return SimpleNamespace(content=content, stop_reason=self.stop_reason, model=kwargs["model"],
                               stop_details=None)


def _fake_client(reply: dict | None, stop_reason: str = "end_turn"):
    return SimpleNamespace(beta=SimpleNamespace(messages=_FakeMessages(reply, stop_reason)))


SECTIONS = {
    "summary": "The model recommends a rate above the card rate.",
    "why_this_rate": ["Customer segment lowered the rate."],
    "risk_view": "Risk grade is low.",
    "commercial_view": "Commercial pressure moved the rate a little.",
    "customer_options": ["Reduce the loan amount."],
    "reviewer_checks": ["Confirm the competitor quote. A 12.75% quote was mentioned."],
}


@pytest.fixture
def narrated(client):
    fake = _fake_client(SECTIONS)
    narrator = narrative.LoanNarrator(client=fake)
    app.dependency_overrides[deps.get_narrator] = lambda: narrator
    yield fake
    app.dependency_overrides.pop(deps.get_narrator, None)


def test_narrative_returns_the_sections_and_flags_unsupported_figures(client, decision, narrated):
    response = client.post(f"/pricing/explain/{decision['decision_id']}/narrative")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sections"]["summary"] == SECTIONS["summary"]
    assert body["unverified_figures"] == ["12.75"]
    assert body["model"] == narrative.DEFAULT_MODEL
    sent = narrated.beta.messages.calls[0]
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert decision["decision_id"] in sent["messages"][0]["content"]


def test_a_second_request_is_served_from_cache_unless_refreshed(client, decision, narrated):
    path = f"/pricing/explain/{decision['decision_id']}/narrative"
    assert client.post(path).json()["cached"] is False
    assert client.post(path).json()["cached"] is True
    assert client.post(path + "?refresh=true").json()["cached"] is False
    assert len(narrated.beta.messages.calls) == 2


def test_a_declined_request_is_reported_not_crashed(client, decision):
    fake = _fake_client(None, stop_reason="refusal")
    app.dependency_overrides[deps.get_narrator] = lambda: narrative.LoanNarrator(client=fake)
    try:
        response = client.post(f"/pricing/explain/{decision['decision_id']}/narrative")
    finally:
        app.dependency_overrides.pop(deps.get_narrator, None)
    assert response.status_code == 502
    assert "declined" in response.json()["detail"]


def test_without_a_key_the_narrative_is_unavailable_and_says_why(client, decision, monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    keyless = narrative.LoanNarrator(env_file=tmp_path / "missing.env")
    app.dependency_overrides[deps.get_narrator] = lambda: keyless
    try:
        status = client.get(f"/pricing/explain/{decision['decision_id']}").json()["narrative"]
        response = client.post(f"/pricing/explain/{decision['decision_id']}/narrative")
    finally:
        app.dependency_overrides.pop(deps.get_narrator, None)
    assert status["available"] is False and "ANTHROPIC_API_KEY" in status["reason"]
    assert response.status_code == 503


def test_the_key_can_come_from_the_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nANTHROPIC_API_KEY=sk-test-123\nOTHER=1\n", encoding="utf-8")
    assert narrative.LoanNarrator(env_file=env_file).status()["available"] is True
