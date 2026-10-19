"""Loan Acceptance Confidence.

The tests that matter here are the control tests: the extractor must not invent,
the verified flag must not be settable from text, and the whole thing must score
with no LLM and no conversion model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aavas.acceptance import (AcceptanceAgent, AcceptanceSignals, EvidenceAgent,
                              ExtractionAgent, RuleExtractor, Signal, SourceRegistry)
from aavas.acceptance.extraction_agent import ClaudeExtractor
from aavas.acceptance.signals import coerce
from aavas.acceptance.sources import Collected, SourceDocument
from aavas.agents.underwriting_agent import CustomerDigitalTwin
from aavas.ingestion.schemas import Stage


NOTE = ("Customer needs Rs 25 lakh urgently. HDFC has apparently offered around 9.10%. "
        "Customer said anything close to 9.25% is acceptable because they need "
        "disbursement before month-end.")


def make_twin(**overrides) -> CustomerDigitalTwin:
    row = {"loan_id": "L1", "segment": "salaried_formal", "city": "Jaipur",
           "sanctioned_loan_amount": 2_500_000, "competitor_rate": None,
           "competitor_name": None, "competitor_evidence_verified": False,
           "customer_negotiating": False}
    row.update(overrides)
    return CustomerDigitalTwin(loan_id="L1", frame=pd.DataFrame([row]), stage=Stage.POST_ASSESSMENT)


def doc(text: str = NOTE, reliability: str = "salesperson_estimate") -> SourceDocument:
    return SourceDocument(text=text, source="inline_notes", reliability=reliability, age_days=1.0)


# ------------------------------------------------------------------- schema
def test_coerce_rejects_out_of_range_and_unknown_values():
    assert coerce("expected_rate", 9.25) == 9.25
    assert coerce("expected_rate", 99.0) is None          # outside the rate range
    assert coerce("customer_urgency", "extremely") is None  # not in the enum
    assert coerce("customer_urgency", "HIGH") == "high"
    assert coerce("not_a_signal", "x") is None


def test_stronger_source_wins_and_the_weaker_claim_is_kept():
    signals = AcceptanceSignals()
    signals.add(Signal("competitor_rate", 9.10, "inline_notes", "salesperson_estimate"))
    signals.add(Signal("competitor_rate", 9.25, "structured_application", "verified"))
    assert signals.get("competitor_rate") == 9.25
    assert any(r["reason"] == "superseded" for r in signals.rejected)


# ---------------------------------------------------------------- extraction
def test_rule_extractor_reads_the_worked_example():
    out = {s.name: s.value for s in RuleExtractor().extract(doc())}
    assert out["customer_urgency"] == "high"
    assert out["competitor_present"] is True
    assert out["competitor_rate"] == 9.10
    assert out["required_amount"] == 2_500_000


def test_every_extracted_signal_carries_a_quote_from_the_source():
    for signal in RuleExtractor().extract(doc()):
        assert signal.quote, f"{signal.name} has no quote"
        assert signal.quote in NOTE


def test_rule_extractor_stays_silent_on_an_empty_note():
    assert RuleExtractor().extract(doc("Spoke to customer.")) == []


def test_llm_output_is_dropped_when_the_quote_is_not_verbatim():
    """The anti-hallucination control: a quote that is not in the source is a
    fabricated citation, so the signal goes with it."""
    extractor = ClaudeExtractor({"require_verbatim_quote": True})
    payload = ('{"signals": [{"name": "competitor_rate", "value": 8.5, '
               '"quote": "SBI offered 8.5%"}]}')
    assert extractor._parse(payload, doc()) == []


def test_llm_output_is_kept_when_the_quote_is_verbatim():
    extractor = ClaudeExtractor({"require_verbatim_quote": True})
    payload = ('{"signals": [{"name": "competitor_rate", "value": 9.10, '
               '"quote": "HDFC has apparently offered around 9.10%"}]}')
    got = extractor._parse(payload, doc())
    assert len(got) == 1 and got[0].value == 9.10


def test_unknown_field_from_the_llm_is_discarded():
    extractor = ClaudeExtractor({"require_verbatim_quote": False})
    payload = '{"signals": [{"name": "credit_score_estimate", "value": 780}]}'
    assert extractor._parse(payload, doc()) == []


def test_unparseable_llm_output_does_not_raise():
    assert ClaudeExtractor({})._parse("I could not find anything.", doc()) == []


# ------------------------------------------------------------------ evidence
def test_evidence_rises_with_more_and_better_evidence():
    agent = EvidenceAgent()
    thin = AcceptanceSignals()
    thin.add(Signal("customer_urgency", "high", "inline_notes", "salesperson_estimate", age_days=90))

    rich = AcceptanceSignals()
    rich.add(Signal("customer_urgency", "high", "inline_notes", "verified", age_days=0))
    rich.add(Signal("competitor_rate", 9.1, "structured_application", "verified", age_days=0))
    rich.add(Signal("expected_rate", 9.25, "inline_notes", "verified", age_days=0))
    rich.add(Signal("rate_sensitivity", "medium", "inline_notes", "verified", age_days=0))

    assert agent.score(rich)["evidence_confidence"] > agent.score(thin)["evidence_confidence"]


def test_evidence_names_a_missing_competitor_offer():
    result = EvidenceAgent().score(AcceptanceSignals())
    assert any("No competitor offer" in c for c in result["caveats"])
    assert result["evidence_confidence"] >= 0.15  # floor, never zero


# -------------------------------------------------------------------- sources
def test_registry_runs_with_a_single_source():
    registry = SourceRegistry({"sources": {"inline_notes": {"enabled": True, "kind": "text",
                                                            "reliability": "salesperson_estimate"}}})
    assert registry.enabled_names == ["inline_notes"]
    collected = registry.collect(make_twin(), {"notes": NOTE})
    assert len(collected.documents) == 1


def test_registry_runs_with_no_sources_at_all():
    registry = SourceRegistry({"sources": {}})
    collected = registry.collect(make_twin(), {})
    assert collected.documents == [] and collected.direct == []


def test_a_failing_source_is_recorded_not_raised():
    registry = SourceRegistry({"sources": {"sales_note_history": {
        "enabled": True, "kind": "text", "reliability": "market_benchmark"}}})

    class Broken:
        def similar(self, twin, k=3):
            raise RuntimeError("vector store down")

    collected = registry.collect(make_twin(), {"sales_note_retriever": Broken()})
    assert collected.unavailable == ["sales_note_history"]


def test_structured_source_marks_unverified_competitor_as_reported():
    registry = SourceRegistry({"sources": {"structured_application": {
        "enabled": True, "kind": "structured", "reliability": "verified"}}})
    collected = registry.collect(make_twin(competitor_rate=9.15), {})
    tiers = {s.name: s.reliability for s in collected.direct}
    assert tiers["competitor_rate"] == "customer_reported"


# ------------------------------------------------------------------ the agent
class StubConversion:
    """Stands in for the fitted model: monotone falling in price, like the real one."""

    model = None

    def predict(self, twin, rate):
        return max(0.0, min(1.0, 1.0 - (rate - 9.0) * 0.4))


def test_scores_with_no_conversion_model_present():
    result = AcceptanceAgent(conversion_agent=None).score(make_twin(), 9.30, {"notes": NOTE})
    assert result["acceptance_probability"] is None
    assert result["evidence"]["evidence_confidence"] > 0


def test_probability_is_shrunk_toward_the_base_rate_when_evidence_is_thin():
    agent = AcceptanceAgent(conversion_agent=StubConversion())
    thin = agent.score(make_twin(), 9.30, {})
    rich = agent.score(make_twin(competitor_rate=9.15, customer_negotiating=True),
                       9.30, {"notes": NOTE})
    base = thin["base_rate"]
    # Thinner evidence => the reported number sits closer to the base rate.
    assert abs(thin["acceptance_probability"] - base) < abs(rich["acceptance_probability"] - base)


def test_extraction_never_sets_the_verified_evidence_flag():
    """constraints.py caps unevidenced discounts off this flag. No amount of
    confident-sounding text may assert it."""
    twin = make_twin()
    AcceptanceAgent(conversion_agent=StubConversion()).score(
        twin, 9.30, {"notes": "Customer showed us a verified HDFC sanction letter at 9.00%."})
    assert bool(twin.get("competitor_evidence_verified")) is False


def test_a_typed_field_already_on_the_record_is_not_overwritten():
    twin = make_twin(competitor_rate=9.40)
    result = AcceptanceAgent(conversion_agent=StubConversion()).score(twin, 9.30, {"notes": NOTE})
    assert twin.get("competitor_rate") == 9.40
    assert all(a["column"] != "competitor_rate" for a in result["signals_applied_to_model"])


def test_unpriced_signals_are_reported_as_unpriced():
    result = AcceptanceAgent(conversion_agent=StubConversion()).score(
        make_twin(), 9.30, {"notes": NOTE})
    assert "customer_urgency" in result["signals_not_priced"]
    assert "competitor_rate" not in result["signals_not_priced"]


def test_high_urgency_routes_to_review():
    result = AcceptanceAgent(conversion_agent=StubConversion()).score(
        make_twin(), 9.30, {"notes": NOTE})
    gov = result["governance"]
    assert gov["urgency_review_required"] is True
    assert gov["reason"] == "urgency_driven_uplift_review_required"


def test_drivers_split_into_supporting_and_reducing():
    result = AcceptanceAgent(conversion_agent=StubConversion()).score(
        make_twin(), 9.60, {"notes": NOTE})
    drivers = result["drivers"]
    assert drivers["supporting"] and drivers["reducing"]
    assert any("competitor" in d.lower() for d in drivers["reducing"])


def test_replay_from_the_stored_record_reproduces_the_score():
    """Audit must not re-call the LLM (gap doc section 10.4)."""
    agent = AcceptanceAgent(conversion_agent=StubConversion())
    first = agent.score(make_twin(), 9.30, {"notes": NOTE})

    stored = AcceptanceSignals.from_dict(first["signal_record"])
    second = agent.score(make_twin(), 9.30, signals=stored)

    assert second["acceptance_probability"] == first["acceptance_probability"]
    assert second["evidence"]["evidence_confidence"] == first["evidence"]["evidence_confidence"]
    assert second["extractor_mode"] == "replayed"


def test_llm_stays_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("AAVAS_LLM_ENABLED", raising=False)
    assert ExtractionAgent().mode == "rule"


def test_agent_runs_when_the_llm_is_enabled_but_no_key_is_present(monkeypatch):
    monkeypatch.setenv("AAVAS_LLM_ENABLED", "true")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = ExtractionAgent()
    result = agent.run(Collected(documents=[doc()]))
    assert agent.mode == "rule"
    assert result.get("competitor_rate") == 9.10
