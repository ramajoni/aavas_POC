"""Sales-note extraction: registry, rates, evidence, roles, merging, quality, cache.

The extractor is deterministic, so these are exact-value tests. Two of them are
controls rather than features: the extractor must never read the latent truth
columns, and the historical batch must never call the LLM.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from aavas.intelligence import note_extraction as nx
from aavas.intelligence.quality import quality_report
from aavas.intelligence.registry import CompetitorRegistry, load_config


@pytest.fixture(scope="module")
def registry():
    return CompetitorRegistry()


@pytest.fixture(scope="module")
def extractor(registry):
    return nx.NoteExtractor(registry=registry, allow_llm=False)


def claims(extractor, text):
    note = extractor.extract(text, 0)
    return {c.competitor: c for c in note.competitors().values()}, note


# ------------------------------------------------------------------ registry
def test_structured_canonical_names_are_in_the_registry(registry):
    for name in ["HDFC", "LIC HF", "Bajaj HF", "Local Bank", "Shubham HF", "Aadhar HF"]:
        assert name in registry.lenders
        assert registry.resolve(name) == name
    assert registry.resolve("Aavas") is None and registry.is_self("Aavas Financiers")


@pytest.mark.parametrize("written,canonical", [
    ("LICHFL", "LIC HF"), ("LIC Housing Finance", "LIC HF"), ("Bajaj Finserv", "Bajaj HF"),
    ("HDFC Ltd", "HDFC"), ("hdfc", "HDFC"), ("Aadhaar Housing", "Aadhar HF"),
    ("a nationalised bank", "Local Bank"), ("the co-operative bank", "Local Bank"),
    ("HomeFirst", "Home First"), ("State Bank", "SBI"), ("ICICI Home Finance", "ICICI HFC"),
])
def test_aliases_resolve_in_text(extractor, written, canonical):
    found, _ = claims(extractor, f"Customer says {written} is offering 9.75%.")
    assert set(found) == {canonical}
    assert found[canonical].rate == 9.75


def test_ambiguous_words_need_a_lending_sentence(extractor):
    found, _ = claims(extractor, "Aadhaar card submitted. LIC policy assigned. Shubham is the RM on this file.")
    assert found == {}
    found, _ = claims(extractor, "Customer aware of Shubham at ROI 11.13 but prefers our service.")
    assert found["Shubham HF"].match == "ambiguous" and found["Shubham HF"].rate == 11.13


def test_misspelling_is_fuzzy_matched_and_says_so(extractor):
    found, _ = claims(extractor, "Claims Shubam Housing quoted 10.41%, no document shared.")
    claim = found["Shubham HF"]
    assert claim.match == "fuzzy" and claim.alias == "Shubam Housing" and claim.fuzzy_ratio >= 0.86


def test_unknown_lender_is_surfaced_not_mapped(extractor):
    found, note = claims(extractor, "Customer mentioned Ramesh Housing Finance has also called with 11.99%.")
    assert found == {}
    assert [u.alias for u in note.unrecognised] == ["Ramesh Housing Finance"]
    assert note.unrecognised[0].competitor is None and note.unrecognised[0].rate == 11.99
    # Same group name, different lender: reviewed, never folded into Muthoot HF.
    found, note = claims(extractor, "Customer says Muthoot Fincorp is offering 12%.")
    assert found == {} and note.unrecognised[0].alias == "Muthoot Fincorp"


def test_registry_aliases_are_not_copied_from_the_generator():
    """The registry is scored against the generator's truth; a copied alias list
    would make that score a tautology. Misspellings live in the fuzzy matcher."""
    lenders = load_config()["lenders"]
    written = {a.lower() for spec in lenders.values() for a in spec.get("aliases", []) + spec.get("ambiguous", [])}
    assert "shubam housing" not in written and "adhar housing" not in written


# --------------------------------------------------------------------- rates
@pytest.mark.parametrize("text,value", [
    ("8.5%", 8.5), ("8.50 pc", 8.5), ("@8.50", 8.5), ("ROI 8.50", 8.5), ("8.50 percent", 8.5),
    ("11.20% p.a.", 11.2), ("rate 9.71", 9.71), ("@ 10.3", 10.3),
])
def test_rate_formats(text, value):
    assert [r.value for r in nx.parse_rates(text)] == [value]


@pytest.mark.parametrize("text", ["Requesting 14 bps relaxation", "loan of 25 lakh", "tenure 20 years",
                                  "rate 30%", "Rs 110.5% nonsense", "2.5 lakh"])
def test_numbers_that_are_not_rates(text):
    assert nx.parse_rates(text) == []


def test_hinglish(extractor):
    found, _ = claims(extractor, "Customer bol raha hai Aadhar HF 10.52% p.a. de raha hai.")
    assert found["Aadhar HF"].rate == 10.52
    assert found["Aadhar HF"].evidence == "undocumented" and found["Aadhar HF"].evidence_cue == "bol raha"
    found, _ = claims(extractor, "Rate kam nahi hua toh Bajaj Housing mein BT karayega.")
    assert set(found) == {"Bajaj HF"} and found["Bajaj HF"].rate is None


# ------------------------------------------------------------------ evidence
@pytest.mark.parametrize("text,lender,label,cue", [
    ("Customer forwarded LIC Housing Finance sanction letter, rate 9.71 pc", "LIC HF", "documented", "sanction letter"),
    ("Copy of hdfc offer letter shared, ROI 11.21", "HDFC", "documented", "offer letter"),
    ("WhatsApp screenshot of Bajaj Housing quote at 10.37%, not verified.", "Bajaj HF", "undocumented", None),
    ("Claims Shubham HF quoted 12.34%, no document shared.", "Shubham HF", "undocumented", None),
    ("HDFC verbal quote 10.14% as per customer", "HDFC", "undocumented", "verbal"),
    ("Also has a quote from SBI at 9.57%.", "SBI", "unknown", None),
    ("Customer mentioned ICICI HFC has also called with 11.99%. Not verified.", "ICICI HFC", "undocumented", None),
])
def test_evidence_classes(extractor, text, lender, label, cue):
    found, _ = claims(extractor, text)
    assert found[lender].evidence == label
    if cue:
        assert found[lender].evidence_cue.lower() == cue


def test_no_document_is_not_read_as_document_shared(extractor):
    found, _ = claims(extractor, "Claims HDFC quoted 9.10%, no document shared.")
    assert found["HDFC"].evidence == "undocumented"


def test_two_lenders_in_one_sentence_keep_their_own_evidence(extractor):
    found, _ = claims(extractor, "Customer showed HDFC Bank sanction letter at 8.65%, "
                                 "also says LICHFL offering 8.9 pc verbally")
    assert (found["HDFC"].rate, found["HDFC"].evidence) == (8.65, "documented")
    assert (found["LIC HF"].rate, found["LIC HF"].evidence) == (8.9, "undocumented")
    assert found["LIC HF"].alias == "LICHFL"


# --------------------------------------------------------------------- roles
def test_customer_ask_is_not_a_competitor_quote(extractor):
    found, note = claims(extractor, "Customer bargaining on every point, wants 12.96% or close.")
    assert found == {} and [a["rate"] for a in note.customer_asks] == [12.96]
    found, note = claims(extractor, "Third round of negotiation. Customer firm on 10.77%, will not sign above "
                                    "that. Also has a quote from SBI at 10.93 percent.")
    assert found["SBI"].rate == 10.93 and [a["rate"] for a in note.customer_asks] == [10.77]


def test_ask_quote_and_our_rate_in_one_sentence(extractor):
    found, note = claims(extractor, "Customer wants 9.5% as HDFC is at 9.6%, our rate 9.9%")
    assert found["HDFC"].rate == 9.6
    assert [a["rate"] for a in note.customer_asks] == [9.5]
    assert [o["rate"] for o in note.our_rates] == [9.9]


def test_our_rate_is_never_a_competitor_quote(extractor):
    found, note = claims(extractor, "Aavas offered 10.2% while LIC quoted 9.9%.")
    assert found["LIC HF"].rate == 9.9 and [o["rate"] for o in note.our_rates] == [10.2]
    found, note = claims(extractor, "Repayment record clean. Worth retaining at ROI 11.74.")
    assert found == {} and [o["rate"] for o in note.our_rates] == [11.74]


def test_rule_mode_expected_rate_comes_from_the_ask_parser():
    """The acceptance rule extractor reads a lone percentage as the customer's
    ask; with an unlisted lender that would turn a quote into an ask."""
    result = nx.extract_current(["Customer says Shubham HF is offering 10.87%."], allow_llm=False)
    assert "expected_rate" not in result["customer_signals"]
    result = nx.extract_current(["Customer firm on 10.77%, will not sign above that. "
                                 "Needs disbursement before registration next week."], allow_llm=False)
    assert result["customer_signals"]["expected_rate"]["value"] == 10.77
    assert result["customer_signals"]["customer_urgency"]["value"] == "high"


# ----------------------------------------------------------- current loan
def test_current_loan_merge_latest_wins_and_conflicts():
    notes = ["Customer says HDFC is offering 9.10%.",
             "Customer showed HDFC sanction letter at 9.00%. Also has a quote from SBI at 8.95%.",
             "Called customer, still comparing with HDFC. Will revert."]
    result = nx.extract_current(notes, {"competitor_name": "HDFC", "competitor_rate": 9.10,
                                        "competitor_evidence_verified": False}, allow_llm=False)
    hdfc = next(c for c in result["competitors"] if c["competitor"] == "HDFC")
    assert hdfc["rate"] == 9.0 and hdfc["rate_note_index"] == 1
    assert hdfc["evidence"] == "documented" and len(hdfc["claims"]) == 3
    types = {c["type"] for c in result["conflicts"]}
    assert {"rate_mismatch", "evidence_not_recorded", "notes_disagree", "additional_competitor"} <= types
    assert result["method"]["customer_signals"] == "rule"
    assert "none" in result["pricing_effect"]


def test_verified_flag_against_a_verbal_note():
    result = nx.extract_current(["LIC Housing Finance verbal quote 12.64% as per customer."],
                                {"competitor_name": "LIC HF", "competitor_rate": 12.64,
                                 "competitor_evidence_verified": True}, allow_llm=False)
    conflict = next(c for c in result["conflicts"] if c["type"] == "evidence_mismatch")
    assert conflict["notes"]["cue"] == "verbal"


def test_unrecognised_structured_competitor_is_flagged():
    result = nx.extract_current(["Rate discussed."], {"competitor_name": "Other housing finance company",
                                                      "competitor_rate": 11.5}, allow_llm=False)
    assert result["conflicts"][0]["type"] == "unrecognised_structured_competitor"


class _FakeClaude:
    def __init__(self, signals=None, explode=False):
        self.signals, self.explode, self.calls = signals or [], explode, 0

    def extract(self, doc):
        self.calls += 1
        if self.explode:
            raise AssertionError("the historical batch must never call the LLM")
        return self.signals


class _FakeAgent:
    llm_enabled = True
    mode = "claude"

    def __init__(self, claude):
        from aavas.acceptance.extraction_agent import RuleExtractor

        self.claude, self.rule = claude, RuleExtractor()


def test_llm_is_used_for_current_customer_signals_only(registry):
    from aavas.acceptance.signals import Signal

    claude = _FakeClaude([Signal("customer_urgency", "high", "sales_note", "salesperson_estimate",
                                 quote="this week", extractor="claude")])
    current = nx.NoteExtractor(registry=registry, agent=_FakeAgent(claude), allow_llm=True)
    result = nx.extract_current(["Customer needs it this week. HDFC at 9.1%."], extractor=current)
    assert result["method"]["customer_signals"] == "claude" and claude.calls == 1
    assert result["competitors"][0]["rate"] == 9.1          # competitors stay deterministic

    batch = nx.NoteExtractor(registry=registry, agent=_FakeAgent(_FakeClaude(explode=True)), allow_llm=False)
    signals, _ = nx.extract_history(pd.DataFrame({"note_id": ["N1"], "loan_id": ["L1"],
                                                  "note_text": ["HDFC at 9.1%"]}), extractor=batch)
    assert signals["signals_method"].tolist() == ["rule"]


# ----------------------------------------------------- latent truth control
def _notes_frame():
    return pd.DataFrame({
        "note_id": ["L1-N1", "L1-N2", "L2-N1"], "loan_id": ["L1", "L1", "L2"],
        "note_date": ["2025-01-01", "2025-01-05", "2025-02-01"], "city": ["Jaipur", "Jaipur", "Kota"],
        "note_text": ["Customer says HDFC is offering 9.5%.", "Also has a quote from SBI at 9.2%.",
                      "Customer firm on 10.1%."],
        "true_competitors": ["HDFC", "SBI", None], "true_competitor_rates": ["9.5", "9.2", None],
    })


class _TruthGuard(pd.DataFrame):
    """A frame that fails loudly if anything reads a latent column."""

    @property
    def _constructor(self):
        return pd.DataFrame

    def __getitem__(self, key):
        keys = key if isinstance(key, (list, tuple, pd.Index)) else [key]
        if any(str(k).startswith("true_") for k in keys):
            raise AssertionError(f"extractor read latent column(s) {keys}")
        return super().__getitem__(key)

    def __getattr__(self, name):
        if name.startswith("true_"):
            raise AssertionError(f"extractor read latent column {name}")
        return super().__getattr__(name)


def test_extractor_never_reads_latent_truth(extractor):
    guarded = _TruthGuard(_notes_frame())
    signals, mentions = nx.extract_history(guarded, extractor=extractor)
    assert not any(c.startswith("true_") for c in [*signals.columns, *mentions.columns])

    scrambled = _notes_frame().assign(true_competitors="Aptus", true_competitor_rates="1.0")
    dropped = _notes_frame().drop(columns=["true_competitors", "true_competitor_rates"])
    a, _ = nx.extract_history(scrambled, extractor=extractor)
    b, _ = nx.extract_history(dropped, extractor=extractor)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(AssertionError):
        nx._assert_no_latent(["note_text", "true_competitors"])


# ------------------------------------------------------------------ quality
def test_quality_metrics_on_a_hand_built_truth_set():
    signals = pd.DataFrame({
        "note_id": ["n1", "n2", "n3", "n4", "n5"],
        "competitors": ["HDFC", "HDFC|SBI", "", "Bajaj HF", "LIC HF|Aptus"],
        "rates": ["9.5", "9.1|9.2", "", "", "|10.5"],
        "evidence": ["documented", "undocumented|unknown", "", "unknown", "unknown|unknown"],
        "unrecognised": ["", "", "Xyz Housing", "", ""],
    })
    notes = pd.DataFrame({
        "note_id": ["n1", "n2", "n3", "n4", "n5"],
        "true_competitors": ["HDFC", "HDFC", "LIC HF", "Bajaj HF", "LIC HF|Aptus"],
        "true_competitor_rates": ["9.5", "9.0", "10.0", "11.0", "10.5"],   # n5: slot dropped -> by value
        "competitor_name": ["HDFC", "HDFC", "LIC HF", "Bajaj HF", "LIC HF"],
        "evidence_verified": [True, True, False, False, False],
    })
    q = quality_report(signals, notes)
    micro = q["competitor_identification"]["micro"]
    assert (micro["tp"], micro["fp"], micro["fn"]) == (5, 1, 1)
    assert micro["precision"] == pytest.approx(5 / 6, abs=1e-4)
    assert micro["recall"] == pytest.approx(5 / 6, abs=1e-4)
    assert micro["f1"] == pytest.approx(5 / 6, abs=1e-4)
    assert q["competitor_identification"]["per_competitor"]["SBI"]["fp"] == 1
    rates = q["rates"]
    assert (rates["exact"], rates["wrong"], rates["missing"], rates["scored"]) == (2, 1, 1, 4)
    assert rates["exact_match_accuracy"] == 0.5 and rates["truth_unaligned_notes"] == 1
    ev = q["evidence_vs_structured_flag"]
    assert ev["classified"] == 2 and ev["agreement"] == 0.5 and ev["unknown_class"] == 2
    assert ev["confusion"]["undocumented_but_flag_verified"] == 1
    assert q["unrecognised_lenders"] == [{"name": "Xyz Housing", "notes": 1}]
    assert q["note_set_exact_match"] == pytest.approx(3 / 5)
    assert "synthetic" in q["caveat"]


def test_quality_without_truth_says_so():
    q = quality_report(pd.DataFrame({"note_id": ["n1"], "competitors": ["HDFC"]}),
                       pd.DataFrame({"note_id": ["n1"], "note_text": ["x"]}))
    assert q["available"] is False and "cannot be measured" in q["reason"]


# -------------------------------------------------------------------- cache
def test_history_cache_is_reused_and_rebuilt_when_notes_change(tmp_path, monkeypatch):
    source = tmp_path / "notes.csv"
    _notes_frame().to_csv(source, index=False)
    signals, mentions = nx.load_history(source=source, cache_dir=tmp_path)
    assert (tmp_path / nx.SIGNALS_FILE).exists() and len(signals) == 3
    assert set(mentions["competitor"]) == {"HDFC", "SBI"}
    assert not any(c.startswith("true_") for c in signals.columns)

    def boom(*a, **k):
        raise AssertionError("cache should have been used")

    monkeypatch.setattr(nx, "extract_history", boom)
    cached, _ = nx.load_history(source=source, cache_dir=tmp_path)
    assert cached["competitors"].tolist() == signals["competitors"].tolist()

    later = time.time() + 5
    os.utime(source, (later, later))
    with pytest.raises(AssertionError, match="cache should have been used"):
        nx.load_history(source=source, cache_dir=tmp_path)


# ---------------------------------------------------------------- retriever
def test_retriever_ids_are_note_ids_and_outcomes_join_on_loan():
    from aavas.knowledge.sales_note_retriever import SalesNoteRetriever
    from aavas.knowledge.vector_store import InMemoryVectorStore

    notes = _notes_frame().drop(columns=["true_competitors", "true_competitor_rates"])
    notes.loc[len(notes)] = ["L2-N2", "L2", "2025-02-03", "Kota", "Customer showed SBI sanction letter at 9.0%."]
    outcomes = pd.DataFrame({"loan_id": ["L1", "L2"], "disbursed": [True, False],
                             "override_bps": [-10.0, 0.0], "sanctioned_rate": [9.6, 10.2]})
    store = InMemoryVectorStore()
    retriever = SalesNoteRetriever(store).index(notes, outcomes)
    ids = [d.doc_id for d in store.documents]
    assert ids == notes["note_id"].tolist() and len(set(ids)) == len(ids)

    hits = retriever.search_notes("SBI sanction letter", k=3)
    top = hits[0]
    assert top["doc_id"] == "L2-N2" and top["loan_id"] == "L2"
    assert top["outcome"]["disbursed"] is False and top["competitors"][0]["competitor"] == "SBI"
    assert all(h["loan_id"] != "L2" for h in retriever.search_notes("SBI sanction letter", k=3,
                                                                       exclude_loan="L2"))
    intel = retriever.competitor_intelligence(k=10)
    # SBI exists only in the text - the structured fields never had a slot for it.
    assert intel["competitors"]["SBI"]["mentions"] == 2
    assert intel["competitors"]["SBI"]["evidenced"] == 1
    assert "extracted" in intel["source"]


def test_retriever_without_note_ids_still_has_unique_ids():
    from aavas.knowledge.sales_note_retriever import SalesNoteRetriever
    from aavas.knowledge.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    SalesNoteRetriever(store).index(pd.DataFrame({"loan_id": ["L1", "L1"], "note_text": ["customer negotiating rate", "sanction letter shared"]}))
    ids = [d.doc_id for d in store.documents]
    assert len(set(ids)) == 2
