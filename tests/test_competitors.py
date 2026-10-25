"""Competitor analysis, per-loan context and the competitor endpoints.

The analysis tests run on a small hand-built book whose every aggregate can be
worked out on paper; the endpoint tests run on the real (synthetic) data and
skip when it has not been generated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aavas.intelligence.competitors import CompetitorAnalysis

SETTINGS = {"min_n_cell": 3, "min_n_rate": 3, "min_n_benchmark": 3,
            "gap_buckets_bps": [-75, -50, -25, 0, 25], "plausibility_percentile": 5,
            "comparable_negotiations": 3}


@pytest.fixture(scope="module")
def book():
    loans = pd.DataFrame([{
        "loan_id": f"L{i:02d}", "city": "Jaipur" if i < 40 else "Kota", "state": "Rajasthan",
        "branch_id": "BR01", "segment": "salaried_formal" if i % 2 == 0 else "self_employed_professional",
        "risk_grade": "A" if i % 2 == 0 else "B", "sanctioned_rate": 10.0, "policy_card_rate": 10.2,
        # HDFC files are won only when their quote is not below ours; the rest win 2 in 3.
        "disbursed": (i % 10 >= 5) if i < 30 else (i % 3 != 0),
        "was_overridden": i % 4 == 0, "override_bps": -10.0 if i % 4 == 0 else 0.0,
        "application_month": "2025-01" if i < 30 else "2025-02", "competitor_name": "HDFC",
        "true_pd": 0.5,                       # latent: must not be needed by anything here
    } for i in range(60)])
    rows = []
    for i in range(30):
        rows.append({"note_id": f"L{i:02d}-N2", "loan_id": f"L{i:02d}", "note_date": "2025-01-05",
                     "city": "Jaipur", "state": "Rajasthan", "mention_index": 0, "competitor": "HDFC",
                     "lender_type": "bank", "alias": "HDFC Bank", "match": "exact",
                     "rate": round(9.5 + (i % 10) * 0.1, 2),
                     "evidence": "documented" if i % 5 == 0 else "undocumented"})
    # An older, different quote on L00: the latest note's quote is the one kept.
    rows.append({**rows[0], "note_id": "L00-N1", "note_date": "2025-01-01", "rate": 9.0, "evidence": "undocumented"})
    for i in range(40, 46):
        rows.append({"note_id": f"L{i}-N1", "loan_id": f"L{i}", "note_date": "2025-02-02", "city": "Kota",
                     "state": "Rajasthan", "mention_index": 0, "competitor": "SBI", "lender_type": "bank",
                     "alias": "State Bank", "match": "exact", "rate": 9.8, "evidence": "unknown"})
    rows.append({"note_id": "L50-N1", "loan_id": "L50", "note_date": "2025-02-09", "city": "Kota",
                 "state": "Rajasthan", "mention_index": 0, "competitor": "", "lender_type": "",
                 "alias": "Ramesh Housing Finance", "match": "unrecognised", "rate": 12.0, "evidence": "unknown"})
    return CompetitorAnalysis(pd.DataFrame(rows), loans, settings=SETTINGS)


def row(analysis, name):
    return next(r for r in analysis.league() if r["competitor"] == name)


def test_league_aggregates(book):
    hdfc = row(book, "HDFC")
    assert hdfc["loans_mentioned"] == 30 and hdfc["share_of_loans"] == 0.5
    assert hdfc["in_structured_data"] is True
    assert hdfc["quotes"]["median"] == 9.95 and hdfc["quotes"]["n"] == 30
    assert hdfc["gap_to_our_rate_bps"]["median"] == -5.0
    assert hdfc["gap_to_card_bps"]["median"] == -25.0
    assert hdfc["evidence"]["documented"] == 6 and hdfc["evidence"]["documented_share"] == 0.2
    assert hdfc["win_rate"]["rate"] == 0.5
    assert hdfc["win_rate"]["base_rate"] == pytest.approx(35 / 60, abs=1e-4)
    assert hdfc["win_rate"]["lift_pp"] == pytest.approx(-8.3, abs=0.05)
    ovr = hdfc["override_when_cited"]
    assert ovr["rate"] == pytest.approx(8 / 30, abs=1e-4) and ovr["base_rate"] == 0.25
    assert ovr["median_depth_bps"] == -10.0 and ovr["n_overridden"] == 8


def test_latest_quote_per_file_wins(book):
    l00 = book.lc[(book.lc["loan_id"] == "L00") & (book.lc["competitor"] == "HDFC")].iloc[0]
    assert l00["quote"] == 9.5 and l00["notes"] == 2


def test_gap_curve_shows_acceptance_falling_as_they_undercut(book):
    curve = {c["bucket"]: c for c in book.gap_curve(book.lc[book.lc["competitor"] == "HDFC"])}
    assert curve["≤ -75"]["n"] == 0 and curve["≤ -75"]["rate"] is None
    assert (curve["-75 to -50"]["n"], curve["-75 to -50"]["rate"]) == (3, 0.0)
    assert (curve["-50 to -25"]["n"], curve["-50 to -25"]["rate"]) == (6, 0.0)
    assert (curve["-25 to +0"]["n"], curve["-25 to +0"]["rate"]) == (9, pytest.approx(1 / 3, abs=1e-4))
    assert curve["+0 to +25"]["rate"] == 1.0 and curve["> +25"]["rate"] == 1.0


def test_small_cells_are_suppressed_not_shown():
    loans = pd.DataFrame({"loan_id": ["A", "B"], "city": ["Jaipur"] * 2, "state": ["Rajasthan"] * 2,
                          "sanctioned_rate": [10.0, 10.0], "policy_card_rate": [10.2, 10.2],
                          "disbursed": [True, False], "was_overridden": [False, True],
                          "override_bps": [0.0, -5.0], "application_month": ["2025-01"] * 2,
                          "competitor_name": ["HDFC", "HDFC"]})
    mentions = pd.DataFrame({"note_id": ["A-1", "B-1"], "loan_id": ["A", "B"], "note_date": ["2025-01-01"] * 2,
                             "city": ["Jaipur"] * 2, "mention_index": [0, 0], "competitor": ["HDFC"] * 2,
                             "match": ["exact"] * 2, "alias": ["HDFC"] * 2, "rate": [9.1, 9.2],
                             "evidence": ["documented"] * 2})
    analysis = CompetitorAnalysis(mentions, loans)               # configured thresholds (20 / 30)
    hdfc = analysis.league()[0]
    assert hdfc["quotes"] == {"n": 2, "median": None, "p10": None, "p90": None, "suppressed": True}
    assert hdfc["win_rate"]["rate"] is None and hdfc["win_rate"]["suppressed"] is True
    assert hdfc["evidence"]["documented_share"] is None


def test_note_only_and_unrecognised_lenders(book):
    emerging = book.emerging()
    assert [e["competitor"] for e in emerging] == ["SBI"]
    assert emerging[0]["first_seen"] == "2025-02-02" and emerging[0]["loans_mentioned"] == 6
    assert emerging[0]["monthly_loans"] == [{"month": "2025-02", "loans": 6}]
    assert row(book, "SBI")["in_structured_data"] is False
    assert book.unrecognised() == [{"name": "Ramesh Housing Finance", "notes": 1, "loans": 1,
                                    "first_seen": "2025-02-09", "cities": ["Kota"], "median_quote": 12.0}]
    assert "Ramesh Housing Finance" not in {r["competitor"] for r in book.league()}


def test_profile_breakdowns(book):
    profile = book.profile("HDFC Ltd")                          # an alias resolves
    assert profile["competitor"] == "HDFC"
    assert profile["by_city"][0]["city"] == "Jaipur" and profile["by_city"][0]["loans_mentioned"] == 30
    assert profile["by_city"][0]["share_of_loans"] == 0.75      # 30 of 40 Jaipur files
    assert {m["month"] for m in profile["monthly"]} == {"2025-01", "2025-02"}
    assert profile["aliases_seen"] == {"HDFC Bank": 31}
    assert book.profile("Nobody Finance") is None


def test_benchmark_narrows_when_the_sample_allows(book):
    bench = book.benchmark("HDFC", city="Jaipur", state="Rajasthan", segment="salaried_formal", grade="A")
    assert bench["level"] == "city+segment+grade" and bench["n"] == 15
    assert book.benchmark("HDFC", city="Jaipur", exclude_loan="L00")["n"] == 29
    assert book.benchmark("HDFC", city="Udaipur", state="Rajasthan")["level"] == "state"


def test_plausibility_flag(book):
    app = {"loan_id": "NEW", "city": "Jaipur", "segment": "salaried_formal", "risk_grade": "A", "our_rate": 10.0,
           "sales_context": {"competitor_name": "HDFC Bank", "competitor_rate": 9.2,
                             "competitor_evidence_verified": False}}
    context = book.context_for(app, [])
    hdfc = context["competitors"][0]
    assert hdfc["quote"] == 9.2 and hdfc["quote_source"] == "structured" and hdfc["evidence"] == "undocumented"
    assert hdfc["plausibility"]["flag"] is True and hdfc["plausibility"]["p5"] == 9.5
    assert hdfc["plausibility"]["message"] == "unusually low - request the sanction letter"
    assert hdfc["gap_to_our_rate_bps"] == -80.0 and hdfc["win_rate_at_gap"]["bucket"] == "≤ -75"
    # No past HDFC file was undercut by more than 50 bps: not "a similar gap".
    assert hdfc["win_rate_at_gap"]["beyond_history"] is True
    assert hdfc["win_rate_at_gap"]["observed_range_bps"] == [-50.0, 40.0]
    assert "none" in context["pricing_effect"]

    documented = book.context_for({**app, "sales_context": {**app["sales_context"],
                                                            "competitor_evidence_verified": True}}, [])
    assert documented["competitors"][0]["plausibility"]["message"].endswith("verify the sanction letter on file")

    fine = book.context_for(app, ["Customer showed HDFC sanction letter at 9.80%."])
    claim = fine["competitors"][0]
    assert claim["quote"] == 9.8 and claim["quote_source"] == "notes" and claim["evidence"] == "documented"
    assert claim["plausibility"]["flag"] is False
    assert claim["win_rate_at_gap"]["bucket"] == "-25 to +0"
    assert claim["win_rate_at_gap"]["beyond_history"] is False
    assert claim["win_rate_at_gap"]["all_cities"]["rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert any(c["type"] == "rate_mismatch" for c in fine["conflicts"])


def test_context_without_our_rate_or_known_lender(book):
    context = book.context_for({"city": "Jaipur", "sales_context": {"competitor_name": "HDFC",
                                                                    "competitor_rate": 9.9}}, [])
    assert context["competitors"][0]["win_rate_at_gap"]["available"] is False
    unknown = book.context_for({"city": "Jaipur", "sales_context": {
        "competitor_name": "Other housing finance company", "competitor_rate": 11.5}}, [])
    assert unknown["competitors"] == []
    assert unknown["conflicts"][0]["type"] == "unrecognised_structured_competitor"


def test_analysis_reads_no_latent_columns(book):
    assert not any(c.startswith("true_") for c in book.loans.columns)


# ------------------------------------------------------------------ endpoints
@pytest.fixture(scope="module")
def client(training_data):
    from fastapi.testclient import TestClient

    from aavas.api.main import app
    from aavas.ingestion import loader

    if loader.load_notes().empty:
        pytest.skip("sales notes not generated")
    return TestClient(app)


def test_notes_extract_endpoint(client):
    from aavas.api.competitors import EXTRACT_EXAMPLE

    response = client.post("/notes/extract", json=EXTRACT_EXAMPLE)
    assert response.status_code == 200, response.text
    body = response.json()
    by_name = {c["competitor"]: c for c in body["competitors"]}
    assert (by_name["HDFC"]["rate"], by_name["HDFC"]["evidence"]) == (8.65, "documented")
    assert (by_name["LIC HF"]["rate"], by_name["LIC HF"]["evidence"]) == (8.9, "undocumented")
    assert [a["rate"] for a in body["customer_asks"]] == [8.75]
    assert {"rate_mismatch", "evidence_not_recorded"} <= {c["type"] for c in body["conflicts"]}
    assert body["method"]["competitors"].startswith("rule")
    assert client.post("/notes/extract", json={"notes": []}).status_code == 422


def test_notes_summary_endpoint(client):
    body = client.get("/notes/summary").json()
    assert body["coverage"]["notes"] > 0 and body["coverage"]["loans"] > 0
    quality = body["quality"]
    assert quality["available"] is True and "synthetic" in quality["caveat"]
    assert 0.0 <= quality["competitor_identification"]["micro"]["f1"] <= 1.0
    assert body["method"]["customer_signals"].startswith("rule")


def test_competitor_endpoints(client):
    body = client.get("/competitors").json()
    names = {r["competitor"] for r in body["league"]}
    assert {"HDFC", "LIC HF", "Bajaj HF", "Local Bank", "Shubham HF", "Aadhar HF"} <= names
    assert all(not e["competitor"] in {"HDFC", "LIC HF"} for e in body["emerging"])
    assert body["gap_win_curve_all"] and "none" in body["pricing_effect"]

    profile = client.get("/competitors/LICHFL")
    assert profile.status_code == 200 and profile.json()["competitor"] == "LIC HF"
    missing = client.get("/competitors/Nobody%20Finance")
    assert missing.status_code == 404 and "known" in missing.json()["detail"]


def test_context_endpoint(client):
    from aavas.api.competitors import CONTEXT_EXAMPLE

    response = client.post("/competitors/context", json=CONTEXT_EXAMPLE)
    assert response.status_code == 200, response.text
    body = response.json()
    shubham = body["competitors"][0]
    assert shubham["competitor"] == "Shubham HF" and shubham["quote"] == 12.47
    assert shubham["benchmark"]["level"] is not None and shubham["benchmark"]["n"] > 0
    assert shubham["plausibility"]["checked"] is True
    assert shubham["win_rate_at_gap"]["available"] is True
    assert all(h["loan_id"] != "AAV0009223" for h in body["comparable_negotiations"])
