"""Risk-aware pricing and the four-lever offer engine under it.

What must hold: the band position and borderline flag follow the policy cut-offs; EMI and
FOIR at a rate come from the pipeline's own code; PD is rescored at each offer's FOIR; the
rates table holds the floor, ask, recommendation and card inside the policy band and the
envelope; the policy envelope moves with grade and tenure; an amount above eligibility, or
a stressed FOIR above the grade cap, fails; the fee changes value and APR and reaches no
model except through the one documented APR assumption; a batch scores like single offers;
and the analysis degrades, saying so, when there is no risk model. The recommendation, floor
and envelope are never changed by any of it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aavas.analytics import offer_evaluator as oe
from aavas.analytics import risk_pricing as rp
from aavas.api import access
from aavas.api.main import app
from aavas.config import policy
from aavas.features import stage2_features
from aavas.optimization import constraints

SNAP = {"loan_id": "T-1", "segment": "salaried_formal", "city": "Rajkot", "channel": "connector",
        "property_type": "resale_flat", "declared_monthly_income": 160000.0, "requested_loan_amount": 2800000.0,
        "requested_tenure_years": 7, "applicant_age": 45, "declared_obligations": 0.0, "bureau_score": 800,
        "new_to_credit": False, "verified_monthly_income": 158000.0, "appraised_property_value": 5900000.0,
        "sanctioned_loan_amount": 2800000.0, "tenure_years": 7, "bureau_vintage_months": 100,
        "delinquency_count_24m": 0, "banking_behaviour_score": 92.0, "deviation_count": 0, "legal_flag": 0,
        "technical_flag": 0, "risk_grade": "A", "competitor_rate": 13.9, "competitor_name": "ICICI Bank",
        "competitor_evidence_verified": True, "customer_negotiating": True, "prior_negotiation_rounds": 1,
        "existing_relationship": True, "ai_recommended_rate": 13.92}
PAYLOAD = {"application_snapshot": SNAP,
           "risk": {"pd": 0.0171, "lifetime_pd": 0.0685, "lgd": 0.18, "risk_grade": "A", "expected_loss_pct": 0.31,
                    "breakeven_rate": 9.01, "source": "model"},
           "envelope": {"card_rate": 14.5, "min_rate": 13.9, "max_rate": 14.5},
           "recommendation": {"recommended_rate": 13.92, "economics": {"expected_life_years": 6.9}}}
REQUEST = {"request_id": "RRQ-T", "requested_rate": 13.62, "recommended_rate": 13.92, "loan_amount": 2800000.0}


class StubRisk:
    """PD rises with the share of income going to EMIs (the direction the brief expects)."""
    card = None

    def __init__(self, grade: str | None = None):
        self.fixed_grade, self.seen = grade, []

    def assess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.seen.append(df.copy())
        pd_value = np.clip(0.01 + 0.05 * (df["foir"].to_numpy(float) - 0.3), 0.002, 0.2)
        cuts = [("A+", 0.01), ("A", 0.025), ("B", 0.05), ("C", 0.09), ("D", 1.0)]
        grade = [self.fixed_grade or next(g for g, c in cuts if p <= c) for p in pd_value]
        return pd.DataFrame({"pd": pd_value, "lifetime_pd": pd_value * 4, "lgd": 0.2, "risk_grade": grade,
                             "expected_loss_pct": pd_value * 0.2 * 100}, index=df.index)


class Recorder:
    """A model whose transformer remembers what it was given."""

    def __init__(self, fn):
        self.fn, self.seen = fn, []
        self.transformer = SimpleNamespace(transform=self._take)
        self.model = SimpleNamespace(predict_proba=self._proba)

    def _take(self, df):
        self.seen.append(df.copy())
        return df

    def _proba(self, X):
        p = np.clip(self.fn(X), 0.001, 0.999)
        return np.column_stack([1 - p, p])

    def life_from_probability(self, p):
        return np.clip(1.0 / np.maximum(1 - (1 - np.asarray(p)) ** 6, 1e-3), 0.5, 8.0)


def stub_bundle(grade: str | None = None, risk: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        risk=StubRisk(grade) if risk else None,
        conversion=Recorder(lambda X: 1 / (1 + np.exp(X["offered_rate"].to_numpy(float) - 14.0))),
        prepayment=Recorder(lambda X: 0.1 + 0.05 * (X["offered_rate"].to_numpy(float) - 13.0)))


def offers(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


# ------------------------------------------------------------------ band and borderline

@pytest.mark.parametrize("pd_value,position,borderline", [
    (0.0175, 0.5, False),      # middle of grade A (1.0% to 2.5%)
    (0.0105, 0.0333, True),    # within 10% of the band's width of A+
    (0.0240, 0.9333, True),    # within 10% of the band's width of B
    (0.0250, 1.0, True),       # exactly on the cut-off: still A, on the edge
    (0.0100, 0.0, True),       # the A+ cut-off: A+ is up to and including it
])
def test_band_position_and_borderline_flag_at_the_cutoffs(pd_value, position, borderline) -> None:
    grade = "A+" if pd_value <= 0.01 else "A"
    got = rp.band(SNAP, {"pd": pd_value, "risk_grade": grade}, 84)
    if grade == "A+":
        assert got["pd_low"] == 0.0 and got["pd_high"] == 0.01 and got["position_in_band"] == 1.0
        assert got["better_grade"] is None and got["worse_grade"] == "A"
        return
    assert got["pd_low"] == 0.01 and got["pd_high"] == 0.025
    assert got["position_in_band"] == pytest.approx(position, abs=1e-3)
    assert got["borderline"] is borderline


def test_a_grade_move_changes_the_card_by_the_policy_matrix_not_a_constant() -> None:
    got = rp.band(SNAP, {"pd": 0.0175, "risk_grade": "A"}, 84)
    matrix = policy()["pricing"]["matrix"]
    assert got["card_rate_this_grade"] == matrix["A"][1] and got["card_rate_better_grade"] == matrix["A+"][1]
    assert got["card_rate_worse_grade"] == matrix["B"][1]
    assert got["card_change_if_better_pts"] == pytest.approx(matrix["A+"][1] - matrix["A"][1])
    # another tenure column moves all three
    assert rp.band(SNAP, {"pd": 0.0175, "risk_grade": "A"}, 120)["card_rate_this_grade"] == matrix["A"][2]


# ------------------------------------------------------------------ the offer engine

def test_emi_and_foir_at_a_rate_come_from_the_pipelines_own_code() -> None:
    out = oe.evaluate_offers(PAYLOAD, offers(rate=[13.5, 14.5]), bundle=stub_bundle())
    for _, row in out.iterrows():
        frame = pd.DataFrame([{**SNAP, "ai_recommended_rate": row["rate"]}]).drop(columns=[], errors="ignore")
        assert row["emi"] == pytest.approx(float(constraints._emi(np.array([2800000.0]), np.array([row["rate"]]), np.array([7.0]))[0]), abs=1)
        assert row["foir"] == pytest.approx(float(stage2_features.build(frame)["foir"].iloc[0]), abs=1e-4)
        stressed = float(constraints._emi(np.array([2800000.0]), np.array([row["rate"] + 2.0]), np.array([7.0]))[0]) / 158000.0
        assert row["stressed_foir"] == pytest.approx(stressed, abs=1e-4)


def test_the_models_see_the_pipelines_foir_so_pd_moves_with_amount_and_tenure_not_with_the_rate() -> None:
    """The pipeline computes FOIR at its own assessment rate; the engine feeds the models exactly that."""
    fallback = pd.DataFrame([{k: v for k, v in SNAP.items() if k != "ai_recommended_rate"}])
    stored = float(stage2_features.build(fallback)["foir"].iloc[0])             # what the decision's FOIR is
    payload = {**PAYLOAD, "application_snapshot": {**SNAP, "foir": stored}}
    assert oe.foir_mode(payload["application_snapshot"]) == "fallback"
    b = stub_bundle()
    out = oe.evaluate_offers(payload, offers(rate=[13.0, 14.0, 15.0, 16.0]), bundle=b)
    assert out["foir"].is_monotonic_increasing, "affordability still follows the offer's rate"
    assert (out["model_foir"] == stored).all() or np.allclose(out["model_foir"], stored, atol=1e-4)
    assert out["pd_annual"].nunique() == 1 and out["acceptance_model"].is_monotonic_decreasing
    assert np.allclose(b.risk.seen[0]["foir"], stored, atol=1e-4), "the risk model was handed the pipeline's FOIR"
    moved = oe.evaluate_offers(payload, offers(rate=[14.0, 14.0], loan_amount=[2.8e6, 4.0e6], tenure_months=[84, 120]), bundle=stub_bundle())
    assert moved["model_foir"].iloc[0] == pytest.approx(stored, abs=1e-4)
    assert moved["model_foir"].iloc[1] != moved["model_foir"].iloc[0] and moved["pd_annual"].iloc[0] != moved["pd_annual"].iloc[1]
    assert moved["ltv"].iloc[1] > moved["ltv"].iloc[0] and moved["tenure_months"].tolist() == [84, 120]


def test_the_envelope_moves_with_tenure_band_and_grade() -> None:
    matrix = policy()["pricing"]["matrix"]
    by_tenure = oe.evaluate_offers(PAYLOAD, offers(rate=[14.5] * 3, tenure_months=[60, 84, 120]), bundle=stub_bundle("A"))
    assert by_tenure["card_rate"].tolist() == matrix["A"]
    assert by_tenure["floor_rate"].tolist() == [round(max(c - 0.6, 13.0), 2) for c in matrix["A"]]
    for g in ("A+", "B"):
        got = oe.evaluate_offers(PAYLOAD, offers(rate=[14.5]), bundle=stub_bundle(g))
        assert got["card_rate"].iloc[0] == matrix[g][1] and got["risk_grade"].iloc[0] == g
    plus = oe.evaluate_offers(PAYLOAD, offers(rate=[13.0, 13.5, 13.4]), bundle=stub_bundle("A+"))
    assert plus["floor_rate"].iloc[0] == 13.0                       # never below the product floor
    assert plus["in_band"].tolist() == [True, True, True]
    assert oe.evaluate_offers(PAYLOAD, offers(rate=[13.0]), bundle=stub_bundle("A"))["in_band"].iloc[0] is np.False_
    assert "rate_below_floor" in oe.evaluate_offers(PAYLOAD, offers(rate=[13.0]), bundle=stub_bundle("A"))["failures"].iloc[0]


def test_an_amount_above_eligibility_fails() -> None:
    out = oe.evaluate_offers(PAYLOAD, offers(rate=[14.0, 14.0], loan_amount=[2.8e6, 9.0e6]), bundle=stub_bundle())
    assert out["failures"].iloc[0] == [] and bool(out["eligible"].iloc[0])
    assert "amount_over_max_loan" in out["failures"].iloc[1] and not out["eligible"].iloc[1]
    assert "ltv_over_cap" in out["failures"].iloc[1]


def test_tenure_outside_policy_and_age_at_maturity_fail() -> None:
    out = oe.evaluate_offers(PAYLOAD, offers(rate=[14.5, 14.5], tenure_months=[24, 360]), bundle=stub_bundle())
    assert "tenure_outside_policy" in out["failures"].iloc[0] and "tenure_outside_policy" in out["failures"].iloc[1]
    assert "age_at_maturity_over_limit" in out["failures"].iloc[1]        # 45 + 30 years against the age limit


def test_stressed_foir_fails_above_the_grade_cap() -> None:
    plain = float(constraints._emi(np.array([2.8e6]), np.array([14.5]), np.array([7.0]))[0])
    payload = {**PAYLOAD, "application_snapshot": {**SNAP, "verified_monthly_income": plain / 0.485}}
    out = oe.evaluate_offers(payload, offers(rate=[14.5]), bundle=stub_bundle("A"))     # grade A cap is 50%
    row = out.iloc[0]
    assert row["foir"] <= row["foir_cap"] < row["stressed_foir"]
    assert "stressed_foir_over_cap" in row["failures"] and "foir_over_cap" not in row["failures"]


def test_the_fee_changes_value_and_apr_and_reaches_no_model_but_through_the_apr_assumption() -> None:
    published = policy()["economics"]["processing_fee_pct"]
    b = stub_bundle()
    out = oe.evaluate_offers(PAYLOAD, offers(rate=[14.0] * 4, fee_pct=[published, 0.0, 0.25, published + 0.4]), bundle=b)
    same, none, half, over = (out.iloc[i] for i in range(4))
    assert over["fee_pct"] == published and bool(over["fee_clipped"]), "never above the published fee"
    assert over["value_inr"] == same["value_inr"] and over["apr_pct"] == same["apr_pct"]
    # a bigger fee is more income if the loan is booked, and a dearer all-in cost lowers acceptance
    per_booked = lambda r: r["value_inr"] / r["acceptance_model"]                      # noqa: E731
    assert per_booked(same) > per_booked(half) > per_booked(none) and same["apr_pct"] > half["apr_pct"] > none["apr_pct"]
    assert same["acceptance_model"] < half["acceptance_model"] < none["acceptance_model"]
    assert len({same["value_inr"], half["value_inr"], none["value_inr"]}) == 3
    assert same["acceptance_rate_used"] == 14.0, "at the published fee the model sees the plain rate"
    assert none["acceptance_rate_used"] < half["acceptance_rate_used"] < 14.0
    # every model input other than the offered rate (and the gap to the competitor it drives) is identical
    scoring = b.conversion.seen[0].reset_index(drop=True)
    drop = ["offered_rate", "competitor_gap_bps"]
    for i in (1, 2, 3):
        pd.testing.assert_series_equal(scoring.drop(columns=drop).iloc[i], scoring.drop(columns=drop).iloc[0], check_names=False)
    for seen in b.prepayment.seen + b.risk.seen:
        rest = seen.drop(columns=drop, errors="ignore").reset_index(drop=True)
        for i in (1, 2, 3):
            pd.testing.assert_series_equal(rest.iloc[i], rest.iloc[0], check_names=False)
    assert "fee" in " ".join(out.attrs["limitations"]).lower()


def test_a_batch_scores_like_single_offers() -> None:
    many = offers(rate=[13.6, 13.92, 14.25, 14.5, 15.0], loan_amount=[2.8e6, 3.0e6, 2.5e6, 2.8e6, 3.2e6],
                  tenure_months=[84, 120, 60, 84, 96], fee_pct=[0.5, 0.2, 0.5, 0.0, 0.5])
    batch = oe.evaluate_offers(PAYLOAD, many, bundle=stub_bundle())
    for i in range(len(many)):
        single = oe.evaluate_offers(PAYLOAD, many.iloc[[i]], bundle=stub_bundle()).iloc[0]
        for column in ("emi", "foir", "stressed_foir", "pd_annual", "card_rate", "expected_loss_pct", "acceptance_model",
                       "apr_pct", "value_inr", "expected_life_years"):
            assert batch.iloc[i][column] == pytest.approx(single[column], abs=1e-6), column
        assert batch.iloc[i]["failures"] == single["failures"]


def test_missing_models_degrade_and_say_so() -> None:
    out = oe.evaluate_offers(PAYLOAD, offers(rate=[14.0, 14.5]), bundle=stub_bundle(risk=False))
    assert out["pd_annual"].tolist() == [0.0171, 0.0171] and out["risk_grade"].tolist() == ["A", "A"]
    assert any("risk model" in t for t in out.attrs["limitations"])


# ------------------------------------------------------------------ rates, peers, verdict

def test_the_rates_table_holds_the_four_marked_rates_inside_the_band_and_envelope() -> None:
    env = {"min_rate": 13.9, "max_rate": 14.5}
    grid = rp.rate_grid(env, 13.62, 13.92, 14.5)
    rates = dict(grid)
    assert rates[13.62] == ["ask"] and rates[13.9] == ["floor"] and rates[13.92] == ["recommended"] and rates[14.5] == ["card"]
    assert all(13.0 <= r <= 17.0 for r, _ in grid)
    steps = [r for r, marks in grid if not marks]
    assert steps == [14.0, 14.25] and all(13.9 <= r <= 14.5 for r in steps)
    # an ask outside the band is shown and scored at its own rate, never moved into it; marks at one rate share a row
    outside = dict(rp.rate_grid({"min_rate": 13.0, "max_rate": 13.5}, 12.0, 13.0, 13.5))
    assert outside[12.0] == ["ask"] and outside[13.0] == ["floor", "recommended"]
    assert all(13.0 <= r <= 13.5 for r, marks in outside.items() if not marks)


def test_percentile_maths() -> None:
    rates = np.array([13.0, 13.5, 14.0, 14.5, 15.0])
    assert rp.percentile(rates, 14.0) == 60.0 and rp.percentile(rates, 12.0) == 0.0
    assert rp.percentile(rates, 15.0) == 100.0 and rp.percentile(rates, None) is None
    assert rp.percentile(np.array([]), 14.0) is None


def test_peers_by_grade_with_the_tenure_band_only_when_there_are_enough_files() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for grade, base, n in (("A+", 13.4, 120), ("A", 14.6, 120), ("B", 15.8, 80), ("C", 16.5, 40)):
        for i in range(n):
            rows.append({"_grade": grade, "sanctioned_rate": base + rng.normal(0, 0.3), "defaulted": i % 10 == 0,
                         "tenure_years": 7 if i < 90 else 10, "_pd": 0.02, "_margin": 5.0})
    book = pd.DataFrame(rows)
    got = rp.peers(SNAP, {"risk_grade": "A"}, 84, 14.3, 14.6, book)
    assert [g["grade"] for g in got["grades"]] == ["A+", "A", "B", "C"]
    a = next(g for g in got["grades"] if g["grade"] == "A")
    assert a["this_file"] and a["files"] == 120 and a["realised_default_rate"] == pytest.approx(0.1)
    assert a["card_rate"] == 14.5 and a["floor_rate"] == 13.9
    assert a["rate_p25"] < a["median_rate"] < a["rate_p75"]
    here = got["this_file"]
    assert here["ask_percentile_same_grade"] < here["recommended_percentile_same_grade"]
    assert here["files_same_grade_and_tenure"] == 90 and here["ask_percentile_same_grade_and_tenure"] is not None
    few = rp.peers(SNAP, {"risk_grade": "A"}, 120, 14.3, 14.6, book)["this_file"]       # 30 files at 10 years
    assert few["files_same_grade_and_tenure"] == 30
    thin = rp.peers(SNAP, {"risk_grade": "A"}, 36, 14.3, 14.6, book)["this_file"]       # none up to 60 months
    assert thin["ask_percentile_same_grade_and_tenure"] is None and thin["ask_percentile_same_grade"] is not None


def _row(rate, pd_annual=0.0140, ok=True, margin=4.7):
    return {"rate": rate, "pd_annual": pd_annual, "stressed_foir_ok": ok, "stressed_foir": 0.52, "foir_cap": 0.5,
            "margin_over_risk_cost": margin}


QUIET = {"borderline": False}


def test_verdict_rules() -> None:
    below = rp.verdict(ask=13.6, floor=13.9, grade="A", band_info=QUIET, rows=[], ask_row=_row(13.6), rec_row=_row(13.92),
                       book_margin=4.8, recommended=13.92)
    assert below["tone"] == "bad" and below["text"].startswith("The ask of 13.60% is 0.30 points below the risk-based floor")
    fail = rp.verdict(ask=14.0, floor=13.9, grade="A", band_info=QUIET, rows=[_row(14.0), _row(14.5, ok=False)],
                      ask_row=_row(14.0), rec_row=_row(13.92), book_margin=4.7, recommended=13.92)
    assert fail["tone"] == "bad" and "14.50%" in fail["text"] and "limit" in fail["text"]
    edge = {"borderline": True, "pd_distance_to_better_grade": 0.0004, "pd_distance_to_worse_grade": 0.01,
            "better_grade": "A+", "worse_grade": "B", "card_change_if_better_pts": -1.0, "card_change_if_worse_pts": 1.0}
    near = rp.verdict(ask=14.0, floor=13.9, grade="A", band_info=edge, rows=[], ask_row=_row(14.0, margin=4.0),
                      rec_row=_row(13.92), book_margin=5.0, recommended=13.92)
    assert "grade A+ boundary" in near["text"] and "-1.00" in near["text"] and "below the 5.00" in near["points"][1]
    good = rp.verdict(ask=14.0, floor=13.9, grade="A", band_info=QUIET, rows=[], ask_row=_row(14.0, margin=5.5),
                      rec_row=_row(13.92), book_margin=5.0, recommended=13.92)
    assert good["tone"] == "good" and "above the 5.00" in good["text"]
    # the risk model does not move with the rate, so the verdict says nothing about PD and the EMI
    quiet = rp.verdict(ask=13.62, floor=13.0, grade="A", band_info=QUIET, rows=[], ask_row=_row(13.62, 0.0140),
                       rec_row=_row(13.92, 0.0140), book_margin=None, recommended=13.92)
    assert "lower EMI" not in quiet["text"] and quiet["text"]
    assert len(rp.verdict(ask=13.0, floor=13.9, grade="A", band_info=edge, rows=[_row(14, ok=False)], ask_row=_row(13.0),
                          rec_row=_row(13.92), book_margin=5.0, recommended=13.92)["text"].split(". ")) <= 2


# ------------------------------------------------------------------ the whole analysis

@pytest.fixture
def stubbed(monkeypatch):
    """The analysis end to end on stub models and a small book, no trained artefacts needed."""
    from aavas.governance import model_registry

    HELD.clear()
    book = pd.DataFrame([{"_grade": g, "sanctioned_rate": r, "defaulted": False, "tenure_years": 7, "_pd": 0.02,
                          "_margin": 5.0} for g in ("A+", "A", "B", "C") for r in (13.5, 14.0, 14.5, 15.0)])
    monkeypatch.setattr(model_registry, "load_bundle", lambda *a, **k: HELD['bundle'])
    monkeypatch.setattr(rp, "_book", lambda trained_at: (book, {}))
    monkeypatch.setattr(rp, "_shift", lambda city: (0.0, None))


HELD: dict = {}


def test_the_whole_analysis_without_a_risk_model_degrades_and_says_so(stubbed) -> None:
    HELD['bundle'] = stub_bundle(risk=False)
    out = rp.risk_pricing(REQUEST, PAYLOAD, {"rate": 14.0})
    assert any("risk model is not available" in t for t in out["limitations"])
    assert out["drivers"] == [] and out["score"]["pd_annual"] == 0.0171 and out["score"]["risk_grade"] == "A"
    assert {p for r in out["rates"] for p in r["marks"]} == {"floor", "ask", "recommended", "card"}
    assert len({r["pd_annual"] for r in out["rates"]}) == 1, "the fixed risk is used at every rate"
    assert out["note"].startswith("Decision support")
    # the engine's own numbers are untouched
    assert out["recommended_rate"] == 13.92 and out["floor_rate"] == 13.9 and out["card_rate"] == 14.5
    assert PAYLOAD["envelope"] == {"card_rate": 14.5, "min_rate": 13.9, "max_rate": 14.5}


def test_the_whole_analysis_with_a_risk_model(stubbed) -> None:
    HELD['bundle'] = stub_bundle()
    out = rp.risk_pricing(REQUEST, PAYLOAD, {"rate": 14.0})
    rows = {tuple(r["marks"]): r for r in out["rates"]}
    assert rows[("ask",)]["failures"] == ["rate_below_floor"] and rows[("recommended",)]["in_band"]
    assert len({r["pd_annual"] for r in out["rates"]}) == 1, "risk is read at a fixed assessment rate"
    assert "fixed assessment rate" in out["pd_note"]
    assert out["peers"]["this_file"]["grade"] == "A" and out["verdict"]["text"]
    assert all(13.0 <= r["rate"] <= 17.0 for r in out["rates"])


# ------------------------------------------------------------------ the route

PASSWORDS = {"rohit": "Rohit@2026", "meera": "Meera@2026"}
FILE = {k: SNAP[k] for k in ("segment", "city", "channel", "property_type", "declared_monthly_income", "applicant_age",
                              "declared_obligations", "bureau_score", "verified_monthly_income", "appraised_property_value",
                              "sanctioned_loan_amount", "tenure_years", "bureau_vintage_months", "delinquency_count_24m",
                              "banking_behaviour_score", "deviation_count", "legal_flag", "technical_flag")}
FILE.update({"requested_loan_amount": 2000000, "requested_tenure_years": 20, "estimated_property_value": 2900000,
             "declared_monthly_income": 55000, "verified_monthly_income": 53000, "appraised_property_value": 2900000,
             "sanctioned_loan_amount": 2000000, "tenure_years": 20, "declared_obligations": 5000, "bureau_score": 790,
             "city": "Jaipur", "applicant_age": 35,
             "sales_context": {"competitor_rate": 11.0, "competitor_name": "HDFC", "competitor_evidence_verified": False}})


@pytest.fixture
def client(monkeypatch, bundle) -> TestClient:   # noqa: F811 - the session fixture of the same name
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def who(client, name) -> dict[str, str]:
    token = client.post("/auth/login", json={"username": name, "password": PASSWORDS[name]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_the_area_manager_gets_the_analysis_and_the_rm_does_not(client) -> None:
    rohit = who(client, "rohit")
    priced = client.post("/pricing/final-assessment", json={"loan_id": "RSK-1", **FILE}, headers=rohit)
    assert priced.status_code == 200, priced.text
    rec = priced.json()["recommendation"]["recommended_rate"]
    asked = client.post("/pricing/rate-requests", headers=rohit,
                        json={"loan_id": "RSK-1", "rate": round(rec - 0.30, 2), "reason": "competitor_lower_rate", "note": "HDFC at 11.0"})
    assert asked.status_code == 200, asked.text
    request_id = asked.json()["request_id"]
    before = client.get(f"/pricing/rate-requests/{request_id}/scenarios", headers=who(client, "meera")).json()
    got = client.get(f"/pricing/rate-requests/{request_id}/risk-pricing", headers=who(client, "meera"))
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["recommended_rate"] == rec and body["score"]["risk_grade"] and body["verdict"]["text"]
    assert {m for r in body["rates"] for m in r["marks"]} >= {"ask", "recommended"}
    assert [g["grade"] for g in body["peers"]["grades"]] == ["A+", "A", "B", "C"]
    # the other screen's numbers did not move, and its rows match at the shared rates
    after = client.get(f"/pricing/rate-requests/{request_id}/scenarios", headers=who(client, "meera")).json()
    assert before["table"] == after["table"] and before["recommended_rate"] == after["recommended_rate"]
    shared = {r["rate"]: r for r in after["table"]}
    for r in body["rates"]:
        if r["rate"] in shared and r["acceptance_model"] is not None:
            assert r["acceptance_model"] == pytest.approx(shared[r["rate"]]["acceptance_model"], abs=0.02)
    assert client.get(f"/pricing/rate-requests/{request_id}/risk-pricing", headers=rohit).status_code == 403
    assert client.get("/pricing/rate-requests/RRQ-NOPE/risk-pricing", headers=who(client, "meera")).status_code == 404


def test_the_engine_reproduces_the_decision_and_the_scenario_table_on_a_real_request(client) -> None:
    """One FOIR convention: at the file's own amount, tenure and fee the engine gives the scenario
    table's acceptance (model and branch) at the ask and the recommendation, and the decision's PD."""
    from aavas.analytics import branch_scenarios
    from aavas.governance import decision_logger
    from aavas.intake import offers as stored_offers

    rohit, meera = who(client, "rohit"), who(client, "meera")
    priced = client.post("/pricing/final-assessment", json={"loan_id": "CNS-1", **FILE}, headers=rohit)
    assert priced.status_code == 200, priced.text
    rec = priced.json()["recommendation"]["recommended_rate"]
    made = client.post("/pricing/rate-requests", headers=rohit,
                       json={"loan_id": "CNS-1", "rate": round(rec - 0.30, 2), "reason": "competitor_lower_rate", "note": "HDFC at 11.0"})
    assert made.status_code == 200, made.text
    request_id = made.json()["request_id"]
    scen = client.get(f"/pricing/rate-requests/{request_id}/scenarios", headers=meera).json()
    row = stored_offers.request(request_id)
    payload = decision_logger.get_decision(row["decision_id"])["payload"]
    out = oe.evaluate_offers(payload, pd.DataFrame({"rate": [row["requested_rate"], rec]}))
    shift, problem = rp._shift(payload["application_snapshot"].get("city"))
    assert problem is None
    for i, mark in ((0, "requested"), (1, "recommended")):
        table = next(r for r in scen["table"] if r["mark"] == mark)
        p = float(out["acceptance_model"].iloc[i])
        assert p == pytest.approx(table["acceptance_model"], abs=0.0005), mark
        branch = float(branch_scenarios._sigmoid(branch_scenarios._logit(np.array([p])) + shift)[0])
        assert branch == pytest.approx(table["acceptance_branch"], abs=0.0005), mark
    assert float(out["pd_annual"].iloc[1]) == pytest.approx(payload["risk"]["pd"], abs=1e-6)
    assert float(out["pd_annual"].iloc[0]) == pytest.approx(payload["risk"]["pd"], abs=1e-6)
    assert float(out["model_foir"].iloc[0]) == pytest.approx(payload["application_snapshot"]["foir"], abs=1e-6)
