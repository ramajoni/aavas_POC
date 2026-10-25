"""The pricing uplift model: contract, metric, learners, served curve and endpoint.

The properties that make it usable rather than merely fitted: it sees only what
the RM sees; Qini is computed correctly (checked by hand); every learner's curve
is monotone in the discount; the synthetic truth never reaches a feature; and the
endpoint refuses plainly (503) when there is no model instead of guessing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aavas.config import feature_config
from aavas.features.feature_pipeline import FeatureTransformer, spec
from aavas.ingestion.schemas import DECISION_COLUMNS, LATENT_COLUMNS, OUTCOME_COLUMNS
from aavas.models import base
from aavas.models.uplift import evaluation as ev
from aavas.models.uplift import learners as lr
from aavas.models.uplift import train as up

# What the RM sees at the moment of deciding includes the engine's recommendation
# and the card rate; every other decision column is the decision itself.
PRE_DECISION = {"ai_recommended_rate", "policy_card_rate"}
POST_DECISION_DERIVED = {"offered_rate", "was_overridden", "override_direction",
                         "total_discount_bps", "sanctioned", "treated"}


# ------------------------------------------------------------------ contract

def test_the_uplift_contract_is_pre_decision_context_only() -> None:
    columns = set(spec("uplift").columns)
    forbidden = (set(OUTCOME_COLUMNS) | set(LATENT_COLUMNS) | POST_DECISION_DERIVED
                 | (set(DECISION_COLUMNS) - PRE_DECISION))
    assert not columns & forbidden, columns & forbidden
    assert {"ai_recommended_rate", "competitor_gap_bps", "customer_negotiating"} <= columns


def test_the_discount_is_the_treatment_with_its_constraint_in_the_contract() -> None:
    block = feature_config()["uplift"]["dose"]
    assert block["column"] == lr.DOSE and block["monotone"] == 1
    assert lr.DOSE not in spec("uplift").columns


def test_the_competitor_gap_is_measured_at_the_recommendation_not_after_the_discount() -> None:
    frame = pd.DataFrame({"ai_recommended_rate": [11.0], "sanctioned_rate": [10.6], "ltv": [0.7],
                          "competitor_rate": [10.5], "competitor_gap_bps": [10.0],
                          "override_bps": [-40.0]})
    built = up.features(frame)
    assert built["competitor_gap_bps"].iloc[0] == pytest.approx(50.0)


def test_rows_keep_discounts_and_untouched_files_and_drop_premiums() -> None:
    frame = pd.DataFrame({"override_bps": [-12.0, 0.0, 8.0, np.nan]})
    out = up.rows(frame)
    assert out.index.tolist() == [0, 1]
    assert out["treated"].tolist() == [1, 0] and out[lr.DOSE].tolist() == [12.0, 0.0]


def test_latent_truth_never_reaches_the_features(training_data) -> None:
    sample = up.rows(training_data.head(400))
    transformer = FeatureTransformer(spec("uplift"))
    with_latent = transformer.fit_transform(up.features(sample))
    without = transformer.transform(up.features(sample.drop(columns=LATENT_COLUMNS)))
    pd.testing.assert_frame_equal(with_latent, without)


# -------------------------------------------------------------------- Qini

def test_qini_on_a_hand_built_example() -> None:
    # Ranked 4 > 3 > 2 > 1. Treated at ranks 1 and 3, control at 2 and 4; only the
    # first treated and the last control accept. Normalised by the treated count
    # (2), the Qini curve is 0, .5, .5, .5, 0 at 0, 1/4, 1/2, 3/4, 1: area .375,
    # random line ends at 0, so the coefficient is .375.
    y, t, score = [1, 0, 0, 1], [1, 0, 1, 0], [4, 3, 2, 1]
    x, gain = ev.qini_curve(y, t, score)
    np.testing.assert_allclose(x, [0, .25, .5, .75, 1])
    np.testing.assert_allclose(gain, [0, .5, .5, .5, 0])
    assert ev.qini(y, t, score) == pytest.approx(0.375)
    # Uplift curve (rate_t - rate_c) * x is 0, 0, .5, .375, 0 -> area .21875.
    assert ev.auuc(y, t, score) == pytest.approx(0.21875)
    # The reverse ranking puts the accepting control first.
    assert ev.qini(y, t, [1, 2, 3, 4]) < 0


def test_qini_is_zero_when_the_model_cannot_rank() -> None:
    y, t = [1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0]
    assert ev.qini(y, t, [0.3] * 6) == pytest.approx(0.0)


def test_qini_weights_are_scale_free_and_matter() -> None:
    rng = np.random.default_rng(0)
    y, t, s = rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.random(200)
    w = rng.uniform(0.5, 3.0, 200)
    assert ev.qini(y, t, s, w) == pytest.approx(ev.qini(y, t, s, 7 * w))
    assert ev.qini(y, t, s, w) != pytest.approx(ev.qini(y, t, s))


def test_a_ranking_by_the_true_effect_beats_random() -> None:
    rng = np.random.default_rng(1)
    n = 20000
    effect = rng.uniform(0, 0.5, n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.25 + t * effect).astype(int)
    assert ev.qini(y, t, effect) > 0.02
    assert abs(ev.qini(y, t, rng.random(n))) < 0.02


# ----------------------------------------------------------- synthetic truth

def test_true_uplift_is_zero_at_zero_and_grows_with_the_discount(training_data) -> None:
    sample = up.rows(training_data.head(300))
    assert np.allclose(ev.true_uplift(sample, 0.0), 0.0)
    small, large = ev.true_uplift(sample, 10.0), ev.true_uplift(sample, 40.0)
    assert (small > 0).all() and (large >= small).all()


def test_true_uplift_is_unavailable_on_real_data(training_data) -> None:
    assert ev.true_uplift(training_data.head(5).drop(columns=["price_elasticity"]), 10.0) is None


# ----------------------------------------------------------------- learners

@pytest.fixture(scope="module")
def small_fit(training_data):
    """Every learner fitted on a small slice with small trees - for shape, not skill."""
    sample = up.rows(training_data).head(3000)
    transformer = FeatureTransformer(spec("uplift"))
    X = transformer.fit_transform(up.features(sample))
    t = sample["treated"].to_numpy(int)
    d = sample[lr.DOSE].to_numpy(float)
    y = sample["disbursed"].astype(int).to_numpy()
    _, e = lr.propensity(X, t, {"max_iter": 60, "max_depth": 3}, 2, (0.03, 0.97))
    return X, t, d, y, e


@pytest.mark.parametrize("name", lr.LEARNERS)
def test_every_learner_draws_a_curve_monotone_in_the_discount(small_fit, name) -> None:
    X, t, d, y, e = small_fit
    params = {"max_iter": 40, "max_depth": 3, "min_samples_leaf": 40}
    learner = lr.make(name, "hist_gradient_boosting", params, folds=2).fit(X, t, d, y, e)
    curve = np.vstack([learner.uplift(X.head(300), dose, e[:300]) for dose in (0, 5, 15, 30, 60)])
    assert (curve >= 0).all()
    assert (np.diff(curve, axis=0) >= -1e-12).all(), f"{name} bends down somewhere"
    assert np.allclose(curve[0], 0.0)
    base_rate = learner.baseline(X.head(300), e[:300])
    assert ((base_rate >= 0) & (base_rate <= 1)).all()


def test_a_selection_for_an_unknown_learner_falls_back(monkeypatch) -> None:
    from aavas.automl import portfolio

    monkeypatch.setattr(portfolio, "selected", lambda name: None)
    assert up.choice()["selection"]["status"] == "no_selection"
    monkeypatch.setattr(portfolio, "selected", lambda name: {
        "family": "hist_gradient_boosting", "params": {"learner": "nope"}, "report": "x"})
    chosen = up.choice()
    assert chosen["selection"]["status"] == "fallback"
    assert chosen["learner"] == up.cfg()["fallback_learner"]


# ------------------------------------------------------------------- search

def test_the_search_report_matches_the_portfolio_schema(training_data, tmp_path, monkeypatch) -> None:
    from aavas.automl import portfolio

    subset = training_data.sort_values("application_date").iloc[::3]
    report = up.search(subset, budget=1, learners=["dose", "t"],
                       families=["hist_gradient_boosting"], record=False)
    for key in ("model", "task", "target", "framework", "families_considered", "families_skipped",
                "trials", "leaderboard", "best_per_family", "selected", "rule", "windows",
                "elapsed_seconds"):
        assert key in report, key
    assert report["task"] == "uplift" and report["metric"] == "qini"
    assert report["selected"]["params"]["learner"] in {"dose", "t"}
    assert {"family", "params", "validation"} <= set(report["selected"])
    assert report["windows"]["holdout"]["shown_to_search"] is False
    assert report["windows"]["valid"]["to"] < report["windows"]["holdout"]["from"]
    assert all(w["valid"]["to"] < report["windows"]["holdout"]["from"]
               for w in report["windows"]["rolling"])

    # Recording merges into selection.json without touching other models' entries.
    (tmp_path / "selection.json").write_text(json.dumps({"risk": {"family": "xgboost"}}))
    portfolio.record(report, directory=tmp_path)
    merged = json.loads((tmp_path / "selection.json").read_text())
    assert merged["risk"] == {"family": "xgboost"}
    assert merged["uplift"]["params"]["learner"] == report["selected"]["learner"]


# ---------------------------------------------------------------- served model

@pytest.fixture(scope="module")
def served():
    if not base.exists("uplift"):
        pytest.skip("uplift artefact not trained")
    from aavas.models.uplift.predict import UpliftModel

    return UpliftModel.load()


@pytest.fixture(scope="module")
def holdout_files(training_data):
    _, holdout = up.split(training_data)
    return holdout.head(200)


def test_the_served_curve_is_monotone_and_bounded(served, holdout_files) -> None:
    curve = served.curve(holdout_files, [0, 5, 10, 20, 30, 45, 60])
    wide = curve.pivot(index="row", columns="discount_bps", values="uplift_pts")
    assert (wide.diff(axis=1).iloc[:, 1:] >= -1e-9).all().all()
    assert (curve["uplift_low_pts"] <= curve["uplift_pts"] + 1e-9).all()
    assert (curve["uplift_pts"] <= curve["uplift_high_pts"] + 1e-9).all()
    assert curve.loc[curve["discount_bps"] > served.estimator.support["max_bps"],
                     "beyond_support"].all()


def test_assess_reports_lift_interval_label_and_value(served, holdout_files) -> None:
    from aavas.models.prepayment.predict import PrepaymentModel

    prepayment = PrepaymentModel.load() if base.exists("prepayment") else None
    out = served.assess(holdout_files.head(1), 15.0, expected_loss_pct=0.4,
                        prepayment_model=prepayment, expected_life_years=6.0)
    low, high = out["interval_pts"]
    assert low <= out["uplift_pts"] <= high
    assert out["label"] in {"discount buys little", "discount buys a modest lift",
                            "discount moves acceptance"}
    assert out["proposed_rate"] == pytest.approx(out["recommended_rate"] - 0.15, abs=0.006)
    assert out["value"]["available"] is True
    value = out["value"]
    assert value["value_effect_inr"] == pytest.approx(
        value["margin_given_up_inr"] + value["acceptance_bought_inr"], abs=2)
    no_value = served.assess(holdout_files.head(1), 15.0)
    assert no_value["value"]["available"] is False and no_value["value"]["reason"]
    with pytest.raises(ValueError):
        served.assess(holdout_files.head(1), -10.0)


def test_the_card_carries_the_truth_check_labelled_as_synthetic(served) -> None:
    metrics = served.card.metrics
    for key in ("qini", "auuc", "ate_pts", "naive_difference_pts", "ipw_difference_pts"):
        assert key in metrics
    assert metrics["monotonicity_violations_pct"] == 0.0
    truth = [k for k in metrics if "truth" in k]
    assert truth and all(k.startswith("synthetic_truth_") for k in truth)
    assert any("Synthetic-only" in line for line in served.card.limitations)


# ------------------------------------------------------------------ endpoint

@pytest.fixture
def client():
    from aavas.api.main import app

    yield TestClient(app)
    app.dependency_overrides.clear()


def test_the_endpoint_returns_503_without_an_artefact(client) -> None:
    from aavas.api.deps import get_orchestrator
    from aavas.api.main import app

    app.dependency_overrides[get_orchestrator] = lambda: SimpleNamespace(
        bundle=SimpleNamespace(uplift=None, risk=None, prepayment=None))
    response = client.post("/pricing/uplift", json={"decision_id": "DEC-X", "proposed_rate": 11.5})
    assert response.status_code == 503
    assert "not trained" in response.json()["detail"]["error"]


def test_the_endpoint_assesses_a_recorded_decision(client, bundle) -> None:
    if getattr(bundle, "uplift", None) is None:
        pytest.skip("uplift artefact not trained")
    from aavas.api import examples

    priced = client.post("/pricing/final-assessment", json=examples.FINAL_ASSESSMENT)
    assert priced.status_code == 200, priced.text
    decision = priced.json()
    rate = decision["recommendation"]["recommended_rate"]
    body = client.post("/pricing/uplift", json={"decision_id": decision["decision_id"],
                                                "proposed_rate": round(rate - 0.15, 2)})
    assert body.status_code == 200, body.text
    out = body.json()
    assert out["discount_bps"] == pytest.approx(15.0) and out["source"] == "decision"
    assert out["interval_pts"][0] <= out["uplift_pts"] <= out["interval_pts"][1]
    assert out["model"]["version"] and out["method"]
    assert out["value"]["available"] is True
    lifts = [p["uplift_pts"] for p in out["curve"]]
    assert lifts == sorted(lifts)

    premium = client.post("/pricing/uplift", json={"decision_id": decision["decision_id"],
                                                   "proposed_rate": round(rate + 0.2, 2)})
    assert premium.status_code == 422
    both = client.post("/pricing/uplift", json={"proposed_rate": 11.0})
    assert both.status_code == 422
    missing = client.post("/pricing/uplift", json={"decision_id": "DEC-NOPE", "proposed_rate": 11.0})
    assert missing.status_code == 404
