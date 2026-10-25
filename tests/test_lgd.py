"""The LGD model and its integration into the risk model.

What has to hold: the contract can only see what is known when the file is
priced; the model is judged against the policy table on the same rows and the
card says honestly which one won; and the risk model degrades to the table -
saying so per row - when the artefact is missing or cannot score a frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aavas.config import model_config
from aavas.features import stage2_features
from aavas.features.feature_pipeline import FeatureTransformer, spec
from aavas.ingestion.schemas import (DECISION_COLUMNS, LATENT_COLUMNS, OUTCOME_COLUMNS,
                                     SALES_CONTEXT_COLUMNS, TARGETS)
from aavas.models import base
from aavas.models.lgd import train as lgd_train
from aavas.models.lgd.predict import LGDModel
from aavas.models.risk import predict as risk_predict


# ------------------------------------------------------------------ contract

def test_the_lgd_contract_holds_no_outcome_latent_or_decision_column() -> None:
    columns = set(spec("lgd").columns)
    forbidden = set(OUTCOME_COLUMNS) | set(LATENT_COLUMNS) | set(DECISION_COLUMNS) | {
        "realized_lgd", "true_lgd", "defaulted", "risk_grade"}
    assert not columns & forbidden, columns & forbidden
    # Commercial pressure says nothing about what a recovery will fetch.
    assert not columns & set(SALES_CONTEXT_COLUMNS)


def test_the_lgd_contract_is_collateral_and_location() -> None:
    columns = set(spec("lgd").columns)
    assert {"ltv", "property_type", "legal_flag", "technical_flag", "city"} <= columns
    assert spec("lgd").monotone.get("ltv") == 1


def test_the_target_is_the_observed_severity() -> None:
    assert TARGETS["lgd"] == "realized_lgd"


def test_rows_are_defaulted_disbursed_files_with_an_observed_severity() -> None:
    frame = pd.DataFrame({
        "disbursed": [True, True, True, False],
        "defaulted": [True, False, True, False],
        "realized_lgd": [0.31, np.nan, np.nan, np.nan],
    })
    assert lgd_train.rows(frame).index.tolist() == [0]


# ------------------------------------------------------------- the benchmark

@pytest.fixture(scope="module")
def fresh_card():
    """Trained in memory on the real table - about two seconds, nothing saved."""
    try:
        return lgd_train.train(save=False)
    except FileNotFoundError:
        pytest.skip("dataset not generated")


def test_the_policy_table_is_scored_on_exactly_the_holdout_rows(fresh_card) -> None:
    _, holdout = lgd_train.split(lgd_train.table())
    y = holdout[TARGETS["lgd"]].to_numpy(float)
    table = stage2_features.lgd(stage2_features.build(holdout)).to_numpy(float)
    assert fresh_card.n_test == len(holdout)
    assert fresh_card.metrics["policy_mae_pts"] == pytest.approx(np.abs(table - y).mean() * 100)
    assert fresh_card.metrics["policy_rmse_pts"] == pytest.approx(
        np.sqrt(((table - y) ** 2).mean()) * 100)


def test_the_card_says_honestly_whether_the_model_beats_the_table(fresh_card) -> None:
    m = fresh_card.metrics
    text = " ".join(fresh_card.limitations)
    for metric, label in (("mae", "MAE"), ("rmse", "RMSE")):
        gain = m[f"policy_{metric}_pts"] - m[f"{metric}_pts"]
        assert m[f"beats_policy_{metric}"] == float(gain > 0)
        assert f"{label} {m[f'{metric}_pts']:.1f} vs {m[f'policy_{metric}_pts']:.1f} pts " \
               f"({'better' if gain > 0 else 'worse'}" in text
        assert m[f"{metric}_gain_low_pts"] <= m[f"{metric}_gain_pts"] <= m[f"{metric}_gain_high_pts"]
    assert "Thin data" in text


def test_the_headline_metrics_the_portfolio_page_reads_exist(fresh_card) -> None:
    for key in (model_config().get("portfolio") or {}).get("headline_metrics", {}).get(
            "lgd", ["mae_pts", "rmse_pts", "r2"]):
        assert key in fresh_card.metrics


def test_selection_data_never_shows_the_holdout() -> None:
    data = lgd_train.selection_data()
    assert data.task == "regression" and data.metric == "mae_pts" and data.unit() == "pts"
    assert data.window["holdout"]["shown_to_search"] is False
    assert data.window["valid"]["to"] < data.window["holdout"]["from"]
    assert list(data.X_fit.columns) == spec("lgd").columns


# ----------------------------------------------------------- scoring + risk

class _Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class _Failing:
    def predict(self, df):
        raise KeyError("appraised_property_value")


@pytest.fixture(scope="module")
def applications(training_data) -> pd.DataFrame:
    return training_data[training_data["disbursed"].astype(bool)].head(25).copy()


def _lgd_model(value: float) -> LGDModel:
    transformer = FeatureTransformer(spec("lgd"))
    frame = stage2_features.build(pd.DataFrame({
        "ltv": [0.6, 0.8], "appraised_property_value": [3e6, 2e6],
        "sanctioned_loan_amount": [1.8e6, 1.6e6], "legal_flag": [0, 1], "technical_flag": [0, 0],
        "property_type": ["resale_flat", "builder_flat"], "city": ["Jaipur", "Indore"],
        "segment": ["salaried_formal", "salaried_informal"]}))
    transformer.fit(frame)
    return LGDModel(_Constant(value), transformer, clip=(0.0, 0.95))


def test_predictions_are_clipped_to_the_configured_range(applications) -> None:
    assert (_lgd_model(1.7).predict(applications) == 0.95).all()
    assert (_lgd_model(-0.2).predict(applications) == 0.0).all()


def _risk(lgd_model) -> risk_predict.RiskModel:
    if not base.exists("risk"):
        pytest.skip("risk artefact not trained")
    model = risk_predict.RiskModel.load()
    model.lgd_model = lgd_model
    return model


def test_risk_falls_back_to_the_policy_table_without_an_lgd_model(applications) -> None:
    risk = _risk(None)
    out = risk.assess(applications)
    table = stage2_features.lgd(stage2_features.build(applications)).to_numpy(float)
    assert (out["lgd_source"] == "policy_table").all()
    np.testing.assert_allclose(out["lgd"], np.round(np.maximum(table, risk.lgd_floor), 4))
    np.testing.assert_allclose(out["expected_loss_pct"],
                               out["pd"] * out["lgd"] * 100, atol=1e-3)  # inputs are rounded


def test_risk_uses_the_lgd_model_and_floors_it(applications) -> None:
    floor = float(model_config()["risk"]["lgd_floor"])
    assert 0 < floor < 0.5
    high = _risk(_lgd_model(0.5)).assess(applications)
    assert (high["lgd_source"] == "model").all() and (high["lgd"] == 0.5).all()
    low = _risk(_lgd_model(0.01)).assess(applications)
    assert (low["lgd"] == floor).all(), "the regulatory floor must bind on a model output below it"
    # The PD maths and its units are untouched by where LGD came from.
    np.testing.assert_allclose(high["pd"], low["pd"])
    np.testing.assert_allclose(high["expected_loss_pct"], high["pd"] * 0.5 * 100, atol=1e-3)


def test_risk_degrades_to_the_table_when_the_lgd_model_cannot_score(applications) -> None:
    out = _risk(_Failing()).assess(applications)
    assert (out["lgd_source"] == "policy_table").all()


def test_risk_load_falls_back_when_the_lgd_artefact_is_absent(monkeypatch, applications) -> None:
    if not base.exists("risk"):
        pytest.skip("risk artefact not trained")
    real_exists = base.exists
    monkeypatch.setattr(base, "exists", lambda name: False if name == "lgd" else real_exists(name))
    risk = risk_predict.RiskModel.load()
    assert risk.lgd_model is None
    assert (risk.assess(applications)["lgd_source"] == "policy_table").all()


def test_the_saved_artefact_loads_and_scores(applications) -> None:
    if not base.exists("lgd"):
        pytest.skip("lgd artefact not trained")
    model = LGDModel.load()
    values = model.predict(applications)
    assert ((values >= 0) & (values <= 0.95)).all()
    assert model.card.name == "lgd" and model.card.dataset.get("dataset_id")
    assert model.importance.get("method") and model.importance.get("features")
    assert model.calibration.get("model") and model.calibration.get("policy_table")
