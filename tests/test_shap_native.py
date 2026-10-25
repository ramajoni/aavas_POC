"""Exact per-loan SHAP for every family the portfolio search can select.

The property that matters is reconciliation: base value + the contributions must
equal the model's own output for that loan - the rate for a regressor, the
log-odds margin for a boosted or linear classifier, the probability for a forest.
An explanation that does not add up to the number on the screen is not one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aavas.automl import portfolio
from aavas.features.feature_pipeline import FeatureSpec, FeatureTransformer
from aavas.models import base
from aavas.xai.shap_explainer import PricingExplainer, global_weights

SPEC = FeatureSpec("native", numeric=["price", "income", "score"], categorical=["city"])
EXPECTED_ENGINE = {
    "xgboost": "xgboost-native-treeshap", "xgboost_quantile": "xgboost-native-treeshap",
    "lightgbm": "lightgbm-native-treeshap", "lightgbm_quantile": "lightgbm-native-treeshap",
    "catboost": "catboost-native-shap", "catboost_quantile": "catboost-native-shap",
    "ridge": "linear-exact", "logistic_regression": "linear-exact",
}


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(11)
    n = 900
    raw = pd.DataFrame({"price": rng.uniform(8, 14, n), "income": rng.lognormal(10, 0.4, n),
                        "score": rng.normal(700, 50, n),
                        "city": rng.choice(["Jaipur", "Pune", "Indore", "Surat"], n)})
    logit = -1.1 * (raw["price"] - 11) + 0.004 * (raw["score"] - 700) + 0.4 * (raw["city"] == "Pune")
    raw["y_bin"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    raw["y_rate"] = 9 + 0.004 * (800 - raw["score"]) + 0.3 * (raw["city"] == "Pune") \
        + rng.normal(0, 0.15, n)
    transformer = FeatureTransformer(SPEC).fit(raw)
    return raw, transformer, transformer.transform(raw)


def _small(fam):
    return {k: (40 if k in ("n_estimators", "max_iter", "iterations") else v)
            for k, v in fam.defaults.items()}


def _cases():
    for task in portfolio.TASKS:
        for fam in portfolio.families(task):
            yield pytest.param(task, fam.name, id=f"{task}-{fam.name}")


@pytest.mark.parametrize("task,name", list(_cases()))
def test_contributions_reconcile_with_the_model_output(frame, task, name):
    raw, transformer, X = frame
    fam = portfolio.family(task, name)
    if fam.available():
        pytest.skip(fam.available())
    est = fam.construct(_small(fam), alpha=0.5 if task == "quantile" else None,
                        categorical=(3,))
    y = raw["y_bin"] if task == "binary" else raw["y_rate"]
    est.fit(X, y)
    explainer = PricingExplainer(est, transformer, is_classifier=task == "binary")

    assert explainer.method() == "shap", explainer.fallback_reason
    if name in EXPECTED_ENGINE:
        assert explainer.engine == EXPECTED_ENGINE[name]
    rows = X.iloc[:60]
    values = explainer.contributions(rows)
    assert values.shape == rows.shape
    reconstructed = explainer.base_value + values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, explainer.raw_output(rows), atol=1e-4)
    if task != "binary":
        # For a regressor the raw output is the prediction itself.
        np.testing.assert_allclose(reconstructed, est.predict(rows), atol=1e-4)
    expected_units = ("output" if task != "binary" else
                      "probability" if name in ("random_forest", "extra_trees") else "log_odds")
    assert explainer.units == expected_units


def test_one_loan_reconciles_through_explain(frame):
    """The per-loan path (what the pricing screen calls) adds up, not just the batch."""
    raw, transformer, X = frame
    import xgboost

    est = xgboost.XGBRegressor(n_estimators=40, max_depth=3).fit(X, raw["y_rate"])
    explainer = PricingExplainer(est, transformer)
    one = raw.iloc[[7]]
    total = explainer.base_value + sum(a.contribution for a in explainer.explain(one))
    assert total == pytest.approx(float(est.predict(transformer.transform(one))[0]), abs=1e-4)


def test_calibration_wrappers_are_reached_through(frame):
    raw, transformer, X = frame
    import lightgbm

    model = base.calibrate(lightgbm.LGBMClassifier(n_estimators=30, verbose=-1), X,
                           raw["y_bin"].to_numpy(), "isotonic")
    explainer = PricingExplainer(model, transformer, is_classifier=True)
    assert explainer.engine == "lightgbm-native-treeshap" and explainer.units == "log_odds"
    inner = base.base_estimator(model)
    rows = X.iloc[:20]
    np.testing.assert_allclose(explainer.base_value + explainer.contributions(rows).sum(axis=1),
                               inner.predict(rows, raw_score=True), atol=1e-6)


def test_catboost_with_monotone_constraints_falls_back_and_says_why(frame):
    """Its SHAP values depend on the batch they are computed over, so it is not
    used for an explanation - and the explainer says which method ran instead."""
    raw, transformer, X = frame
    fam = portfolio.family("binary", "catboost")
    if fam.available():
        pytest.skip(fam.available())
    est = fam.construct(_small(fam), monotone=[-1, 0, 0, 0]).fit(X, raw["y_bin"])
    explainer = PricingExplainer(est, transformer, is_classifier=True)
    if explainer.method() == "shap":
        pytest.skip("this catboost build computes batch-independent SHAP under constraints")
    assert explainer.method() == "ablation"
    assert "batch-dependent" in explainer.describe()["fallback_reason"]


def test_the_ablation_fallback_names_itself_and_still_reconciles(frame):
    raw, transformer, X = frame
    from sklearn.ensemble import HistGradientBoostingRegressor

    est = HistGradientBoostingRegressor(max_iter=30).fit(X, raw["y_rate"])
    explainer = PricingExplainer(est, transformer, backend="ablation")
    assert explainer.method() == "ablation" and explainer.engine == "ablation"
    assert explainer.describe()["fallback_reason"]
    rows = X.iloc[:15]
    np.testing.assert_allclose(explainer.base_value + explainer.contributions(rows).sum(axis=1),
                               est.predict(rows), atol=1e-6)


def test_global_weights_are_shares_of_mean_absolute_shap(frame):
    raw, transformer, X = frame
    import xgboost

    est = xgboost.XGBRegressor(n_estimators=40, max_depth=3).fit(X, raw["y_rate"])
    result = global_weights(PricingExplainer(est, transformer), X)
    shares = [f["share_pct"] for f in result["features"]]
    assert sum(shares) == pytest.approx(100, abs=0.1)
    assert shares == sorted(shares, reverse=True)
    assert result["features"][0]["feature"] == "score"      # the strongest driver of y_rate
    assert result["engine"] == "xgboost-native-treeshap" and result["rows"] == len(X)


def test_the_trained_rate_model_is_explained_with_shap_not_ablation(bundle):
    """The regression this fixes: XGBoost 3.x's base_score broke shap's parser and
    the Stage-2 explanation silently became ablation."""
    explainer = PricingExplainer(bundle.stage2.model, bundle.stage2.transformer)
    assert explainer.method() == "shap", explainer.fallback_reason
    assert explainer.engine != "ablation"
