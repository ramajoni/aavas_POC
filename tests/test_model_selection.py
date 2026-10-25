"""The AutoML portfolio search: out of time, every trial recorded, the deck's rule
applied literally, and the selection actually being what the trainers fit.

The search itself is exercised on a small synthetic frame so the suite stays
fast; the windows and the trainer hand-off are checked on the real table.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aavas.automl import portfolio
from aavas.features.feature_pipeline import FeatureSpec, spec
from aavas.models import base


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def report_dir(tmp_path, monkeypatch):
    """Selections and reports go to a temp dir, never reports/automl/portfolio."""
    monkeypatch.setattr(portfolio, "REPORT_DIR", tmp_path)
    return tmp_path


def _frame(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"price": rng.uniform(8, 14, n), "income": rng.lognormal(10, 0.4, n),
                      "score": rng.normal(700, 50, n), "city": rng.integers(0, 5, n).astype(float)})
    logit = -1.2 * (X["price"] - 11) + 0.002 * (X["score"] - 700) + 0.3 * (X["city"] == 2)
    y_bin = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    y_rate = 9 + 0.004 * (800 - X["score"]) + 0.2 * X["city"] + rng.normal(0, 0.2, n)
    return X, y_bin, y_rate.to_numpy()


def _data(task: str, monotone: dict | None = None, n: int = 600) -> portfolio.SelectionData:
    X, y_bin, y_rate = _frame(n)
    y = y_bin if task == "binary" else y_rate
    cut = int(n * 0.75)
    feature_spec = FeatureSpec("synthetic", numeric=["price", "income", "score"],
                               categorical=["city"], monotone=monotone or {})
    return portfolio.SelectionData(
        name="synthetic", task=task, X_fit=X.iloc[:cut], y_fit=y[:cut], X_valid=X.iloc[cut:],
        y_valid=y[cut:], spec=feature_spec, target="y",
        quantiles=[0.1, 0.5, 0.9] if task == "quantile" else None)


# ------------------------------------------------------------------ catalogue
def test_every_task_has_the_required_families():
    names = {task: {f.name for f in portfolio.families(task)} for task in portfolio.TASKS}
    assert {"ridge", "random_forest", "extra_trees", "hist_gradient_boosting", "xgboost",
            "lightgbm", "catboost"} <= names["regression"]
    assert {"logistic_regression", "random_forest", "extra_trees", "hist_gradient_boosting",
            "xgboost", "lightgbm", "catboost"} <= names["binary"]
    assert {"gradient_boosting_quantile", "hist_gradient_boosting_quantile",
            "lightgbm_quantile", "xgboost_quantile", "catboost_quantile"} <= names["quantile"]


@pytest.mark.parametrize("name,key", [
    ("gradient_boosting_quantile", "alpha"), ("hist_gradient_boosting_quantile", "quantile"),
    ("lightgbm_quantile", "alpha"), ("xgboost_quantile", "quantile_alpha")])
def test_quantile_families_carry_alpha(name, key):
    fam = portfolio.family("quantile", name)
    if fam.available():
        pytest.skip(fam.available())
    assert fam.construct(fam.defaults, alpha=0.9).get_params()[key] == 0.9


def test_catboost_quantile_carries_alpha_in_its_loss():
    fam = portfolio.family("quantile", "catboost_quantile")
    if fam.available():
        pytest.skip(fam.available())
    assert fam.construct(fam.defaults, alpha=0.1).get_params()["loss_function"] == "Quantile:alpha=0.1"


# --------------------------------------------------------------------- search
def test_every_trial_is_recorded_with_params_metrics_and_time():
    data = _data("regression")
    report = portfolio.search(data, budget=2, only=["ridge", "hist_gradient_boosting"])
    assert report["trials"] == len(report["leaderboard"]) == 4
    for row in report["leaderboard"]:
        assert row["status"] == "ok"
        assert row["params"] and row["fit_seconds"] is not None
        assert {"mae_bps", "rmse_bps", "r2"} <= set(row["metrics"])
    assert report["metric"] == "mae_bps"
    assert "AutoGluon" not in report["framework"] and "aavas.automl.portfolio" in report["framework"]


def test_quantile_and_binary_metrics_are_the_ones_the_deck_names():
    q = portfolio.search(_data("quantile"), budget=1, only=["hist_gradient_boosting_quantile"])
    assert q["metric"] == "pinball_bps"
    assert {"coverage_pct", "mean_width_bps", "nominal_coverage_pct"} <= set(q["selected"]["validation"])
    b = portfolio.search(_data("binary"), budget=1, only=["hist_gradient_boosting"])
    assert b["metric"] == "auc" and b["higher_is_better"]
    assert {"log_loss", "brier", "ece"} <= set(b["selected"]["validation"])


def test_trial_zero_is_the_incumbent_when_it_is_in_the_family():
    data = _data("regression")
    data.incumbent = {"family": "hist_gradient_boosting", "params": {"max_iter": 77}}
    report = portfolio.search(data, budget=2, only=["hist_gradient_boosting"])
    first = next(r for r in report["leaderboard"] if r["trial"] == 0)
    assert first["origin"] == "incumbent" and first["params"] == {"max_iter": 77}


def test_the_search_is_seeded():
    one = portfolio.search(_data("regression"), budget=3, only=["ridge"], seed=7)
    two = portfolio.search(_data("regression"), budget=3, only=["ridge"], seed=7)
    assert [r["params"] for r in one["leaderboard"]] == [r["params"] for r in two["leaderboard"]]


def test_a_missing_library_is_skipped_with_a_reason_not_raised(monkeypatch):
    real = portfolio._catalogue

    def without_xgboost():
        cat = real()
        for fam in cat["regression"]:
            if fam.name == "xgboost":
                fam.library = "xgboost_is_not_installed_here"
        return cat

    monkeypatch.setattr(portfolio, "_catalogue", without_xgboost)
    report = portfolio.search(_data("regression"), budget=1, only=["xgboost", "ridge"])
    skipped = {s["family"]: s for s in report["families_skipped"]}
    assert skipped["xgboost"]["kind"] == "unavailable" and "not importable" in skipped["xgboost"]["reason"]
    assert report["families_considered"] == ["ridge"]


def test_a_monotone_contract_excludes_families_that_cannot_carry_it_and_says_why():
    data = _data("binary", monotone={"price": -1})
    report = portfolio.search(data, budget=1, only=["logistic_regression", "catboost",
                                                    "hist_gradient_boosting"])
    excluded = {s["family"]: s["reason"] for s in report["families_skipped"] if s["kind"] == "excluded"}
    assert "logistic_regression" in excluded and "price" in excluded["logistic_regression"]
    assert "catboost" in excluded and "SHAP" in excluded["catboost"]
    assert report["families_considered"] == ["hist_gradient_boosting"]
    assert report["monotone_constraints"] == {"price": -1}


def test_every_monotone_capable_family_honours_the_constraint():
    """Not a parameter check: the fitted model is scored along price and must
    never rise. A constraint that is set but not obeyed is worse than none."""
    data = _data("binary", monotone={"price": -1}, n=800)
    columns = list(data.X_fit.columns)
    vector = portfolio._monotone(data.spec, columns)
    grid = pd.DataFrame(np.repeat(data.X_valid.iloc[:40].to_numpy(), 25, axis=0), columns=columns)
    grid["price"] = np.tile(np.linspace(8, 14, 25), 40)
    for fam in portfolio.families("binary"):
        if fam.available() or fam.monotone is None or fam.monotone_unexplainable:
            continue
        params = {k: (40 if k in ("n_estimators", "max_iter", "iterations") else v)
                  for k, v in fam.defaults.items()}
        est = fam.construct(params, monotone=vector).fit(data.X_fit, data.y_fit)
        p = est.predict_proba(grid)[:, 1].reshape(40, 25)
        assert (np.diff(p, axis=1) <= 1e-9).all(), f"{fam.name} rose with price"


def test_failed_trials_are_recorded_not_raised(monkeypatch):
    real = portfolio._catalogue

    def broken():
        cat = real()
        for fam in cat["regression"]:
            if fam.name == "hist_gradient_boosting":
                fam.defaults = {"max_iter": -5}
        return cat

    monkeypatch.setattr(portfolio, "_catalogue", broken)
    report = portfolio.search(_data("regression"), budget=1,
                              only=["hist_gradient_boosting", "ridge"])
    failed = [r for r in report["leaderboard"] if r["status"] == "failed"]
    assert failed and failed[0]["error"] and report["trials_failed"] == 1
    assert report["selected"]["family"] == "ridge"


# ----------------------------------------------------------------------- rule
def _best(family, tier, value):
    return {"family": family, "explainability_tier": tier, "metrics": {"auc": value, "mae": value}}


def test_best_score_wins_outside_the_tolerance():
    winner, how = portfolio.choose([_best("xgboost", 2, 0.80), _best("logistic_regression", 1, 0.70)],
                                   "auc", True, 0.01)
    assert winner["family"] == "xgboost" and how["decided_by"] == "validation score"


def test_within_the_tolerance_the_more_explainable_family_wins():
    winner, how = portfolio.choose([_best("extra_trees", 3, 0.800), _best("xgboost", 2, 0.797),
                                    _best("logistic_regression", 1, 0.795)], "auc", True, 0.01)
    assert winner["family"] == "logistic_regression"
    assert how["decided_by"].startswith("explainability")
    assert set(how["within_tolerance"]) == {"extra_trees", "xgboost", "logistic_regression"}


def test_the_tie_break_works_for_a_loss_too():
    winner, _ = portfolio.choose([_best("catboost", 2, 10.0), _best("ridge", 1, 10.05),
                                  _best("random_forest", 3, 9.99)], "mae", False, 0.01)
    assert winner["family"] == "ridge"


# --------------------------------------------------------- record and build
def test_record_writes_the_report_and_the_selection(report_dir):
    report = portfolio.search(_data("regression"), budget=1, only=["ridge"])
    path = portfolio.record(report)
    assert path == report_dir / "synthetic.json" and json.loads(path.read_text())["model"] == "synthetic"
    entry = portfolio.selected("synthetic")
    assert entry["family"] == "ridge" and entry["metric"] == "mae_bps"
    assert {"family", "params", "metric", "value", "selected_at", "report"} <= set(entry)


def test_record_keeps_other_models_selections(report_dir):
    for name in ("one", "two"):
        report = portfolio.search(_data("regression"), budget=1, only=["ridge"])
        report["model"] = name
        portfolio.record(report)
    assert set(portfolio.selections()) == {"one", "two"}


def test_build_falls_back_when_nothing_was_selected(report_dir):
    est, label = portfolio.build("nothing", "binary", None, lambda: ("fallback", "incumbent"))
    assert (est, label) == ("fallback", "incumbent")


def test_build_constructs_the_selected_family_with_the_contract_constraints(report_dir):
    (report_dir / "selection.json").write_text(json.dumps(
        {"conv": {"family": "hist_gradient_boosting", "task": "binary",
                  "params": {"max_iter": 33}}}))
    contract = FeatureSpec("conv", numeric=["price", "income"], categorical=[],
                           monotone={"price": -1})
    est, label = portfolio.build("conv", "binary", contract, lambda: (None, "fallback"))
    assert label == "hist_gradient_boosting"
    assert est.get_params()["max_iter"] == 33 and list(est.get_params()["monotonic_cst"]) == [-1, 0]


def test_build_falls_back_when_the_selected_library_is_missing(report_dir, monkeypatch):
    (report_dir / "selection.json").write_text(json.dumps(
        {"m": {"family": "lightgbm", "task": "regression", "params": {}}}))
    monkeypatch.setattr(portfolio.Family, "available", lambda self: "not installed in this test")
    assert portfolio.build("m", "regression", None, lambda: ("fb", "hist"))[1] == "hist"


def test_build_refuses_a_selection_that_cannot_carry_the_monotone_contract(report_dir):
    (report_dir / "selection.json").write_text(json.dumps(
        {"m": {"family": "logistic_regression", "task": "binary", "params": {}}}))
    contract = FeatureSpec("m", numeric=["price"], monotone={"price": -1})
    assert portfolio.build("m", "binary", contract, lambda: ("fb", "hgb"))[1] == "hgb"


# ------------------------------------------------------- real windows / trainers
@pytest.mark.parametrize("module_name", ["stage1", "stage2", "risk", "conversion"])
def test_selection_data_never_touches_the_trainers_holdout(training_data, module_name):
    import importlib

    module = importlib.import_module(f"aavas.models.{module_name}.train")
    data = module.selection_data(training_data)
    _, holdout = module.split(training_data)
    seen = set(data.X_fit.index) | set(data.X_valid.index)
    assert not seen & set(holdout.index), "holdout rows reached the search"
    w = data.window
    assert w["fit"]["to"] < w["valid"]["from"], "a date sits on both sides of fit/valid"
    assert w["valid"]["to"] < w["holdout"]["from"], "the search saw a holdout date"
    assert w["holdout"]["rows"] == len(holdout) and not w["holdout"]["shown_to_search"]
    assert list(data.X_fit.columns) == list(data.X_valid.columns)
    assert data.incumbent and data.target


def test_selection_frames_are_the_trainers_frames(training_data):
    """Same builder, same contract, transformer fitted on the fit slice only."""
    from aavas.models.risk import train as risk_train

    data = risk_train.selection_data(training_data)
    assert list(data.X_fit.columns) == spec("risk").columns
    dev, holdout = risk_train.split(training_data)
    fit, valid = portfolio.split(dev, holdout_df=holdout)
    transformer, ((X_fit, _), (X_valid, _)) = base.design(
        spec("risk"), risk_train.features, "defaulted", fit, valid, binary=True)
    pd.testing.assert_frame_equal(X_fit, data.X_fit)
    pd.testing.assert_frame_equal(X_valid, data.X_valid)


def test_a_trainer_fits_the_recorded_selection_and_its_card_says_so(training_data, report_dir):
    from aavas.models.risk import train as risk_train

    (report_dir / "selection.json").write_text(json.dumps(
        {"risk": {"family": "lightgbm", "task": "binary", "params": {"n_estimators": 40},
                  "metric": "auc", "value": 0.7, "report": "reports/automl/portfolio/risk.json"}}))
    card = risk_train.train(training_data.sample(4000, random_state=1), save=False)
    assert card.learner == "lightgbm"
    assert card.selection["status"] == "selected" and card.selection["report"].endswith("risk.json")


def test_without_a_selection_the_trainer_fits_its_fallback(training_data, report_dir):
    from aavas.models.conversion import train as conversion_train

    card = conversion_train.train(training_data.sample(4000, random_state=2), save=False)
    assert card.learner == "hist_gradient_boosting_classifier"
    assert card.selection["status"] == "no_selection"
    # The trainer's own checks survive the refactor.
    assert "monotonicity_violations_pct" in card.metrics and card.metrics["monotonicity_violations_pct"] == 0


def test_stage1_fits_one_selected_model_per_quantile(training_data, report_dir, monkeypatch):
    from aavas.models.stage1 import train as stage1_train

    (report_dir / "selection.json").write_text(json.dumps(
        {"stage1": {"family": "hist_gradient_boosting_quantile", "task": "quantile",
                    "params": {"max_iter": 40}}}))
    saved = {}
    monkeypatch.setattr(base, "save", lambda name, payload, card: saved.update(payload=payload))
    card = stage1_train.train(training_data.sample(4000, random_state=3), save=True)
    assert card.learner == "hist_gradient_boosting_quantile"
    models = saved["payload"]["models"]
    assert sorted(m.get_params()["quantile"] for m in models.values()) == [0.1, 0.5, 0.9]
    assert {"coverage_pct", "mean_width_bps", "median_mae_bps"} <= set(card.metrics)
