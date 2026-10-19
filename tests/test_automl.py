"""Agentic AutoML.

The tests that earn their place are the ones guarding a property the search would
otherwise lose quietly: the monotone contract, calibration measured off the rows
it was fitted on, and tracking that never takes a search down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aavas.automl import (ALGORITHMS, AutoMLSearch, SearchConfig, WoEScorecard,
                          calibration, ensembles, fairness, hpo, imbalance,
                          inventory, locality, registry, selection, summary, tracking)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    n = 1200
    X = pd.DataFrame({
        "rate": rng.uniform(9.0, 14.0, n),
        "bureau": rng.normal(700, 60, n),
        "ltv": rng.uniform(0.4, 0.9, n),
    })
    logit = 3.0 - 0.45 * X["rate"] + 0.004 * (X["bureau"] - 700) - 1.2 * X["ltv"]
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    groups = rng.choice(["salaried", "self_employed"], n)
    cut = int(n * 0.7)
    return dict(Xtr=X.iloc[:cut], ytr=y[:cut], Xva=X.iloc[cut:], yva=y[cut:],
                gva=groups[cut:])


# ------------------------------------------------------------------ registry
def test_every_catalogue_family_is_registered():
    expected = {"linear_scorecard", "tree", "boosting", "interpretable", "kernel",
                "instance", "probabilistic", "neural_tabular", "foundation",
                "sequential", "graph", "imbalanced", "cost_sensitive"}
    assert expected.issubset(set(registry.FAMILIES))
    assert len(ALGORITHMS) >= 40


def test_unavailable_algorithms_say_why_rather_than_vanishing():
    """The reason must also say whether it is *fixable*.

    A missing package is an install. A missing adapter is work someone could do.
    A missing data shape is not fixable without inventing the data - and the three
    used to read alike, which made the last one look like an oversight.
    """
    rows = {r["name"]: r for r in inventory()}
    assert all(r["available"] or r["reason"] for r in inventory())

    assert rows["gcn"]["reason"].startswith("no graph data in the schema")
    assert "identity columns" in rows["gcn"]["reason"], (
        "the graph reason must say what the only available edges are, and why they "
        "are held out")

    assert rows["lstm"]["reason"].startswith("no sequence data in the schema")
    assert "per-period rows per loan" in rows["lstm"]["reason"], (
        "the sequence reason must say what data would make it available")

    # Availability first: an available algorithm has `reason` None, and `in None`
    # raises rather than evaluating false. Written the other way round, this test
    # passed only while xgboost happened to be missing.
    assert rows["xgboost"]["available"] or "pip install" in rows["xgboost"]["reason"]
    assert rows["tabnet"]["available"] or "pytorch_tabnet" in rows["tabnet"]["reason"], (
        "an unimplemented adapter must name the package it would need")


def test_summary_counts_are_consistent():
    s = summary()
    assert s["registered"] == len(ALGORITHMS)
    assert s["available"] <= s["registered"]


def test_monotone_filter_excludes_learners_that_cannot_carry_constraints():
    names = {a.name for a in registry.candidates(monotone_required=True)}
    assert "knn" not in names and "svm_rbf" not in names
    assert names, "no monotone-capable learner is available"


# ------------------------------------------------------------- WoE scorecard
def test_woe_scorecard_fits_and_produces_a_points_table(data):
    model = WoEScorecard(n_bins=5).fit(data["Xtr"], data["ytr"])
    p = model.predict_proba(data["Xva"])[:, 1]
    assert p.min() >= 0 and p.max() <= 1
    table = model.scorecard(list(data["Xtr"].columns))
    assert {"feature", "bin", "woe", "points"} <= set(table[0])


# ------------------------------------------------------------------- HPO
@pytest.mark.parametrize("strategy", ["grid", "random", "tpe", "bayesian",
                                      "hyperband", "successive_halving", "evolutionary"])
def test_every_hpo_strategy_finds_the_optimum(strategy):
    result = hpo.optimize({"x": ("float", 0.0, 10.0)},
                          lambda p: -((p["x"] - 3.0) ** 2),
                          strategy=strategy, budget=25)
    assert abs(result.best_params["x"] - 3.0) < 1.5
    assert result.trials


def test_hpo_defaults_to_a_chronological_split():
    from sklearn.model_selection import TimeSeriesSplit

    assert isinstance(hpo.cv(3), TimeSeriesSplit)


def test_empty_search_space_is_not_an_error():
    assert hpo.optimize({}, lambda p: 1.0, strategy="random").best_params == {}


# -------------------------------------------------------------- imbalance
def test_diagnose_does_not_recommend_resampling_a_balanced_book():
    y = np.random.default_rng(0).binomial(1, 0.66, 500)
    assert imbalance.diagnose(y)["recommended_strategy"] == "none"


@pytest.mark.parametrize("strategy", ["random_oversample", "random_undersample",
                                      "class_weight", "scale_pos_weight", "smote"])
def test_every_imbalance_strategy_flags_that_recalibration_is_required(strategy, data):
    out = imbalance.apply(data["Xtr"], data["ytr"], strategy)
    assert out.recalibration_required is True


def test_none_strategy_leaves_the_data_and_the_base_rate_alone(data):
    out = imbalance.apply(data["Xtr"], data["ytr"], "none")
    assert out.applied is False and out.recalibration_required is False
    assert len(out.y) == len(data["ytr"])


# ------------------------------------------------------------- calibration
@pytest.mark.parametrize("method", ["platt", "isotonic", "beta"])
def test_calibration_is_monotone_so_the_price_constraint_survives(method):
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 800)
    y = rng.binomial(1, np.clip(p * 0.8 + 0.1, 0, 1))
    assert calibration.is_monotone(calibration.fit(p, y, method))


def test_calibration_improves_a_deliberately_miscalibrated_score():
    rng = np.random.default_rng(5)
    y = rng.binomial(1, 0.5, 1500)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.4, 1500), 0.001, 0.999)  # badly scaled
    fitted = calibration.fit(p, y, "isotonic")
    assert fitted.metrics["ece_after"] <= fitted.metrics["ece_before"]


# ---------------------------------------------------------------- ensembles
def test_ensemble_refuses_to_mix_monotone_and_non_monotone_members():
    with pytest.raises(ValueError, match="not monotone-capable"):
        ensembles.build("voting_soft", ["hist_gradient_boosting", "knn"],
                        monotone_required=True)


def test_ensemble_of_monotone_members_is_allowed_and_reports_monotone():
    spec = ensembles.build("weighted", ["hist_gradient_boosting", "woe_scorecard"],
                           monotone_required=True)
    assert spec.monotone is True


def test_weighted_ensemble_predicts_in_range(data):
    spec = ensembles.build("weighted", ["logistic_regression", "decision_tree"])
    spec.estimator.fit(data["Xtr"], data["ytr"])
    p = spec.estimator.predict_proba(data["Xva"])[:, 1]
    assert p.min() >= 0 and p.max() <= 1


# ----------------------------------------------------------------- fairness
def test_calibration_gap_is_caught_when_parity_looks_perfect():
    """Why calibration is the primary criterion: parity can be flawless while the
    model is badly wrong for one group."""
    rng = np.random.default_rng(0)
    n = 800
    g = rng.choice(["a", "b"], n)
    y = rng.binomial(1, np.where(g == "a", 0.3, 0.6))
    p = np.full(n, 0.3) + rng.normal(0, 0.01, n)
    result = fairness.evaluate(y, p, g)
    assert result["passed"] is False
    assert result["gaps"]["selection_rate_gap"] == 0.0
    assert result["gaps"]["calibration_gap"] > 0.1


def test_thin_groups_return_insufficient_evidence():
    y = np.array([0, 1] * 10)
    p = np.full(20, 0.5)
    g = np.array(["tiny"] * 20)
    per_group = fairness.by_group(y, p, g)
    assert per_group["tiny"]["verdict"] == "insufficient evidence"


# ---------------------------------------------------------------- selection
def test_selection_is_not_argmax_auc():
    strong_but_miscalibrated = selection.Candidate(
        "a", "boosting", metrics={"auc": 0.90, "ks": 0.5, "ece": 0.30, "psi": 0.01})
    weaker_but_calibrated = selection.Candidate(
        "b", "linear_scorecard", metrics={"auc": 0.82, "ks": 0.45, "ece": 0.01, "psi": 0.01})
    result = selection.select([strong_but_miscalibrated, weaker_but_calibrated])
    assert result.champion.name == "b"          # gated out on calibration, not outranked
    assert result.referred_to_review[0]["name"] == "a"


def test_a_gate_failure_refers_rather_than_promotes():
    bad = selection.Candidate("x", "tree", metrics={"auc": 0.55, "ece": 0.01, "psi": 0.01})
    result = selection.select([bad])
    assert result.champion is None
    assert "AUC" in result.referred_to_review[0]["failures"][0]


def test_monotone_gate_blocks_a_non_monotone_learner():
    candidate = selection.Candidate("knn", "instance", monotone=False,
                                    metrics={"auc": 0.95, "ece": 0.01, "psi": 0.01})
    failures = selection.check_gates(candidate, {"require_monotone": True})
    assert any("monotone" in f for f in failures)


def test_pareto_front_is_not_empty_when_profit_is_absent():
    a = selection.Candidate("a", "f", metrics={"auc": 0.8, "ece": 0.02})
    b = selection.Candidate("b", "f", metrics={"auc": 0.7, "ece": 0.01})
    assert selection.pareto_front([a, b])


# ----------------------------------------------------------------- tracking
def test_tracking_status_reports_the_connection():
    status = tracking.status()
    assert set(status) >= {"mlflow_installed", "tracking_enabled", "tracking_uri"}


def test_tracker_is_inert_when_tracking_is_disabled(monkeypatch):
    monkeypatch.setenv("AAVAS_TRACKING_ENABLED", "false")
    with tracking.Tracker("t") as tracker:
        assert tracker.active is False
        tracker.log_metrics({"auc": 0.9})       # must not raise
        with tracker.child("c") as child:
            child.log_params({"a": 1})


# -------------------------------------------------------------------- search
def test_search_scopes_candidates_to_the_contract(data):
    cfg = SearchConfig(contract="conversion", hpo_budget=2,
                       include=["hist_gradient_boosting", "woe_scorecard", "knn"])
    result = AutoMLSearch(cfg).run(data["Xtr"], data["ytr"], data["Xva"], data["yva"])
    names = {row["name"] for row in result.selection.leaderboard}
    assert "knn" not in names, "a non-monotone learner reached a monotone contract"
    assert any(s["name"] == "knn" for s in result.candidates_skipped)


def test_search_runs_the_whole_flow_and_records_each_step(data):
    cfg = SearchConfig(contract="generic", hpo_budget=2,
                       include=["logistic_regression", "decision_tree"],
                       ensemble_methods=["weighted"])
    result = AutoMLSearch(cfg).run(data["Xtr"], data["ytr"], data["Xva"], data["yva"],
                                   groups_valid=data["gva"])
    steps = {s["step"] for s in result.steps}
    assert {"algorithm_selection", "imbalance", "validation_split",
            "champion_selection"} <= steps
    assert result.selection.leaderboard


def test_calibration_is_not_measured_on_the_rows_it_was_fitted_on(data):
    """A calibrator fitted and scored on the same rows reports ~0 ECE for anything."""
    cfg = SearchConfig(contract="generic", hpo_budget=2, include=["logistic_regression"])
    result = AutoMLSearch(cfg).run(data["Xtr"], data["ytr"], data["Xva"], data["yva"])
    split = next(s for s in result.steps if s["step"] == "validation_split")
    assert split["calibration_rows"] > 0 and split["evaluation_rows"] > 0
    ece = result.selection.leaderboard[0]["ece"]
    assert ece > 0.0, "ECE of exactly zero means it was measured in-sample"


def test_search_survives_a_learner_that_cannot_fit(data):
    cfg = SearchConfig(contract="generic", hpo_budget=2,
                       include=["logistic_regression", "gaussian_nb"])
    bad = data["Xtr"].copy()
    result = AutoMLSearch(cfg).run(bad, data["ytr"], data["Xva"], data["yva"])
    assert result.selection.leaderboard


def test_config_loads_per_contract_overrides():
    cfg = SearchConfig.from_yaml("prepayment")
    assert cfg.contract == "prepayment"
    assert cfg.monotone_required is True
    assert "beta" in cfg.calibration_methods


# ------------------------------------------------------------ branch level
def _branch_data(seed=11, n=2400, weak_branch=True):
    rng = np.random.default_rng(seed)
    branch = rng.choice([f"BR{i:03d}" for i in range(1, 9)], n)
    signal = rng.normal(0, 1, n)
    # One branch is deliberately unpredictable: its outcome ignores the signal.
    strength = np.where((branch == "BR001") & weak_branch, 0.0, 2.0)
    y = rng.binomial(1, 1 / (1 + np.exp(-(strength * signal))))
    p = 1 / (1 + np.exp(-(2.0 * signal)))
    return y, p, branch


def test_branch_table_reports_every_unit_worst_first():
    y, p, branch = _branch_data()
    per_unit = locality.by_unit(y, p, branch)
    rows = locality.table(per_unit)
    assert len(rows) == len(set(branch))
    aucs = [r["auc"] for r in rows if r.get("auc") is not None]
    assert aucs == sorted(aucs), "table should be worst-first"


def test_a_global_pass_with_one_failing_branch_is_a_fail():
    """The whole point: the headline number hides the branch being priced badly."""
    y, p, branch = _branch_data()
    result = locality.evaluate(y, p, branch)
    assert result["global"]["auc"] > 0.70, "the global model should look healthy"
    assert result["passed"] is False
    assert any(f["unit"] == "BR001" for f in result["failing_units"])
    assert "Do not promote on the global number alone" in result["verdict"]


def test_a_uniformly_good_model_passes_every_branch():
    y, p, branch = _branch_data(weak_branch=False)
    assert locality.evaluate(y, p, branch)["passed"] is True


def test_thin_branches_return_insufficient_evidence_not_a_number():
    y, p, branch = _branch_data(n=300)
    per_unit = locality.by_unit(y, p, branch, min_size=200)
    assert all(v.get("verdict") == "insufficient evidence" for v in per_unit.values())


def test_variance_names_the_worst_branch():
    y, p, branch = _branch_data()
    spread = locality.variance(locality.by_unit(y, p, branch))
    assert spread["auc"]["worst_unit"] == "BR001"
    assert spread["auc"]["spread"] > 0


def test_locality_failure_blocks_promotion_through_the_gate():
    y, p, branch = _branch_data()
    candidate = selection.Candidate(
        "m", "boosting", metrics={"auc": 0.78, "ece": 0.01, "psi": 0.01},
        locality=locality.evaluate(y, p, branch))
    failures = selection.check_gates(candidate)
    assert any("branch BR001" in f for f in failures)
    assert selection.select([candidate]).champion is None


def test_search_accepts_units_and_carries_the_branch_verdict(data):
    units = np.array(["BR001", "BR002"] * (len(data["yva"]) // 2 + 1))[: len(data["yva"])]
    cfg = SearchConfig(contract="generic", hpo_budget=2, include=["logistic_regression"],
                       gates={"require_locality_pass": False})
    result = AutoMLSearch(cfg).run(data["Xtr"], data["ytr"], data["Xva"], data["yva"],
                                   units_valid=units)
    assert "locality_passed" in result.selection.leaderboard[0]


def test_locality_config_block_is_present():
    from aavas.config import load

    gates = load("automl")["defaults"]["gates"]
    assert gates["require_locality_pass"] is True
    assert load("automl")["defaults"]["locality"]["unit_column"] == "branch_id"


def test_a_fully_referred_search_still_names_what_would_have_won():
    """When every candidate fails a gate, "nothing was admitted" is only half the
    answer. The review question is which candidate led and which gate stopped it."""
    weak = selection.Candidate("a", "f", metrics={"auc": 0.80, "ece": 0.40})
    weaker = selection.Candidate("b", "f", metrics={"auc": 0.62, "ece": 0.45})
    chosen = selection.select([weak, weaker], gates={"max_ece": 0.05})

    assert chosen.champion is None
    assert chosen.best_referred is weak
    assert chosen.as_dict()["best_referred"] == "a"
    assert {r["name"] for r in chosen.referred_to_review} == {"a", "b"}


def test_best_referred_is_empty_when_a_champion_was_admitted():
    good = selection.Candidate("a", "f", metrics={"auc": 0.80, "ece": 0.01, "psi": 0.01})
    chosen = selection.select([good])
    assert chosen.champion is good and chosen.best_referred is None
