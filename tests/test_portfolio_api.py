"""GET /models/portfolio: the five deck models, in deck order, with the evidence
behind each - and an honest status when a piece of that evidence is missing or
out of step with another."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aavas.api import portfolio as portfolio_api
from aavas.api.main import app
from aavas.automl import portfolio
from aavas.config import model_config

DECK_ORDER = ["Indicative pricing", "Risk", "Risk-adjusted pricing", "Acceptance",
              "Pricing uplift"]


@pytest.fixture()
def empty(tmp_path, monkeypatch):
    """No reports, no weights, no artefacts."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(portfolio, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(portfolio_api, "WEIGHTS_FILE", tmp_path / "reports" / "global_weights.json")
    monkeypatch.setenv("AAVAS_MODEL_DIR", str(tmp_path / "models"))
    return tmp_path


def _card(directory, name, learner, trained_at="2026-10-20T10:00:00+00:00", **metrics):
    version = model_config()["registry"]["version"]
    (directory / "models" / f"{name}__{version}.card.json").write_text(json.dumps(
        {"name": name, "version": version, "learner": learner, "target": "defaulted",
         "trained_at": trained_at, "n_train": 100, "n_test": 20,
         "metrics": metrics or {"auc": 0.74, "brier": 0.08, "calibration_gap": 0.01}}))


def _report(directory, name, family, value=0.744):
    report = {"model": name, "task": "binary", "target": "defaulted", "metric": "auc",
              "higher_is_better": True, "units": "bps", "framework": "aavas.automl.portfolio (test)",
              "families_considered": ["logistic_regression", "xgboost"],
              "families_skipped": [{"family": "catboost", "kind": "unavailable", "reason": "x"}],
              "trials": 16, "selected": {"family": family, "label": portfolio.label(family),
                                         "params": {"C": 1.0}, "validation": {"auc": value}},
              "best_per_family": [{"family": family, "metrics": {"auc": value}}],
              "rule": {"decided_by": "validation score", "within_tolerance": [family]},
              "windows": {"holdout": {"rows": 2637, "from": "2025-12-24", "to": "2026-04-22"}},
              "generated_at": "2026-10-20T12:00:00+00:00"}
    portfolio.record(report)


def _weights(directory, name, trained_at="2026-10-20T10:00:00+00:00"):
    (directory / "reports" / "global_weights.json").write_text(json.dumps({
        "generated_at": "2026-10-20T13:00:00+00:00",
        "models": {name: {"artefact": {"learner": "logistic_regression", "trained_at": trained_at},
                          "method": "shap", "engine": "linear-exact", "units": "log_odds",
                          "holdout": {"rows": 2637},
                          "top": [{"feature": f"f{i}", "label": f"F{i}", "share_pct": 30 - i,
                                   "rank": i + 1} for i in range(7)]}},
        "skipped": {}}))


def test_the_five_deck_models_come_back_in_deck_order(empty):
    body = TestClient(app).get("/models/portfolio").json()
    assert [m["title"] for m in body["models"]] == DECK_ORDER
    risk = next(m for m in body["models"] if m["key"] == "risk")
    assert [c["model"] for c in risk["components"]] == ["risk", "lgd"]
    assert all(m["question"] for m in body["models"])
    assert {r["key"] for r in body["rules"]} == {"optimisation", "guardrails"}
    assert [s["model"] for s in body["supporting"]] == ["prepayment", "override"]


def test_titles_and_questions_come_from_config(empty, monkeypatch):
    cfg = model_config()["portfolio"]
    monkeypatch.setitem(cfg["models"][0], "question", "A question set in config")
    body = TestClient(app).get("/models/portfolio").json()
    assert body["models"][0]["question"] == "A question set in config"


def test_nothing_on_disk_is_reported_as_missing_not_invented(empty):
    body = TestClient(app).get("/models/portfolio").json()
    for model in body["models"]:
        assert model["status"] == "missing"
        for c in model["components"]:
            assert c["automl"] is None and c["card"] is None
            assert any("run scripts/run_model_selection.py" in i for i in c["issues"])
    assert body["automl_table"] == []


def test_a_complete_model_shows_its_evidence(empty):
    _card(empty, "risk", "logistic_regression")
    _report(empty, "risk", "logistic_regression")
    _weights(empty, "risk")
    body = TestClient(app).get("/models/portfolio").json()
    risk = next(c for m in body["models"] for c in m["components"] if c["model"] == "risk")
    assert risk["status"] == "ready" and not risk["issues"]
    assert risk["automl"]["selected"]["label"] == "Logistic regression"
    assert risk["automl"]["trials"] == 16 and risk["automl"]["value"] == 0.744
    assert risk["automl"]["metric_text"] == "AUC 0.744"
    assert set(risk["card"]["holdout_metrics"]) == {"auc", "brier", "calibration_gap"}
    assert len(risk["weights"]["top"]) == model_config()["portfolio"]["top_weights"]
    row = next(r for r in body["automl_table"] if r["model"] == "risk")
    assert row["deck_model"] == "Risk" and row["trials"] == 16 and row["families"] == 2
    assert body["selection_rule"]["holdout"]["from"] == "2025-12-24"


def test_an_artefact_that_is_not_the_selection_is_flagged(empty):
    _card(empty, "risk", "hist_gradient_boosting_classifier")
    _report(empty, "risk", "logistic_regression")
    _weights(empty, "risk")
    risk = next(c for m in TestClient(app).get("/models/portfolio").json()["models"]
                for c in m["components"] if c["model"] == "risk")
    assert risk["status"] == "partial"
    assert any("retrain to fit the selection" in i for i in risk["issues"])


def test_weights_from_an_older_artefact_are_flagged(empty):
    _card(empty, "risk", "logistic_regression", trained_at="2026-10-21T09:00:00+00:00")
    _report(empty, "risk", "logistic_regression")
    _weights(empty, "risk", trained_at="2026-10-20T10:00:00+00:00")
    risk = next(c for m in TestClient(app).get("/models/portfolio").json()["models"]
                for c in m["components"] if c["model"] == "risk")
    assert any("earlier artefact" in i for i in risk["issues"])


def test_rule_components_carry_their_live_settings(empty):
    body = TestClient(app).get("/models/portfolio").json()
    optimisation = next(r for r in body["rules"] if r["key"] == "optimisation")
    values = {s["key"]: s["value"] for s in optimisation["settings"]}
    assert values["optimizer.min_conversion_probability"] is not None
