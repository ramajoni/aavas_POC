"""The feature-engineering funnel.

The funnel is a measurement, so the tests are about what makes a count honest:
every dataset column has a disposition and a reason, nothing forbidden reaches the
generators, nothing is approved without a human-authored label, a contract feature
that fails a screen is shown failing it, and the slide's "rejected" table holds
exactly what its title says. Contract sizes are read from the config, never
hard-coded - the contracts are expected to change.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aavas.api.features import feature_approvals
from aavas.api.main import app
from aavas.features import discovery, funnel
from aavas.features.feature_pipeline import spec
from aavas.ingestion import loader, schemas
from aavas.paths import CONFIG_DIR

MODELS = ["stage2", "risk", "conversion", "stage1"]
SAMPLE = 2000


def _config_bytes() -> str:
    return hashlib.sha256((CONFIG_DIR / "feature_config.yaml").read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return loader.build_training_table(save=False)


@pytest.fixture(scope="module")
def runs(frame):
    before = _config_bytes()
    reports = {m: funnel.run(m, frame=frame, sample=SAMPLE) for m in ("stage2", "stage1")}
    return {"reports": reports, "config_before": before, "config_after": _config_bytes()}


@pytest.fixture(scope="module")
def stage2(runs):
    return runs["reports"]["stage2"]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _view(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    return funnel.model_view(frame, model)


# ----------------------------------------------- 1. raw attributes and exclusions

@pytest.mark.parametrize("model", MODELS)
def test_every_column_has_a_disposition_and_every_exclusion_a_reason(frame, model) -> None:
    view = _view(frame, model)
    raw, excluded = funnel.classify_columns(view, model)
    kept = [r["column"] for r in raw]
    dropped = [e["column"] for e in excluded]
    assert sorted(kept + dropped) == sorted(view.columns), "the partition is not exact"
    assert not set(kept) & set(dropped)
    for e in excluded:
        assert e["reason"] in funnel.EXCLUSIONS, e
        assert len(e["why"]) > 20, e


@pytest.mark.parametrize("model", MODELS)
def test_outcome_latent_and_decision_columns_never_reach_the_raw_attributes(frame, model) -> None:
    """The deck's leakage evidence: true PD, LGD, acceptance and every
    post-decision field are excluded, and the exclusion says why."""
    view = _view(frame, model)
    raw, excluded = funnel.classify_columns(view, model)
    kept = {r["column"] for r in raw}
    reason = {e["column"]: e["reason"] for e in excluded}
    upstream = set(funnel.UPSTREAM_DECISIONS.get(model, {}))
    for column in schemas.OUTCOME_COLUMNS:
        if column in view.columns:
            assert column not in kept and reason[column] == "outcome", column
    for column in schemas.LATENT_COLUMNS:
        if column in view.columns:
            assert column not in kept and reason[column] == "latent", column
    for column in schemas.DECISION_COLUMNS:
        if column in view.columns and column not in upstream:
            assert column not in kept and reason[column] == "decision", column
    for column in ("true_pd", "true_lgd", "true_conversion_probability"):
        assert column in reason, f"{column} is not in the frame to be excluded"


def test_post_assessment_columns_are_excluded_for_stage1_only(frame) -> None:
    raw1, excluded1 = funnel.classify_columns(_view(frame, "stage1"), "stage1")
    reasons1 = {e["column"]: e["reason"] for e in excluded1}
    for column in schemas.POST_ASSESSMENT_COLUMNS:
        if column in frame.columns:
            assert reasons1.get(column) == "post_assessment", column
    assert not {r["column"] for r in raw1} & set(schemas.POST_ASSESSMENT_COLUMNS)

    _, excluded2 = funnel.classify_columns(_view(frame, "stage2"), "stage2")
    assert not [e for e in excluded2 if e["reason"] == "post_assessment"]


def test_a_decision_derived_column_is_excluded_unless_it_is_the_simulated_price(frame) -> None:
    """offered_rate is the sanctioned rate. For the pricing models it is the answer;
    for the conversion model it is the price the optimiser asks about, which the
    contract declares by constraining the response to it."""
    _, excluded = funnel.classify_columns(_view(frame, "stage2"), "stage2")
    reasons = {e["column"]: e["reason"] for e in excluded}
    assert reasons["offered_rate"] == "decision_derived"
    assert reasons["competitor_gap_bps"] == "decision_derived"

    raw, _ = funnel.classify_columns(_view(frame, "conversion"), "conversion")
    scenario = {r["column"] for r in raw if r["group"] == "scenario_input"}
    assert scenario <= funnel.SCENARIO_COLUMNS
    assert scenario <= set(spec("conversion").monotone), \
        "a scenario input needs a monotone contract"


def test_the_risk_model_does_not_see_the_grade_built_from_its_own_output(frame) -> None:
    _, excluded = funnel.classify_columns(_view(frame, "risk"), "risk")
    assert {e["column"]: e["reason"] for e in excluded}["risk_grade"] == "own_output"


def test_a_column_nobody_declared_is_excluded_rather_than_admitted(frame) -> None:
    probe = frame.head(300).assign(mystery_score=np.arange(300, dtype=float))
    raw, excluded = funnel.classify_columns(probe, "stage2")
    assert "mystery_score" not in {r["column"] for r in raw}
    assert {e["column"]: e["reason"] for e in excluded}["mystery_score"] == "unclassified"


@pytest.mark.parametrize("model", MODELS)
def test_the_target_is_excluded_and_marked_as_the_target(frame, model) -> None:
    _, excluded = funnel.classify_columns(_view(frame, model), model)
    targets = [e for e in excluded if e["is_target"]]
    assert [e["column"] for e in targets] == [funnel.resolve_target(model)]


def test_a_negative_bureau_score_is_read_as_missing(stage2, frame) -> None:
    """-1 is the new-to-credit code. Read as a number it is the worst score on the
    book, and every ratio built on it is arithmetic on a code."""
    tests = {c["name"]: c["tests"] for c in stage2["candidates"]}
    measured = float((pd.to_numeric(frame["bureau_score"], errors="coerce") >= 0).mean())
    assert tests["bureau_score"]["coverage"] == pytest.approx(measured, abs=1e-4)
    for name, t in tests.items():
        if "bureau_score" in name and name != "bureau_score":
            assert t["coverage"] <= measured + 1e-4, name
    row = next(r for r in stage2["stages"]["in_model"]["reconciliation"]
               if r["feature"] == "bureau_score")
    if discovery.THIN in row.get("failed", []):
        assert row["missing_explained_by"] == "new_to_credit"


# --------------------------------------------------------- 2-5. the funnel itself

def test_the_counts_are_monotone_where_they_should_be(runs) -> None:
    """Derived outnumbers raw (generation expands), and in-model can exceed
    screened-and-approved (a contract feature can fail a screen). Everything else
    narrows."""
    for model, report in runs["reports"].items():
        stages = report["stages"]
        counts = {f["stage"]: f["count"] for f in report["funnel"]}
        tested = stages["screening"]["tested"]
        assert tested == counts["raw_attributes"] + counts["derived_candidates"], model
        assert tested >= counts["passed_screening"] >= counts["approved"], model
        assert stages["in_model"]["screened_and_approved"] <= counts["approved"], model
        assert stages["in_model"]["screened_and_approved"] <= counts["in_model"], model
        assert counts["in_model"] == len(spec(model).columns), model
        assert counts["raw_attributes"] == (len(stages["raw_attributes"]["numeric"])
                                            + len(stages["raw_attributes"]["categorical"]))


def test_a_contract_feature_that_fails_a_screen_is_shown_failing_it(runs) -> None:
    for model, report in runs["reports"].items():
        rows = report["stages"]["in_model"]["reconciliation"]
        assert [r["feature"] for r in rows] == spec(model).columns, model
        items = {c["name"]: c for c in report["candidates"]}
        for row in rows:
            if row["verdict"] == "failed_screening":
                assert row["failed"] and not row["passed_screening"] and not row["approved"], row
            if row["verdict"] == "screened_and_approved":
                assert not row["failed"] and row["reason_label"], row
            if row["feature"] in items:
                assert row["passed_screening"] == items[row["feature"]]["passed_screening"]
        by_verdict = report["stages"]["in_model"]["by_verdict"]
        assert sum(by_verdict.values()) == len(rows)


def test_nothing_is_approved_without_a_reason_label(runs) -> None:
    for model, report in runs["reports"].items():
        review = report["stages"]["review"]
        items = {c["name"]: c for c in report["candidates"]}
        for name in review["approved_names"]:
            assert discovery.has_reason_label(name), name
            assert items[name]["passed_screening"], name
        expected = {n for n, c in items.items() if c["passed_screening"] and c["reason_label"]}
        assert set(review["approved_names"]) == expected, model
        assert review["approved_derived"] == sum(
            1 for n in expected if items[n]["kind"] == "derived")


def test_the_funnel_writes_no_label_and_changes_no_contract(runs) -> None:
    assert runs["config_before"] == runs["config_after"]


def test_a_governance_approval_without_a_label_does_not_count(runs, frame) -> None:
    """The label is the admission artefact. An approval on record for a feature
    nobody has written a sentence for is shown - and flagged - not counted."""
    stage1 = runs["reports"]["stage1"]
    assert stage1["stages"]["review"]["governance"]["consulted"] is False
    name = stage1["rejected_unexplainable"][0]["name"]
    report = funnel.run("stage1", frame=frame, sample=SAMPLE,
                        approvals={"admitted_proposals": {}, "approvals": {name: "APR-TEST"}})
    review = report["stages"]["review"]
    assert name not in review["approved_names"]
    assert name in review["governance"]["approved_without_label"]
    assert review["governance"]["consulted"] is True
    assert name in {r["name"] for r in report["rejected_unexplainable"]}


def test_the_rejected_table_is_exactly_strong_stable_and_unexplainable(runs) -> None:
    for model, report in runs["reports"].items():
        rules = report["thresholds"]
        items = {c["name"]: c for c in report["candidates"]}
        table = report["rejected_unexplainable"]
        expected = {n for n, c in items.items()
                    if c["kind"] == "derived" and c["passed_screening"] and not c["reason_label"]}
        assert {r["name"] for r in table} == expected, model
        signals = [abs(r["signal"]) for r in table]
        assert signals == sorted(signals, reverse=True), "not ranked by |signal|"
        for row in table:
            assert items[row["name"]]["failed"] == []
            assert abs(row["signal"]) >= rules["min_abs_target_corr"]
            assert row["coverage"] >= rules["min_coverage"]
            assert row["psi"] is None or row["psi"] <= rules["max_psi"]
            assert row["overlap"] is None or row["overlap"] <= rules["max_abs_correlation"]
            for key in ("signal", "overlap", "psi"):
                assert key in row


def test_every_screen_threshold_is_reported(stage2) -> None:
    for key, value in discovery.DEFAULTS.items():
        assert stage2["thresholds"][key] == value
    assert stage2["rows"]["correlation_sample"] == SAMPLE
    assert stage2["rows"]["seed"] is not None
    for stage in ("raw_attributes", "derived_candidates", "screening", "review", "in_model"):
        assert "seconds" in stage2["stages"][stage]
    assert stage2["target"] == schemas.TARGETS["stage2"]


def test_the_latent_audit_evaluates_without_feeding_anything(stage2) -> None:
    """Latent columns may evaluate, never feed: no latent column is a raw
    attribute, a derived input, or a tested item."""
    latent = set(schemas.LATENT_COLUMNS)
    for item in stage2["candidates"]:
        assert item["name"] not in latent and not set(item["inputs"]) & latent, item["name"]
    for row in stage2["latent_audit"]["rows"]:
        assert row["latent"] in latent and row["column"] not in latent


# ------------------------------------------------------------- the trend family

def test_a_trend_candidate_reads_only_strictly_earlier_applications(frame) -> None:
    """Changing a value on or after a date must not move any baseline on or
    before it - the same day included, which a naive window would leak."""
    sub = frame.sort_values("application_date").head(4000).reset_index(drop=True)
    cut = sub["application_date"].iloc[2500]
    columns = ["competitor_rate"]
    before = discovery.gen_trend(sub, columns, time=sub["application_date"])[0].values
    moved = sub.copy()
    later = moved["application_date"] >= cut
    moved.loc[later, "competitor_rate"] = moved.loc[later, "competitor_rate"] + 5.0
    after = discovery.gen_trend(moved, columns, time=moved["application_date"])[0].values

    baseline_before = sub["competitor_rate"] - before
    baseline_after = moved["competitor_rate"] - after
    upto = sub["application_date"] <= cut
    pd.testing.assert_series_equal(baseline_before[upto], baseline_after[upto])
    assert not baseline_before[~upto].equals(baseline_after[~upto]), "the perturbation did nothing"


def test_trend_is_available_to_discover_without_changing_its_defaults(frame) -> None:
    report = discovery.discover(frame, stage="stage2", generators=["trend"])
    assert report.candidates and {c.generator for c in report.candidates} == {"trend"}
    assert all(discovery.NO_LABEL in c.refusals for c in report.candidates)
    assert "trend" in discovery.FUNCTION_REGISTRY and "trend" in discovery.summary()["generators"]
    assert discovery.gen_trend(frame, ["competitor_rate"]) == [], "no date, no trend"


# --------------------------------------------------------------------- the API

def test_the_api_serves_a_fresh_saved_report(client, stage2, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(funnel, "REPORT_DIR", tmp_path)
    report = dict(stage2)
    # Re-stamped so a concurrent edit to a fingerprinted file since the fixture ran
    # cannot turn this into a test of staleness; that is the next test.
    report["fingerprint"] = funnel.fingerprint("stage2", sample=SAMPLE,
                                               seed=report["rows"]["seed"],
                                               approvals=feature_approvals("stage2"))
    funnel.save(report)

    body = client.get("/features/funnel", params={"model": "stage2"}).json()
    assert body["served"]["from"] == "saved_report" and body["served"]["fresh"] is True
    assert [f["count"] for f in body["funnel"]] == [f["count"] for f in stage2["funnel"]]
    assert "candidates" not in body
    assert "stage2" in body["models"]
    detailed = client.get("/features/funnel", params={"model": "stage2", "detail": True}).json()
    assert len(detailed["candidates"]) == stage2["stages"]["screening"]["tested"]


def test_a_stale_report_is_recomputed_and_says_why(client, stage2, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(funnel, "REPORT_DIR", tmp_path)
    report = dict(stage2)
    report["fingerprint"] = {**funnel.fingerprint("stage2", sample=SAMPLE,
                                                  seed=report["rows"]["seed"],
                                                  approvals=feature_approvals("stage2")),
                             "contract": "an-older-contract"}
    funnel.save(report)
    calls = []

    def fake_run(model, sample, seed, approvals=None):
        calls.append((model, sample, seed))
        assert approvals is not None, "the API must hand the governance evidence in"
        return {**stage2, "computed_at": "just now"}

    monkeypatch.setattr(funnel, "run", fake_run)
    funnel._MEMO.clear()
    body = client.get("/features/funnel", params={"model": "stage2"}).json()
    assert calls == [("stage2", SAMPLE, report["rows"]["seed"])], "recomputed with the saved params"
    assert body["served"]["from"] == "computed"
    assert funnel.FINGERPRINT_LABELS["contract"] in body["served"]["stale_because"]


def test_an_unknown_model_is_a_404(client) -> None:
    response = client.get("/features/funnel", params={"model": "no_such_contract"})
    assert response.status_code == 404
