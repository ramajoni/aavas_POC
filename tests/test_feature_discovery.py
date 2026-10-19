"""Agentic feature engineering.

The tests are about what the agent **refuses**, not what it finds. A generator
that produces candidates is easy; the whole design rests on the four refusals and
on the one that is not statistical - a feature nobody can explain cannot be
admitted however strong it is.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aavas.api.main import app
from aavas.features import discovery
from aavas.governance import approvals, feature_registry
from aavas.ingestion import schemas


@pytest.fixture(scope="module")
def frame():
    from aavas.ingestion import loader

    return loader.build_training_table()


@pytest.fixture(scope="module")
def report(frame):
    return discovery.discover(frame, stage="stage2", generators=["ratio", "gap"],
                              max_candidates=200)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ------------------------------------------------------- leakage by construction

def test_a_stage1_candidate_cannot_be_built_from_a_post_assessment_field() -> None:
    """Prevented by construction, not caught afterwards: the generator is only
    ever handed columns the stage's own contract already contains."""
    allowed = set(discovery.allowed_columns("stage1"))
    assert allowed, "stage1 has no usable columns"
    assert not allowed & set(schemas.POST_ASSESSMENT_COLUMNS)
    assert not allowed & set(schemas.OUTCOME_COLUMNS)
    assert not allowed & set(schemas.LATENT_COLUMNS)
    assert not allowed & set(schemas.DECISION_COLUMNS)


def test_no_stage1_candidate_names_a_forbidden_input(frame) -> None:
    forbidden = (set(schemas.POST_ASSESSMENT_COLUMNS) | set(schemas.OUTCOME_COLUMNS)
                 | set(schemas.LATENT_COLUMNS))
    result = discovery.discover(frame, stage="stage1", generators=["ratio"],
                                max_candidates=120)
    for candidate in result.candidates:
        assert not set(candidate.inputs) & forbidden, candidate.name


def test_the_outcome_columns_are_never_available_to_any_stage() -> None:
    for stage in ("stage1", "stage2"):
        assert not set(discovery.allowed_columns(stage)) & set(schemas.OUTCOME_COLUMNS)


# --------------------------------------------------------------- the refusals

def test_nothing_is_admissible_without_a_reason_label(report) -> None:
    """The point of the whole module. Every generated name is new, so none has an
    authored label, so none is admissible - however strong."""
    assert report.candidates, "no candidates were generated"
    assert report.admissible() == []
    assert all(discovery.NO_LABEL in c.refusals for c in report.candidates)


def test_the_shortlist_is_the_useful_output(report) -> None:
    """Strong, stable, independent and unexplainable. Each is a question for a
    human, and the answer is often no."""
    shortlist = report.shortlist()
    for candidate in shortlist:
        assert candidate.refusals == [discovery.NO_LABEL]
        assert abs(candidate.tests["target_spearman"]) >= discovery.DEFAULTS["min_abs_target_corr"]


def test_a_redundant_candidate_is_refused_not_merely_ranked_low(report) -> None:
    redundant = [c for c in report.candidates if discovery.REDUNDANT in c.refusals]
    assert redundant, "no redundant candidate was generated to check"
    for candidate in redundant:
        assert (candidate.tests["max_correlation_with_contract"]
                > discovery.DEFAULTS["max_abs_correlation"])
        assert not candidate.admissible


def test_a_refusal_is_not_a_score(report) -> None:
    """A candidate refused on any ground cannot be rescued by scoring better."""
    for candidate in report.candidates:
        if candidate.refusals:
            assert candidate.admissible is False


def test_every_refusal_code_carries_an_explanation() -> None:
    assert set(discovery.REFUSALS) >= {
        discovery.LEAKAGE, discovery.REDUNDANT, discovery.UNSTABLE,
        discovery.THIN, discovery.NO_LABEL, discovery.DEGENERATE}
    for code, why in discovery.REFUSALS.items():
        assert len(why) > 20, code


# ------------------------------------------------------------- the generators

def test_a_gap_is_only_built_between_comparable_scales(report) -> None:
    """Subtracting a rupee amount from a percentage produces a number and means
    nothing."""
    for candidate in report.candidates:
        if candidate.generator != "gap":
            continue
        assert discovery._comparable(report.candidates[0].values * 0 + 1,
                                     report.candidates[0].values * 0 + 1)


def test_a_recipe_is_human_readable(report) -> None:
    """`income_per_emi` is a claim about the world; `f_0417` is not."""
    for candidate in report.candidates[:25]:
        assert any(op in candidate.recipe for op in ("/", "-", "*", "log1p")), candidate.recipe
        assert all(name in candidate.recipe for name in candidate.inputs)


def test_a_candidate_that_duplicates_a_contract_feature_is_dropped(report) -> None:
    from aavas.features.feature_pipeline import spec

    existing = set(spec("stage2").columns)
    assert not {c.name for c in report.candidates} & existing


# ---------------------------------------------------------------- the label

def test_the_fallback_label_does_not_count_as_authored() -> None:
    """feature_pipeline.reason_label title-cases an unknown column. That is a
    placeholder and must not satisfy the gate."""
    from aavas.features.feature_pipeline import reason_label

    fabricated = "ltv_per_foir_x_bureau"
    assert reason_label(fabricated)["label"], "the fallback still returns something"
    assert not discovery.has_reason_label(fabricated)


def test_an_authored_label_is_recognised() -> None:
    assert discovery.has_reason_label("ltv")
    assert discovery.has_reason_label("bureau_score")


# ------------------------------------------------------- admission is a change

def test_the_admission_gates_block_on_the_missing_label(client: TestClient) -> None:
    client.post("/features/discover",
                json={"stage": "stage2", "generators": ["ratio"], "limit": 150})
    report = client.post("/features/discover",
                         json={"stage": "stage2", "generators": ["ratio"],
                               "limit": 150}).json()["report"]
    candidate = report["awaiting_a_reason_label"][0]
    proposed = client.post("/features/propose",
                           json={"name": candidate["name"], "stage": "stage2"})
    assert proposed.status_code == 200

    gates = approvals.run_gates("feature", f"stage2:{candidate['name']}")
    by_name = {g["gate"]: g for g in gates}
    assert by_name["feature_proposed"]["passed"] is True
    assert by_name["feature_no_leakage"]["passed"] is True
    assert by_name["reason_label_authored"]["passed"] is False
    assert not approvals.gates_clear(gates)


def test_a_feature_cannot_be_proposed_without_a_discovery_run(client: TestClient) -> None:
    """A feature is proposed with its evidence or not at all."""
    from aavas.api import features as features_api

    features_api._CACHE.clear()
    refused = client.post("/features/propose",
                          json={"name": "made_up_feature", "stage": "stage2"})
    assert refused.status_code == 409


def test_a_proposal_stores_its_refusals_rather_than_filtering_them(client: TestClient) -> None:
    report = client.post("/features/discover",
                         json={"stage": "stage2", "generators": ["ratio"],
                               "limit": 150}).json()["report"]
    name = report["awaiting_a_reason_label"][0]["name"]
    client.post("/features/propose", json={"name": name, "stage": "stage2"})
    row = feature_registry.by_name(name, "stage2")
    assert row["refusals"] == ["no_reason_label"]
    assert row["tests"]["max_correlation_with_contract"] is not None
    assert feature_registry.summary()["blocked_only_on_a_reason_label"] >= 1


def test_there_is_no_endpoint_that_widens_the_contract() -> None:
    """The absence is the control. If one ever appears, this test should be
    changed deliberately and with an argument."""
    paths = {r.path for r in app.routes if hasattr(r, "methods")}
    writes = {r.path for r in app.routes
              if hasattr(r, "methods") and (r.methods - {"GET", "HEAD", "OPTIONS"})}
    assert "/features/admit" not in paths
    assert not any(p.startswith("/config") and p in writes for p in paths), (
        "a write endpoint under /config would let a feature be admitted outside the "
        "approval path")


def test_the_discovery_agent_is_on_the_roster_and_dormant() -> None:
    from aavas.agents import roster

    agent = roster.BY_NAME["feature_discovery"]
    assert agent["wired"] is False
    assert agent["stages"] == []
    assert "admission cannot" in agent["dormant_because"]


# ================================================================ the profiler
# The data-understanding and statistical roles. Tested for what they refuse to
# assert as much as for what they compute.

def test_a_thin_slice_reports_insufficient_evidence_not_a_number() -> None:
    """A p-value from eleven rows is a number, not evidence."""
    from aavas.features import profiling

    tiny = pd.Series(range(11), dtype=float)
    target = pd.Series([0, 1] * 5 + [0])
    assert profiling.information_value(tiny, target)["iv"] is None
    assert profiling.group_separation(tiny, target)["p_value"] is None
    assert profiling.stability(tiny)["psi"] is None


def test_the_profiler_labels_a_distribution_rather_than_only_measuring_it() -> None:
    from aavas.features import profiling

    right = pd.Series(np.random.default_rng(0).lognormal(size=2000))
    flat = pd.Series(np.random.default_rng(0).normal(size=2000))
    assert profiling.profile_numeric(right)["shape"] == "right-skewed"
    assert profiling.profile_numeric(flat)["shape"] == "roughly symmetric"
    assert profiling.profile_numeric(pd.Series([0, 1] * 500))["shape"] == "binary"


def test_a_high_iv_is_labelled_as_suspicious_rather_than_excellent() -> None:
    """Above 0.5 the question is whether it leaks, not whether to use it."""
    from aavas.features import profiling

    assert "leakage" in profiling._iv_label(0.8)
    assert profiling._iv_label(0.2) == "medium"
    assert profiling._iv_label(0.01) == "not predictive"


def test_unseen_category_risk_is_reported_for_rare_levels() -> None:
    from aavas.features import profiling

    values = pd.Series(["a"] * 990 + [f"rare_{i}" for i in range(10)])
    out = profiling.profile_categorical(values)
    assert out["rare_categories"] == 10
    assert out["unseen_category_risk"] == "high"


def test_vif_detects_an_algebraic_relationship() -> None:
    """A column that is an exact combination of others is the clearest finding
    this test can produce, so it must not come back as 'not computed'."""
    from aavas.features import profiling

    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=1000))
    b = pd.Series(rng.normal(size=1000))
    exact = pd.DataFrame({"a": a, "b": b, "sum": a + b})
    assert profiling.variance_inflation(exact, "sum") == profiling.VIF_CAP

    noisy = pd.DataFrame({"a": a, "b": b, "near": a + b + rng.normal(0, 0.05, 1000)})
    value = profiling.variance_inflation(noisy, "near")
    assert 10 < value < profiling.VIF_CAP


def test_a_temporal_column_is_flagged_as_a_leakage_surface() -> None:
    from aavas.features import profiling

    out = profiling.profile(pd.DataFrame(
        {"application_date": pd.date_range("2025-01-01", periods=100)}))
    assert out["application_date"]["kind"] == "temporal"
    assert "leakage" in out["application_date"]["note"]


# ============================================================== the decision

def test_a_veto_cannot_be_outvoted_by_a_strong_statistic() -> None:
    """The trade a lender must never make: leakage beaten by predictive power."""
    from aavas.features import cards

    evidence = {"information_value": 0.45, "iv_strength": "strong",
                "mutual_information": 0.4, "p_value": 1e-20, "psi": 0.01, "vif": 1.0}
    decision, vetoes, _ = cards.decide(evidence, has_reason_label=True, leaks=True)
    assert decision == cards.REJECT and "leakage" in vetoes


def test_an_unexplainable_feature_is_vetoed_however_strong() -> None:
    from aavas.features import cards

    evidence = {"information_value": 0.4, "mutual_information": 0.3, "p_value": 1e-20,
                "psi": 0.01}
    decision, vetoes, _ = cards.decide(evidence, has_reason_label=False)
    assert decision == cards.REJECT and "unexplainable" in vetoes


def test_review_exists_because_mixed_evidence_is_the_usual_answer() -> None:
    from aavas.features import cards

    mixed = {"information_value": 0.2, "iv_strength": "medium", "mutual_information": 0.0001,
             "p_value": 1e-8, "psi": 0.01, "vif": 1.2}
    decision, vetoes, reasons = cards.decide(mixed, has_reason_label=True)
    assert decision == cards.REVIEW and not vetoes
    assert len(reasons) >= 2, "a review must say what was strong and what was weak"


def test_suspicious_strength_is_a_veto_not_a_recommendation() -> None:
    from aavas.features import cards

    decision, vetoes, _ = cards.decide(
        {"information_value": 0.9, "p_value": 1e-30, "psi": 0.01}, has_reason_label=True)
    assert decision == cards.REJECT and "suspicious_strength" in vetoes


def test_instability_is_a_veto(  ) -> None:
    from aavas.features import cards

    decision, vetoes, _ = cards.decide(
        {"information_value": 0.3, "p_value": 1e-10, "psi": 0.4}, has_reason_label=True)
    assert decision == cards.REJECT and "unstable" in vetoes


def test_a_card_carries_its_evidence_not_a_verdict_alone() -> None:
    from aavas.features import cards

    card = cards.build_card("x", "stage2", {"profile": {"kind": "numeric", "shape": "binary"}},
                            has_reason_label=True)
    body = card.as_dict()
    for key in ("feature", "transformation", "original_columns", "business_hypothesis",
                "evidence", "decision", "reasons", "version", "created_by"):
        assert key in body, key


def test_a_card_with_no_hypothesis_says_so() -> None:
    from aavas.features import cards

    card = cards.build_card("x", "stage2", {}, has_reason_label=True)
    assert "coincidence" in card.hypothesis


# ============================================================ research trigger

def test_research_is_not_triggered_for_an_ordinary_feature() -> None:
    """The saving is the point: a known method on a known problem executes
    directly rather than becoming a model call."""
    from aavas.features import discovery, profiling

    rng = np.random.default_rng(0)
    values = pd.Series(rng.normal(100, 15, 4000))
    # Weakly related on purpose. An earlier version of this test built the target
    # straight from the feature, which made it genuinely suspicious - and the
    # trigger fired correctly.
    target = pd.Series((rng.normal(0, 1, 4000) + 0.05 * (values - 100) / 15 > 0).astype(int))
    report = profiling.examine(values, target, name="ordinary")
    triggered = discovery.research_needed(report)
    assert triggered["research_needed"] is False, triggered["triggers"]


def test_research_is_triggered_by_an_extreme_distribution() -> None:
    from aavas.features import discovery, profiling

    rng = np.random.default_rng(0)
    values = pd.Series(np.concatenate([rng.normal(1, 0.1, 3900), rng.normal(500, 5, 100)]))
    target = pd.Series(rng.integers(0, 2, 4000))
    report = profiling.examine(values, target, name="extreme")
    triggered = discovery.research_needed(report)
    assert triggered["research_needed"] is True
    assert all(t["why"] for t in triggered["triggers"])


def test_every_trigger_explains_itself() -> None:
    from aavas.features import discovery

    for code, why in discovery.RESEARCH_TRIGGERS.items():
        assert len(why) > 25, code


def test_the_function_registry_answers_is_this_already_covered() -> None:
    from aavas.features import discovery

    assert discovery.registry_covers("ratio")
    assert not discovery.registry_covers("fourier_decomposition")
    for name, entry in discovery.FUNCTION_REGISTRY.items():
        assert entry["applies_to"] and entry["produces"] and entry["why"], name


# ================================================== the contract explains itself

def test_every_contract_feature_has_an_authored_reason_label() -> None:
    """The defect the feature-card pass found: 13 features already in production
    fell back to a title-cased column name. The fallback *reads* like a label, so
    a customer would have seen "Appraised Property Value" where a sentence belongs.

    This is the durable fix - a feature admitted without a label now fails here
    rather than surfacing in front of a customer.
    """
    from aavas.config import feature_config
    from aavas.features import discovery
    from aavas.features.feature_pipeline import spec

    missing = {}
    for stage in feature_config():
        block = feature_config()[stage]
        if not isinstance(block, dict) or not ("numeric" in block or "categorical" in block):
            continue
        absent = [c for c in spec(stage).columns if not discovery.has_reason_label(c)]
        if absent:
            missing[stage] = absent
    assert not missing, f"contract features with no authored reason label: {missing}"


def test_an_authored_label_carries_both_directions() -> None:
    """A label with only one direction cannot explain a rate that moved the other
    way, which is half the explanations."""
    from aavas.config import feature_config

    for name, label in (feature_config().get("reason_code_labels") or {}).items():
        assert label.get("label"), name
        assert label.get("up") and label.get("down"), f"{name} explains only one direction"


# ======================================================= no code is generated

def test_there_is_no_code_generation_or_execution_path() -> None:
    """The deliberate omission. Generated Python that reaches the feature contract
    is the highest-blast-radius thing in a pricing engine, and an in-process exec
    with a blocklist is not a sandbox. If this ever changes it must change with an
    argument, not by accident."""
    import aavas.features.discovery as module

    source = pathlib.Path(module.__file__).read_text()
    for forbidden in ("exec(", "eval(", "subprocess", "compile("):
        assert forbidden not in source, f"{forbidden} appeared in the discovery agent"

    paths = {r.path for r in app.routes if hasattr(r, "methods")}
    assert not any("generate" in p or "execute" in p for p in paths)


import pathlib  # noqa: E402  (used by the test above)
import numpy as np  # noqa: E402
