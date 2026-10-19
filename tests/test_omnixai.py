"""The OmniXAI layer's contract with the models.

The explainers themselves are OmniXAI's problem. What this repo has to get
right is the pairing between a trained artefact, the frame builder that feeds
it and the feature contract it was fitted on - get that wrong and the analysis
confidently explains a model on features it never saw. Those parts need no
omnixai import, so they are tested with the plain environment; anything that
does need it skips when the overlay venv is not the interpreter in use.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aavas.xai import omnixai_analysis as oa


def _shap_or_skip():
    """shap is an optional extra AND a common ABI casualty: a shap built against
    another numpy raises AttributeError, not ImportError, so importorskip is not
    enough here."""
    try:
        import shap
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"shap unusable in this interpreter ({type(exc).__name__}); "
                    f"the explainability extras live in .venv-xai")
    return shap


def _bundles(names=oa.MODELS):
    from aavas.models import base

    missing = [n for n in names if not base.exists(n)]
    if missing:
        pytest.skip(f"models not trained ({missing}); run: python scripts/train_all.py")
    return [oa.load_bundle(n) for n in names]


def test_every_model_has_a_bundle():
    for bundle in _bundles():
        assert bundle.mode in ("classification", "regression")
        assert bundle.spec.columns, f"{bundle.name} has an empty feature contract"
        if bundle.is_classification:
            assert bundle.class_names and len(bundle.class_names) == 2


def test_human_frame_matches_the_feature_contract(training_data):
    """Column-for-column, in order: this is what stops an explanation from
    describing a feature the model was not fitted on."""
    for bundle in _bundles():
        rows = bundle.rows(training_data).head(50)
        frame = oa.human_frame(bundle, rows)
        assert list(frame.columns) == list(bundle.spec.columns)
        assert not frame.isna().any().any(), f"{bundle.name}: NaN would break LIME and MACE"
        for col in bundle.spec.categorical:
            assert frame[col].map(type).eq(str).all(), f"{bundle.name}.{col} must reach OmniXAI as labels"


def test_human_frame_scores_identically_to_the_model(training_data):
    """The explained rows must be the scored rows. If the human frame drifted
    from what predict() sees, every attribution would be about a different
    application than the one on the screen."""
    for bundle in _bundles():
        rows = bundle.rows(training_data).head(200)
        direct = bundle.transformer.transform(bundle.build(rows))
        via_human = bundle.transformer.transform(oa.human_frame(bundle, rows))
        pd.testing.assert_frame_equal(direct, via_human, check_dtype=False)


def test_attribution_units_are_labelled():
    """A classifier's KernelSHAP values are log-odds, not probabilities. The
    report has to say so or the numbers get read as percentage points."""
    for bundle in _bundles():
        expected = "log-odds" if bundle.is_classification else bundle.units
        assert bundle.attribution_units == expected


def test_instance_selection_covers_both_tails():
    scores = np.linspace(0.0, 1.0, 100)
    rng = np.random.default_rng(0)
    picks = oa._select_instances(scores, 5, rng)
    labels = {label for label, _ in picks}
    assert {"highest predicted", "lowest predicted", "median predicted"} <= labels
    assert len(picks) == 5
    assert len({i for _, i in picks}) == 5, "the same row must not be explained twice"


def test_curve_effects_reads_direction_from_the_curve():
    class _Explanation:
        @staticmethod
        def get_explanations():
            return {
                "ltv": {"values": [0.4, 0.6, 0.8], "scores": np.array([10.0, 11.0, 13.0])},
                "segment": {"values": ["a", "b"], "scores": np.array([2.0, 1.0])},
            }

    effects = oa._curve_effects(_Explanation(), categorical={"segment"})
    assert effects["ltv"]["direction"] == "increases"
    assert effects["ltv"]["monotone"] is True
    assert effects["ltv"]["swing"] == pytest.approx(3.0)
    # An unordered categorical has no "up": first vs last category is arbitrary.
    assert effects["segment"]["direction"] == "varies by category"
    assert effects["segment"]["peak"]["value"] == "a"


def test_shap_shim_restores_the_per_class_shape(monkeypatch):
    """omnixai 1.3.1 indexes KernelSHAP output as ``shap_values[label][row]``.
    shap >= 0.45 returns one ``(rows, features, classes)`` array instead, which
    turns that index into a *row* lookup - wrong numbers, no error. The shim has
    to hand back the per-class list."""
    shap = _shap_or_skip()

    values = np.arange(24, dtype=float).reshape(2, 4, 3)     # rows, features, classes
    monkeypatch.setattr(shap.KernelExplainer, "shap_values",
                        lambda self, X, **kwargs: values, raising=True)
    oa._patch_shap_legacy_output()

    restored = shap.KernelExplainer.shap_values(object(), None)
    assert isinstance(restored, list) and len(restored) == 3
    assert restored[1].shape == (2, 4)
    np.testing.assert_allclose(restored[1], values[..., 1])


def test_shap_shim_leaves_single_output_alone(monkeypatch):
    """A regression model's SHAP output is already (rows, features)."""
    shap = _shap_or_skip()

    values = np.ones((3, 5))
    monkeypatch.setattr(shap.KernelExplainer, "shap_values",
                        lambda self, X, **kwargs: values, raising=True)
    oa._patch_shap_legacy_output()
    assert shap.KernelExplainer.shap_values(object(), None).shape == (3, 5)


def test_shap_shim_is_applied_once():
    shap = _shap_or_skip()

    oa._patch_shap_legacy_output()
    oa._patch_shap_legacy_output()
    assert getattr(shap.KernelExplainer.shap_values, "_aavas_legacy_shape", False)


# --------------------------------------------------------------- credit score


def _credit_frame(n_thin=120, n_scored=880, seed=0):
    rng = np.random.default_rng(seed)
    scores = np.concatenate([np.full(n_thin, float(oa.NEW_TO_CREDIT_SENTINEL)),
                             rng.integers(560, 900, n_scored).astype(float)])
    frame = pd.DataFrame({oa.CREDIT_SCORE_FEATURE: scores})
    # A PD that falls with the score, and thin files priced like a middling one.
    predictions = np.where(scores < 0, 0.18, 0.5 - (scores - 560) / 340 * 0.45)
    return frame, predictions


def test_instance_selection_reaches_both_ends_of_the_credit_score():
    frame = pd.DataFrame({oa.CREDIT_SCORE_FEATURE: [-1, -1, 640, 720, 880, 705, 601]})
    picks = oa._select_instances(np.linspace(0, 1, 7), 7, np.random.default_rng(0), frame)
    labels = " | ".join(label for label, _ in picks)
    assert "lowest credit score" in labels and "highest credit score" in labels
    assert dict((label, i) for label, i in picks if "highest credit score" in label)
    assert len({i for _, i in picks}) == len(picks), "no row explained twice"


def test_the_new_to_credit_sentinel_is_never_the_lowest_credit_score():
    """-1 is "no bureau history", not a bad score. Ranking the two together is
    how a thin file ends up explained as the worst credit in the book."""
    frame = pd.DataFrame({oa.CREDIT_SCORE_FEATURE: [-1.0, -1.0, 612.0, 790.0]})
    picks = dict((label, i) for label, i in oa._credit_score_extremes(frame))
    assert picks["lowest credit score"] == 2
    assert picks["highest credit score"] == 3


def test_one_row_that_is_two_extremes_keeps_both_labels():
    """The worst credit score often is the highest predicted PD. Explaining it
    once under both names beats spending two instances on the same row."""
    frame = pd.DataFrame({oa.CREDIT_SCORE_FEATURE: [900.0, 700.0, 600.0]})
    picks = oa._select_instances(np.array([0.1, 0.5, 0.9]), 3, np.random.default_rng(0), frame)
    merged = [label for label, _ in picks if "+" in label]
    assert merged and "lowest credit score" in merged[0] and "highest predicted" in merged[0]
    assert len(picks) == 3


def test_instance_selection_without_a_frame_is_unchanged():
    picks = oa._select_instances(np.linspace(0, 1, 50), 5, np.random.default_rng(0))
    assert {label for label, _ in picks} == {"highest predicted", "lowest predicted",
                                             "median predicted", "randomly sampled"}


def test_credit_band_separates_thin_files_from_low_scores():
    frame, predictions = _credit_frame()
    pdp = {oa.CREDIT_SCORE_FEATURE: {"points": [
        {"value": -1.0, "score": 0.20}, {"value": 560.0, "score": 0.49},
        {"value": 700.0, "score": 0.30}, {"value": 880.0, "score": 0.05}]}}
    bundle = SimpleNamespace(units="probability", output_label="Lifetime PD")
    band = oa._credit_score_band(bundle, frame, predictions, pdp, [])

    observed = band["observed"]
    assert observed["score_min"] >= oa.CREDIT_SCORE_FLOOR, "the sentinel is not a score"
    assert observed["new_to_credit"]["files"] == 120
    assert observed["floor_cohort"]["mean_prediction"] > observed["ceiling_cohort"]["mean_prediction"]
    # The thin cohort prices better than the bottom decile, which is exactly why
    # it must not be folded into it.
    assert observed["new_to_credit"]["mean_prediction"] < observed["floor_cohort"]["mean_prediction"]

    curve = band["curve"]
    assert curve["sentinel_points_dropped"] == [-1.0]
    assert curve["at_min"]["value"] == 560.0 and curve["at_max"]["value"] == 880.0
    assert curve["monotone"] is True
    assert "new-to-credit" in band["reading"]


def test_credit_band_flags_a_curve_that_turns():
    """A non-monotone response to the credit score is a policy question, so the
    band has to report it rather than average it away."""
    frame, predictions = _credit_frame()
    pdp = {oa.CREDIT_SCORE_FEATURE: {"points": [
        {"value": 560.0, "score": 0.40}, {"value": 700.0, "score": 0.46},
        {"value": 880.0, "score": 0.05}]}}
    band = oa._credit_score_band(SimpleNamespace(units="probability", output_label="Lifetime PD"),
                                 frame, predictions, pdp, [])
    assert band["curve"]["monotone"] is False
    assert band["curve"]["peak"]["value"] == 700.0
    assert "turns at 700" in band["reading"]


def test_credit_band_is_absent_when_the_contract_has_no_score():
    frame = pd.DataFrame({"ltv": [0.6, 0.7]})
    assert oa._credit_score_band(SimpleNamespace(units="probability", output_label="x"),
                                 frame, np.array([0.1, 0.2]), {}, []) is None


def test_curve_effects_carries_the_points_the_band_re_reads():
    class _Explanation:
        @staticmethod
        def get_explanations():
            return {"bureau_score": {"values": [-1.0, 700.0, 880.0],
                                     "scores": np.array([0.2, 0.1, 0.05])}}

    points = oa._curve_effects(_Explanation())["bureau_score"]["points"]
    assert [p["value"] for p in points] == [-1.0, 700.0, 880.0]
