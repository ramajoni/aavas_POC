"""Stage-2 is an anchor for the optimizer, not the final price - and its
feature contract must stay separable from commercial pressure."""

from __future__ import annotations

import numpy as np
import pytest

from aavas.config import policy
from aavas.features.feature_pipeline import FeatureTransformer, spec
from aavas.ingestion import schemas


def test_stage2_sees_more_than_stage1():
    stage1, stage2 = spec("stage1"), spec("stage2")
    assert "verified_monthly_income" in stage2.columns
    assert len(stage2.columns) > len(stage1.columns)


def test_assessment_improves_pricing_accuracy(bundle, training_data):
    """The commercial justification for collecting documents at all."""
    from aavas.models import base

    sample = training_data.tail(2000)
    actual = sample[schemas.TARGETS["stage2"]].to_numpy(float)
    stage2_mae = base.regression_metrics(actual, bundle.stage2.predict_frame(sample).to_numpy())["mae_bps"]
    stage1_mae = base.regression_metrics(
        actual, bundle.stage1.predict_frame(sample)["indicative_rate"].to_numpy())["mae_bps"]
    assert stage2_mae < stage1_mae, (
        f"stage2 ({stage2_mae:.1f} bps) should beat the stage1 anchor ({stage1_mae:.1f} bps)")


def test_anchor_carries_an_uncertainty_interval(bundle, sample_application):
    import pandas as pd

    result = bundle.stage2.predict(pd.DataFrame([sample_application]))
    band = result["search_band"]
    assert band["min_rate"] < result["anchor_rate"] < band["max_rate"]
    assert result["residual_std_bps"] > 0


def test_anchor_stays_inside_the_product_envelope(bundle, training_data):
    pol = policy()["reference"]
    predictions = bundle.stage2.predict_frame(training_data.head(500))
    assert predictions.min() >= pol["card_rate_floor"] - 0.5
    assert predictions.max() <= pol["card_rate_ceiling"] + 0.5


def test_transformer_rejects_a_missing_feature_rather_than_guessing(training_data):
    transformer = FeatureTransformer(spec("stage2"))
    transformer.fit(training_data)
    with pytest.raises(KeyError, match="missing feature columns"):
        transformer.transform(training_data.drop(columns=["ltv"]))


def test_unseen_category_does_not_crash_scoring(bundle, training_data):
    """A new branch city appears every month; scoring must degrade, not fail."""
    sample = training_data.head(5).assign(city="Brand New Town")
    predictions = bundle.stage2.predict_frame(sample)
    assert np.isfinite(predictions.to_numpy()).all()
