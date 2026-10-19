"""Stage-1 must quote a band that is honest, credible and free of hindsight."""

from __future__ import annotations

import numpy as np
import pytest

from aavas.config import policy
from aavas.features import stage1_features
from aavas.ingestion import schemas
from aavas.ingestion.validator import assert_no_leakage
from aavas.models import base


def test_stage1_contract_excludes_post_assessment_information():
    """The guard that stops verified income or valuation leaking into a quote
    made before either exists."""
    spec = stage1_features.stage1_spec()
    assert_no_leakage(spec.columns, "stage1")
    for forbidden in ("verified_monthly_income", "ltv", "risk_grade", "foir", "disbursed"):
        assert forbidden not in spec.columns


def test_leakage_guard_actually_fires():
    with pytest.raises(ValueError, match="leak"):
        assert_no_leakage(["declared_monthly_income", "verified_monthly_income"], "stage1")


def test_band_is_ordered_and_within_policy(bundle, training_data):
    predictions = bundle.stage1.predict_frame(training_data.head(300))
    pol = policy()["reference"]
    assert (predictions["min_rate"] <= predictions["max_rate"]).all()
    assert (predictions["indicative_rate"] >= predictions["min_rate"]).all()
    assert (predictions["indicative_rate"] <= predictions["max_rate"]).all()
    assert (predictions["min_rate"] >= pol["card_rate_floor"] - 1e-9).all()
    assert (predictions["max_rate"] <= pol["card_rate_ceiling"] + 1e-9).all()


def test_band_width_respects_the_credibility_envelope(bundle, training_data):
    cfg = policy()["stage1"]
    widths = bundle.stage1.predict_frame(training_data.head(300))["band_width_bps"]
    assert widths.min() >= cfg["min_band_width_bps"] - 1e-6, "a falsely precise band is not allowed"
    assert widths.max() <= cfg["max_band_width_bps"] + 1e-6, "a band this wide is not a quote"


def test_band_covers_the_eventual_price_about_as_often_as_promised(bundle, training_data):
    """A 10-90 quantile band should contain the final rate roughly 80% of the time.
    Materially less and the quote misleads the customer."""
    sample = training_data.tail(2000)
    predictions = bundle.stage1.predict_frame(sample)
    actual = sample[schemas.TARGETS["stage1"]].to_numpy(float)
    metrics = base.interval_metrics(actual, predictions["min_rate"].to_numpy(),
                                    predictions["max_rate"].to_numpy())
    assert metrics["coverage_pct"] >= 65.0, f"band coverage too low: {metrics['coverage_pct']:.1f}%"
    assert metrics["inverted_bands"] == 0


def test_confidence_falls_when_the_file_is_thinner(bundle, training_data):
    known = training_data[training_data["bureau_score"] > 0].head(200)
    thin = known.assign(bureau_score=-1, new_to_credit=True, competitor_rate=np.nan)
    assert bundle.stage1.predict_frame(thin)["confidence"].mean() < \
           bundle.stage1.predict_frame(known)["confidence"].mean()


def test_eligibility_falls_as_the_rate_rises(training_data):
    sample = stage1_features.build(training_data.head(200))
    pol = policy()
    cheap = stage1_features.eligibility(sample, np.full(len(sample), 9.0), pol)
    dear = stage1_features.eligibility(sample, np.full(len(sample), 14.0), pol)
    assert (dear <= cheap + 1e-6).all(), "a higher rate cannot make a customer eligible for more"


def test_quote_survives_a_minimal_enquiry(orchestrator):
    """A walk-in with almost nothing on file must still get a usable quote."""
    result = orchestrator.pre_assessment({
        "loan_id": "TEST-MIN", "segment": "self_employed_non_professional", "city": "Jaipur",
        "channel": "branch_walkin", "property_type": "self_construction",
        "declared_monthly_income": 32000, "requested_loan_amount": 900000,
        "requested_tenure_years": 20, "applicant_age": 34, "declared_obligations": 0,
        "estimated_property_value": 1300000, "bureau_score": -1, "new_to_credit": True,
    }, persist=False)
    assert result["band"]["min_rate"] < result["band"]["max_rate"]
    assert 0 < result["confidence"] <= 1
    assert result["eligible_amount"]["max"] > 0
    assert result["caveats"], "a thin file must carry caveats"
    assert "indicative" in result["disclaimer"].lower()
