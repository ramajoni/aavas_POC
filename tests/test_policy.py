"""Policy is the boundary the engine may not cross - so it is tested hardest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aavas.config import policy
from aavas.optimization import constraints


@pytest.fixture
def base_row() -> pd.DataFrame:
    return pd.DataFrame([{
        "segment": "salaried_formal", "risk_grade": "A", "property_type": "resale_flat",
        "ltv": 0.65, "deviation_count": 0, "new_to_credit": False,
        "sanctioned_loan_amount": 1_200_000, "verified_monthly_income": 60_000,
        "declared_obligations": 0.0, "tenure_years": 20,
    }])


def test_card_rate_is_deterministic(base_row):
    assert constraints.card_rate(base_row)[0] == constraints.card_rate(base_row)[0]


def test_card_rate_rises_with_risk_grade(base_row):
    rates = []
    for grade in ["A", "B", "C", "D", "E"]:
        row = base_row.assign(risk_grade=grade)
        rates.append(float(constraints.card_rate(row)[0]))
    assert rates == sorted(rates), f"card rate must be non-decreasing in risk grade, got {rates}"
    assert rates[-1] > rates[0]


def test_card_rate_rises_with_ltv(base_row):
    low = float(constraints.card_rate(base_row.assign(ltv=0.55))[0])
    high = float(constraints.card_rate(base_row.assign(ltv=0.88))[0])
    assert high > low


def test_new_to_credit_carries_minimum_spread(base_row):
    pol = policy()
    established = float(constraints.card_rate(base_row)[0])
    ntc = float(constraints.card_rate(base_row.assign(new_to_credit=True))[0])
    assert round((ntc - established) * 100) == pol["constraints"]["new_to_credit_min_spread_bps"]


def test_missing_columns_do_not_break_the_policy_engine():
    """A partially populated application must still get a card rate, not a crash."""
    sparse = pd.DataFrame([{"segment": "salaried_formal", "requested_loan_amount": 900_000,
                            "declared_monthly_income": 50_000}])
    rate = constraints.card_rate(sparse)
    assert np.isfinite(rate).all()
    band = constraints.permissible_band(sparse)
    assert band["min_rate"].iloc[0] <= band["max_rate"].iloc[0]


def test_band_never_escapes_the_product_envelope(base_row):
    pol = policy()["reference"]
    band = constraints.permissible_band(base_row).iloc[0]
    assert band["min_rate"] >= pol["card_rate_floor"] - 1e-9
    assert band["max_rate"] <= pol["card_rate_ceiling"] + 1e-9


def test_discount_beyond_policy_is_rejected(base_row):
    card = float(constraints.card_rate(base_row)[0])
    limit = policy()["constraints"]["absolute_discount_floor_bps"]
    ok = constraints.check(base_row, card - (limit - 5) / 100)
    too_far = constraints.check(base_row, card - (limit + 10) / 100)
    assert ok.allowed
    assert not too_far.allowed
    assert any("discount" in v for v in too_far.violations)


def test_approval_authority_escalates_with_discount():
    levels = [constraints.approval_level(bps) for bps in (5, 20, 35, 55, 200)]
    assert levels[0] == "sales_officer"
    assert levels[-1] == "pricing_committee"
    assert len(set(levels)) > 2, "authority must escalate, not collapse to one level"


def test_unevidenced_competitor_claim_cannot_drive_a_large_discount(base_row):
    pol = policy()["override"]
    card = float(constraints.card_rate(base_row)[0])
    big_discount = card - (pol["max_unevidenced_discount_bps"] + 10) / 100

    unevidenced = constraints.check(base_row, big_discount,
                                    override_reason="competitor_lower_rate", evidence_verified=False)
    evidenced = constraints.check(base_row, big_discount,
                                  override_reason="competitor_lower_rate", evidence_verified=True)
    assert not unevidenced.allowed
    assert evidenced.allowed, "an evidenced competitor quote should clear the same discount"


def test_small_unevidenced_discount_is_allowed_but_warned(base_row):
    pol = policy()["override"]
    card = float(constraints.card_rate(base_row)[0])
    check = constraints.check(base_row, card - (pol["max_unevidenced_discount_bps"] - 5) / 100,
                              override_reason="competitor_lower_rate", evidence_verified=False)
    assert check.allowed
    assert check.warnings


def test_affordability_caps_the_upper_end():
    """A thin file cannot be priced up indefinitely: the EMI breaches FOIR first."""
    thin = pd.DataFrame([{
        "segment": "salaried_informal", "risk_grade": "C", "property_type": "self_construction",
        "ltv": 0.85, "deviation_count": 1, "new_to_credit": False,
        "sanctioned_loan_amount": 1_500_000, "verified_monthly_income": 30_000,
        "declared_obligations": 3_000.0, "tenure_years": 15,
    }])
    band = constraints.permissible_band(thin).iloc[0]
    max_foir = policy()["constraints"]["max_foir"]

    assert band["affordability_binding"] or band["infeasible"]
    if not band["infeasible"]:
        emi = constraints._emi(np.array([1_500_000.0]), np.array([band["max_rate"]]), np.array([15.0]))
        foir = (emi[0] + 3_000.0) / 30_000.0
        assert foir <= max_foir + 1e-6, "the band's upper end must remain affordable"


def _thin_file(income: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "segment": "salaried_informal", "risk_grade": "C", "property_type": "self_construction",
        "ltv": 0.85, "deviation_count": 1, "new_to_credit": False,
        "sanctioned_loan_amount": 1_500_000, "verified_monthly_income": income,
        "declared_obligations": 3_000.0, "tenure_years": 15,
    }])


def test_rate_above_affordability_is_rejected():
    """An affordable price exists inside the band, so one that breaks FOIR is blocked."""
    thin = _thin_file(40_000)
    band = constraints.permissible_band(thin).iloc[0]
    assert not band["resize_required"] and band["affordability_binding"]
    check = constraints.check(thin, float(band["max_affordable_rate"]) + 0.5)
    assert not check.allowed
    assert check.verdict == "block"
    assert any("FOIR" in v for v in check.violations)


def test_unaffordable_file_is_priced_on_its_risk_band_conditional_on_resize():
    """When no permissible rate clears FOIR, the band must not collapse to a point:
    the file is priced on its risk band and referred with the resize that fixes it."""
    thin = _thin_file(30_000)
    band = constraints.permissible_band(thin).iloc[0]
    assert band["resize_required"]
    assert band["max_rate"] > band["min_rate"], "the band collapsed instead of pricing the risk"

    rate = float(band["card_rate"])
    check = constraints.check(thin, rate)
    assert check.allowed and check.verdict == "refer"
    assert check.conditions and "resize" in check.conditions[0]
    terms = check.resize
    assert terms["possible"] and terms["max_affordable_amount"] < terms["current_amount"]
    emi = constraints._emi(np.array([terms["max_affordable_amount"]]), np.array([rate]), np.array([15.0]))[0]
    assert (emi + 3_000.0) / 30_000.0 <= policy()["constraints"]["max_foir"] + 1e-9
