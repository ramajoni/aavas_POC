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
        "declared_obligations": 0.0, "tenure_years": 10,
    }])


def test_card_rate_is_deterministic(base_row):
    assert constraints.card_rate(base_row)[0] == constraints.card_rate(base_row)[0]


def _offered_grades() -> list[str]:
    """Grades the policy prices, best first (a null loading is no offer)."""
    loadings = policy()["risk_grade_loading_bps"]
    return sorted((g for g, bps in loadings.items() if bps is not None), key=lambda g: loadings[g])


def test_card_rate_rises_with_risk_grade(base_row):
    rates = [float(constraints.card_rate(base_row.assign(risk_grade=g))[0]) for g in _offered_grades()]
    assert rates == sorted(rates), f"card rate must be non-decreasing in risk grade, got {rates}"
    assert rates[-1] > rates[0]


def test_card_rate_carries_exactly_the_configured_ltv_loading(base_row):
    """LTV moves the card rate by the policy's own LTV loading - and by nothing else.
    (Under a grade-matrix policy that loading is zero: LTV is priced through the grade.)"""
    bands = sorted(policy()["ltv_loading_bps"], key=lambda b: b["max_ltv"])
    low = float(constraints.card_rate(base_row.assign(ltv=0.55))[0])
    high = float(constraints.card_rate(base_row.assign(ltv=0.88))[0])
    loading = lambda ltv: next(b["bps"] for b in bands if ltv <= b["max_ltv"])  # noqa: E731
    assert round((high - low) * 100) == loading(0.88) - loading(0.55)


def test_card_rate_is_the_pricing_matrix(base_row):
    """Grade base ROI plus the tenure column, for every offered grade and tenure band."""
    pricing = policy()["pricing"]
    for grade, row in pricing["matrix"].items():
        for column, band in zip(row, pricing["tenure_adjustment"]):
            if column is None:
                continue
            for months in (band["min_months"], band["max_months"]):
                file = base_row.assign(risk_grade=grade, tenure_years=months / 12)
                assert float(constraints.card_rate(file)[0]) == pytest.approx(column), (grade, months)


def test_a_grade_with_no_offer_is_declined_not_priced(base_row):
    declined = [g for g, bps in policy()["risk_grade_loading_bps"].items() if bps is None]
    assert declined, "the policy names at least one grade with no offer"
    file = base_row.assign(risk_grade=declined[0])
    assert np.isnan(constraints.card_rate(file)[0])
    band = constraints.permissible_band(file).iloc[0]
    assert band["no_offer"] and np.isnan(band["min_rate"]) and np.isnan(band["max_rate"])
    check = constraints.check(file, policy()["reference"]["card_rate_ceiling"])
    assert not check.allowed and check.verdict == "block"
    assert any("no offer" in v for v in check.violations)


def test_a_grid_rate_above_the_ceiling_is_no_offer_never_capped(base_row):
    """Weakest offered grade, longest tenure, new to credit: the grid exceeds the ceiling.
    The policy's answer is no standard offer at that tenure - not a quote at the ceiling."""
    pol = policy()
    file = base_row.assign(risk_grade=_offered_grades()[-1], new_to_credit=True,
                           tenure_years=pol["constraints"]["max_tenure_years"])
    if constraints.grid_rate(file)[0] <= pol["reference"]["card_rate_ceiling"]:
        pytest.skip("this policy's grid never exceeds its ceiling")
    assert np.isnan(constraints.card_rate(file)[0])
    assert "ceiling" in constraints.no_offer_reason(file)


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
    # Within the discount limit, and never below the product's absolute floor.
    ok = constraints.check(base_row, max(card - (limit - 5) / 100, policy()["reference"]["card_rate_floor"]))
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


THIN_TENURE_YEARS = 10.0   # the product maximum: the longest a thin file can stretch


def _thin_file(income: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "segment": "salaried_informal", "risk_grade": "C", "property_type": "self_construction",
        "ltv": 0.85, "deviation_count": 1, "new_to_credit": False,
        "sanctioned_loan_amount": 1_500_000, "verified_monthly_income": income,
        "declared_obligations": 3_000.0, "tenure_years": THIN_TENURE_YEARS,
    }])


def _stressed_foir(amount: float, rate: float, income: float) -> float:
    """The policy's affordability test: EMI at the quoted rate plus the stress add-on."""
    emi = constraints._emi(np.array([amount]), np.array([rate + constraints.stress_add_pct()]),
                           np.array([THIN_TENURE_YEARS]))[0]
    return (emi + 3_000.0) / income


def test_affordability_caps_the_upper_end():
    """A thin file cannot be priced up indefinitely: the EMI breaches FOIR first."""
    thin = _thin_file(30_000)
    band = constraints.permissible_band(thin).iloc[0]
    assert band["affordability_binding"] or band["infeasible"]
    if not band["infeasible"]:
        assert _stressed_foir(1_500_000.0, band["max_rate"], 30_000) <= constraints.grade_foir_cap(thin)[0] + 1e-6


def test_affordability_is_tested_at_the_stressed_rate_against_the_grade_cap():
    """At the most affordable rate the stressed FOIR sits on the grade's ceiling (C: its
    own foir_max, not the product-wide one); 5 bps more and it breaks through."""
    thin = _thin_file(77_000)
    cap = constraints.grade_foir_cap(thin)[0]
    assert cap == policy()["eligibility_limits"]["C"]["foir_max"] < policy()["constraints"]["max_foir"]
    rate = float(constraints.max_affordable_rate(thin)[0])
    assert _stressed_foir(1_500_000.0, rate, 77_000) <= cap + 1e-4
    assert _stressed_foir(1_500_000.0, rate + 0.05, 77_000) > cap


def test_rate_above_affordability_is_rejected():
    """An affordable price exists inside the band, so one that breaks FOIR is blocked."""
    thin = _thin_file(77_000)
    band = constraints.permissible_band(thin).iloc[0]
    assert not band["resize_required"] and band["affordability_binding"]
    check = constraints.check(thin, float(band["card_rate"]))       # the card rate is above the affordable one
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
    assert terms["stressed_rate"] == pytest.approx(rate + constraints.stress_add_pct())
    assert _stressed_foir(terms["max_affordable_amount"], rate, 30_000) <= constraints.grade_foir_cap(thin)[0] + 1e-9


def test_no_premium_above_the_matrix_rate(base_row):
    """The matrix rate is the ceiling: a rate above the card rate is blocked, whoever approves."""
    card = float(constraints.card_rate(base_row)[0])
    band = constraints.permissible_band(base_row).iloc[0]
    assert band["max_rate"] <= card + 1e-9
    check = constraints.check(base_row, card + 0.10)
    assert not check.allowed and any("above the card rate" in v for v in check.violations)


def test_a_bureau_score_below_the_minimum_is_no_offer(base_row):
    minimum = policy()["constraints"]["min_credit_score"]
    low = base_row.assign(bureau_score=minimum - 1)
    assert np.isnan(constraints.card_rate(low)[0])
    check = constraints.check(low, 15.0)
    assert not check.allowed and any("bureau score" in v for v in check.violations)
    assert np.isfinite(constraints.card_rate(base_row.assign(bureau_score=minimum))[0])


def test_new_to_credit_is_referred_for_manual_review_not_declined(base_row):
    """No bureau file is not a score below the minimum: the file is priced, then reviewed by hand."""
    ntc = base_row.assign(new_to_credit=True, bureau_score=-1)
    card = float(constraints.card_rate(ntc)[0])
    assert np.isfinite(card)
    check = constraints.check(ntc, card)
    assert check.allowed and check.verdict == "refer"
    assert any("new to credit" in c for c in check.conditions)
