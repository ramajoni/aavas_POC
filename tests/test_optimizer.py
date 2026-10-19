"""The optimizer decides the price, so its economics and guard-rails are tested
directly rather than only through the orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aavas.config import policy
from aavas.optimization import price_optimizer, scenario_generator
from aavas.optimization import yield_calculator as yc


# ------------------------------------------------------------------ economics


def test_value_rises_with_rate_when_conversion_is_held_fixed():
    values = [yc.compute(rate=r, conversion_probability=0.7, expected_loss_pct=0.8,
                         expected_life_years=6.0, loan_amount=1_000_000).expected_value_inr
              for r in (10.0, 10.5, 11.0)]
    assert values == sorted(values)


def test_value_falls_when_the_loan_leaves_early():
    long_life = yc.compute(rate=11.0, conversion_probability=0.8, expected_loss_pct=0.8,
                           expected_life_years=8.0, loan_amount=1_000_000)
    short_life = yc.compute(rate=11.0, conversion_probability=0.8, expected_loss_pct=0.8,
                            expected_life_years=1.2, loan_amount=1_000_000)
    assert short_life.expected_value_inr < long_life.expected_value_inr, (
        "an early balance transfer must destroy value, which is the whole reason "
        "the prepayment model exists")


def test_a_rate_below_breakeven_loses_money():
    expected_loss = 1.0
    breakeven = yc.breakeven_rate(expected_loss)
    below = yc.compute(rate=breakeven - 1.0, conversion_probability=1.0, expected_loss_pct=expected_loss,
                       expected_life_years=6.0, loan_amount=1_000_000)
    assert below.net_margin_pct < 0
    assert below.expected_value_inr < 0


def test_higher_price_is_not_automatically_better():
    """Price uplift versus yield maximization, in one assertion: with a realistic
    demand curve the value-maximising rate is strictly inside the band."""
    rates = np.arange(10.0, 13.01, 0.05)
    # Conversion falls ~7 points per 25 bps - in line with the fitted model.
    conversions = np.clip(0.92 - (rates - 10.0) * 0.28, 0.02, 1.0)
    values = [yc.compute(rate=float(r), conversion_probability=float(p), expected_loss_pct=0.8,
                         expected_life_years=6.0, loan_amount=1_000_000).expected_value_inr
              for r, p in zip(rates, conversions)]
    best = rates[int(np.argmax(values))]
    assert rates[0] < best < rates[-1], f"optimum collapsed to a boundary at {best}"


# ------------------------------------------------------------------ optimizer


def _scenarios(rates, conversions, bt=0.1, life=6.0, value=None) -> pd.DataFrame:
    rows = []
    for rate, conversion in zip(rates, conversions):
        result = yc.compute(rate=float(rate), conversion_probability=float(conversion),
                            expected_loss_pct=0.8, expected_life_years=life,
                            loan_amount=1_000_000, bt_out_probability=bt)
        rows.append(result.as_dict())
    return pd.DataFrame(rows)


def test_optimizer_picks_the_value_peak():
    rates = np.arange(10.0, 13.01, 0.05)
    conversions = np.clip(0.92 - (rates - 10.0) * 0.28, 0.02, 1.0)
    scenarios = _scenarios(rates, conversions)
    result = price_optimizer.optimize(scenarios)
    peak = float(scenarios.loc[scenarios["expected_value_inr"].idxmax(), "rate"])
    assert abs(result.recommended_rate - peak) <= 0.25


def test_conversion_floor_rejects_unconvertible_prices():
    floor = policy()["optimizer"]["min_conversion_probability"]
    rates = np.array([10.0, 11.0, 12.0])
    conversions = np.array([0.9, 0.5, floor / 2])
    result = price_optimizer.optimize(_scenarios(rates, conversions))
    assert result.rejected.get("below_min_conversion", 0) >= 1
    assert result.recommended_rate != 12.0


def test_balance_transfer_ceiling_is_enforced():
    ceiling = policy()["optimizer"]["max_bt_out_probability"]
    scenarios = _scenarios([10.0, 11.0], [0.8, 0.75], bt=ceiling + 0.1)
    result = price_optimizer.optimize(scenarios)
    assert result.rejected.get("above_max_bt_out", 0) >= 1
    assert result.binding_constraint


def test_tie_break_prefers_the_lower_rate():
    """Two economically indistinguishable prices: the customer gets the cheaper one.

    Expected values are set directly so the test exercises the tie-break rule
    rather than a hand-tuned demand curve.
    """
    scenarios = _scenarios([10.50, 10.55, 10.60], [0.80, 0.79, 0.78])
    tolerance = policy()["optimizer"]["ev_tolerance_pct"] / 100.0
    peak = 100_000.0
    scenarios["expected_value_inr"] = [peak * (1 - tolerance / 2), peak, peak * (1 - tolerance / 2)]

    result = price_optimizer.optimize(scenarios)
    assert result.recommended_rate == 10.50, "within tolerance, the cheaper price must win"
    assert "within" in (result.binding_constraint or "")


def test_tie_break_does_not_give_away_real_value():
    """The tie-break is a tolerance, not a preference for cheap: a materially
    better price still wins."""
    scenarios = _scenarios([10.50, 10.55, 10.60], [0.80, 0.79, 0.78])
    scenarios["expected_value_inr"] = [80_000.0, 100_000.0, 90_000.0]
    result = price_optimizer.optimize(scenarios)
    assert result.recommended_rate == 10.55


def test_optimizer_always_returns_something():
    """Even when every scenario breaches a guard-rail, a branch needs an answer
    plus a loud explanation - not an exception."""
    scenarios = _scenarios([12.0, 12.5], [0.05, 0.02])
    result = price_optimizer.optimize(scenarios)
    assert result.recommended_rate in (12.0, 12.5)
    assert "guard-rail" in (result.binding_constraint or "")


def test_empty_scenarios_raise():
    with pytest.raises(ValueError):
        price_optimizer.optimize(pd.DataFrame())


# ------------------------------------------------------------ scenario search


def test_scenario_grid_stays_inside_policy(training_data):
    application = training_data.head(1)
    scenarios = scenario_generator.generate(application)
    assert scenarios.rates.min() >= scenarios.band.min_rate - 1e-9
    assert scenarios.rates.max() <= scenarios.band.max_rate + 1e-9
    assert len(scenarios.rates) > 1


def test_search_is_trimmed_to_where_the_model_has_data(training_data):
    """Guards the failure mode where a monotone GBM extrapolates flat above every
    rate it ever saw and the optimizer walks off to the ceiling chasing an
    artefact of that flat line."""
    application = training_data.head(1)
    unrestricted = scenario_generator.generate(application)

    # A support window that genuinely overlaps this file's permissible band.
    low = unrestricted.band.min_rate
    cap = round(low + (unrestricted.band.max_rate - low) / 2, 2)
    trimmed = scenario_generator.generate(application, support_band=(low, cap))

    assert trimmed.rates.max() <= cap + 1e-9
    assert trimmed.rates.max() < unrestricted.rates.max()
    assert any("extrapolating" in note for note in trimmed.notes)


def test_no_model_support_downgrades_confidence_rather_than_refusing(training_data):
    """If the model has no data anywhere in the permissible band, the branch still
    needs a price - but the record has to say the response is extrapolated."""
    application = training_data.head(1)
    envelope = scenario_generator.generate(application)
    far_away = (envelope.band.min_rate - 3.0, envelope.band.min_rate - 2.0)
    scenarios = scenario_generator.generate(application, support_band=far_away)
    assert len(scenarios.rates) > 1, "a price must still be produced"
    assert any("low confidence" in note for note in scenarios.notes)


def test_unmatchable_competitor_quote_is_called_out(training_data):
    application = training_data.head(1)
    envelope = scenario_generator.generate(application)
    scenarios = scenario_generator.generate(application, competitor_rate=envelope.band.min_rate - 1.0)
    assert any("cannot be matched" in note for note in scenarios.notes)


def test_end_to_end_recommendation_is_permissible(decision):
    """Whatever the optimizer picks, the policy agent must have cleared it."""
    rate = decision["recommendation"]["recommended_rate"]
    envelope = decision["envelope"]
    assert envelope["min_rate"] - 1e-9 <= rate <= envelope["max_rate"] + 1e-9
    assert decision["policy"]["allowed"]
    assert decision["policy"]["required_approval"]
    assert decision["human_decision_required"] is True


# ------------------------------------------------- vectorised scoring contract


def test_scalar_and_array_inputs_broadcast_instead_of_truncating():
    """Regression: the rate grid is an array while expected loss and loan amount
    are per-file scalars. Zipping that mix silently truncated the whole grid to
    one row, collapsing the scenario table to a single choice."""
    rates = np.arange(10.0, 12.01, 0.05)
    conversions = np.linspace(0.9, 0.3, len(rates))
    results = yc.compute(
        rate=rates, conversion_probability=conversions,
        expected_loss_pct=0.8,            # scalar, per file
        expected_life_years=6.0,          # scalar
        loan_amount=1_000_000,            # scalar
        tenure_years=20.0,
        bt_out_probability=0.1,
    )
    assert isinstance(results, list)
    assert len(results) == len(rates), f"expected {len(rates)} scenarios, got {len(results)}"
    assert [r.rate for r in results] == pytest.approx(list(rates))


def test_mismatched_array_lengths_fail_loudly():
    with pytest.raises(ValueError, match="broadcast"):
        yc.compute(rate=np.array([10.0, 10.5, 11.0]),
                   conversion_probability=np.array([0.8, 0.7]),
                   expected_loss_pct=0.8, expected_life_years=6.0, loan_amount=1_000_000)


def test_scenario_table_covers_the_whole_grid(bundle, training_data):
    """End-to-end guard on the same failure: the scored table must have one row
    per candidate rate."""
    from aavas.optimization import scenario_generator

    application = training_data.head(1)
    scenarios = scenario_generator.generate(
        application, support_band=bundle.conversion.supported_rate_band(application))
    scored = price_optimizer.evaluate_scenarios(
        application=application, rates=scenarios.rates,
        conversion_model=bundle.conversion, prepayment_model=bundle.prepayment,
        expected_loss_pct=0.9, loan_amount=1_200_000, tenure_years=20)
    assert len(scored) == len(scenarios.rates) > 1


def test_vectorised_scoring_matches_per_rate_scoring(bundle, training_data):
    """The optimisation must not change the answer, only the speed."""
    application = training_data.head(1)
    rates = np.round(np.arange(10.5, 11.51, 0.05), 2)
    batched = bundle.conversion.predict_over_rates(application, rates)
    one_at_a_time = np.array([bundle.conversion.predict_at_rate(application, float(r))[0] for r in rates])
    assert batched == pytest.approx(one_at_a_time, abs=1e-9)
