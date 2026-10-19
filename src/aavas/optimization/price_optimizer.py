"""Choose the price.

The optimizer maximises expected risk-adjusted value across the simulated
scenarios, subject to guard-rails that exist because a pure EV maximiser will
happily trade away conversion and portfolio quality for a thin margin gain:

* a minimum conversion probability (do not win the price and lose the customer),
* a maximum balance-transfer probability (do not book a loan that leaves),
* a tie-break toward the *lower* rate when two prices are economically
  indistinguishable - better for the customer, thinner tail risk, and it keeps
  the engine from drifting upward one basis point at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import policy
from ..logging_utils import get_logger
from . import yield_calculator as yc

log = get_logger(__name__)


@dataclass
class OptimizationResult:
    recommended_rate: float
    scenarios: pd.DataFrame
    best: yc.YieldResult
    rejected: dict[str, int] = field(default_factory=dict)
    binding_constraint: str | None = None
    marginal: dict[str, float] = field(default_factory=dict)
    comparisons: dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "recommended_rate": round(self.recommended_rate, 2),
            "economics": self.best.as_dict(),
            "value_decomposition": yc.decompose(self.best),
            "binding_constraint": self.binding_constraint,
            "scenarios_rejected": self.rejected,
            "marginal_analysis": self.marginal,
            "comparisons": self.comparisons,
        }


def evaluate_scenarios(
    application: pd.DataFrame,
    rates: np.ndarray,
    conversion_model,
    prepayment_model,
    expected_loss_pct: float,
    loan_amount: float,
    tenure_years: float,
) -> pd.DataFrame:
    """Score every candidate price. One vectorised pass per model, not per rate.

    Scoring rate-by-rate cost seconds per decision - almost all of it sklearn
    input validation on single-row frames, not inference. The whole grid now goes
    through each model once.
    """
    rates = np.asarray(rates, dtype=float)
    if len(rates) == 0:
        return pd.DataFrame()

    p_conv = conversion_model.predict_over_rates(application, rates)
    p_bt = prepayment_model.predict_over_rates(application, rates)
    life = prepayment_model.life_from_probability(p_bt)

    results = yc.compute(
        rate=rates, conversion_probability=p_conv, expected_loss_pct=expected_loss_pct,
        expected_life_years=life, loan_amount=loan_amount, tenure_years=tenure_years,
        bt_out_probability=p_bt,
    )
    return pd.DataFrame([r.as_dict() for r in results])


def optimize(scenarios: pd.DataFrame, anchors: dict[str, float] | None = None) -> OptimizationResult:
    """Pick the value-maximising permissible price from a scored scenario table."""
    cfg = policy()["optimizer"]
    if scenarios.empty:
        raise ValueError("no scenarios to optimize over")

    feasible = scenarios.copy()
    rejected: dict[str, int] = {}

    low_conversion = feasible["conversion_probability"] < cfg["min_conversion_probability"]
    if low_conversion.any():
        rejected["below_min_conversion"] = int(low_conversion.sum())
    high_bt = feasible["bt_out_probability"] > cfg["max_bt_out_probability"]
    if high_bt.any():
        rejected["above_max_bt_out"] = int(high_bt.sum())

    allowed = feasible[~(low_conversion | high_bt)]
    binding = None
    if allowed.empty:
        # Never return nothing: fall back to the price that best satisfies the
        # guard-rail we care most about, and say so loudly.
        allowed = feasible.nlargest(1, "conversion_probability")
        binding = "all scenarios breached a guard-rail; fell back to the most convertible price"
        log.warning(binding)
    elif len(allowed) < len(feasible):
        binding = "conversion floor" if low_conversion.any() else "balance-transfer ceiling"

    best_ev = allowed["expected_value_inr"].max()
    tolerance = abs(best_ev) * cfg["ev_tolerance_pct"] / 100.0
    near_best = allowed[allowed["expected_value_inr"] >= best_ev - tolerance]
    chosen = near_best.nsmallest(1, "rate").iloc[0]
    if len(near_best) > 1:
        binding = binding or f"{len(near_best)} prices within {cfg['ev_tolerance_pct']}% of best value; chose the lowest"

    best = yc.YieldResult(**{k: chosen[k] for k in yc.YieldResult.__dataclass_fields__})
    result = OptimizationResult(
        recommended_rate=float(chosen["rate"]), scenarios=scenarios, best=best,
        rejected=rejected, binding_constraint=binding,
        marginal=_marginal_analysis(scenarios, float(chosen["rate"])),
        comparisons=_comparisons(scenarios, anchors or {}, float(chosen["rate"])),
    )
    return result


def _marginal_analysis(scenarios: pd.DataFrame, chosen_rate: float) -> dict[str, float]:
    """What the next step of price buys, and what it costs - the number that
    answers 'why not just charge more?'.

    The step is the configured pricing increment, not the adjacent row: anchor
    rates are folded into the grid, so neighbouring rows can be 1 bp apart and
    would make the marginal cost look absurdly steep.
    """
    cfg = policy()["optimizer"]
    step = cfg["step_bps"] / 100.0
    ordered = scenarios.sort_values("rate").reset_index(drop=True)
    current = ordered.iloc[(ordered["rate"] - chosen_rate).abs().idxmin()]

    out: dict[str, float] = {}
    for label, target in (("next", chosen_rate + step), ("previous", chosen_rate - step)):
        if not (ordered["rate"].min() - 1e-9 <= target <= ordered["rate"].max() + 1e-9):
            continue
        neighbour = ordered.iloc[(ordered["rate"] - target).abs().idxmin()]
        if abs(neighbour["rate"] - current["rate"]) < 1e-9:
            continue
        out[f"{label}_step_bps"] = round((neighbour["rate"] - current["rate"]) * 100, 1)
        out[f"{label}_step_conversion_change_pts"] = round(
            (neighbour["conversion_probability"] - current["conversion_probability"]) * 100, 2)
        out[f"{label}_step_value_change_inr"] = round(
            neighbour["expected_value_inr"] - current["expected_value_inr"], 0)

    peak = ordered.loc[ordered["expected_value_inr"].idxmax()]
    out["unconstrained_peak_rate"] = round(float(peak["rate"]), 2)
    out["value_at_peak_inr"] = round(float(peak["expected_value_inr"]), 0)
    return out


def _comparisons(scenarios: pd.DataFrame, anchors: dict[str, float], chosen_rate: float) -> dict[str, dict]:
    """Value lost or gained versus each anchor a human might argue for."""
    ordered = scenarios.sort_values("rate").reset_index(drop=True)
    chosen = ordered.iloc[(ordered["rate"] - chosen_rate).abs().idxmin()]
    out: dict[str, dict] = {}
    for name, rate in anchors.items():
        if not np.isfinite(rate):
            continue
        nearest = ordered.iloc[(ordered["rate"] - rate).abs().idxmin()]
        if abs(nearest["rate"] - rate) > 0.051:
            continue      # anchor sits outside the permissible band
        out[name] = {
            "rate": round(float(nearest["rate"]), 2),
            "conversion_probability": round(float(nearest["conversion_probability"]), 4),
            "expected_value_inr": round(float(nearest["expected_value_inr"]), 0),
            "value_vs_recommended_inr": round(float(nearest["expected_value_inr"] - chosen["expected_value_inr"]), 0),
        }
    return out
