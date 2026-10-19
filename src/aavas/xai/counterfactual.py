"""Counterfactuals: what would have to change, and what happens if we move.

Two questions a branch actually asks:

* "What rate do I need to quote to have a real chance of closing this?"
* "The customer will not accept this rate - what can they change to earn a
  better one?"

The second matters for fairness as much as for sales: a customer is entitled to
know which of the drivers behind their price is something they can act on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import policy

# Levers the customer or the file can genuinely move, and how far.
ACTIONABLE = {
    "ltv": [-0.10, -0.05],
    "sanctioned_loan_amount": [-0.10, -0.05],      # proportional
    "tenure_years": [-5, -3],
    "deviation_count": [-2, -1],
    "declared_obligations": [-0.5],                # proportional
}


def rate_for_conversion(application: pd.DataFrame, conversion_model, target_probability: float,
                        band: tuple[float, float] | None = None, step_bps: int = 1) -> float | None:
    """Highest permissible rate still reaching ``target_probability`` of conversion.

    Conversion is monotone decreasing in rate by construction, so the answer is
    the last rate on a grid that still clears the target. A grid scored in one
    vectorised pass beats a bisection here: 25 sequential one-row predictions
    cost far more than a single call over 1-bp resolution.
    """
    pol = policy()["reference"]
    low, high = band or (pol["card_rate_floor"], pol["card_rate_ceiling"])
    rates = np.round(np.arange(low, high + 1e-9, step_bps / 100.0), 2)
    if len(rates) == 0:
        return None

    probabilities = conversion_model.predict_over_rates(application, rates)
    clearing = np.flatnonzero(probabilities >= target_probability)
    if len(clearing) == 0:
        return None                                  # unreachable even at the floor
    return round(float(rates[clearing[-1]]), 2)


def price_sensitivity_table(application: pd.DataFrame, conversion_model,
                            centre_rate: float, spread_bps: int = 50, step_bps: int = 10) -> pd.DataFrame:
    """The negotiation table: conversion at each step around the recommendation."""
    offsets = np.arange(-spread_bps, spread_bps + 1, step_bps) / 100.0
    rates = np.round(centre_rate + offsets, 2)
    probabilities = conversion_model.predict_over_rates(application, rates)
    frame = pd.DataFrame({"rate": rates, "conversion_probability": np.round(probabilities, 4)})
    frame["change_vs_recommended_pts"] = np.round(
        (frame["conversion_probability"] - frame.loc[frame["rate"] == round(centre_rate, 2), "conversion_probability"].mean()) * 100, 2)
    return frame


def actionable_improvements(application: pd.DataFrame, stage2_model, top_k: int = 3) -> list[dict]:
    """Which realistic change to the file would most improve the indicated price."""
    base_rate = float(stage2_model.predict_frame(application).iloc[0])

    # Build every what-if first, then score them in one pass.
    candidates: list[dict] = []
    rows: list[pd.Series] = []
    row = application.iloc[0]
    for feature, deltas in ACTIONABLE.items():
        if feature not in application.columns:
            continue
        current = float(pd.to_numeric(application[feature], errors="coerce").iloc[0])
        for delta in deltas:
            new_value = current * (1 + delta) if feature in ("sanctioned_loan_amount", "declared_obligations") \
                else current + delta
            if new_value <= 0:
                continue
            candidate = row.copy()
            candidate[feature] = new_value
            if feature == "sanctioned_loan_amount" and "appraised_property_value" in application.columns:
                candidate["ltv"] = new_value / float(application["appraised_property_value"].iloc[0])
            rows.append(candidate)
            candidates.append({"feature": feature, "from": round(current, 4), "to": round(new_value, 4)})

    if not rows:
        return []

    predicted = stage2_model.predict_frame(pd.DataFrame(rows).reset_index(drop=True)).to_numpy()

    results: list[dict] = []
    for candidate, new_rate in zip(candidates, predicted):
        improvement_bps = round((base_rate - float(new_rate)) * 100, 1)
        if improvement_bps <= 0.5:
            continue
        results.append({
            **candidate,
            "rate_improvement_bps": improvement_bps,
            "text": _phrase(candidate["feature"], candidate["from"], candidate["to"], improvement_bps),
        })

    results.sort(key=lambda r: r["rate_improvement_bps"], reverse=True)
    return results[:top_k]


def _phrase(feature: str, current: float, new: float, bps: float) -> str:
    if feature == "ltv":
        return f"Reducing LTV from {current:.0%} to {new:.0%} would improve the indicated rate by about {bps:.0f} bps"
    if feature == "sanctioned_loan_amount":
        return f"A loan amount of Rs {new:,.0f} instead of Rs {current:,.0f} would improve the rate by about {bps:.0f} bps"
    if feature == "tenure_years":
        return f"A {new:.0f}-year tenure instead of {current:.0f} years would improve the rate by about {bps:.0f} bps"
    if feature == "deviation_count":
        return f"Clearing {current - new:.0f} credit deviation(s) would improve the rate by about {bps:.0f} bps"
    if feature == "declared_obligations":
        return f"Closing existing obligations would improve the rate by about {bps:.0f} bps"
    return f"Changing {feature} from {current} to {new} would improve the rate by about {bps:.0f} bps"
