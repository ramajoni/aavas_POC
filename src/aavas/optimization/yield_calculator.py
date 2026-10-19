"""Risk-adjusted economics of a single proposed price.

This is where the business statement's central distinction lives:

    Risk-Adjusted Value = P(conversion) x Expected Loan Economics
                          - Credit Loss - Funding/Servicing Cost - Prepayment/BT Impact

A higher rate raises the carry on every rupee but lowers the probability there
are any rupees at all, and shortens the life over which the carry is earned. All
three effects are priced here, in rupees, so the optimizer compares like with
like.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..config import policy


@dataclass
class YieldResult:
    rate: float
    conversion_probability: float
    bt_out_probability: float
    expected_life_years: float
    expected_loss_pct: float
    gross_spread_pct: float          # rate - funding - servicing - capital
    net_margin_pct: float            # ...after expected credit loss
    lifetime_value_pct: float        # value if the customer converts, % of disbursal
    expected_value_pct: float        # ...times conversion probability
    expected_value_inr: float
    contractual_yield_pct: float     # the naive "uplift" number, for contrast

    def as_dict(self) -> dict:
        out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}
        out["expected_value_inr"] = round(self.expected_value_inr, 0)
        return out


def average_outstanding_factor(life_years: np.ndarray | float, tenure_years: np.ndarray | float) -> np.ndarray:
    """Average balance over the held period, as a fraction of the original loan.

    A home loan amortises slowly at first, so a loan that exits early sits on
    almost its full principal, while one held to term averages far less. Getting
    this roughly right matters: it is the reason an early balance transfer costs
    more than the annualised yield suggests.
    """
    life = np.asarray(life_years, dtype=float)
    tenure = np.maximum(np.asarray(tenure_years, dtype=float), 1.0)
    fraction_of_term = np.clip(life / tenure, 0.0, 1.0)
    # Linear-ish decline from ~97% for a very short hold to ~58% held to term.
    return np.clip(0.97 - 0.39 * fraction_of_term, 0.55, 0.99)


def compute(
    rate: float | np.ndarray,
    conversion_probability: float | np.ndarray,
    expected_loss_pct: float | np.ndarray,
    expected_life_years: float | np.ndarray,
    loan_amount: float | np.ndarray,
    tenure_years: float | np.ndarray = 20.0,
    bt_out_probability: float | np.ndarray = 0.0,
) -> YieldResult | list[YieldResult]:
    """Economics of offering ``rate``. Scalars in, scalar result out."""
    econ = policy()["economics"]

    rate_a = np.asarray(rate, dtype=float)
    p_conv = np.clip(np.asarray(conversion_probability, dtype=float), 0.0, 1.0)
    el = np.asarray(expected_loss_pct, dtype=float)
    life = np.clip(np.asarray(expected_life_years, dtype=float), 0.25, econ["behavioural_horizon_years"])
    amount = np.asarray(loan_amount, dtype=float)

    gross_spread = rate_a - econ["funding_cost_pct"] - econ["servicing_cost_pct"] - econ["capital_charge_pct"]
    net_margin = gross_spread - el

    avg_outstanding = average_outstanding_factor(life, tenure_years)
    disc = econ["discount_rate_pct"] / 100.0
    # Continuous-time annuity factor: present value of 1%/yr earned over `life`.
    annuity = (1 - np.exp(-disc * life)) / disc

    lifetime_value = (
        net_margin * avg_outstanding * annuity
        + econ["processing_fee_pct"]
        - econ["acquisition_cost_pct"]
    )
    expected_value_pct = p_conv * lifetime_value
    expected_value_inr = expected_value_pct / 100.0 * amount

    # Broadcast explicitly. Some arguments are naturally scalar (expected loss and
    # loan amount are per-file, rates are per-scenario), and zipping a mix of
    # lengths silently truncates to the shortest - which would quietly collapse a
    # whole scenario grid to a single row and hand the optimizer one choice.
    columns = _broadcast(
        rate_a, p_conv, np.asarray(bt_out_probability, dtype=float), life, el,
        gross_spread, net_margin, lifetime_value, expected_value_pct, expected_value_inr,
    )

    results = [
        YieldResult(
            rate=float(r), conversion_probability=float(pc), bt_out_probability=float(bt),
            expected_life_years=float(lf), expected_loss_pct=float(e),
            gross_spread_pct=float(gs), net_margin_pct=float(nm),
            lifetime_value_pct=float(lv), expected_value_pct=float(ev), expected_value_inr=float(evi),
            contractual_yield_pct=float(r - econ["funding_cost_pct"]),
        )
        for r, pc, bt, lf, e, gs, nm, lv, ev, evi in zip(*columns)
    ]
    return results[0] if np.ndim(rate_a) == 0 else results


def _broadcast(*arrays: np.ndarray) -> list[np.ndarray]:
    """Stretch every argument to the longest length, or fail loudly."""
    expanded = [np.atleast_1d(a) for a in arrays]
    length = max(len(a) for a in expanded)
    out = []
    for array in expanded:
        if len(array) == length:
            out.append(array)
        elif len(array) == 1:
            out.append(np.repeat(array, length))
        else:
            raise ValueError(
                f"cannot broadcast an input of length {len(array)} to {length}; "
                "yield inputs must be scalar or the same length as the rate grid"
            )
    return out


def breakeven_rate(expected_loss_pct: float) -> float:
    """The rate below which the loan loses money even if it converts and is held."""
    econ = policy()["economics"]
    return round(
        econ["funding_cost_pct"] + econ["servicing_cost_pct"] + econ["capital_charge_pct"] + expected_loss_pct, 2
    )


def decompose(result: YieldResult) -> dict[str, float]:
    """Where the money goes, in percentage points - for the explanation payload."""
    econ = policy()["economics"]
    return {
        "customer_rate": round(result.rate, 2),
        "less_funding_cost": -econ["funding_cost_pct"],
        "less_servicing_cost": -econ["servicing_cost_pct"],
        "less_capital_charge": -econ["capital_charge_pct"],
        "less_expected_credit_loss": -round(result.expected_loss_pct, 3),
        "net_annual_margin": round(result.net_margin_pct, 3),
        "expected_life_years": round(result.expected_life_years, 2),
        "lifetime_value_if_converted_pct": round(result.lifetime_value_pct, 3),
        "conversion_probability": round(result.conversion_probability, 4),
        "risk_adjusted_value_pct": round(result.expected_value_pct, 3),
    }
