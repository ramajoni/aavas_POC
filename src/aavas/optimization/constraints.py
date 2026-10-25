"""The policy envelope: what price is *allowed*, before asking what price is best.

This module is the single source of truth for the pricing grid. The synthetic
generator, the policy agent and the scenario generator all call it, so the
history the models learn from and the rules the engine enforces cannot drift
apart.

Nothing here is fitted. Every number comes from ``config/pricing_policy.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import policy
from ..kernel.columns import boolean_column, numeric_column
from ..ingestion.schemas import RateBand


# ------------------------------------------------------------------- card rate


def card_rate(df: pd.DataFrame) -> np.ndarray:
    """Policy card rate: floor + segment + risk grade + LTV + property + deviations."""
    pol = policy()
    n = len(df)
    floor = pol["reference"]["card_rate_floor"]

    bps = _map(df, "segment", pol["base_spread_bps"], default=np.mean(list(pol["base_spread_bps"].values())))
    bps = bps + _map(df, "risk_grade", pol["risk_grade_loading_bps"], default=pol["risk_grade_loading_bps"]["C"])
    bps = bps + _map(df, "property_type", pol["property_type_loading_bps"], default=0.0)

    ltv = _numeric(df, "ltv", np.nan)
    if np.isnan(ltv).any():
        ltv = np.where(np.isnan(ltv), _numeric(df, "estimated_ltv", 0.70), ltv)
    ltv_bps = np.full(n, float(pol["ltv_loading_bps"][-1]["bps"]))
    for band in sorted(pol["ltv_loading_bps"], key=lambda b: b["max_ltv"], reverse=True):
        ltv_bps = np.where(ltv <= band["max_ltv"], float(band["bps"]), ltv_bps)
    bps = bps + ltv_bps

    dev_cfg = pol["deviation_loading_bps"]
    deviations = _numeric(df, "deviation_count", 0.0)
    bps = bps + np.minimum(deviations * dev_cfg["per_deviation"], dev_cfg["max_total"])

    ntc = _boolean(df, "new_to_credit")
    bps = bps + np.where(ntc, pol["constraints"]["new_to_credit_min_spread_bps"], 0.0)

    return np.round(floor + bps / 100.0, 2)


def _map(df: pd.DataFrame, column: str, table: dict, default: float) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), float(default))
    return df[column].map(table).astype(float).fillna(default).to_numpy()


def _numeric(df: pd.DataFrame, column: str, default: float) -> np.ndarray:
    return numeric_column(df, column, default).to_numpy(dtype=float)


def _boolean(df: pd.DataFrame, column: str, default: bool = False) -> np.ndarray:
    return boolean_column(df, column, default).to_numpy()


# ------------------------------------------------------------ affordability cap


def max_affordable_rate(df: pd.DataFrame, foir_cap: float | None = None) -> np.ndarray:
    """Highest rate at which the file still clears the FOIR cap.

    Raising the price raises the EMI, which can push a thin file past the
    affordability limit. Bisection keeps this exact without inverting the EMI
    formula analytically.
    """
    pol = policy()
    cap = pol["constraints"]["max_foir"] if foir_cap is None else foir_cap
    principal = _numeric(df, "sanctioned_loan_amount", np.nan)
    principal = np.where(np.isnan(principal), _numeric(df, "requested_loan_amount", 0.0), principal)
    income = _numeric(df, "verified_monthly_income", np.nan)
    income = np.where(np.isnan(income), _numeric(df, "declared_monthly_income", 0.0), income)
    obligations = _numeric(df, "declared_obligations", 0.0)
    tenure = _numeric(df, "tenure_years", np.nan)
    tenure = np.where(np.isnan(tenure), _numeric(df, "requested_tenure_years", 20.0), tenure)

    lo = np.full(len(df), pol["reference"]["card_rate_floor"])
    hi = np.full(len(df), pol["reference"]["card_rate_ceiling"])
    for _ in range(40):
        mid = (lo + hi) / 2
        foir = (_emi(principal, mid, tenure) + obligations) / np.maximum(income, 1.0)
        affordable = foir <= cap
        lo = np.where(affordable, mid, lo)
        hi = np.where(affordable, hi, mid)
    return np.round(lo, 2)


def _emi(principal: np.ndarray, annual_rate: np.ndarray, years: np.ndarray) -> np.ndarray:
    r = np.asarray(annual_rate, dtype=float) / 100.0 / 12.0
    n = np.asarray(years, dtype=float) * 12
    growth = (1 + r) ** n
    return principal * r * growth / np.maximum(growth - 1, 1e-12)


# --------------------------------------------------------------- permissible band


def permissible_band(df: pd.DataFrame, card: np.ndarray | None = None) -> pd.DataFrame:
    """The rate interval a file may legitimately be priced in.

    Affordability caps the top of the band when some permissible rate still keeps
    the file inside the FOIR cap. When none does - the EMI breaches FOIR even at
    the policy floor - price cannot fix the file and capping the band would only
    collapse it to a single point below card. That is a loan-sizing problem, so
    the file is priced on its risk-based band and flagged ``resize_required``:
    every rate in the band carries the condition that the ticket or tenure
    changes (see ``resize_terms``). ``infeasible`` is kept as the same flag for
    callers that predate the resize path.
    """
    pol = policy()
    card = card_rate(df) if card is None else np.asarray(card, dtype=float)
    max_discount = pol["constraints"]["absolute_discount_floor_bps"]
    max_premium = max(level["max_premium_bps"] for level in pol["approval_authority"])

    lower = np.maximum(card - max_discount / 100.0, pol["reference"]["card_rate_floor"])
    policy_upper = np.minimum(card + max_premium / 100.0, pol["reference"]["card_rate_ceiling"])
    affordable = max_affordable_rate(df)
    resize_required = affordable < lower
    upper = np.where(resize_required, policy_upper, np.minimum(policy_upper, affordable))

    return pd.DataFrame({
        "card_rate": card,
        "min_rate": np.round(lower, 2),
        "max_rate": np.round(upper, 2),
        "max_affordable_rate": affordable,
        "affordability_binding": affordable < policy_upper,
        "resize_required": resize_required,
        "infeasible": resize_required,
    }, index=df.index)


def resize_terms(application: pd.DataFrame, rate: float) -> dict:
    """What has to change for this file to clear the FOIR cap at ``rate``.

    Two levers a credit officer actually has: a smaller ticket, or a longer tenure
    within the product maximum. Both are computed at the rate being priced, since
    the EMI - and so the answer - moves with it. Amounts round *down* to the
    nearest Rs 1,000: rounding up would hand back a ticket that breaches again.
    """
    pol = policy()
    row = application.iloc[[0]] if isinstance(application, pd.DataFrame) else pd.DataFrame([application])
    cap = pol["constraints"]["max_foir"]
    max_tenure = int(pol["constraints"]["max_tenure_years"])
    principal = float(_numeric(row, "sanctioned_loan_amount", np.nan)[0])
    if np.isnan(principal):
        principal = float(_numeric(row, "requested_loan_amount", 0.0)[0])
    income = float(_numeric(row, "verified_monthly_income", np.nan)[0])
    if np.isnan(income):
        income = float(_numeric(row, "declared_monthly_income", 0.0)[0])
    obligations = float(_numeric(row, "declared_obligations", 0.0)[0])
    tenure = float(_numeric(row, "tenure_years", np.nan)[0])
    if np.isnan(tenure):
        tenure = float(_numeric(row, "requested_tenure_years", 20.0)[0])

    def emi(amount: float, years: float) -> float:
        return float(_emi(np.array([amount]), np.array([rate]), np.array([years]))[0])

    headroom = cap * income - obligations
    terms = {
        "rate": round(float(rate), 2),
        "foir_cap": cap,
        "foir_at_rate": round((emi(principal, tenure) + obligations) / max(income, 1.0), 4),
        "current_amount": round(principal, 0),
        "current_tenure_years": int(tenure),
    }
    if headroom <= 0:
        terms.update({"possible": False,
                      "reason": "existing obligations already use the whole FOIR allowance"})
        return terms
    per_rupee = emi(1.0, tenure)
    max_amount = float(np.floor(headroom / per_rupee / 1000.0) * 1000.0)
    tenure_needed = next((years for years in range(int(np.ceil(tenure)), max_tenure + 1)
                          if emi(principal, years) <= headroom + 1e-6), None)
    terms.update({
        "possible": True,
        "max_affordable_amount": max_amount,
        "reduce_by": round(max(principal - max_amount, 0.0), 0),
        "tenure_needed_years": tenure_needed,
        "max_tenure_years": max_tenure,
    })
    return terms


def _resize_condition(terms: dict) -> str:
    if not terms.get("possible"):
        return f"no loan amount clears the FOIR cap: {terms.get('reason')}"
    text = (f"resize required: at {terms['rate']:.2f}% the FOIR is {terms['foir_at_rate']:.0%} against a "
            f"{terms['foir_cap']:.0%} cap - reduce the loan to Rs {terms['max_affordable_amount']:,.0f}")
    if terms.get("tenure_needed_years"):
        text += f" or extend the tenure to {terms['tenure_needed_years']} years"
    return text


# ------------------------------------------------------------ approval authority


def approval_level(discount_bps: float) -> str:
    """Lowest authority that may sign off this deviation from the card rate.

    ``discount_bps`` is positive for a discount, negative for a premium.
    """
    pol = policy()
    levels = sorted(pol["approval_authority"], key=lambda level: level["max_discount_bps"])
    if discount_bps <= 0:
        premium = -discount_bps
        for level in sorted(levels, key=lambda level: level["max_premium_bps"]):
            if premium <= level["max_premium_bps"]:
                return level["level"]
        return levels[-1]["level"]
    for level in levels:
        if discount_bps <= level["max_discount_bps"]:
            return level["level"]
    return levels[-1]["level"]


# --------------------------------------------------------------- policy check


@dataclass
class PolicyCheck:
    allowed: bool
    rate: float
    card_rate: float
    discount_bps: float
    required_approval: str
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    band: RateBand | None = None
    # Conditions do not block the price; they must be met before the file
    # proceeds. Today the only one is a resize when no rate is affordable.
    conditions: list[str] = field(default_factory=list)
    resize: dict | None = None

    @property
    def verdict(self) -> str:
        """Pass, refer or block - the three outcomes the guardrail can return.

        Block is any violation. Refer is a price that is allowed but cannot be
        released by the lowest authority alone: it carries a condition, or it
        needs a higher approval level.
        """
        if self.violations:
            return "block"
        lowest = min(policy()["approval_authority"], key=lambda level: level["max_discount_bps"])["level"]
        if self.conditions or self.required_approval != lowest:
            return "refer"
        return "pass"

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "verdict": self.verdict,
            "rate": round(self.rate, 2),
            "card_rate": round(self.card_rate, 2),
            "discount_bps": round(self.discount_bps, 1),
            "required_approval": self.required_approval,
            "violations": self.violations,
            "conditions": self.conditions,
            "warnings": self.warnings,
            "resize": self.resize,
            "permissible_band": self.band.as_dict() if self.band else None,
            "policy_version": policy()["policy_version"],
        }


def check(application: pd.DataFrame, rate: float, override_reason: str | None = None,
          evidence_verified: bool = False) -> PolicyCheck:
    """Validate one proposed rate for one application."""
    pol = policy()
    row = application.iloc[[0]] if isinstance(application, pd.DataFrame) else pd.DataFrame([application])
    bands = permissible_band(row)
    band_row = bands.iloc[0]
    card = float(band_row["card_rate"])
    discount_bps = round((card - rate) * 100, 1)

    violations: list[str] = []
    warnings: list[str] = []

    if rate < pol["reference"]["card_rate_floor"]:
        violations.append(f"rate {rate:.2f}% is below the absolute floor {pol['reference']['card_rate_floor']:.2f}%")
    if rate > pol["reference"]["card_rate_ceiling"]:
        violations.append(f"rate {rate:.2f}% is above the product ceiling {pol['reference']['card_rate_ceiling']:.2f}%")
    if discount_bps > pol["constraints"]["absolute_discount_floor_bps"]:
        violations.append(
            f"discount of {discount_bps:.0f} bps exceeds the maximum permissible "
            f"{pol['constraints']['absolute_discount_floor_bps']} bps from card rate"
        )
    conditions: list[str] = []
    resize = None
    if bool(band_row["resize_required"]):
        # Price cannot restore affordability here, so a rate above the affordable
        # one is not the human's error: the ticket is. Allowed on condition.
        resize = resize_terms(row, rate)
        conditions.append(_resize_condition(resize))
    elif rate > float(band_row["max_affordable_rate"]) + 1e-9:
        # An affordable price exists inside the band, so choosing one that breaks
        # FOIR is a pricing decision, and it is blocked.
        violations.append(
            f"rate {rate:.2f}% breaches the FOIR cap of {pol['constraints']['max_foir']:.0%} "
            f"(maximum affordable rate {band_row['max_affordable_rate']:.2f}%)"
        )

    # Unverified competitive intelligence is captured as context, never priced in
    # automatically beyond a small tolerance.
    if override_reason in pol["override"]["evidence_required_reasons"] and not evidence_verified:
        limit = pol["override"]["max_unevidenced_discount_bps"]
        if discount_bps > limit:
            violations.append(
                f"reason '{override_reason}' requires evidence for a discount above {limit} bps "
                f"(requested {discount_bps:.0f} bps)"
            )
        else:
            warnings.append(f"reason '{override_reason}' is unevidenced; discount capped at {limit} bps")

    ltv = float(_numeric(row, "ltv", 0.0)[0])
    if ltv > pol["constraints"]["max_ltv"] + 1e-9:
        violations.append(f"LTV {ltv:.2%} exceeds the policy maximum {pol['constraints']['max_ltv']:.0%}")
    if bool(band_row["affordability_binding"]) and not bool(band_row["resize_required"]):
        warnings.append("affordability, not policy, is the binding constraint on the upper end of the band")

    return PolicyCheck(
        allowed=not violations,
        rate=rate,
        card_rate=card,
        discount_bps=discount_bps,
        required_approval=approval_level(discount_bps),
        violations=violations,
        warnings=warnings,
        band=RateBand(float(band_row["min_rate"]), float(band_row["max_rate"])),
        conditions=conditions,
        resize=resize,
    )
