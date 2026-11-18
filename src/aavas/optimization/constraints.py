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
    """Policy card rate: floor + segment + risk grade + LTV + property + tenure + deviations.

    NaN where the policy makes no standard offer (``no_offer``): a grade loaded as
    null, or a grid rate above the product ceiling (rate outside 13-17% is
    ``no_standard_offer``, never capped at the ceiling). Pricing such a file on
    another grade's loading, or at the ceiling, would quote a loan the policy
    declines. A grade the table does not name at all is priced as C, the neutral
    grade used before the risk agent has graded the file.
    """
    grid = grid_rate(df)
    return np.where(no_offer(df, grid), np.nan, grid)


def grid_rate(df: pd.DataFrame) -> np.ndarray:
    """The rate the grid builds, before asking whether the policy offers it at all."""
    pol = policy()
    n = len(df)
    floor = pol["reference"]["card_rate_floor"]
    grades = pol["risk_grade_loading_bps"]

    bps = _map(df, "segment", pol["base_spread_bps"], default=np.mean(list(pol["base_spread_bps"].values())))
    bps = bps + _map(df, "risk_grade", {g: v for g, v in grades.items() if v is not None}, default=grades["C"])
    bps = bps + _map(df, "property_type", pol["property_type_loading_bps"], default=0.0)
    bps = bps + _tenure_bps(df, pol.get("tenure_loading_bps") or [])

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


def declined_grade(df: pd.DataFrame) -> np.ndarray:
    """True where the file's grade has no offer under standard policy (a null loading)."""
    if "risk_grade" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    declined = [g for g, v in policy()["risk_grade_loading_bps"].items() if v is None]
    return df["risk_grade"].astype(str).isin(declined).to_numpy()


def bureau_below_minimum(df: pd.DataFrame) -> np.ndarray:
    """True where a scored applicant's bureau score is under constraints.min_credit_score.

    A new-to-credit file (no bureau file: score below 0 or flagged) is not "below the
    minimum" - it has no score, and goes to manual review instead (see ``check``).
    """
    minimum = policy()["constraints"].get("min_credit_score")
    if minimum is None or "bureau_score" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    score = _numeric(df, "bureau_score", np.nan)
    scored = (score >= 0) & ~_boolean(df, "new_to_credit")
    return scored & (score < float(minimum))


def no_offer(df: pd.DataFrame, grid: np.ndarray | None = None) -> np.ndarray:
    """True where there is no standard offer: a declined grade, a bureau score under the
    policy minimum, or a grid rate above the ceiling."""
    grid = grid_rate(df) if grid is None else np.asarray(grid, dtype=float)
    return (declined_grade(df) | bureau_below_minimum(df)
            | (grid > policy()["reference"]["card_rate_ceiling"] + 1e-9))


def no_offer_reason(df: pd.DataFrame) -> str | None:
    """Why the first row has no standard offer, in words, or None when it has one."""
    row = df.iloc[[0]]
    pol = policy()
    if bureau_below_minimum(row)[0]:
        return (f"bureau score {int(_numeric(row, 'bureau_score', 0)[0])} is below the policy minimum of "
                f"{pol['constraints']['min_credit_score']}: declined under standard policy")
    if declined_grade(row)[0]:
        return (f"risk grade {row['risk_grade'].iloc[0]} has no offer under standard policy "
                f"({pol['policy_version']}); the file is declined, not priced")
    grid = float(grid_rate(row)[0])
    ceiling = pol["reference"]["card_rate_ceiling"]
    if grid > ceiling + 1e-9:
        return (f"the grid prices this file at {grid:.2f}%, above the {ceiling:.2f}% ceiling: no standard "
                f"offer at this tenure - shorten the tenure or decline, never cap at the ceiling")
    return None


def _tenure_bps(df: pd.DataFrame, bands: list[dict]) -> np.ndarray:
    """Tenure adjustment from ``tenure_loading_bps``: the pricing matrix's column.

    Booked tenure when there is one, else the requested tenure. A file with
    neither carries no adjustment (the shortest column) rather than a guess.
    """
    if not bands:
        return np.zeros(len(df))
    years = _numeric(df, "tenure_years", np.nan)
    years = np.where(np.isnan(years), _numeric(df, "requested_tenure_years", np.nan), years)
    months = years * 12
    ordered = sorted(bands, key=lambda b: b["max_months"])
    out = np.full(len(df), float(ordered[-1]["bps"]))
    for band in reversed(ordered):
        out = np.where(months <= band["max_months"], float(band["bps"]), out)
    return np.where(np.isnan(months), 0.0, out)


def _default_tenure_years() -> float:
    """Assumed tenure for a file that states none: the product maximum."""
    return float(policy()["constraints"]["max_tenure_years"])


def _map(df: pd.DataFrame, column: str, table: dict, default: float) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), float(default))
    return df[column].map(table).astype(float).fillna(default).to_numpy()


def _numeric(df: pd.DataFrame, column: str, default: float) -> np.ndarray:
    return numeric_column(df, column, default).to_numpy(dtype=float)


def _boolean(df: pd.DataFrame, column: str, default: bool = False) -> np.ndarray:
    return boolean_column(df, column, default).to_numpy()


# ------------------------------------------------------------ affordability cap


def grade_foir_cap(df: pd.DataFrame) -> np.ndarray:
    """Each file's FOIR ceiling: its grade's ``eligibility_limits.foir_max``.

    A file with no grade yet, or a grade the limits do not name, gets the
    product-wide ``constraints.max_foir``.
    """
    pol = policy()
    default = float(pol["constraints"]["max_foir"])
    caps = {g: float(v["foir_max"]) for g, v in (pol.get("eligibility_limits") or {}).items() if v}
    if "risk_grade" not in df.columns:
        return np.full(len(df), default)
    return df["risk_grade"].astype(str).map(caps).astype(float).fillna(default).to_numpy()


def stress_add_pct() -> float:
    """Affordability is tested at the quoted rate plus this (``pricing.stress_test_add_pct``)."""
    return float((policy().get("pricing") or {}).get("stress_test_add_pct", 0.0))


def max_affordable_rate(df: pd.DataFrame, foir_cap: float | None = None) -> np.ndarray:
    """Highest quoted rate at which the file still passes the affordability test.

    The test is the policy's: the EMI at the quoted rate *plus the stress add-on*,
    with existing obligations, must stay within the grade's FOIR ceiling. Raising the
    price raises the EMI, which can push a thin file past it. Bisection keeps this
    exact without inverting the EMI formula analytically.
    """
    pol = policy()
    cap = grade_foir_cap(df) if foir_cap is None else np.full(len(df), float(foir_cap))
    stress = stress_add_pct()
    principal = _numeric(df, "sanctioned_loan_amount", np.nan)
    principal = np.where(np.isnan(principal), _numeric(df, "requested_loan_amount", 0.0), principal)
    income = _numeric(df, "verified_monthly_income", np.nan)
    income = np.where(np.isnan(income), _numeric(df, "declared_monthly_income", 0.0), income)
    obligations = _numeric(df, "declared_obligations", 0.0)
    tenure = _numeric(df, "tenure_years", np.nan)
    tenure = np.where(np.isnan(tenure), _numeric(df, "requested_tenure_years", _default_tenure_years()), tenure)

    lo = np.full(len(df), pol["reference"]["card_rate_floor"])
    hi = np.full(len(df), pol["reference"]["card_rate_ceiling"])
    for _ in range(40):
        mid = (lo + hi) / 2
        foir = (_emi(principal, mid + stress, tenure) + obligations) / np.maximum(income, 1.0)
        affordable = foir <= cap
        lo = np.where(affordable, mid, lo)
        hi = np.where(affordable, hi, mid)
    # The search starts at the floor, so it ends on the floor both when the floor is the
    # highest affordable rate and when no rate is affordable at all. Tell them apart: a
    # file that fails even at the floor has no affordable rate (NaN) and must be resized.
    # Without this, a grade priced at the floor (A+, 36-60 months) passed unaffordable.
    floor = pol["reference"]["card_rate_floor"]
    at_floor = (_emi(principal, np.full(len(df), floor + stress), tenure) + obligations) \
        / np.maximum(income, 1.0) <= cap
    return np.where(at_floor, np.round(lo, 2), np.nan)


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

    A file whose grade has no offer (``no_offer``) has no band: every rate is NaN.
    """
    pol = policy()
    card = card_rate(df) if card is None else np.asarray(card, dtype=float)
    max_discount = pol["constraints"]["absolute_discount_floor_bps"]
    max_premium = max(level["max_premium_bps"] for level in pol["approval_authority"])

    lower = np.maximum(card - max_discount / 100.0, pol["reference"]["card_rate_floor"])
    policy_upper = np.minimum(card + max_premium / 100.0, pol["reference"]["card_rate_ceiling"])
    affordable = max_affordable_rate(df)
    unaffordable = np.isnan(affordable)                   # no rate in the envelope passes
    resize_required = unaffordable | (affordable < lower)
    upper = np.where(resize_required, policy_upper, np.minimum(policy_upper, affordable))

    return pd.DataFrame({
        "card_rate": card,
        "min_rate": np.round(lower, 2),
        "max_rate": np.round(upper, 2),
        "max_affordable_rate": affordable,
        "affordability_binding": unaffordable | (affordable < policy_upper),
        "resize_required": resize_required,
        "infeasible": resize_required,
        "no_offer": np.isnan(card),
    }, index=df.index)


def resize_terms(application: pd.DataFrame, rate: float) -> dict:
    """What has to change for this file to pass the affordability test at ``rate``.

    Two levers a credit officer actually has: a smaller ticket, or a longer tenure
    within the product maximum. Both are computed at the stressed rate (``rate``
    plus the stress add-on) against the grade's FOIR ceiling, since that is the
    test the file must pass. Amounts round *down* to the nearest Rs 1,000:
    rounding up would hand back a ticket that breaches again.
    """
    pol = policy()
    row = application.iloc[[0]] if isinstance(application, pd.DataFrame) else pd.DataFrame([application])
    cap = float(grade_foir_cap(row)[0])
    stressed = float(rate) + stress_add_pct()
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
        tenure = float(_numeric(row, "requested_tenure_years", _default_tenure_years())[0])

    def emi(amount: float, years: float) -> float:
        return float(_emi(np.array([amount]), np.array([stressed]), np.array([years]))[0])

    headroom = cap * income - obligations
    terms = {
        "rate": round(float(rate), 2),
        "stressed_rate": round(stressed, 2),
        "foir_cap": cap,
        "foir_at_rate": round((emi(principal, tenure) + obligations) / max(income, 1.0), 4),  # at the stressed rate
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
    text = (f"resize required: at {terms['rate']:.2f}% (tested at {terms['stressed_rate']:.2f}%) the FOIR is "
            f"{terms['foir_at_rate']:.0%} against the grade's {terms['foir_cap']:.0%} cap - reduce the loan "
            f"to Rs {terms['max_affordable_amount']:,.0f}")
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

    if bool(band_row["no_offer"]):
        violations.append(no_offer_reason(row) or "no standard offer under policy")
    if rate < pol["reference"]["card_rate_floor"]:
        violations.append(f"rate {rate:.2f}% is below the absolute floor {pol['reference']['card_rate_floor']:.2f}%")
    if rate > pol["reference"]["card_rate_ceiling"]:
        violations.append(f"rate {rate:.2f}% is above the product ceiling {pol['reference']['card_rate_ceiling']:.2f}%")
    if discount_bps > pol["constraints"]["absolute_discount_floor_bps"]:
        violations.append(
            f"discount of {discount_bps:.0f} bps exceeds the maximum permissible "
            f"{pol['constraints']['absolute_discount_floor_bps']} bps from card rate"
        )
    max_premium = max(level["max_premium_bps"] for level in pol["approval_authority"])
    if -discount_bps > max_premium + 1e-9:
        violations.append(
            f"rate {rate:.2f}% is {-discount_bps:.0f} bps above the card rate {card:.2f}% for this grade and "
            f"tenure; the most any authority may add is {max_premium} bps"
            + (" - the matrix rate is the ceiling" if max_premium == 0 else "")
        )
    conditions: list[str] = []
    if _boolean(row, "new_to_credit")[0] and not pol["constraints"].get("new_to_credit_allowed", True):
        # No thin-file model is approved: the file is not declined, it is reviewed by hand
        # (mandatory_rules.bureau.no_score: separate_model_or_manual).
        conditions.append("new to credit: no approved thin-file model, so a credit officer reviews the file "
                          "by hand before the price is released")
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
            f"rate {rate:.2f}% breaches the grade's FOIR cap of {grade_foir_cap(row)[0]:.0%} when tested at "
            f"{rate + stress_add_pct():.2f}% (maximum affordable rate {band_row['max_affordable_rate']:.2f}%)"
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
