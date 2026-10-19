"""Dataset contract and the canonical lifecycle vocabulary.

One ``loan_id`` threads the whole journey, which is what makes every stage
comparable:

    enquiry -> stage1 band -> assessment -> stage2 recommendation
            -> human decision -> sanction -> disbursal -> behaviour -> yield
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --------------------------------------------------------------------- lifecycle


class Stage(str, Enum):
    PRE_ASSESSMENT = "pre_assessment"
    POST_ASSESSMENT = "post_assessment"
    HUMAN_DECISION = "human_decision"
    OUTCOME = "outcome"


class CustomerResponse(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEGOTIATED = "negotiated"
    DEFERRED = "deferred"


LIFECYCLE_ORDER = [
    "policy_card_rate",      # what the grid allows
    "ai_recommended_rate",   # what the engine recommended
    "sales_offered_rate",    # what the human offered
    "sanctioned_rate",       # what credit sanctioned
    "realized_yield_pct",    # what the book actually earned
]

# ------------------------------------------------------------------- columns

IDENTITY_COLUMNS = ["loan_id", "application_date", "state", "city", "branch_id", "sales_officer_id"]

PRE_ASSESSMENT_COLUMNS = [
    "segment", "channel", "property_type", "applicant_age", "declared_monthly_income",
    "new_to_credit", "bureau_score", "declared_obligations", "requested_loan_amount",
    "estimated_property_value", "estimated_ltv", "requested_tenure_years",
]

POST_ASSESSMENT_COLUMNS = [
    "verified_monthly_income", "income_verification_haircut", "bureau_vintage_months",
    "delinquency_count_24m", "banking_behaviour_score", "appraised_property_value",
    "tenure_years", "sanctioned_loan_amount", "ltv", "deviation_count", "legal_flag",
    "technical_flag", "foir", "risk_grade",
]

SALES_CONTEXT_COLUMNS = [
    "competitor_rate", "competitor_name", "competitor_evidence_verified", "days_to_month_end",
    "month_end", "sales_target_pct", "branch_target_pct", "existing_relationship",
    "prior_negotiation_rounds", "customer_negotiating",
]

DECISION_COLUMNS = [
    "policy_card_rate", "ai_recommended_rate", "override_bps", "override_reason",
    "sales_offered_rate", "sanctioned_rate", "credit_clawback",
]

OUTCOME_COLUMNS = [
    "customer_response", "disbursed", "bt_out_24m", "delinquent_12m", "defaulted",
    "months_to_exit", "realized_life_years", "realized_yield_pct", "realized_value_pct",
    "realized_value_inr",
]

# Ground truth that only exists because the data is synthetic. Never a feature.
LATENT_COLUMNS = [
    "true_pd", "true_lgd", "price_elasticity", "true_conversion_probability", "true_bt_probability",
]

ALL_COLUMNS = (
    IDENTITY_COLUMNS + PRE_ASSESSMENT_COLUMNS + POST_ASSESSMENT_COLUMNS
    + SALES_CONTEXT_COLUMNS + DECISION_COLUMNS + OUTCOME_COLUMNS + LATENT_COLUMNS
    + ["stage1_min_rate", "stage1_max_rate", "sanctioned"]
)

BOOL_COLUMNS = [
    "new_to_credit", "competitor_evidence_verified", "month_end", "existing_relationship",
    "customer_negotiating", "disbursed", "bt_out_24m", "delinquent_12m", "defaulted",
    "credit_clawback", "sanctioned",
]

# ------------------------------------------------------------------- targets

TARGETS = {
    "stage1": "sanctioned_rate",      # quantile band over the price the file finally clears at
    "stage2": "sanctioned_rate",
    "conversion": "disbursed",
    "prepayment": "bt_out_24m",
    "risk": "defaulted",
    "override_flag": "was_overridden",
    "override_magnitude": "override_bps",
}

# The shorthand column list used in the original solution outline, mapped onto the
# canonical names above. Kept so external specs and this codebase stay traceable.
OUTLINE_FEATURE_ALIASES = {
    "monthly_income": "declared_monthly_income",
    "loan_amount": "requested_loan_amount",
    "credit_score": "bureau_score",
    "verified_income": "verified_monthly_income",
    "final_risk": "risk_grade",
    "stage2_ai_rate": "ai_recommended_rate",
    "sales_final_rate": "sanctioned_rate",
    "realized_yield": "realized_yield_pct",
}


@dataclass(frozen=True)
class RateBand:
    """A permissible price interval. Used everywhere a single rate would be a lie."""

    min_rate: float
    max_rate: float

    def __post_init__(self) -> None:
        if self.min_rate > self.max_rate:
            raise ValueError(f"invalid band: {self.min_rate} > {self.max_rate}")

    @property
    def width_bps(self) -> float:
        return round((self.max_rate - self.min_rate) * 100, 1)

    def contains(self, rate: float, tol: float = 1e-9) -> bool:
        return self.min_rate - tol <= rate <= self.max_rate + tol

    def clamp(self, rate: float) -> float:
        return min(max(rate, self.min_rate), self.max_rate)

    def as_dict(self) -> dict[str, float]:
        return {"min_rate": round(self.min_rate, 2), "max_rate": round(self.max_rate, 2),
                "width_bps": self.width_bps}
