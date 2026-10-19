"""Request and response contracts for the LOS / Salesforce integration.

Stage-1 asks for what a branch genuinely has at enquiry. Stage-2 asks for the
assessment outcome. Anything optional has a defensible default, because a field
the branch cannot fill must not block a quote.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .examples import FINAL_ASSESSMENT, OUTCOME, OVERRIDE, PRE_ASSESSMENT

Segment = Literal["salaried_formal", "salaried_informal",
                  "self_employed_professional", "self_employed_non_professional"]
Channel = Literal["branch_walkin", "dsa", "connector", "digital"]
PropertyType = Literal["resale_flat", "builder_flat", "self_construction", "plot_plus_construction"]


class SalesContext(BaseModel):
    competitor_rate: float | None = Field(None, description="Rate the customer says a competitor offered")
    competitor_name: str | None = None
    competitor_evidence_verified: bool = Field(
        False, description="Has a competitor sanction letter or written quote been seen?")
    sales_target_pct: float = Field(100.0, ge=0, le=300)
    branch_target_pct: float = Field(100.0, ge=0, le=300)
    customer_negotiating: bool = False
    prior_negotiation_rounds: int = Field(0, ge=0)
    existing_relationship: bool = False
    month_end: bool | None = None
    days_to_month_end: int | None = Field(None, ge=0, le=31)


class Stage1Request(BaseModel):
    """Pre-assessment: self-declared, unverified, property possibly unidentified."""

    model_config = ConfigDict(json_schema_extra={"examples": [PRE_ASSESSMENT]})

    loan_id: str
    segment: Segment
    city: str
    channel: Channel = "branch_walkin"
    property_type: PropertyType = "resale_flat"
    declared_monthly_income: float = Field(..., gt=0)
    requested_loan_amount: float = Field(..., gt=0)
    requested_tenure_years: int = Field(20, ge=1, le=30)
    applicant_age: int = Field(35, ge=18, le=75)
    declared_obligations: float = Field(0.0, ge=0)
    estimated_property_value: float | None = Field(None, gt=0)
    bureau_score: int | None = Field(None, description="Soft-pull score: 300–900; omit or use -1 for new to credit", examples=[750])
    new_to_credit: bool | None = None
    sales_context: SalesContext = Field(default_factory=SalesContext)

    @field_validator("bureau_score")
    @classmethod
    def _score_range(cls, value: int | None) -> int | None:
        if value is not None and value != -1 and not (300 <= value <= 900):
            raise ValueError("bureau_score must be between 300 and 900, or -1 for new to credit")
        return value

    def to_row(self) -> dict[str, Any]:
        row = self.model_dump(exclude={"sales_context"})
        row.update(self.sales_context.model_dump())
        if row.get("bureau_score") is None:
            row["bureau_score"] = -1
        if row.get("new_to_credit") is None:
            row["new_to_credit"] = row["bureau_score"] < 0
        if row.get("estimated_property_value") is None:
            row["estimated_property_value"] = row["requested_loan_amount"] / 0.70
        row["estimated_ltv"] = row["requested_loan_amount"] / row["estimated_property_value"]
        return row


class Stage2Request(Stage1Request):
    """Post-assessment: verified income, valuation, bureau depth, legal, technical."""

    model_config = ConfigDict(json_schema_extra={"examples": [FINAL_ASSESSMENT]})

    verified_monthly_income: float = Field(..., gt=0)
    appraised_property_value: float = Field(..., gt=0)
    sanctioned_loan_amount: float = Field(..., gt=0)
    tenure_years: int = Field(..., ge=1, le=30)
    bureau_vintage_months: int = Field(0, ge=0)
    delinquency_count_24m: int = Field(0, ge=0)
    banking_behaviour_score: float = Field(50.0, ge=0, le=100)
    deviation_count: int = Field(0, ge=0)
    legal_flag: int = Field(0, ge=0, le=1)
    technical_flag: int = Field(0, ge=0, le=1)
    risk_grade: str | None = Field(None, description="Optional; the risk agent re-grades the file anyway")

    def to_row(self) -> dict[str, Any]:
        row = super().to_row()
        row["ltv"] = row["sanctioned_loan_amount"] / row["appraised_property_value"]
        row["income_verification_haircut"] = max(
            0.0, (row["declared_monthly_income"] - row["verified_monthly_income"])
            / max(row["declared_monthly_income"], 1.0))
        return row


class OverrideRequest(BaseModel):
    """A human decision against the recommendation. The reason is mandatory."""

    model_config = ConfigDict(json_schema_extra={"examples": [OVERRIDE]})

    loan_id: str
    human_rate: float = Field(..., gt=0, lt=30)
    reason: str
    reason_detail: str | None = None
    approved_by: str | None = Field(
        None, description="'name:level', e.g. 'r.sharma:branch_manager'")
    evidence_verified: bool = False


class OutcomeRequest(BaseModel):
    """What actually happened. Fed from the LOS at sanction and the LMS later."""

    model_config = ConfigDict(json_schema_extra={"examples": [OUTCOME]})

    loan_id: str
    disbursed: bool
    customer_response: Literal["accepted", "rejected", "negotiated", "deferred"] | None = None
    sanctioned_rate: float | None = Field(None, gt=0, lt=30)
    bt_out: bool = False
    delinquent: bool = False
    months_to_exit: int | None = Field(None, ge=0)
    realized_yield_pct: float | None = None
    # The join the learning loop needs. Optional: an outcome can arrive from an
    # LMS feed that never saw a decision id, and the record says which happened.
    decision_id: str | None = Field(
        None, description="The decision this outcome followed. Omit and the most recent "
                          "final assessment for the file is used, recorded as inferred.")
    # Why the file was lost. "rejected" says nothing an underwriter can act on.
    loss_reason: str | None = Field(
        None, description="One of the reasons listed by GET /pricing/outcome/loss-reasons")
    lost_to: str | None = Field(None, description="Which lender won the file, if known")
    lost_to_rate: float | None = Field(None, gt=0, lt=30,
                                       description="The rate they won it at, if known")


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, str]
    models_missing: list[str]
    versions: dict[str, str]
    decisions_recorded: int
