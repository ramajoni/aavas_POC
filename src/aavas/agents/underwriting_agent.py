"""Underwriting agent - builds the customer digital twin.

The twin is the consolidated, single-row view of everything currently known
about one application: declared and verified figures, property, bureau, sales
context, and which of those are actually present. Every other agent reads and
enriches this one object, so no agent has to re-derive the file and no two
agents can disagree about what the file says.

It also decides which stage the file is genuinely at. That is a data question,
not a workflow flag: a file whose income is unverified and whose property is
unvisited is a Stage-1 file no matter what the calling system claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..features import sales_features, stage2_features
from ..ingestion.schemas import Stage
from ..logging_utils import get_logger

log = get_logger(__name__)

# Evidence that the assessment has actually happened.
POST_ASSESSMENT_EVIDENCE = [
    "verified_monthly_income", "appraised_property_value", "ltv", "banking_behaviour_score",
    "bureau_vintage_months", "legal_flag", "technical_flag", "deviation_count",
]


@dataclass
class CustomerDigitalTwin:
    """Consolidated state of one application as it moves through the journey."""

    loan_id: str
    frame: pd.DataFrame
    stage: Stage
    completeness: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] | None = None
    sales_context: dict[str, Any] | None = None
    envelope: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ access
    def get(self, column: str, default: Any = None) -> Any:
        if column not in self.frame.columns:
            return default
        value = self.frame[column].iloc[0]
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        return default if (value is None or (isinstance(value, float) and np.isnan(value))) else value

    def note(self, message: str) -> "CustomerDigitalTwin":
        """Record an observation once. Several agents run more than once per
        decision (the sales agent re-reads context after the price is chosen),
        and a decision record that repeats itself is harder to review."""
        if message not in self.notes:
            self.notes.append(message)
        return self

    def set(self, **columns: Any) -> "CustomerDigitalTwin":
        for key, value in columns.items():
            self.frame[key] = value
        return self

    @property
    def loan_amount(self) -> float:
        return float(self.get("sanctioned_loan_amount") or self.get("requested_loan_amount") or 0.0)

    @property
    def tenure(self) -> float:
        return float(self.get("tenure_years") or self.get("requested_tenure_years") or 20.0)

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe copy of the file exactly as it was priced.

        Stored on the decision so a later override check, a re-run, or an audit
        works from the real application rather than a reconstruction of it.
        """
        record = self.frame.iloc[0].to_dict()
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            elif isinstance(value, np.bool_):
                value = bool(value)
            elif isinstance(value, (pd.Timestamp,)):
                value = value.isoformat()
            if isinstance(value, float) and np.isnan(value):
                value = None
            clean[key] = value
        return clean

    def summary(self) -> dict[str, Any]:
        return {
            "loan_id": self.loan_id,
            "stage": self.stage.value,
            "segment": self.get("segment"),
            "city": self.get("city"),
            "loan_amount": self.loan_amount,
            "ltv": self.get("ltv") or self.get("estimated_ltv"),
            "tenure_years": self.tenure,
            "completeness": self.completeness,
            "risk": self.risk,
            "sales_context": self.sales_context,
            "notes": self.notes,
        }


class UnderwritingAgent:
    """Assembles the twin and reports what is still unknown."""

    def build(self, application: dict | pd.Series | pd.DataFrame,
              stage: Stage | None = None) -> CustomerDigitalTwin:
        frame = _as_frame(application)
        loan_id = str(frame.get("loan_id", pd.Series(["UNKNOWN"])).iloc[0])

        present = [c for c in POST_ASSESSMENT_EVIDENCE if c in frame.columns and frame[c].notna().all()]
        ratio = len(present) / len(POST_ASSESSMENT_EVIDENCE)
        inferred = Stage.POST_ASSESSMENT if ratio >= 0.75 else Stage.PRE_ASSESSMENT
        resolved = stage or inferred

        notes: list[str] = []
        if stage is not None and stage != inferred:
            notes.append(
                f"caller requested {stage.value} but only {len(present)}/{len(POST_ASSESSMENT_EVIDENCE)} "
                f"post-assessment fields are present"
            )

        # Enrich in the order the file is actually built up.
        if resolved == Stage.POST_ASSESSMENT:
            frame = stage2_features.build(frame)
        frame = sales_features.build(frame, reference_rate_col="policy_card_rate"
                                     if "policy_card_rate" in frame.columns else "ai_recommended_rate")

        haircut = frame.get("income_verification_haircut")
        if haircut is not None and float(pd.to_numeric(haircut, errors="coerce").fillna(0).iloc[0]) > 0.30:
            notes.append("verified income is more than 30% below declared income")
        if float(pd.to_numeric(frame.get("deviation_count", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) >= 3:
            notes.append("three or more credit deviations on file")
        for flag, message in (("legal_flag", "legal exception on the property"),
                              ("technical_flag", "technical exception on the property")):
            if flag in frame.columns and float(pd.to_numeric(frame[flag], errors="coerce").fillna(0).iloc[0]) > 0:
                notes.append(message)

        twin = CustomerDigitalTwin(
            loan_id=loan_id, frame=frame, stage=resolved,
            completeness={
                "post_assessment_fields_present": len(present),
                "post_assessment_fields_expected": len(POST_ASSESSMENT_EVIDENCE),
                "completeness_ratio": round(ratio, 2),
                "missing": [c for c in POST_ASSESSMENT_EVIDENCE if c not in present],
                "inferred_stage": inferred.value,
            },
            notes=notes,
        )
        return twin


def _as_frame(application) -> pd.DataFrame:
    if isinstance(application, pd.DataFrame):
        return application.head(1).copy().reset_index(drop=True)
    if isinstance(application, pd.Series):
        return application.to_frame().T.reset_index(drop=True)
    return pd.DataFrame([application])
