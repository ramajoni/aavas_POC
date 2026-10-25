"""Risk agent - turns the file into PD, grade and expected loss in money terms.

The grade it assigns is the grade the policy card rate is then built from, so
this agent sits upstream of pricing rather than beside it.
"""

from __future__ import annotations

from ..logging_utils import get_logger
from ..optimization import yield_calculator
from .underwriting_agent import CustomerDigitalTwin

log = get_logger(__name__)


class RiskAgent:
    def __init__(self, risk_model):
        self.model = risk_model

    def assess(self, twin: CustomerDigitalTwin) -> dict:
        if self.model is None:
            # No risk artefact: fall back to the grade on the file, if any, and
            # say so rather than silently pricing at zero expected loss.
            grade = twin.get("risk_grade", "C")
            assessment = {"pd": None, "lgd": None, "risk_grade": grade,
                          "expected_loss_pct": 0.0, "source": "unavailable"}
            twin.note("risk model unavailable; using the grade supplied on the file")
            twin.risk = assessment
            return assessment

        row = self.model.assess(twin.frame).iloc[0]
        assessment = {
            "pd": float(row["pd"]),
            "lifetime_pd": float(row["lifetime_pd"]),
            "lgd": float(row["lgd"]),
            "risk_grade": str(row["risk_grade"]),
            "expected_loss_pct": float(row["expected_loss_pct"]),
            "breakeven_rate": yield_calculator.breakeven_rate(float(row["expected_loss_pct"])),
            "source": "model",
        }
        # Which severity priced the file, and what the policy table would have said,
        # so the record can answer "why is the expected loss lower than the grid?".
        if "lgd_source" in row.index:
            assessment["lgd_source"] = str(row["lgd_source"])
        if "lgd_policy_table" in row.index:
            assessment["lgd_policy_table"] = float(row["lgd_policy_table"])
        # The model's grade, not the caller's, drives the policy card rate.
        supplied = twin.get("risk_grade")
        if supplied and supplied != assessment["risk_grade"]:
            twin.note(
                f"risk grade moved from {supplied} (supplied) to {assessment['risk_grade']} (model)")
        twin.set(risk_grade=assessment["risk_grade"])
        twin.risk = assessment
        return assessment
