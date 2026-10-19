"""Sales context agent - the commercial environment around the decision.

Competitor quotes, target pressure, negotiation history and how much of that is
actually evidenced. The agent deliberately separates *capturing* competitive
intelligence from *acting* on it: an unverified claim that a competitor offered
9.05% is recorded and shown, but it cannot move the price beyond the tolerance
the policy allows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import policy
from ..features import sales_features
from ..logging_utils import get_logger
from .underwriting_agent import CustomerDigitalTwin

log = get_logger(__name__)


class SalesContextAgent:
    def __init__(self, override_model=None, note_retriever=None):
        self.override_model = override_model
        self.note_retriever = note_retriever

    def build(self, twin: CustomerDigitalTwin, reference_rate: float | None = None) -> dict:
        pol = policy()
        frame = twin.frame
        if reference_rate is not None:
            frame = frame.copy()
            frame["ai_recommended_rate"] = reference_rate
            frame = sales_features.build(frame, reference_rate_col="ai_recommended_rate")
            twin.frame = frame

        competitor_rate = twin.get("competitor_rate")
        evidence = bool(twin.get("competitor_evidence_verified", False))
        context = {
            "competitor_rate": competitor_rate,
            "competitor_name": twin.get("competitor_name"),
            "competitor_evidence_verified": evidence,
            "competitor_gap_bps": twin.get("competitor_gap_bps", 0.0),
            "sales_target_pct": twin.get("sales_target_pct"),
            "branch_target_pct": twin.get("branch_target_pct"),
            "days_to_month_end": twin.get("days_to_month_end"),
            "month_end": bool(twin.get("month_end", False)),
            "customer_negotiating": bool(twin.get("customer_negotiating", False)),
            "existing_relationship": bool(twin.get("existing_relationship", False)),
            "prior_negotiation_rounds": twin.get("prior_negotiation_rounds", 0),
            "pressure_index": float(sales_features.pressure_index(twin.frame).iloc[0]),
        }

        if competitor_rate is not None and not evidence:
            limit = pol["override"]["max_unevidenced_discount_bps"]
            context["competitor_claim_status"] = "unverified"
            context["max_discount_on_this_claim_bps"] = limit
            twin.note(
                f"competitor quote of {competitor_rate:.2f}% is unverified; it informs the "
                f"recommendation but cannot justify more than {limit} bps of discount on its own"
            )
        elif competitor_rate is not None:
            context["competitor_claim_status"] = "evidenced"

        if self.override_model is not None and reference_rate is not None:
            prediction = self.override_model.predict(twin.frame, recommended_rate=reference_rate).iloc[0]
            context["override_risk"] = {
                "probability": float(prediction["override_probability"]),
                "expected_override_bps": float(prediction["expected_override_bps"]),
                "anticipated_human_rate": float(prediction["anticipated_human_rate"]),
            }
            if prediction["override_probability"] > 0.6:
                twin.note(
                    "this file sits in territory where humans usually renegotiate the price; "
                    "the recommendation should be defensible in a negotiation, not just optimal"
                )

        if self.note_retriever is not None:
            try:
                context["similar_situations"] = self.note_retriever.similar(twin, k=3)
            except Exception as exc:                        # noqa: BLE001
                log.warning("note retrieval failed: %s", exc)

        twin.sales_context = context
        return context
