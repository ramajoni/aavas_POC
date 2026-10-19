"""Pre-assessment agent - the early, credible answer.

The customer wants a number before anyone has verified anything. The honest
answer is a band, an eligibility range and a confidence level, clamped into
policy so the branch never quotes a rate the file cannot be sanctioned at.
"""

from __future__ import annotations

import pandas as pd

from ..config import policy
from ..ingestion.schemas import Stage
from ..logging_utils import get_logger
from ..optimization import constraints
from .underwriting_agent import CustomerDigitalTwin, UnderwritingAgent

log = get_logger(__name__)


class PreAssessmentAgent:
    def __init__(self, stage1_model, underwriting_agent: UnderwritingAgent | None = None):
        self.model = stage1_model
        self.underwriting = underwriting_agent or UnderwritingAgent()

    def quote(self, application: dict | pd.DataFrame) -> dict:
        twin = self.underwriting.build(application, stage=Stage.PRE_ASSESSMENT)
        if self.model is None:
            raise RuntimeError("stage1 model artefact is missing; run scripts/train_all.py")

        quote = self.model.predict(twin.frame)
        band = quote["band"]

        # Clamp the customer-facing band into what policy could actually sanction.
        # Quoting outside it would be a promise the underwriter has to break.
        pol = policy()
        floor = pol["reference"]["card_rate_floor"]
        ceiling = pol["reference"]["card_rate_ceiling"]
        clamped_min = max(band["min_rate"], floor)
        clamped_max = min(band["max_rate"], ceiling)
        if (clamped_min, clamped_max) != (band["min_rate"], band["max_rate"]):
            twin.note("indicative band clamped to the product rate envelope")

        confidence = quote["confidence"]
        caveats = self._caveats(twin, quote)

        return {
            "loan_id": twin.loan_id,
            "stage": Stage.PRE_ASSESSMENT.value,
            "band": {"min_rate": round(clamped_min, 2), "max_rate": round(clamped_max, 2),
                     "width_bps": round((clamped_max - clamped_min) * 100, 1)},
            "indicative_rate": quote["indicative_rate"],
            "confidence": confidence,
            "confidence_label": _confidence_label(confidence),
            "eligible_amount": quote["eligible_amount"],
            "caveats": caveats,
            "completeness": twin.completeness,
            "notes": twin.notes,
            "model_versions": {"stage1": quote.get("model_version")},
            "disclaimer": (
                "Indicative only. The final rate is determined after income verification, "
                "property valuation, and legal and technical assessment."
            ),
        }

    def _caveats(self, twin: CustomerDigitalTwin, quote: dict) -> list[str]:
        caveats: list[str] = []
        if twin.get("new_to_credit", False):
            caveats.append("No bureau history yet; the final rate depends on the full credit assessment.")
        if twin.get("competitor_rate") is None:
            caveats.append("No competitor quote on record; the band does not reflect any matching offer.")
        if quote["confidence"] < 0.5:
            caveats.append("Limited information on file; treat this band as a wide early indication.")
        haircut_risk = twin.get("segment") in ("self_employed_non_professional", "salaried_informal")
        if haircut_risk:
            caveats.append(
                "Income is self-declared. For this profile, verified income is typically lower, "
                "which can move both the rate and the eligible amount."
            )
        return caveats


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "moderate"
    return "low"
