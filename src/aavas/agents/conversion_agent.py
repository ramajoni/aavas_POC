"""Conversion agent - the customer's price response.

Answers the question the business statement says is currently unanswerable:
how much conversion does this *particular* customer cost us per basis point,
rather than applying one commercial treatment to everyone.
"""

from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger
from ..xai import counterfactual
from .underwriting_agent import CustomerDigitalTwin

log = get_logger(__name__)

# Elasticity bands used to describe a customer in words on the sales screen.
SENSITIVITY_BANDS = [(2.0, "highly price sensitive"), (1.0, "price sensitive"),
                     (0.4, "moderately price sensitive"), (0.0, "relatively price insensitive")]


class ConversionAgent:
    def __init__(self, conversion_model):
        self.model = conversion_model

    def predict(self, twin: CustomerDigitalTwin, rate: float) -> float:
        if self.model is None:
            return float("nan")
        return float(self.model.predict_at_rate(twin.frame, rate)[0])

    def profile(self, twin: CustomerDigitalTwin, centre_rate: float) -> dict:
        """Elasticity, the negotiation table, and the price that still converts."""
        if self.model is None:
            twin.note("conversion model unavailable; price response unknown")
            return {"available": False}

        # Search only inside the permissible band: a rate the policy would never
        # allow is not an answer to "what rate still converts", and scanning the
        # whole product envelope at 1 bp costs more than the rest of the profile.
        envelope = twin.envelope or {}
        band = None
        if envelope.get("min_rate") is not None and envelope.get("max_rate") is not None:
            band = (float(envelope["min_rate"]), float(envelope["max_rate"]))

        elasticity = self.model.elasticity(twin.frame, centre_rate, bump_bps=25)
        per_100 = elasticity * 4
        label = next(text for threshold, text in SENSITIVITY_BANDS if per_100 >= threshold)

        table = counterfactual.price_sensitivity_table(twin.frame, self.model, centre_rate)
        profile = {
            "available": True,
            "conversion_at_recommended": round(self.predict(twin, centre_rate), 4),
            "points_lost_per_25bps": elasticity,
            "points_lost_per_100bps": round(per_100, 2),
            "sensitivity_label": label,
            "rate_for_70pct_conversion": counterfactual.rate_for_conversion(
                twin.frame, self.model, 0.70, band=band),
            "rate_for_50pct_conversion": counterfactual.rate_for_conversion(
                twin.frame, self.model, 0.50, band=band),
            "search_band": {"min_rate": band[0], "max_rate": band[1]} if band else None,
            "negotiation_table": table.to_dict(orient="records"),
        }
        return profile
