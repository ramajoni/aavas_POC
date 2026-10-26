"""Pricing agent - scenario simulation and the yield-optimal permissible price.

This is the step the business statement is really asking for: not "predict the
rate" but "given what this file costs us, how this customer responds to price,
and how long the loan is likely to live, which permissible price creates the most
risk-adjusted value?".

The Stage-2 model contributes an anchor and an uncertainty interval. The decision
itself comes from the simulated scenarios.
"""

from __future__ import annotations

import numpy as np

from ..ingestion.schemas import RateBand
from ..logging_utils import get_logger
from ..optimization import price_optimizer, scenario_generator
from .underwriting_agent import CustomerDigitalTwin

log = get_logger(__name__)


class PricingAgent:
    def __init__(self, stage2_model, conversion_model, prepayment_model, override_model=None):
        self.stage2 = stage2_model
        self.conversion = conversion_model
        self.prepayment = prepayment_model
        self.override = override_model

    def recommend(self, twin: CustomerDigitalTwin, expected_loss_pct: float,
                  quoted_band: RateBand | None = None) -> dict:
        """``quoted_band`` is the indicative range the customer was given at Stage 1;
        the search stays inside it (see ``optimizer.honour_indicative_band``)."""
        anchor = self._anchor(twin)
        anticipated = self._anticipated_human_rate(twin, anchor["anchor_rate"])

        scenarios = scenario_generator.generate(
            twin.frame,
            stage2_band=RateBand(**{k: anchor["search_band"][k] for k in ("min_rate", "max_rate")}),
            competitor_rate=twin.get("competitor_rate"),
            anticipated_human_rate=anticipated,
            support_band=self._support_band(twin),
            quoted_band=quoted_band,
        )

        scored = price_optimizer.evaluate_scenarios(
            application=twin.frame,
            rates=scenarios.rates,
            conversion_model=self.conversion,
            prepayment_model=self.prepayment,
            expected_loss_pct=expected_loss_pct,
            loan_amount=twin.loan_amount,
            tenure_years=twin.tenure,
        )
        result = price_optimizer.optimize(scored, anchors=scenarios.anchors)

        payload = result.as_dict()
        payload["stage2_anchor"] = anchor
        payload["scenarios"] = scenarios.as_dict()
        payload["scenario_table"] = scored.round(4).to_dict(orient="records")
        payload["rate_vs_anchor_bps"] = round((result.recommended_rate - anchor["anchor_rate"]) * 100, 1)
        return payload

    # ------------------------------------------------------------------ helpers
    def _support_band(self, twin: CustomerDigitalTwin) -> tuple[float, float] | None:
        if self.conversion is None or not hasattr(self.conversion, "supported_rate_band"):
            return None
        band = self.conversion.supported_rate_band(twin.frame)
        if band is None:
            twin.note("conversion model has no recorded support range; searching the full band")
        return band

    def _anchor(self, twin: CustomerDigitalTwin) -> dict:
        if self.stage2 is None:
            envelope = twin.envelope or {}
            card = envelope.get("card_rate", 11.5)
            twin.note("stage2 model unavailable; anchoring the search on the policy card rate")
            return {"anchor_rate": card, "search_band": {"min_rate": card - 0.5, "max_rate": card + 0.5},
                    "residual_std_bps": None, "source": "card_rate"}
        anchor = self.stage2.predict(twin.frame)
        anchor["source"] = "stage2_model"
        return anchor

    def _anticipated_human_rate(self, twin: CustomerDigitalTwin, anchor_rate: float) -> float | None:
        if self.override is None:
            return None
        prediction = self.override.predict(twin.frame, recommended_rate=anchor_rate).iloc[0]
        if prediction["override_probability"] < 0.5:
            return None
        return float(prediction["anticipated_human_rate"])
