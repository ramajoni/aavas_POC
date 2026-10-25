"""Explanation agent - why this price, in language a branch can use.

Three audiences, one decision:

* the sales officer needs the negotiation position and what the next 5 bps costs,
* the credit reviewer needs the drivers and how much came from commercial
  pressure rather than the customer's own risk,
* the customer is entitled to reason codes and to know what they could change.
"""

from __future__ import annotations

from ..config import policy
from ..features import stage2_features
from ..logging_utils import get_logger
from ..optimization import yield_calculator
from ..xai import counterfactual
from ..xai import reason_codes as rc
from ..xai.shap_explainer import PricingExplainer
from .underwriting_agent import CustomerDigitalTwin

log = get_logger(__name__)


class ExplanationAgent:
    def __init__(self, stage2_model, conversion_model=None):
        self.stage2 = stage2_model
        self.conversion = conversion_model
        self._explainer: PricingExplainer | None = None

    @property
    def explainer(self) -> PricingExplainer | None:
        if self.stage2 is None:
            return None
        if self._explainer is None:
            self._explainer = PricingExplainer(self.stage2.model, self.stage2.transformer)
        return self._explainer

    def explain(self, twin: CustomerDigitalTwin, recommendation: dict) -> dict:
        rate = recommendation["recommended_rate"]
        payload: dict = {
            "headline": None,
            "reason_codes": [],
            "attribution_method": None,
            "value_decomposition": recommendation.get("value_decomposition", {}),
            "marginal_analysis": recommendation.get("marginal_analysis", {}),
            "comparisons": recommendation.get("comparisons", {}),
        }

        explainer = self.explainer
        if explainer is not None:
            attributions = explainer.explain(twin.frame, build=stage2_features.build)
            codes = rc.build(attributions)
            payload["reason_codes"] = [c.as_dict() for c in codes]
            payload["attribution_method"] = explainer.method()
            payload["attribution_scope"] = (
                "Attributions explain the Stage-2 anchor rate - the model's view of the price a file "
                "like this clears at. The recommended rate additionally reflects conversion, expected "
                "loss and loan life, which the value decomposition below covers."
            )
            payload["commercial_vs_risk"] = rc.commercial_share(attributions)
            payload["headline"] = rc.summarise(codes, rate)

        if self.conversion is not None:
            payload["actionable_for_customer"] = counterfactual.actionable_improvements(
                twin.frame, self.stage2) if self.stage2 else []

        payload["negotiation_guidance"] = self._negotiation_guidance(twin, recommendation)
        payload["policy_position"] = self._policy_position(twin, rate)
        payload["fair_pricing_review"] = self._fair_pricing_review(payload)
        payload["customer_caveats"] = self._customer_caveats(twin, recommendation)
        payload["disclaimer"] = (
            "This rate is a recommendation produced with model assistance and reviewed by an "
            "authorised officer. It is subject to final credit approval and to the terms of the "
            "sanction letter. You may ask for the reasons behind it and request a human review."
        )
        return payload

    def _fair_pricing_review(self, payload: dict) -> dict:
        """Flag files where commercial pressure priced the customer UP.

        Commercial context normally moves a price down - a competitor bid, a
        branch chasing target. When the same block moves it up, the customer is
        paying for something that is not their own credit risk. That is a
        legitimate commercial decision in some cases and a fair-lending problem in
        others, and the difference is a human judgement, so it is surfaced for
        review rather than decided here.
        """
        share = payload.get("commercial_vs_risk") or {}
        uplift_bps = float(share.get("commercial_net_bps") or 0.0)
        threshold = policy()["fair_pricing"]["commercial_uplift_review_bps"]
        flagged = uplift_bps > threshold
        return {
            "flagged": flagged,
            "commercial_net_bps": round(uplift_bps, 1),
            "threshold_bps": threshold,
            "reason": (
                f"Commercial factors priced this customer {uplift_bps:+.0f} bps, above the "
                f"{threshold} bps review threshold. Confirm the uplift reflects a commercial "
                "decision and not the customer's inability to shop."
            ) if flagged else "Commercial factors did not price this customer up.",
        }

    def _customer_caveats(self, twin: CustomerDigitalTwin, recommendation: dict) -> list[str]:
        """What a customer must be told alongside the number.

        Stage 1 carries caveats because a pre-assessment band is uncertain. Stage 2
        needs its own: the price is conditional on the file not changing.
        """
        caveats = [
            "This rate is subject to final credit approval and to satisfactory legal and "
            "technical clearance of the property.",
            f"The quote is valid for {policy()['fair_pricing']['quote_validity_days']} days from "
            "the date of issue.",
        ]
        if twin.get("deviation_count", 0):
            caveats.append(
                "Credit deviations recorded on this file affect the rate. Clearing them may improve it.")
        if twin.get("legal_flag") or twin.get("technical_flag"):
            caveats.append(
                "A legal or technical exception is recorded against the property; resolving it may "
                "change both the rate and the sanctioned amount.")
        haircut = float(twin.get("income_verification_haircut", 0) or 0)
        if haircut > 0.15:
            caveats.append(
                "The income used for this assessment is the verified figure, which is lower than "
                "the income originally declared.")
        envelope = twin.envelope or {}
        if envelope.get("resize_required"):
            terms = envelope.get("resize_at_card") or {}
            caveats.append(
                "At the current loan amount the EMI exceeds the affordability limit at every "
                "permissible rate. The rate shown applies once the loan is reduced"
                + (f" to about Rs {terms['max_affordable_amount']:,.0f}" if terms.get("max_affordable_amount") else "")
                + (f" or the tenure is extended to {terms['tenure_needed_years']} years"
                   if terms.get("tenure_needed_years") else "") + ".")
        elif envelope.get("affordability_binding"):
            caveats.append(
                "The maximum rate on this file is limited by affordability rules, not by pricing policy.")
        return caveats

    def _negotiation_guidance(self, twin: CustomerDigitalTwin, recommendation: dict) -> dict:
        marginal = recommendation.get("marginal_analysis", {})
        economics = recommendation.get("economics", {})
        envelope = twin.envelope or {}
        guidance = {
            "recommended_rate": recommendation["recommended_rate"],
            "walk_away_rate": envelope.get("min_rate"),
            "conversion_at_recommended": economics.get("conversion_probability"),
            "value_of_next_5bps_inr": marginal.get("next_step_value_change_inr"),
            "conversion_cost_of_next_5bps_pts": marginal.get("next_step_conversion_change_pts"),
        }
        if marginal.get("next_step_value_change_inr", 0) < 0:
            guidance["note"] = (
                "Pricing above the recommendation destroys expected value on this file: the "
                "conversion lost outweighs the extra margin."
            )
        else:
            guidance["note"] = "There is headroom above the recommendation if the customer accepts."
        breakeven = (twin.risk or {}).get("breakeven_rate")
        if breakeven:
            guidance["breakeven_rate"] = breakeven
            guidance["margin_over_breakeven_bps"] = round(
                (recommendation["recommended_rate"] - breakeven) * 100, 1)
        return guidance

    def _policy_position(self, twin: CustomerDigitalTwin, rate: float) -> dict:
        envelope = twin.envelope or {}
        card = envelope.get("card_rate")
        return {
            "card_rate": card,
            "discount_from_card_bps": round((card - rate) * 100, 1) if card else None,
            "permissible_band": {"min_rate": envelope.get("min_rate"), "max_rate": envelope.get("max_rate")},
            "affordability_binding": envelope.get("affordability_binding"),
            "resize_required": envelope.get("resize_required"),
        }
