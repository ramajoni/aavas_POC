"""Pricing orchestrator - the agentic pipeline.

    digital twin -> risk -> sales context -> policy envelope
                 -> scenario simulation -> yield optimization
                 -> policy validation -> explanation -> audited decision

Each step is a separate agent with its own contract, and the result of every step
lands on the shared digital twin. That is what makes this a pipeline rather than
six models in a trench coat: the risk agent's grade determines the card rate the
policy agent enforces, the sales agent's competitor gap moves the conversion
curve the optimizer searches, and the explanation agent reads all of it back.

Nothing here decides anything a human cannot see, override, or reverse.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from ..config import versions
from ..governance import decision_logger, model_registry
from ..ingestion.schemas import Stage
from ..logging_utils import get_logger
from .conversion_agent import ConversionAgent
from .explanation_agent import ExplanationAgent
from .learning_agent import LearningAgent
from .policy_agent import PolicyAgent
from .preassessment_agent import PreAssessmentAgent
from .pricing_agent import PricingAgent
from .risk_agent import RiskAgent
from .sales_context_agent import SalesContextAgent
from .trace import Trace
from .underwriting_agent import CustomerDigitalTwin, UnderwritingAgent

log = get_logger(__name__)


class PricingOrchestrator:
    def __init__(self, bundle=None, note_retriever=None):
        self.bundle = bundle or model_registry.load_bundle()
        self.underwriting = UnderwritingAgent()
        self.risk = RiskAgent(self.bundle.risk)
        self.conversion = ConversionAgent(self.bundle.conversion)
        self.sales = SalesContextAgent(self.bundle.override, note_retriever=note_retriever)
        self.policy = PolicyAgent()
        self.pricing = PricingAgent(self.bundle.stage2, self.bundle.conversion,
                                    self.bundle.prepayment, self.bundle.override)
        self.explanation = ExplanationAgent(self.bundle.stage2, self.bundle.conversion)
        self.preassessment = PreAssessmentAgent(self.bundle.stage1, self.underwriting)
        self.learning = LearningAgent(self.bundle)

    # ------------------------------------------------------------- stage 1
    def pre_assessment(self, application: dict | pd.DataFrame, persist: bool = True) -> dict:
        started = time.perf_counter()
        trace = Trace()
        with trace.step("preassessment", "quote") as step:
            result = self.preassessment.quote(application)
            band = result.get("band") or {}
            step.summary = (f"band {band.get('min_rate')}-{band.get('max_rate')}% "
                            f"({band.get('width_bps')} bps) at confidence "
                            f"{result.get('confidence')} ({result.get('confidence_label')})"
                            if band else "no indicative band produced")
        result["versions"] = versions()
        result["trace"] = trace.as_dict()
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        if persist:
            result["decision_id"] = decision_logger.log_decision(
                result["loan_id"], Stage.PRE_ASSESSMENT.value, result)
        return result

    # ------------------------------------------------------------- stage 2
    def process(self, application: dict | pd.DataFrame, persist: bool = True) -> dict:
        if not self.bundle.ready:
            log.warning("running with missing model artefacts: %s", self.bundle.missing)

        started = time.perf_counter()
        trace = Trace()

        with trace.step("underwriting", "build digital twin") as step:
            twin = self.underwriting.build(application, stage=Stage.POST_ASSESSMENT)
            step.summary = (f"twin for {twin.loan_id}: "
                            f"{len(twin.notes)} note(s) raised")

        # 1. Risk first: the grade it assigns is what the policy card rate is built on.
        with trace.step("risk", "assess") as step:
            risk = self.risk.assess(twin)
            step.summary = (f"grade {risk.get('risk_grade')}, PD {risk.get('pd'):.4f}, "
                            f"expected loss {risk.get('expected_loss_pct'):.3f}% p.a.")

        # 2. Policy envelope, now that the grade is known.
        with trace.step("policy", "envelope") as step:
            envelope = self.policy.envelope(twin)
            step.summary = (f"card {envelope.get('card_rate')}%, band "
                            f"{envelope.get('min_rate')}-{envelope.get('max_rate')}%"
                            + (", affordability binding" if envelope.get("affordability_binding") else "")
                            + (", RESIZE REQUIRED" if envelope.get("resize_required") else ""))

        # 3. Commercial context, anchored on the card rate until we have a recommendation.
        with trace.step("sales_context", "read against the card rate") as step:
            anchored = self.sales.build(twin, reference_rate=envelope["card_rate"])
            step.summary = (f"competitor gap {anchored.get('competitor_gap_bps')} bps"
                            + (f" vs {anchored['competitor_name']}" if anchored.get("competitor_name")
                               else ", no competitor quote on file")
                            + (", evidence verified" if anchored.get("competitor_evidence_verified")
                               else ""))

        # 4. Scenario simulation and yield optimization.
        with trace.step("pricing", "optimise yield") as step:
            recommendation = self.pricing.recommend(
                twin, expected_loss_pct=risk["expected_loss_pct"])
            step.summary = (f"recommends {recommendation.get('recommended_rate')}% from "
                            f"{len(recommendation.get('scenarios') or [])} scenario(s)")

        # 5. Validate and, if necessary, clamp into policy.
        with trace.step("policy", "enforce") as step:
            rate, policy_check = self.policy.enforce(twin, recommendation["recommended_rate"])
            moved = round((rate - recommendation["recommended_rate"]) * 100, 1)
            recommendation["recommended_rate"] = rate
            step.summary = (f"clamped by {moved:+} bps to {rate}%" if moved
                            else f"{rate}% is inside policy, unchanged")

        # 6. Re-read commercial context against the price we are actually recommending.
        with trace.step("sales_context", "re-read against the recommendation") as step:
            sales_context = self.sales.build(twin, reference_rate=rate)
            step.summary = (f"competitor gap {sales_context.get('competitor_gap_bps')} bps "
                            f"at the recommended price")

        # 7. Customer price response around the recommendation.
        with trace.step("conversion", "price response") as step:
            elasticity = self.conversion.profile(twin, rate)
            table = elasticity.get("negotiation_table") or []
            step.summary = (f"P(disbursed) {elasticity.get('conversion_at_recommended')} at the "
                            f"recommended price, {elasticity.get('sensitivity_label')}, "
                            f"{len(table)} negotiation row(s)"
                            if elasticity.get("available") else "price response unavailable")

        # 8. Explain it.
        with trace.step("explanation", "reason codes") as step:
            explanation = self.explanation.explain(twin, recommendation)
            review = (explanation.get("fair_pricing_review") or {}).get("flagged")
            step.summary = (f"{len(explanation.get('reason_codes') or [])} reason code(s) via "
                            f"{explanation.get('attribution_method')}"
                            + (", fair-pricing review flagged" if review else ""))

        payload = {
            "loan_id": twin.loan_id,
            "stage": Stage.POST_ASSESSMENT.value,
            "twin": twin.summary(),
            "application_snapshot": twin.snapshot(),
            "risk": risk,
            "policy": policy_check,
            "envelope": envelope,
            "recommendation": recommendation,
            "price_sensitivity": elasticity,
            "sales_context": sales_context,
            "explanation": explanation,
            "model_versions": self.bundle.versions(),
            "versions": versions(),
            "human_decision_required": True,
            "notes": twin.notes,
            "trace": trace.as_dict(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if persist:
            payload["decision_id"] = decision_logger.log_decision(
                twin.loan_id, Stage.POST_ASSESSMENT.value, payload)

            review = explanation.get("fair_pricing_review") or {}
            if review.get("flagged"):
                # Queue it explicitly: a fair-pricing exception buried in a JSON
                # blob is not a control, it is a number nobody reads.
                from ..governance import audit

                audit.record("pricing_engine", "fair_pricing_review_required", "loan", twin.loan_id,
                             {"decision_id": payload["decision_id"], **review})
                log.warning("fair-pricing review flagged on %s: %+.0f bps of commercial uplift",
                            twin.loan_id, review["commercial_net_bps"])
        return payload

    # ------------------------------------------------- human decision + outcome
    def capture_override(self, loan_id: str, human_rate: float, reason: str,
                         approved_by: str | None = None, reason_detail: str | None = None,
                         evidence_verified: bool = False) -> dict:
        """Record a human decision and tell the user whether it is within authority."""
        decision = decision_logger.latest_for_loan(loan_id, stage=Stage.POST_ASSESSMENT.value)
        if not decision:
            raise ValueError(f"no post-assessment decision on record for {loan_id}")

        payload = decision["payload"]
        ai_rate = payload["recommendation"]["recommended_rate"]
        snapshot = payload.get("application_snapshot")
        twin_frame = pd.DataFrame([snapshot]) if snapshot else None

        from ..optimization import constraints

        discount_bps = round((payload["policy"]["card_rate"] - human_rate) * 100, 1)
        required = constraints.approval_level(discount_bps)
        within_authority = _within_authority(approved_by, required)

        record = self.learning.capture_override(
            loan_id=loan_id, ai_rate=ai_rate, human_rate=human_rate, reason=reason,
            decision_id=decision["decision_id"], reason_detail=reason_detail,
            approved_by=approved_by, required_approval=required,
            within_authority=within_authority, evidence_verified=evidence_verified,
            sales_context=payload.get("sales_context"),
        )
        record["required_approval"] = required
        record["within_authority"] = within_authority
        # Re-validate the human's price against policy, using the file as priced.
        record["policy_check"] = (
            constraints.check(twin_frame, human_rate, override_reason=reason,
                              evidence_verified=evidence_verified).as_dict()
            if twin_frame is not None and not twin_frame.empty else None
        )
        return record

    def capture_outcome(self, loan_id: str, **kwargs: Any) -> dict:
        return self.learning.capture_outcome(loan_id, **kwargs)


def _within_authority(approved_by: str | None, required: str) -> bool | None:
    """``approved_by`` is expected as '<name>:<level>' or just a level."""
    if not approved_by:
        return None
    from ..config import policy

    levels = [level["level"] for level in policy()["approval_authority"]]
    supplied = approved_by.split(":")[-1].strip()
    if supplied not in levels or required not in levels:
        return None
    return levels.index(supplied) >= levels.index(required)
