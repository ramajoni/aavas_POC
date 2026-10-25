"""Policy agent - the boundary the engine may not cross.

It runs twice in every decision: once before the search, to establish the
permissible band the optimizer is allowed to explore, and once after, to validate
the chosen price and route it to the right approval authority. A recommendation
that fails validation is clamped into policy and flagged, never silently served.
"""

from __future__ import annotations

from ..config import policy
from ..logging_utils import get_logger
from ..optimization import constraints
from .underwriting_agent import CustomerDigitalTwin

log = get_logger(__name__)


class PolicyAgent:
    def envelope(self, twin: CustomerDigitalTwin) -> dict:
        band = constraints.permissible_band(twin.frame).iloc[0]
        envelope = {
            "card_rate": float(band["card_rate"]),
            "min_rate": float(band["min_rate"]),
            "max_rate": float(band["max_rate"]),
            "max_affordable_rate": float(band["max_affordable_rate"]),
            "affordability_binding": bool(band["affordability_binding"]),
            "infeasible": bool(band["infeasible"]),
            "resize_required": bool(band["resize_required"]),
            "policy_version": policy()["policy_version"],
        }
        if envelope["resize_required"]:
            # Sized at the card rate so the twin carries the condition from the
            # start; the recommendation's own policy check re-sizes at its rate.
            envelope["resize_at_card"] = constraints.resize_terms(twin.frame, envelope["card_rate"])
            twin.note(
                "no permissible rate keeps this file inside the FOIR cap at its current size; it is "
                "priced on its risk band, conditional on reducing the loan or extending the tenure"
            )
        twin.envelope = envelope
        return envelope

    def validate(self, twin: CustomerDigitalTwin, rate: float, override_reason: str | None = None,
                 evidence_verified: bool | None = None) -> dict:
        if evidence_verified is None:
            evidence_verified = bool(twin.get("competitor_evidence_verified", False))
        check = constraints.check(twin.frame, rate, override_reason=override_reason,
                                  evidence_verified=evidence_verified)
        return check.as_dict()

    def enforce(self, twin: CustomerDigitalTwin, rate: float) -> tuple[float, dict]:
        """Validate and, if needed, clamp into the permissible band."""
        check = self.validate(twin, rate)
        if check["allowed"]:
            return rate, check

        envelope = twin.envelope or self.envelope(twin)
        clamped = min(max(rate, envelope["min_rate"]), envelope["max_rate"])
        log.warning("policy clamp on %s: %.2f%% -> %.2f%% (%s)",
                    twin.loan_id, rate, clamped, "; ".join(check["violations"]))
        twin.note(f"recommendation clamped into policy: {rate:.2f}% -> {clamped:.2f}%")
        revalidated = self.validate(twin, clamped)
        revalidated["clamped_from"] = round(rate, 2)
        revalidated["original_violations"] = check["violations"]
        return clamped, revalidated
