"""What each role may see of a pricing decision.

A relationship manager prices and sends offers; the analysis behind the price -
model drivers and their effects, the rate-versus-outcome scenarios, uplift, the
value breakdown - belongs to the area manager who approves a lower rate, and to
risk and model validation. Hiding those panels in the browser is not enough: the
pricing response carries all of it, so the relationship manager is sent a
cut-down copy built here.

``rm_view`` keeps what screen 04 "Price and send offer" needs and nothing else:
the recommended rate and its permitted band, the range quoted at Stage 1, how
many similar customers accept, the competitor claim and the risk grade. Why the
model priced it where it did (the reason codes) and how far above the
recommendation customers still accept (the pricing headroom) are the area
manager's.
"""

from __future__ import annotations

from typing import Any

from . import security
from .security import Principal

# Roles that see the full analysis.
FULL = {"area_manager", "model_validation", "administrator"}


def full_access(principal: Principal) -> bool:
    if principal.mode != security.SESSION:
        return True        # disabled and API-key modes keep their existing behaviour
    return bool(set(principal.roles) & FULL)


def rm_view(payload: dict[str, Any]) -> dict[str, Any]:
    rec = payload.get("recommendation") or {}
    economics = rec.get("economics") or {}
    envelope = payload.get("envelope") or {}
    policy = payload.get("policy") or {}
    sales = payload.get("sales_context") or {}
    risk = payload.get("risk") or {}
    return {
        "view": "relationship_manager",
        "loan_id": payload.get("loan_id"), "decision_id": payload.get("decision_id"),
        "stage": payload.get("stage"), "versions": payload.get("versions"),
        "human_decision_required": payload.get("human_decision_required", True),
        "recommendation": {"recommended_rate": rec.get("recommended_rate"),
                           "conversion_probability": economics.get("conversion_probability")},
        "envelope": {k: envelope.get(k) for k in ("min_rate", "max_rate", "card_rate")},
        "indicative_band": payload.get("indicative_band"),
        "policy": {k: policy.get(k) for k in ("allowed", "verdict", "card_rate", "required_approval",
                                               "violations", "policy_version")},
        "risk": {"risk_grade": risk.get("risk_grade")},
        "sales_context": {k: sales.get(k) for k in ("competitor_rate", "competitor_name",
                                                     "competitor_claim_status",
                                                     "max_discount_on_this_claim_bps")},
        "application_snapshot": payload.get("application_snapshot"),
    }


def for_principal(payload: dict[str, Any], principal: Principal) -> dict[str, Any]:
    return payload if full_access(principal) else rm_view(payload)
