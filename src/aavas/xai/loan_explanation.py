"""One loan, explained in full: every feature's weight in every model that priced it.

The pricing screen shows the five reason codes a branch can read aloud. A credit
reviewer asking "why this number?" needs the rest: all of the Stage-2 features
with their SHAP contribution and share of the total, the same for the risk and
conversion models, and the bridge from a typical customer's rate to the rate
actually recommended.

Everything is recomputed from the application snapshot stored on the decision,
so it explains the file exactly as it was priced. If the models have been
retrained since, the page says so rather than quietly explaining a different
model.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import feature_config
from ..features import stage2_features
from ..features.feature_pipeline import reason_label
from ..ingestion.schemas import Stage
from ..logging_utils import get_logger
from .shap_explainer import Attribution, PricingExplainer

log = get_logger(__name__)

# How far the recomputed Stage-2 rate may sit from the recorded anchor before the
# page warns that it is not explaining the number on the decision. The anchor is
# stored rounded to 2 dp, so half a basis point is rounding, not drift.
ANCHOR_TOLERANCE_BPS = 1.0

MODELS = {
    "stage2": {"title": "Rate model (Stage 2)", "positive_means": "raises the rate",
               "positive_is_adverse": True},
    "risk": {"title": "Risk model (probability of default)", "positive_means": "raises default risk",
             "positive_is_adverse": True},
    "conversion": {"title": "Conversion model (chance the customer accepts)",
                   "positive_means": "raises the chance of acceptance", "positive_is_adverse": False},
}


class NotExplainable(ValueError):
    """The decision exists but has no final rate to explain."""


class LoanExplainer:
    """Holds one SHAP explainer per model; building a TreeExplainer is the slow part."""

    def __init__(self, bundle):
        self.bundle = bundle
        self._explainers: dict[str, PricingExplainer] = {}

    def _explainer(self, name: str) -> PricingExplainer | None:
        model = getattr(self.bundle, name, None)
        if model is None:
            return None
        if name not in self._explainers:
            self._explainers[name] = PricingExplainer(
                model.model, model.transformer, is_classifier=name != "stage2")
        return self._explainers[name]

    # ------------------------------------------------------------------ public
    def explain(self, decision: dict[str, Any]) -> dict[str, Any]:
        payload = decision.get("payload") or {}
        if decision.get("stage") != Stage.POST_ASSESSMENT.value or "recommendation" not in payload:
            raise NotExplainable(
                "Only a final assessment has a recommended rate to explain; this decision is a "
                f"{decision.get('stage', 'unknown')} record.")

        snapshot = payload.get("application_snapshot") or {}
        frame = pd.DataFrame([snapshot])
        recommendation = payload["recommendation"]
        rate = float(recommendation["recommended_rate"])

        models: dict[str, Any] = {}
        builds: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
            "stage2": stage2_features.build,
            "risk": stage2_features.build,
            "conversion": lambda df: _at_offered_rate(df, rate),
        }
        for name, build in builds.items():
            explainer = self._explainer(name)
            if explainer is None:
                continue
            models[name] = _model_view(name, explainer, explainer.explain(frame, build=build))

        warnings = self._warnings(decision, payload, models)
        stage2 = models.get("stage2")
        explanation = payload.get("explanation") or {}
        return {
            "decision_id": decision.get("decision_id"),
            "loan_id": decision.get("loan_id") or payload.get("loan_id"),
            "created_at": decision.get("created_at"),
            "applicant": _applicant(snapshot),
            "rate": {
                "recommended": rate,
                "card": (payload.get("policy") or {}).get("card_rate"),
                "model_anchor": stage2["prediction"] if stage2 else None,
                "typical_customer": stage2["base_value"] if stage2 else None,
                "permissible_band": (payload.get("policy") or {}).get("permissible_band"),
                "required_approval": (payload.get("policy") or {}).get("required_approval"),
            },
            "models": models,
            "bridge": _bridge(stage2, rate) if stage2 else None,
            "outcomes": {
                "risk": payload.get("risk"),
                "conversion_probability": (recommendation.get("economics") or {}).get(
                    "conversion_probability"),
            },
            "context": {
                "headline": explanation.get("headline"),
                "reason_codes": explanation.get("reason_codes") or [],
                "commercial_vs_risk": explanation.get("commercial_vs_risk"),
                "fair_pricing_review": explanation.get("fair_pricing_review"),
                "actionable_for_customer": explanation.get("actionable_for_customer") or [],
                "negotiation_guidance": explanation.get("negotiation_guidance"),
                "customer_caveats": explanation.get("customer_caveats") or [],
                "economics": recommendation.get("economics"),
                "comparisons": recommendation.get("comparisons"),
                "policy": payload.get("policy"),
                "disclaimer": explanation.get("disclaimer"),
            },
            "model_versions": payload.get("model_versions"),
            "warnings": warnings,
        }

    def _warnings(self, decision: dict, payload: dict, models: dict) -> list[str]:
        warnings: list[str] = []
        recorded = payload.get("model_versions") or {}
        current = self.bundle.versions() if hasattr(self.bundle, "versions") else {}
        changed = sorted(k for k in ("stage2", "risk", "conversion")
                         if recorded.get(k) and current.get(k) and recorded[k] != current[k])
        if changed:
            warnings.append(
                "Models retrained since this decision (" + ", ".join(
                    f"{k}: {recorded[k]} -> {current[k]}" for k in changed)
                + "). These attributions explain the current models, not the ones that priced it.")

        anchor = ((payload.get("recommendation") or {}).get("stage2_anchor") or {}).get("anchor_rate")
        stage2 = models.get("stage2")
        if anchor is not None and stage2 is not None:
            drift_bps = abs(stage2["prediction"] - float(anchor)) * 100
            if drift_bps > ANCHOR_TOLERANCE_BPS:
                warnings.append(
                    f"Recomputed Stage-2 anchor {stage2['prediction']:.2f}% differs from the recorded "
                    f"anchor {float(anchor):.2f}% by {drift_bps:.1f} bps.")
        recorded_method = (payload.get("explanation") or {}).get("attribution_method")
        if recorded_method and stage2 is not None and recorded_method != stage2["method"]:
            # Reason codes and the fair-pricing review were stored when the file was
            # priced; recomputing them with a different method moves the numbers.
            warnings.append(
                f"This decision's reason codes and fair-pricing review were recorded with "
                f"{recorded_method}; this page recomputes them with {stage2['method']}, so its "
                "figures can differ from the recorded ones.")
        missing = [name for name in MODELS if name not in models]
        if missing:
            warnings.append("No model artefact loaded for: " + ", ".join(missing) + ".")
        return warnings


# ---------------------------------------------------------------------- views
def _commercial_features() -> set[str]:
    block = feature_config()["sales_context"]
    # The offered rate is the price itself - a commercial choice, not a fact about
    # the customer - so it sits with the commercial block in the conversion model.
    return set(block.get("numeric", [])) | set(block.get("categorical", [])) | {"offered_rate"}


def _model_view(name: str, explainer: PricingExplainer, attributions: list[Attribution]) -> dict:
    meta = MODELS[name]
    commercial = _commercial_features()
    total_abs = sum(abs(a.contribution) for a in attributions) or 1.0
    is_rate = name == "stage2"

    features = []
    for rank, a in enumerate(attributions, start=1):
        labels = reason_label(a.feature)
        direction = "increase" if a.contribution > 0 else "decrease" if a.contribution < 0 else "none"
        row = {
            "rank": rank,
            "feature": a.feature,
            "label": labels["label"],
            "group": "commercial" if a.feature in commercial else "customer_risk",
            "value": _display(a.value),
            "contribution": round(float(a.contribution), 6),
            "share_pct": round(100 * abs(a.contribution) / total_abs, 2),
            "direction": direction,
        }
        if is_rate:
            # The labels are written for the price: "Weaker bureau profile" is why a
            # rate goes up. They would misdescribe a conversion or risk movement.
            row["impact_bps"] = a.bps
            row["effect"] = labels["up"] if a.contribution > 0 else labels["down"]
        features.append(row)

    groups = {}
    for group in ("customer_risk", "commercial"):
        members = [f for f in features if f["group"] == group]
        net = sum(f["contribution"] for f in members)
        groups[group] = {
            "count": len(members),
            "net": round(net, 6),
            "share_pct": round(sum(f["share_pct"] for f in members), 1),
            **({"net_bps": round(net * 100, 1)} if is_rate else {}),
        }

    base = explainer.base_value
    return {
        **meta,
        "method": explainer.method(),
        "units": "rate_pct" if is_rate else explainer.units,
        "base_value": round(base, 6),
        "prediction": round(base + sum(a.contribution for a in attributions), 6),
        "features": features,
        "groups": groups,
    }


def _bridge(stage2: dict, recommended: float) -> dict:
    """Typical customer -> each feature -> model anchor -> recommended rate."""
    return {
        "baseline_rate": stage2["base_value"],
        "model_rate": stage2["prediction"],
        "recommended_rate": recommended,
        "optimisation_bps": round((recommended - stage2["prediction"]) * 100, 2),
        "steps": [{"feature": f["feature"], "label": f["label"], "group": f["group"],
                   "delta_bps": round(f["contribution"] * 100, 4)} for f in stage2["features"]],
    }


def _at_offered_rate(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """The conversion model's view of the file: offered this rate. Mirrors
    ConversionModel.predict_at_rate so the attributions explain that prediction."""
    scored = df.copy()
    scored["offered_rate"] = rate
    if "competitor_rate" in scored.columns:
        competitor = pd.to_numeric(scored["competitor_rate"], errors="coerce")
        scored["competitor_gap_bps"] = ((scored["offered_rate"] - competitor) * 100).fillna(0.0)
    return scored


def _applicant(snapshot: dict) -> dict:
    keys = ("segment", "city", "channel", "property_type", "sanctioned_loan_amount", "tenure_years",
            "ltv", "foir", "bureau_score", "verified_monthly_income", "competitor_rate",
            "competitor_name", "risk_grade")
    return {k: _plain(snapshot.get(k)) for k in keys}


def _display(value):
    """A feature value as a reviewer reads it: 0.6896551724 is 0.6897, not more."""
    value = _plain(value)
    return round(value, 4) if isinstance(value, float) else value


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
