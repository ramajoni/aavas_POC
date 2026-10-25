"""Turn attributions into sentences a sales or credit officer can use.

Raw SHAP values are not an explanation to anyone outside the model team. What a
branch needs is: which five things moved this customer's rate, in which
direction, by how many basis points, and how much of it was the customer's own
risk profile versus commercial pressure.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import feature_config, model_config
from ..features.feature_pipeline import display_value, reason_label, sentinel_rules
from .shap_explainer import Attribution


@dataclass
class ReasonCode:
    code: str
    feature: str
    label: str
    direction: str          # "increase" | "decrease"
    impact_bps: float
    value: object
    text: str

    def as_dict(self) -> dict:
        return {
            "code": self.code, "feature": self.feature, "label": self.label,
            "direction": self.direction, "impact_bps": self.impact_bps,
            "value": display_value(self.feature, _plain(self.value)), "text": self.text,
        }


def build(attributions: list[Attribution], top_k: int | None = None,
          min_bps: float | None = None) -> list[ReasonCode]:
    cfg = model_config()["xai"]
    top_k = cfg["top_k_reason_codes"] if top_k is None else top_k
    min_bps = cfg["min_reason_code_bps"] if min_bps is None else min_bps

    codes: list[ReasonCode] = []
    for attribution in attributions:
        if abs(attribution.bps) < min_bps:
            continue
        labels = reason_label(attribution.feature)
        increases = attribution.contribution > 0
        phrase = labels["up"] if increases else labels["down"]
        value = display_value(attribution.feature, attribution.value)
        sentinel = (sentinel_rules().get(attribution.feature) or {}).get("display")
        if sentinel is not None and value == sentinel:
            # "Weaker bureau profile" would misdescribe a file that has no bureau
            # profile at all, whichever way the imputed score happens to push.
            phrase = sentinel
        arrow = "increase" if increases else "decrease"
        codes.append(ReasonCode(
            code=_code(attribution.feature, increases),
            feature=attribution.feature,
            label=labels["label"],
            direction=arrow,
            impact_bps=abs(attribution.bps),
            value=value,
            text=f"{phrase} ({attribution.bps:+.0f} bps)",
        ))
        if len(codes) >= top_k:
            break
    return codes


def _code(feature: str, increases: bool) -> str:
    return f"{feature.upper()[:18]}_{'UP' if increases else 'DN'}"


def commercial_share(attributions: list[Attribution]) -> dict[str, float]:
    """How much of the price came from commercial pressure rather than the
    customer's own risk profile.

    Governance asks this question, and it is unanswerable from a single SHAP
    plot: it needs the feature contract's split between risk and sales context.
    """
    sales_block = feature_config()["sales_context"]
    sales_features = set(sales_block.get("numeric", [])) | set(sales_block.get("categorical", []))

    commercial = sum(abs(a.contribution) for a in attributions if a.feature in sales_features)
    total = sum(abs(a.contribution) for a in attributions)
    signed = sum(a.contribution for a in attributions if a.feature in sales_features)
    return {
        "commercial_share_pct": round(100 * commercial / total, 1) if total else 0.0,
        "commercial_net_bps": round(signed * 100, 1),
        "risk_net_bps": round(sum(a.contribution for a in attributions if a.feature not in sales_features) * 100, 1),
    }


def summarise(codes: list[ReasonCode], rate: float) -> str:
    """A single paragraph for the sanction letter / CRM note."""
    if not codes:
        return f"Recommended rate {rate:.2f}%. No individual factor moved the price materially."
    ups = [c for c in codes if c.direction == "increase"]
    downs = [c for c in codes if c.direction == "decrease"]
    parts = [f"Recommended rate {rate:.2f}%."]
    if ups:
        parts.append("Priced up by: " + ", ".join(f"{c.label.lower()} ({c.impact_bps:.0f} bps)" for c in ups) + ".")
    if downs:
        parts.append("Priced down by: " + ", ".join(f"{c.label.lower()} ({c.impact_bps:.0f} bps)" for c in downs) + ".")
    return " ".join(parts)


def _plain(value):
    try:
        import numpy as np

        if isinstance(value, (np.floating, np.integer)):
            return value.item()
    except ImportError:
        pass
    return value
