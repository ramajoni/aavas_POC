"""Feature cards, the Keep / Review / Reject decision, and the lineage graph.

A feature store answers *what is this feature*. A feature card answers the
question an auditor actually asks: **why does it exist, what evidence supports
it, and who decided.** The difference is the whole point - source, transformation,
hypothesis, every test, the decision, the reason, and the version.

The decision is deliberately **multi-dimensional and not a weighted score.** A
single number would let a strong statistic outvote a leakage finding, which is
the one trade a lender must never make. Instead each dimension can *veto*, and
what survives every veto is then ranked:

    risk        - leakage, instability, bias. A veto here is final.
    business    - is it explainable, is it available at decision time. Also a veto.
    statistics  - significance, IV, separation. Contributes; cannot rescue.
    predictive  - MI, SHAP, model contribution. Contributes; cannot rescue.

Three outcomes, not two. **Review** exists because "the evidence is mixed" is the
honest answer more often than keep or reject, and collapsing it into one of them
hides exactly the cases a human should look at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..logging_utils import get_logger

log = get_logger(__name__)

KEEP, REVIEW, REJECT = "keep", "review", "reject"

# A dimension that can veto, and what it vetoes on. Kept explicit so the reason
# attached to a decision is the rule that fired, not a narrative.
VETOES = {
    "leakage": "built from information unavailable when the decision is made",
    "unexplainable": "no human-authored reason label, so it cannot appear in a customer explanation",
    "unstable": "its distribution moves between time windows",
    "redundant": "carries what a contract feature already carries",
    "unavailable": "not reliably present at decision time",
    "suspicious_strength": "predictive power high enough to suggest leakage rather than signal",
}

THRESHOLDS = {
    "iv_keep": 0.10,        # medium or better
    "iv_review": 0.02,      # anything below this is not predictive
    "iv_suspicious": 0.50,  # above this, assume leakage until shown otherwise
    "psi_max": 0.25,
    "vif_max": 5.0,
    "p_value_max": 0.01,
    "mi_keep": 0.01,
}


@dataclass
class FeatureCard:
    """Everything known about one feature, in the order a reviewer reads it."""

    name: str
    stage: str
    family: str = "derived"
    recipe: str = ""
    inputs: list[str] = field(default_factory=list)
    hypothesis: str = ""
    expected_direction: str = ""
    created_by: str = "feature_discovery_agent"
    evidence: dict[str, Any] = field(default_factory=dict)
    decision: str = REVIEW
    vetoes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    version: str = "v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.name, "stage": self.stage, "family": self.family,
            "transformation": self.recipe, "original_columns": self.inputs,
            "business_hypothesis": self.hypothesis,
            "expected_relationship": self.expected_direction,
            "created_by": self.created_by, "version": self.version,
            "evidence": self.evidence,
            "decision": self.decision,
            "vetoes": [{"code": v, "why": VETOES[v]} for v in self.vetoes],
            "reasons": self.reasons,
        }


def _flatten(examination: dict[str, Any]) -> dict[str, Any]:
    """The numbers a reviewer reads, pulled out of the battery's nesting."""
    profile = examination.get("profile") or {}
    iv = examination.get("information_value") or {}
    separation = examination.get("separation") or {}
    stability = examination.get("stability") or {}
    return {
        "missing_pct": profile.get("missing_pct"),
        "distribution": profile.get("shape"),
        "outlier_pct": profile.get("outlier_pct"),
        "cardinality": profile.get("cardinality"),
        "information_value": iv.get("iv"),
        "iv_strength": iv.get("strength"),
        "monotonic_woe": iv.get("monotonic_woe"),
        "mutual_information": examination.get("mutual_information"),
        "p_value": separation.get("p_value"),
        "ks_statistic": separation.get("ks_statistic"),
        "effect_size_auc": separation.get("effect_size_auc"),
        "psi": stability.get("psi"),
        "vif": examination.get("vif"),
        "cramers_v": (examination.get("association") or {}).get("cramers_v"),
    }


def decide(evidence: dict[str, Any], *, has_reason_label: bool, leaks: bool = False,
           redundant_with: str | None = None,
           available_at_decision: bool = True,
           thresholds: dict[str, float] | None = None) -> tuple[str, list[str], list[str]]:
    """Keep / Review / Reject, with the rule that fired.

    Vetoes are evaluated first and are final. Nothing in the statistical or
    predictive dimensions can overturn one - a feature that leaks is rejected at
    any IV, which is the trade this ordering exists to prevent.
    """
    rules = {**THRESHOLDS, **(thresholds or {})}
    vetoes, reasons = [], []

    if leaks:
        vetoes.append("leakage")
    if not has_reason_label:
        vetoes.append("unexplainable")
    if not available_at_decision:
        vetoes.append("unavailable")
    if redundant_with:
        vetoes.append("redundant")
        reasons.append(f"carries what {redundant_with} already carries")

    psi = evidence.get("psi")
    if psi is not None and psi > rules["psi_max"]:
        vetoes.append("unstable")
        reasons.append(f"PSI {psi} between halves, above {rules['psi_max']}")

    iv = evidence.get("information_value")
    if iv is not None and iv > rules["iv_suspicious"]:
        vetoes.append("suspicious_strength")
        reasons.append(f"IV {iv} is high enough to suggest leakage rather than signal")

    if vetoes:
        return REJECT, vetoes, reasons + [VETOES[v] for v in vetoes if v not in ("redundant",)]

    # Nothing vetoed. Now weigh the evidence - and let it be mixed.
    strong, weak = [], []
    if iv is not None:
        (strong if iv >= rules["iv_keep"] else weak).append(f"IV {iv} ({evidence.get('iv_strength')})")
    mi = evidence.get("mutual_information")
    if mi is not None:
        (strong if mi >= rules["mi_keep"] else weak).append(f"mutual information {mi}")
    p = evidence.get("p_value")
    if p is not None:
        (strong if p < rules["p_value_max"] else weak).append(f"p = {p}")
    vif = evidence.get("vif")
    if vif is not None and vif > rules["vif_max"]:
        weak.append(f"VIF {vif} above {rules['vif_max']}: its own contribution is unstable")
    if evidence.get("monotonic_woe") is True:
        strong.append("WoE is monotonic across bins, which a credit reviewer can read")

    if strong and not weak:
        return KEEP, [], strong
    if not strong:
        return REJECT, [], weak or ["no dimension showed usable signal"]
    return REVIEW, [], strong + weak


def build_card(name: str, stage: str, examination: dict[str, Any], *,
               recipe: str = "", inputs: list[str] | None = None,
               hypothesis: str = "", family: str = "derived",
               has_reason_label: bool = False, leaks: bool = False,
               redundant_with: str | None = None,
               created_by: str = "feature_discovery_agent") -> FeatureCard:
    evidence = _flatten(examination)
    decision, vetoes, reasons = decide(
        evidence, has_reason_label=has_reason_label, leaks=leaks,
        redundant_with=redundant_with)

    direction = ""
    auc = evidence.get("effect_size_auc")
    if auc is not None:
        direction = ("higher value, higher event rate" if auc > 0.5
                     else "higher value, lower event rate" if auc < 0.5 else "no direction")

    return FeatureCard(
        name=name, stage=stage, family=family, recipe=recipe, inputs=inputs or [],
        hypothesis=hypothesis or "not stated - a candidate with no hypothesis is a coincidence "
                                 "until someone writes one",
        expected_direction=direction, created_by=created_by,
        evidence=evidence, decision=decision, vetoes=vetoes, reasons=reasons)


# ------------------------------------------------------------- lineage graph


def lineage(cards: list[FeatureCard]) -> dict[str, Any]:
    """Source column -> derived feature -> decision.

    Deliberately a small graph over real edges rather than a rendering of an
    idealised one: every edge here exists because a card names it.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    for card in cards:
        nodes.setdefault(card.name, {"id": card.name, "type": "derived",
                                     "family": card.family, "decision": card.decision})
        for source in card.inputs:
            nodes.setdefault(source, {"id": source, "type": "source", "family": "contract"})
            edges.append({"from": source, "to": card.name, "via": card.recipe})
    by_decision: dict[str, int] = {}
    for card in cards:
        by_decision[card.decision] = by_decision.get(card.decision, 0) + 1
    return {
        "nodes": list(nodes.values()), "edges": edges,
        "sources": sorted(n["id"] for n in nodes.values() if n["type"] == "source"),
        "derived": sorted(n["id"] for n in nodes.values() if n["type"] == "derived"),
        "by_decision": by_decision,
        "note": ("Every edge exists because a feature card names it. A column with no "
                 "outgoing edge fed no candidate; a candidate with no incoming edge would be "
                 "a bug."),
    }
