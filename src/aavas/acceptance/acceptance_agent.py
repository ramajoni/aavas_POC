"""Acceptance agent - composes the others into one governed score.

Order, and why:

    sources  ->  extraction  ->  enrichment  ->  conversion model  ->  shrinkage
                      |                                                    |
                   evidence  -------------------------------------------- +

The probability comes from the fitted, isotonically calibrated conversion model,
never from the language model. Extraction supplies typed signals; where a signal
maps onto a column the model was actually trained on, it is written to the frame
and the model re-reads it. Where it does not, the signal informs the evidence
score and the explanation but is honestly reported as not priced.

That last distinction matters. Urgency, timeline pressure and the customer's
stated expected rate are not in any feature contract today, so they cannot move
the probability without retraining and a reason-code label for each
(docs/target-architecture-gap.md section 8.2). Pretending otherwise would put a
number on screen that no model produced.
"""

from __future__ import annotations

from typing import Any

from ..config import load
from ..logging_utils import get_logger
from .evidence_agent import EvidenceAgent
from .extraction_agent import ExtractionAgent
from .signals import AcceptanceSignals
from .sources import SourceRegistry

log = get_logger(__name__)

# Signals that map onto a column the conversion model was trained on. Anything
# outside this map is context, not a model input - stated rather than implied.
PRICED_SIGNALS = {
    "competitor_rate": "competitor_rate",
    "competitor_name": "competitor_name",
}

LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}


class AcceptanceAgent:
    def __init__(self, conversion_agent=None, config: dict[str, Any] | None = None,
                 registry: SourceRegistry | None = None):
        self.cfg = config if config is not None else load("acceptance")
        self.conversion = conversion_agent
        self.registry = registry or SourceRegistry(self.cfg)
        self.extractor = ExtractionAgent(self.cfg)
        self.evidence = EvidenceAgent(self.cfg)
        self.scoring = self.cfg.get("scoring", {})
        self.governance = self.cfg.get("governance", {})

    # ------------------------------------------------------------------- score
    def score(self, twin, rate: float, context: dict[str, Any] | None = None,
              signals: AcceptanceSignals | None = None) -> dict[str, Any]:
        """Acceptance confidence for one application at one proposed rate.

        Pass ``signals`` to replay a stored decision: an audit must reproduce the
        original number from what was persisted, never by re-calling the model
        (gap doc section 10.4).
        """
        replayed = signals is not None
        if signals is None:
            collected = self.registry.collect(twin, context)
            signals = self.extractor.run(collected)

        applied = self._enrich(twin, signals)
        evidence = self.evidence.score(signals)
        weight = float(evidence["evidence_confidence"]) ** float(
            self.scoring.get("shrinkage_exponent", 1.0))

        base_rate = self._base_rate()
        modelled = self._model_probability(twin, rate)

        if modelled is None:
            adjusted = None
            band = None
        else:
            # Thin evidence pulls the estimate toward the portfolio base rate.
            # This is what "be more conservative when confidence is low" means
            # operationally; without it, two engineers implement it two ways.
            adjusted = modelled * weight + base_rate * (1.0 - weight)
            half = float(self.scoring.get("report_band_pts", 0.06)) / 2.0
            band = [round(max(adjusted - half, 0.0), 3), round(min(adjusted + half, 1.0), 3)]

        payload = {
            "loan_id": getattr(twin, "loan_id", None),
            "proposed_rate": round(float(rate), 4),
            "acceptance_probability": None if adjusted is None else round(adjusted, 4),
            "acceptance_band": band,
            "model_probability": None if modelled is None else round(modelled, 4),
            "base_rate": round(base_rate, 4),
            "shrinkage_weight": round(weight, 3),
            "evidence": evidence,
            "extractor_mode": "replayed" if replayed else self.extractor.mode,
            "signals_applied_to_model": applied,
            "signals_not_priced": sorted(
                n for n in signals.signals if n not in PRICED_SIGNALS),
            "drivers": self._drivers(signals, twin, rate, modelled, base_rate),
            "governance": self._governance(signals),
            "signal_record": signals.as_dict() if self.governance.get("persist_signals", True) else None,
            "caveat": ("Acceptance probability is produced by the calibrated conversion model. "
                       "Extracted signals are context and evidence, not a probability."),
        }
        return payload

    # ---------------------------------------------------------------- internals
    def _enrich(self, twin, signals: AcceptanceSignals) -> list[dict[str, Any]]:
        """Write priced signals onto the frame so the model re-reads them.

        Two things are deliberately never written: ``competitor_evidence_verified``
        and anything derived from it. Verification is a statement that a document
        was seen. No extractor may assert it, however confident the note sounds -
        that flag is what ``constraints.py`` uses to cap an unevidenced discount.
        """
        applied: list[dict[str, Any]] = []
        for name, column in PRICED_SIGNALS.items():
            signal = signals.signals.get(name)
            if signal is None:
                continue
            current = twin.get(column)
            if current is not None and str(current).strip() != "":
                continue  # a typed field already on the record wins
            twin.set(**{column: signal.value})
            applied.append({"signal": name, "column": column, "value": signal.value,
                            "reliability": signal.reliability, "source": signal.source})
            twin.note(f"{column} taken from {signal.source} ({signal.reliability})")
        return applied

    def _base_rate(self) -> float:
        """Portfolio acceptance base rate, from the model card when available."""
        fallback = float(self.scoring.get("fallback_base_rate", 0.6585))
        model = getattr(self.conversion, "model", None)
        card = getattr(model, "card", None)
        try:
            metrics = card.get("metrics", {}) if isinstance(card, dict) else card.metrics
            value = float(metrics.get("base_rate"))
            return value if 0.0 < value < 1.0 else fallback
        except Exception:  # noqa: BLE001 - card shape varies; fallback is configured
            return fallback

    def _model_probability(self, twin, rate: float) -> float | None:
        if self.conversion is None:
            log.info("acceptance: no conversion model; probability unavailable")
            return None
        try:
            value = float(self.conversion.predict(twin, float(rate)))
        except Exception as exc:  # noqa: BLE001
            log.warning("acceptance: conversion model failed (%s)", exc)
            return None
        return None if value != value else value  # NaN guard

    def _drivers(self, signals: AcceptanceSignals, twin, rate: float,
                 modelled: float | None, base_rate: float) -> dict[str, list[str]]:
        """Plain-language reasons, split into what helps and what hurts (FR-09)."""
        up: list[str] = []
        down: list[str] = []

        if modelled is not None:
            (up if modelled >= base_rate else down).append(
                f"Modelled acceptance at {rate:.2f}% is "
                f"{'above' if modelled >= base_rate else 'below'} the portfolio base rate "
                f"({base_rate:.0%}).")

        urgency = signals.get("customer_urgency")
        if urgency in {"medium", "high"}:
            up.append(f"Customer urgency reported as {urgency}.")

        intent = signals.get("customer_intent")
        if intent == "positive":
            up.append("Customer has signalled willingness to proceed.")
        elif intent == "negative":
            down.append("Customer has signalled reluctance.")

        sensitivity = signals.get("rate_sensitivity")
        if sensitivity in {"medium", "high"}:
            down.append(f"Rate sensitivity reported as {sensitivity}.")

        expected = signals.get("expected_rate")
        if expected is not None:
            gap = (rate - float(expected)) * 100
            if gap > 5:
                down.append(f"Proposed rate is {gap:.0f} bps above the customer's stated "
                            f"expectation of {float(expected):.2f}%.")
            elif gap < -5:
                up.append(f"Proposed rate is {abs(gap):.0f} bps below the customer's stated "
                          f"expectation of {float(expected):.2f}%.")
            else:
                up.append("Proposed rate is close to the customer's stated expectation.")

        competitor = signals.signals.get("competitor_rate")
        if competitor is not None:
            gap = (rate - float(competitor.value)) * 100
            if gap > 0:
                down.append(f"We are {gap:.0f} bps above a reported competitor rate of "
                            f"{float(competitor.value):.2f}% "
                            f"({'verified' if competitor.reliability == 'verified' else 'unverified'}).")
            else:
                up.append(f"We are {abs(gap):.0f} bps below the reported competitor rate.")

        return {"supporting": up, "reducing": down}

    def _governance(self, signals: AcceptanceSignals) -> dict[str, Any]:
        """Flag uplift that would rest on urgency.

        A customer who needs the money most is not a customer who should be
        charged more for it. This does not block anything; it routes the decision
        to the review path that already exists for commercial uplift.
        """
        urgency = signals.get("customer_urgency")
        timeline = signals.get("timeline_sensitivity")
        pressure = max(LEVEL_ORDER.get(urgency, -1), LEVEL_ORDER.get(timeline, -1))
        flagged = pressure >= LEVEL_ORDER["high"]
        return {
            "urgency_pressure_present": pressure >= LEVEL_ORDER["medium"],
            "urgency_review_required": flagged,
            "max_urgency_attributed_uplift_bps": self.governance.get(
                "max_urgency_attributed_uplift_bps", 10),
            "reason": self.governance.get("urgency_review_reason") if flagged else None,
            "note": ("High urgency detected. Uplift attributed to urgency is capped and routed "
                     "for review: pricing on urgency is pricing on vulnerability.") if flagged else None,
        }
