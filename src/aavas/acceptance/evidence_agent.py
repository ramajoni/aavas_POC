"""Evidence agent - how much the acceptance estimate should be trusted.

This is the second number the business statement asks for, and it is deliberately
separate from the first. An acceptance probability of 82% built on one stale note
and no competitor information is a different object from the same 82% built on a
verified competitor letter and a call from this morning.

The shape mirrors the Stage-1 confidence score that already ships in
``features/stage1_features.py`` - part "how much of what we wanted is present",
part "how good is what is present" - so the two numbers on the two screens mean
the same kind of thing.
"""

from __future__ import annotations

import math
from typing import Any

from ..config import load
from ..logging_utils import get_logger
from .signals import AcceptanceSignals

log = get_logger(__name__)


class EvidenceAgent:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config if config is not None else load("acceptance")
        self.signal_weights = {k: float(v.get("weight", 0.0))
                               for k, v in cfg.get("signals", {}).items()}
        self.settings = cfg.get("evidence", {})
        self.reliability_score = self.settings.get("reliability_score", {})

    # ------------------------------------------------------------------ pieces
    def _recency(self, age_days: float | None) -> float:
        """Exponential decay. A note with no timestamp is treated as moderately
        old rather than fresh, so missing metadata cannot inflate confidence."""
        half_life = float(self.settings.get("recency_half_life_days", 45))
        if age_days is None:
            age_days = float(self.settings.get("unknown_age_days", 30))
        if half_life <= 0:
            return 1.0
        return float(0.5 ** (max(age_days, 0.0) / half_life))

    def _tier(self, reliability: str) -> float:
        return float(self.reliability_score.get(reliability, 0.4))

    # ------------------------------------------------------------------- score
    def score(self, signals: AcceptanceSignals) -> dict[str, Any]:
        total_weight = sum(self.signal_weights.values()) or 1.0

        covered = 0.0
        quality_numerator = 0.0
        detail: dict[str, dict[str, Any]] = {}

        for name, weight in self.signal_weights.items():
            signal = signals.signals.get(name)
            if signal is None:
                continue
            tier = self._tier(signal.reliability)
            recency = self._recency(signal.age_days)
            covered += weight
            quality_numerator += weight * tier * recency
            detail[name] = {
                "reliability": signal.reliability,
                "tier_score": round(tier, 3),
                "recency_score": round(recency, 3),
                "source": signal.source,
                "extractor": signal.extractor,
            }

        coverage = covered / total_weight
        # Quality is averaged over what is present, not over everything we wanted:
        # otherwise coverage would be penalised twice.
        quality = (quality_numerator / covered) if covered > 0 else 0.0

        raw = (float(self.settings.get("coverage_weight", 0.55)) * coverage
               + float(self.settings.get("quality_weight", 0.45)) * quality)
        confidence = min(max(raw, float(self.settings.get("floor", 0.15))),
                         float(self.settings.get("ceiling", 0.95)))

        return {
            "evidence_confidence": round(confidence, 3),
            "coverage": round(coverage, 3),
            "quality": round(quality, 3),
            "signals_present": sorted(detail),
            "signals_missing": [n for n in self.signal_weights if n not in detail],
            "sources_used": list(signals.sources_used),
            "sources_unavailable": list(signals.sources_unavailable),
            "per_signal": detail,
            "caveats": self._caveats(signals, coverage),
        }

    def _caveats(self, signals: AcceptanceSignals, coverage: float) -> list[str]:
        """Plain sentences a branch user can read, not just numbers."""
        out: list[str] = []
        if not signals.has("competitor_rate"):
            out.append("No competitor offer is recorded for this customer.")
        elif (s := signals.signals.get("competitor_rate")) and s.reliability != "verified":
            out.append("The competitor rate is unverified; it is a reported claim, not evidence.")
        if not signals.sources_used:
            out.append("No context source returned anything; this rests on structured data alone.")
        if coverage < 0.4:
            out.append("Fewer than half the expected context signals are present.")
        if signals.rejected:
            out.append(f"{len(signals.rejected)} conflicting or weaker claim(s) were set aside.")
        return out
