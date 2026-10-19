"""The canonical signal schema shared by every acceptance agent.

One rule runs through this module: a signal that cannot say where it came from
does not exist. Every ``Signal`` carries its source, its reliability tier and -
when it was read out of free text - the verbatim span it came from, so a
reviewer can check the claim against the document rather than trusting it.

Absence is represented explicitly. ``AcceptanceSignals.missing`` lists what was
looked for and not found, because "no competitor offer was mentioned" and "we
never asked" are different facts and the evidence score treats them differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Reliability tiers, strongest first (FR-05).
RELIABILITY_ORDER = ["verified", "customer_reported", "salesperson_estimate", "market_benchmark"]

LEVELS = ["low", "medium", "high"]
INTENTS = ["negative", "neutral", "positive"]

# The closed vocabulary the extractor may emit. Anything else is discarded:
# an open schema is how an LLM quietly invents a new field.
SIGNAL_SPEC: dict[str, dict[str, Any]] = {
    "customer_urgency":     {"type": "level",  "allowed": LEVELS},
    "rate_sensitivity":     {"type": "level",  "allowed": LEVELS},
    "timeline_sensitivity": {"type": "level",  "allowed": LEVELS},
    "customer_intent":      {"type": "level",  "allowed": INTENTS},
    "expected_rate":        {"type": "rate",   "min": 5.0, "max": 25.0},
    "competitor_rate":      {"type": "rate",   "min": 5.0, "max": 25.0},
    "expected_emi":         {"type": "amount", "min": 0.0, "max": 10_000_000.0},
    "required_amount":      {"type": "amount", "min": 0.0, "max": 500_000_000.0},
    "competitor_present":   {"type": "flag"},
    "competitor_name":      {"type": "text"},
}


@dataclass
class Signal:
    """One extracted business signal, with its provenance."""

    name: str
    value: Any
    source: str
    reliability: str
    quote: str | None = None
    extractor: str = "rule"
    age_days: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AcceptanceSignals:
    """Everything the extraction step produced for one application."""

    signals: dict[str, Signal] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    sources_unavailable: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ access
    def get(self, name: str, default: Any = None) -> Any:
        signal = self.signals.get(name)
        return default if signal is None else signal.value

    def has(self, name: str) -> bool:
        return name in self.signals

    def add(self, signal: Signal) -> "AcceptanceSignals":
        """Keep the more reliable of two claims about the same thing.

        Two sources disagreeing is normal - the customer says 9.10%, the
        application record says 9.25%. The stronger tier wins, and the weaker
        claim is kept in ``rejected`` so the disagreement stays visible.
        """
        existing = self.signals.get(signal.name)
        if existing is not None:
            if _rank(existing.reliability) <= _rank(signal.reliability):
                self.rejected.append({"reason": "weaker_source", **signal.as_dict()})
                return self
            self.rejected.append({"reason": "superseded", **existing.as_dict()})
        self.signals[signal.name] = signal
        return self

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe form - this is what gets written to the decision record."""
        return {
            "signals": {name: s.as_dict() for name, s in self.signals.items()},
            "missing": list(self.missing),
            "sources_used": list(self.sources_used),
            "sources_unavailable": list(self.sources_unavailable),
            "rejected": list(self.rejected),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AcceptanceSignals":
        """Rebuild from the stored record, so an audit replays from what was
        persisted instead of re-calling the model (gap doc section 10.4)."""
        obj = cls(
            missing=list(payload.get("missing", [])),
            sources_used=list(payload.get("sources_used", [])),
            sources_unavailable=list(payload.get("sources_unavailable", [])),
            rejected=list(payload.get("rejected", [])),
        )
        for name, raw in (payload.get("signals") or {}).items():
            obj.signals[name] = Signal(**raw)
        return obj


def _rank(reliability: str) -> int:
    try:
        return RELIABILITY_ORDER.index(reliability)
    except ValueError:
        return len(RELIABILITY_ORDER)


def coerce(name: str, value: Any) -> Any | None:
    """Validate one raw extracted value against the schema.

    Returns ``None`` when the value is unusable, which is how an out-of-range
    rate or an invented enum level gets dropped rather than priced.
    """
    spec = SIGNAL_SPEC.get(name)
    if spec is None or value is None:
        return None

    kind = spec["type"]
    if kind in {"level"}:
        text = str(value).strip().lower()
        return text if text in spec["allowed"] else None

    if kind in {"rate", "amount"}:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number != number:  # NaN
            return None
        return number if spec["min"] <= number <= spec["max"] else None

    if kind == "flag":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
        return None

    if kind == "text":
        text = str(value).strip()
        return text[:80] or None

    return None
