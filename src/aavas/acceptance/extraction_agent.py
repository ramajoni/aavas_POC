"""Extraction agent - unstructured text to standardised signals.

This agent does the one job the business statement says structured data cannot:
read what the sales officer actually wrote and turn it into typed signals.

Three rules define it, and all three are controls rather than preferences:

1. **It never produces a probability.** The LLM proposes signals; the fitted,
   isotonically calibrated conversion model turns them into an acceptance
   probability. An 82% produced by a language model is a number with no
   frequency behind it (gap doc section 10.3).
2. **It never invents.** The output schema is closed, every value is range- and
   enum-checked, and each claim must quote a span that appears verbatim in the
   source. A quote that is not in the document means the signal is dropped.
3. **It always has a fallback.** With no key, no network or a broken SDK, the
   deterministic rule extractor runs instead and the payload says which was used.
   Nothing in this repository fails because an optional dependency is missing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import env, env_bool, load
from ..logging_utils import get_logger
from .signals import SIGNAL_SPEC, AcceptanceSignals, Signal, coerce
from .sources import Collected, SourceDocument

log = get_logger(__name__)

SYSTEM_PROMPT = """You read Indian home-loan sales notes and extract only what is explicitly stated.

You are an extraction tool, not an analyst. You do not estimate, infer, average or guess.
If the text does not state something, you omit that field entirely. Omission is correct
and expected; a plausible-sounding invented value is a serious error.

Return ONLY a JSON object of this shape, with no commentary:

{"signals": [{"name": "<field>", "value": <value>, "quote": "<verbatim span>"}]}

Allowed fields and values:
  customer_urgency      "low" | "medium" | "high"    how soon they need disbursement
  timeline_sensitivity  "low" | "medium" | "high"    how hard their deadline is
  rate_sensitivity      "low" | "medium" | "high"    how much the rate drives their decision
  customer_intent       "negative" | "neutral" | "positive"   willingness to proceed
  expected_rate         number, % p.a.               the rate the customer says they want
  competitor_rate       number, % p.a.               a rate another lender has quoted
  competitor_present    true | false                 another lender is involved
  competitor_name       string                       that lender's name
  expected_emi          number, rupees               the EMI the customer says they want
  required_amount       number, rupees               the loan amount they need

"quote" must be copied character-for-character from the source text. Do not paraphrase,
do not tidy punctuation, do not translate. A quote that does not appear in the source
causes the signal to be discarded.

Convert Indian number words in the VALUE only: "25 lakh" -> 2500000, "1.2 crore" -> 12000000.
The quote still shows the original wording.
"""

# ------------------------------------------------------------------- rule mode
_RATE = re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*%")
_LAKH = re.compile(r"(?:(?:rs|inr|₹)\s*)?(\d+(?:\.\d+)?)\s*(lakh|lac|crore|cr)\b", re.I)

_URGENCY = {
    "high": ["urgent", "urgently", "immediately", "asap", "before month-end", "before month end",
             "this week", "needs disbursement", "in a hurry", "time critical"],
    "medium": ["soon", "next month", "shortly", "prefers early"],
}
_SENSITIVITY = {
    "high": ["rate sensitive", "only on rate", "lowest rate", "will move for", "bps matters",
             "wants a lower rate", "negotiating hard", "price sensitive"],
    "medium": ["would like a better rate", "asked for a discount", "comparing rates"],
}
_INTENT = {
    "positive": ["ready to proceed", "wants to proceed", "happy to", "will accept", "keen",
                 "agreed", "confirmed", "acceptable"],
    "negative": ["not interested", "declined", "going with", "dropping", "withdrew", "not happy"],
}
_COMPETITORS = ["hdfc", "lic housing", "sbi", "icici", "axis", "bajaj", "pnb housing",
                "indiabulls", "tata capital", "piramal", "shriram", "canfin", "another lender"]


def _find_phrase(text: str, phrases: list[str]) -> str | None:
    low = text.lower()
    for phrase in phrases:
        if phrase in low:
            start = low.index(phrase)
            return text[start:start + len(phrase)]
    return None


def _to_rupees(number: float, unit: str) -> float:
    return number * (10_000_000 if unit.lower().startswith("cr") else 100_000)


class RuleExtractor:
    """Keyword and pattern extraction. Deliberately conservative.

    It is better for this to stay silent and let the evidence score fall than to
    guess a rate. Everything it emits carries the matched span as its quote, so
    rule-derived signals are auditable on exactly the same terms as LLM ones.
    """

    name = "rule"

    def extract(self, doc: SourceDocument) -> list[Signal]:
        text = doc.text
        out: list[Signal] = []

        def emit(name: str, value: Any, quote: str | None) -> None:
            checked = coerce(name, value)
            if checked is not None:
                out.append(Signal(name=name, value=checked, source=doc.source,
                                  reliability=doc.reliability, quote=quote,
                                  extractor=self.name, age_days=doc.age_days))

        for name, table in (("customer_urgency", _URGENCY), ("rate_sensitivity", _SENSITIVITY),
                            ("customer_intent", _INTENT)):
            for level, phrases in table.items():
                hit = _find_phrase(text, phrases)
                if hit:
                    emit(name, level, hit)
                    break

        competitor = _find_phrase(text, _COMPETITORS)
        if competitor:
            emit("competitor_present", True, competitor)
            if competitor != "another lender":
                emit("competitor_name", competitor, competitor)

        rates = _RATE.findall(text)
        if rates:
            # Without a named subject, a bare percentage is ambiguous. Only read
            # it as a competitor quote when a competitor is actually mentioned.
            span = _RATE.search(text)
            if competitor:
                emit("competitor_rate", float(rates[0]), span.group(0) if span else None)
            elif len(rates) == 1:
                emit("expected_rate", float(rates[0]), span.group(0) if span else None)

        amount = _LAKH.search(text)
        if amount:
            emit("required_amount", _to_rupees(float(amount.group(1)), amount.group(2)),
                 amount.group(0))
        return out


# -------------------------------------------------------------------- LLM mode
class ClaudeExtractor:
    """Claude-backed extraction, with the verbatim-quote check applied to output."""

    name = "claude"

    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.model = env("AAVAS_LLM_MODEL") or settings.get("model", "claude-opus-5")
        self._client = None

    def available(self) -> bool:
        """Probe once. Catches Exception, not ImportError: a broken SDK build
        should degrade exactly like a missing one rather than raise at score time."""
        if self._client is not None:
            return True
        if not env("ANTHROPIC_API_KEY"):
            log.info("acceptance: ANTHROPIC_API_KEY not set; using rule extractor")
            return False
        try:
            import anthropic

            self._client = anthropic.Anthropic(
                timeout=float(self.settings.get("timeout_seconds", 20)))
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("acceptance: anthropic SDK unavailable (%s); using rule extractor", exc)
            return False

    def extract(self, doc: SourceDocument) -> list[Signal]:
        if not self.available():
            return []
        text = doc.text[: int(self.settings.get("max_input_chars", 12000))]
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=int(self.settings.get("max_tokens", 1500)),
                temperature=float(self.settings.get("temperature", 0.0)),
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"<sales_note>\n{text}\n</sales_note>"}],
            )
            payload = "".join(block.text for block in response.content
                              if getattr(block, "type", None) == "text")
        except Exception as exc:  # noqa: BLE001 - never take a pricing screen down
            log.warning("acceptance: extraction call failed (%s); falling back", exc)
            return []

        return self._parse(payload, doc)

    def _parse(self, payload: str, doc: SourceDocument) -> list[Signal]:
        raw = _json_object(payload)
        if raw is None:
            log.warning("acceptance: extractor returned unparseable output; discarded")
            return []

        out: list[Signal] = []
        for item in raw.get("signals", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name not in SIGNAL_SPEC:
                continue  # closed vocabulary: an unknown field is a hallucinated field
            value = coerce(name, item.get("value"))
            if value is None:
                continue
            quote = item.get("quote")
            quote = str(quote) if quote else None
            if self.settings.get("require_verbatim_quote", True):
                if not quote or quote not in doc.text:
                    log.info("acceptance: dropped %r - quote not found verbatim in source", name)
                    continue
            out.append(Signal(name=name, value=value, source=doc.source,
                              reliability=doc.reliability, quote=quote,
                              extractor=self.name, age_days=doc.age_days))
        return out


def _json_object(payload: str) -> dict | None:
    """Pull the JSON object out of a response that may be fenced or padded."""
    text = payload.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ----------------------------------------------------------------------- agent
class ExtractionAgent:
    """Runs the configured extractor over every collected document."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config if config is not None else load("acceptance")
        self.settings = cfg.get("extraction", {})
        # Env wins over config so the LLM can be switched on without a file edit.
        self.llm_enabled = env_bool("AAVAS_LLM_ENABLED", self.settings.get("llm_enabled", False))
        self.rule = RuleExtractor()
        self.claude = ClaudeExtractor(self.settings) if self.llm_enabled else None

    @property
    def mode(self) -> str:
        if self.claude is not None and self.claude.available():
            return "claude"
        return "rule"

    def run(self, collected: Collected) -> AcceptanceSignals:
        result = AcceptanceSignals(
            sources_used=list(collected.used),
            sources_unavailable=list(collected.unavailable),
        )

        # Typed fields first: they carry the strongest tier, so anything a
        # document later claims about the same field has to beat them.
        for signal in collected.direct:
            result.add(signal)

        mode = self.mode
        for doc in collected.documents:
            extracted: list[Signal] = []
            if mode == "claude":
                extracted = self.claude.extract(doc)
                if not extracted:
                    extracted = self.rule.extract(doc)  # degraded, not silent
            else:
                extracted = self.rule.extract(doc)
            for signal in extracted:
                result.add(signal)

        result.missing = [name for name in SIGNAL_SPEC if name not in result.signals]
        return result
