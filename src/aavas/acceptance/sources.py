"""Configurable data sources for the acceptance agents.

The scoring path must work with one source, several, or none. Sources are
declared in ``config/acceptance.yaml`` and each is independently switchable, so
adding a CRM feed later is a config change rather than a code change.

Two kinds:

* ``structured`` - already-typed fields on the application record. These become
  signals directly; there is nothing to reason about.
* ``text`` - free text for the extraction agent to read.

A source that fails is recorded as unavailable and the run continues. Losing a
transcript feed must never take a pricing screen down, which is the same rule
the sales-note retriever already follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import load
from ..logging_utils import get_logger
from .signals import Signal, coerce

log = get_logger(__name__)


@dataclass
class SourceDocument:
    """One piece of free text, with where it came from and how old it is."""

    text: str
    source: str
    reliability: str
    age_days: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Collected:
    documents: list[SourceDocument] = field(default_factory=list)
    direct: list[Signal] = field(default_factory=list)
    used: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- sources
class DataSource:
    """Base class. ``name`` and ``reliability`` come from config."""

    kind = "text"

    def __init__(self, name: str, settings: dict[str, Any]):
        self.name = name
        self.settings = settings
        self.reliability = settings.get("reliability", "salesperson_estimate")

    def fetch(self, twin, context: dict[str, Any]) -> list[SourceDocument] | list[Signal]:
        raise NotImplementedError


class StructuredApplicationSource(DataSource):
    """Competitor and negotiation fields already carried on the application.

    These are the most reliable signals available and they need no reasoning, so
    they are emitted directly. An LLM should never be asked to re-read a number
    the system already holds in a typed column.
    """

    kind = "structured"

    FIELDS = {
        "competitor_rate": "competitor_rate",
        "competitor_name": "competitor_name",
        "required_amount": "sanctioned_loan_amount",
    }

    def fetch(self, twin, context: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        verified = bool(twin.get("competitor_evidence_verified", False))
        # An unverified competitor claim is exactly that - the tier records it,
        # and constraints.py already caps what an unevidenced claim may move.
        tier = "verified" if verified else "customer_reported"

        for signal_name, column in self.FIELDS.items():
            value = coerce(signal_name, twin.get(column))
            if value is None:
                continue
            out.append(Signal(name=signal_name, value=value, source=self.name,
                              reliability=tier if "competitor" in signal_name else "verified",
                              extractor="structured", age_days=0.0))

        if twin.get("competitor_rate") is not None:
            out.append(Signal(name="competitor_present", value=True, source=self.name,
                              reliability=tier, extractor="structured", age_days=0.0))

        if twin.get("customer_negotiating"):
            out.append(Signal(name="rate_sensitivity", value="high", source=self.name,
                              reliability="verified", extractor="structured", age_days=0.0))
        return out


class InlineNotesSource(DataSource):
    """Free text handed in on the scoring request - a CRM comment, a branch note."""

    def fetch(self, twin, context: dict[str, Any]) -> list[SourceDocument]:
        raw = context.get("notes") or context.get("inline_notes")
        if not raw:
            return []
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        return [
            SourceDocument(text=str(item), source=self.name, reliability=self.reliability,
                           age_days=context.get("notes_age_days"))
            for item in items if str(item).strip()
        ]


class TranscriptSource(DataSource):
    """Call transcript or chat thread passed on the request.

    Off by default: there is no transcription feed wired up yet, and a source
    that is enabled but never populated would quietly inflate the evidence score.
    """

    def __init__(self, name: str, settings: dict[str, Any], key: str):
        super().__init__(name, settings)
        self.key = key

    def fetch(self, twin, context: dict[str, Any]) -> list[SourceDocument]:
        raw = context.get(self.key)
        if not raw:
            return []
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        return [
            SourceDocument(text=str(item), source=self.name, reliability=self.reliability,
                           age_days=context.get(f"{self.key}_age_days"))
            for item in items if str(item).strip()
        ]


class SalesNoteHistorySource(DataSource):
    """Comparable past negotiations from the knowledge store.

    This is market context, not a claim about *this* customer, so it carries the
    weakest tier. It informs; it does not evidence.
    """

    def fetch(self, twin, context: dict[str, Any]) -> list[SourceDocument]:
        retriever = context.get("sales_note_retriever")
        if retriever is None:
            return []
        k = int(self.settings.get("max_documents", 3))
        hits = retriever.similar(twin, k=k)
        return [
            SourceDocument(text=str(hit.get("text", "")), source=self.name,
                           reliability=self.reliability, age_days=None,
                           metadata={"doc_id": hit.get("doc_id"), "outcome": hit.get("outcome")})
            for hit in hits if str(hit.get("text", "")).strip()
        ]


# -------------------------------------------------------------------- registry
class McpSource(DataSource):
    """A communication channel reached over MCP: WhatsApp, Gmail, Outlook.

    The class is thin on purpose. Everything that decides whether this may run -
    is it switched on, is the server reachable, is consent on record - lives in
    ``connectors.registry`` and is checked before a message is requested, not
    after. What lands here is text, which goes to the extraction agent and comes
    back as signals with a verbatim quote; a connector can never put a value
    straight into a model feature.

    When the connector is blocked or the server silent, this returns nothing and
    the registry records the source as *unavailable*. That distinction is what
    keeps the acceptance score honest: a source that was asked and answered
    nothing is evidence of absence; one that was never asked is not.
    """

    kind = "text"

    def __init__(self, name: str, settings: dict[str, Any], connector_name: str | None = None):
        super().__init__(name, settings)
        self.connector_name = connector_name or settings.get("connector") or name

    def fetch(self, twin, context: dict[str, Any]) -> list[SourceDocument]:
        from ..agents import capabilities
        from ..connectors import all_connectors, call, consent_backend

        # A fourth gate, before the three in the registry: is the agent asking
        # even permitted to reach a connector? This is the one path that reads a
        # customer's own words, so the permission is checked rather than assumed.
        agent = str(context.get("agent") or "extraction")
        try:
            capabilities.check(agent, "connector.mcp", capabilities.READ)
        except PermissionError as exc:
            log.warning("acceptance: %s", exc)
            return []

        connector = all_connectors().get(self.connector_name)
        if connector is None:
            log.warning("acceptance: source %r names unknown connector %r",
                        self.name, self.connector_name)
            return []

        blocked = connector.blocked_reason(consent_backend())
        if blocked:
            log.info("acceptance: connector %r not collected - %s", connector.name, blocked)
            return []
        if connector.consumes != "acceptance_signals":
            # A Salesforce record is not a customer's words. Reading one here
            # would be extracting intent from a typed column.
            log.warning("acceptance: connector %r supplies %r, not acceptance signals",
                        connector.name, connector.consumes)
            return []

        result = call(connector, {**context, "loan_id": twin.get("loan_id")})
        if not result.available:
            log.info("acceptance: connector %r unavailable - %s", connector.name, result.reason)
            return []

        limit = int(self.settings.get("max_documents", 5))
        return [SourceDocument(text=str(doc.get("text", ""))[:12000],
                               source=self.name, reliability=self.reliability,
                               age_days=doc.get("age_days"),
                               metadata={"connector": connector.name,
                                         "server": result.server, "tool": result.tool})
                for doc in result.documents[:limit] if doc.get("text")]


_BUILDERS = {
    "structured_application": lambda n, s: StructuredApplicationSource(n, s),
    "inline_notes": lambda n, s: InlineNotesSource(n, s),
    "sales_note_history": lambda n, s: SalesNoteHistorySource(n, s),
    "call_transcript": lambda n, s: TranscriptSource(n, s, "call_transcript"),
    "whatsapp_thread": lambda n, s: TranscriptSource(n, s, "whatsapp_thread"),
    # Reached over MCP; each names a connector in config/connectors.yaml.
    "whatsapp": lambda n, s: McpSource(n, s, "whatsapp"),
    "gmail": lambda n, s: McpSource(n, s, "gmail"),
    "outlook": lambda n, s: McpSource(n, s, "outlook"),
}


class SourceRegistry:
    """Builds the enabled sources from config and collects from all of them."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config if config is not None else load("acceptance")
        self.settings = cfg.get("sources", {})
        self.sources: list[DataSource] = []
        for name, settings in self.settings.items():
            if not settings.get("enabled", False):
                continue
            builder = _BUILDERS.get(name)
            if builder is None:
                log.warning("acceptance: unknown source %r in config; skipped", name)
                continue
            self.sources.append(builder(name, settings))

    @property
    def enabled_names(self) -> list[str]:
        return [s.name for s in self.sources]

    def collect(self, twin, context: dict[str, Any] | None = None) -> Collected:
        """Pull from every enabled source. Failures are recorded, never raised."""
        context = context or {}
        out = Collected()

        for source in self.sources:
            try:
                result = source.fetch(twin, context)
            except Exception as exc:  # noqa: BLE001 - a bad feed must not stop pricing
                log.warning("acceptance: source %r failed: %s", source.name, exc)
                out.unavailable.append(source.name)
                continue

            if not result:
                out.unavailable.append(source.name)
                continue

            if source.kind == "structured":
                out.direct.extend(result)
            else:
                out.documents.extend(result)
            out.used.append(source.name)

        # A configured-but-disabled source is not "unavailable" - nobody asked
        # for it. Only enabled sources that returned nothing count against
        # evidence, which is what keeps the score honest.
        return out
