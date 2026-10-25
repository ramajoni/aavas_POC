"""Sales notes -> competitor mentions, quoted rates, evidence and customer signals.

What one note says, read deterministically and kept traceable:

* **competitor mentions** - canonical lender, the alias exactly as written and
  its span, and how it matched (exact / ambiguous / fuzzy / unrecognised);
* **the rate attached to that mention**, parsed from the formats branches use
  ("8.5%", "8.50 pc", "@8.50", "ROI 8.50", "8.50 percent", "11.20% p.a.");
* **the role of every rate** - a competitor quote, the customer's own ask
  ("wants 12.96% or close", "firm on 10.77%"), our rate ("worth retaining at
  ROI 11.74", "Aavas offered"), or unattributed. A number is only a competitor
  quote when the text ties it to a lender; a bare percentage is never guessed;
* **evidence per mention** - documented (sanction / offer letter shared or on
  file), undocumented (verbal, WhatsApp screenshot, "no document", "claims",
  "as per customer") or unknown, with the cue that decided it;
* **customer signals** - urgency, rate sensitivity, intent, amount - by reusing
  the acceptance ``ExtractionAgent`` rule extractor unchanged.

Two entry points. ``extract_current`` reads the notes on the file being priced
and merges them (latest note wins, every claim keeps its quote and note index)
and flags where they contradict the structured sales context. ``load_history``
runs the same reader over every historical note and caches a per-note table.

What this module may read is fixed by ``INPUT_COLUMNS``. The notes file carries
latent truth columns (``true_*``) because the data is synthetic; they exist to
*score* this extractor (``quality.py``) and are dropped here before any text is
read, so the score cannot leak into what it measures.

The LLM is optional and narrow: when the acceptance agent's LLM mode is on and a
key is present, the *current-loan* path may use it for customer signals only,
quote-verified exactly as that agent enforces. Competitor identification, rates
and evidence are always the deterministic reader, and the historical batch never
calls the LLM (46k notes is a cost, not a demo). Every payload says which ran.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable

import pandas as pd

from ..acceptance.signals import coerce
from ..acceptance.sources import SourceDocument
from ..config import model_config
from ..logging_utils import get_logger
from ..paths import PROCESSED_DIR, resolve
from .registry import (SELF, UNRECOGNISED, CompetitorRegistry, Mention, config_fingerprint,
                       default_registry, load_config, sentences)

log = get_logger(__name__)

EXTRACTOR_VERSION = "note-x-1"

# The only columns the extractor reads. Everything else - including the latent
# truth columns - is dropped before a single note is parsed.
INPUT_COLUMNS = ["note_id", "loan_id", "note_date", "application_date", "note_type",
                 "city", "state", "branch_id", "note_text"]
LATENT_PREFIX = "true_"

SIGNALS_FILE = "note_signals.csv"
MENTIONS_FILE = "note_mentions.csv"

DOCUMENTED, UNDOCUMENTED, UNKNOWN = "documented", "undocumented", "unknown"
QUOTE, ASK, OURS, UNATTRIBUTED = "competitor_quote", "customer_ask", "our_rate", "unattributed"

# Customer-side signals taken from the acceptance rule extractor. Its competitor
# fields are not used: its lender list is a dozen names and it reads any lone
# percentage as the customer's ask, which is exactly the confusion this module
# exists to avoid.
CUSTOMER_SIGNALS = ("customer_urgency", "timeline_sensitivity", "rate_sensitivity",
                    "customer_intent", "required_amount", "expected_emi")

# ------------------------------------------------------------------- rates
_NUM = r"(?<![\d.])(\d{1,2}(?:\.\d{1,3})?)"
_NOT_A_RATE = (r"(?!\s*(?:bps|bp\b|basis|lakh|lac\b|lacs|cr\b|crore|k\b|years?\b|yrs?\b|months?\b"
               r"|days?\b|x\b|times\b|th\b|st\b|nd\b|rd\b))(?![\d]|\.\d)")
_UNIT = r"\s*(?:%|pc\b|pct\b|percent\b|per\s*cent\b)"
_PA = r"(?:\s*(?:p\.\s?a\.?|pa\b|per\s+annum))?"
_RATE = re.compile(
    rf"@\s*{_NUM}{_NOT_A_RATE}(?:{_UNIT})?{_PA}"
    rf"|\b(?:roi|rate\s+of\s+interest|interest\s+rate|rate)\b\s*(?:of|at|is|=|:|-|@)?\s*{_NUM}{_NOT_A_RATE}(?:{_UNIT})?{_PA}"
    rf"|{_NUM}{_NOT_A_RATE}{_UNIT}{_PA}",
    re.I)
_CLAUSE = re.compile(r"[,;:]\s*|\.\s+|\s+(?:and|but|also|while|whereas|aur|lekin|par)\s+|\s+-\s+", re.I)
_FOLLOWING_LINK = re.compile(r"^\s*(?:from|by|with|of|at|offered\s+by|quoted\s+by|frm)\s+$", re.I)


@dataclass
class RateToken:
    value: float
    text: str
    start: int
    end: int
    role: str = UNATTRIBUTED
    cue: str | None = None
    number_start: int | None = None     # where the digits begin: "our rate 9.9%" keeps "our rate" in the lead-in


def parse_rates(text: str) -> list[RateToken]:
    """Every plausible annual rate in ``text`` (5-25% p.a., the signal schema's
    range). Basis points, amounts and tenors are excluded by construction."""
    out = []
    for match in _RATE.finditer(text or ""):
        group = next(i for i, g in enumerate(match.groups(), start=1) if g is not None)
        value = coerce("competitor_rate", match.group(group))
        if value is None:
            continue
        out.append(RateToken(round(float(value), 3), match.group(0).strip(), match.start(), match.end(),
                             number_start=match.start(group)))
    return out


# ------------------------------------------------------------------ cues
def _cue_regex(phrases: Iterable[str]) -> re.Pattern | None:
    items = sorted({p.lower() for p in phrases if p}, key=len, reverse=True)
    if not items:
        return None
    return re.compile(r"(?<![a-z])(" + "|".join(re.escape(p) for p in items) + r")(?![a-z])", re.I)


class _Cues:
    def __init__(self, cfg: dict[str, Any]):
        ev = cfg.get("evidence_cues") or {}
        self.undocumented = _cue_regex(ev.get("undocumented") or [])
        self.documented = _cue_regex(ev.get("documented") or [])
        self.hearsay = _cue_regex(ev.get("hearsay") or [])
        self.ask = [c.lower() for c in cfg.get("customer_ask_cues") or []]
        self.ours = [c.lower() for c in cfg.get("our_rate_cues") or []]


# ---------------------------------------------------------------- results
@dataclass
class CompetitorClaim:
    """One mention, with its rate and evidence. ``sentence`` is the quote a
    reviewer checks the claim against."""

    competitor: str | None
    lender_type: str | None
    alias: str
    match: str
    span: tuple[int, int]
    rate: float | None = None
    rate_text: str | None = None
    rate_span: tuple[int, int] | None = None
    evidence: str = UNKNOWN
    evidence_cue: str | None = None
    sentence: str = ""
    fuzzy_ratio: float | None = None
    note_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["span"] = list(self.span)
        out["rate_span"] = list(self.rate_span) if self.rate_span else None
        return out


@dataclass
class NoteExtraction:
    text: str
    note_index: int | None = None
    mentions: list[CompetitorClaim] = field(default_factory=list)
    unrecognised: list[CompetitorClaim] = field(default_factory=list)
    customer_asks: list[dict[str, Any]] = field(default_factory=list)
    our_rates: list[dict[str, Any]] = field(default_factory=list)
    unattributed_rates: list[dict[str, Any]] = field(default_factory=list)
    customer_signals: dict[str, dict[str, Any]] = field(default_factory=dict)
    signals_method: str = "rule"

    def competitors(self) -> dict[str, CompetitorClaim]:
        """One claim per canonical lender in this note: the first mention that
        carries a rate (else the first), with the strongest evidence any mention
        of that lender in the note states. Returns copies; ``mentions`` keeps
        each mention exactly as read."""
        out: dict[str, CompetitorClaim] = {}
        for claim in self.mentions:
            current = out.get(claim.competitor)
            if current is None:
                out[claim.competitor] = replace(claim)
            elif current.rate is None and claim.rate is not None:
                merged = replace(claim)
                if merged.evidence == UNKNOWN:
                    merged.evidence, merged.evidence_cue = current.evidence, current.evidence_cue
                out[claim.competitor] = merged
            elif current.evidence == UNKNOWN and claim.evidence != UNKNOWN:
                current.evidence, current.evidence_cue = claim.evidence, claim.evidence_cue
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "note_index": self.note_index,
            "text": self.text,
            "mentions": [m.as_dict() for m in self.mentions],
            "unrecognised_lenders": [m.as_dict() for m in self.unrecognised],
            "customer_asks": self.customer_asks,
            "our_rates": self.our_rates,
            "unattributed_rates": self.unattributed_rates,
            "customer_signals": self.customer_signals,
            "signals_method": self.signals_method,
        }


# ---------------------------------------------------------------- extractor
class NoteExtractor:
    """Deterministic reader. Construct once; ``extract`` is pure per note."""

    def __init__(self, registry: CompetitorRegistry | None = None, agent=None,
                 allow_llm: bool = False):
        self.registry = registry or default_registry()
        self.cues = _Cues(self.registry.config if registry is not None else load_config())
        # The acceptance agent supplies the customer-signal reader. Imported here
        # rather than reimplemented: one set of phrases, one place to change them.
        if agent is None:
            from ..acceptance.extraction_agent import ExtractionAgent
            agent = ExtractionAgent()
        self.agent = agent
        self.allow_llm = allow_llm

    @property
    def signals_mode(self) -> str:
        if self.allow_llm and getattr(self.agent, "llm_enabled", False) and self.agent.mode == "claude":
            return "claude"
        return "rule"

    # -------------------------------------------------------------- one note
    def extract(self, text: str, note_index: int | None = None,
                with_signals: bool = True) -> NoteExtraction:
        text = str(text or "")
        result = NoteExtraction(text=text, note_index=note_index)
        if not text.strip():
            return result
        spans = sentences(text)
        found = self.registry.find(text)
        rates = parse_rates(text)
        claims = [CompetitorClaim(m.competitor, m.lender_type, m.alias, m.match, (m.start, m.end),
                                  fuzzy_ratio=m.fuzzy_ratio, note_index=note_index) for m in found]

        self._attach_rates(text, spans, found, claims, rates)
        self._classify_evidence(text, spans, found, claims)

        for mention, claim in zip(found, claims):
            claim.sentence = _sentence_text(text, spans, mention.start)
            if mention.match == SELF:
                if claim.rate is not None:
                    result.our_rates.append({"rate": claim.rate, "quote": claim.rate_text,
                                             "cue": claim.alias, "note_index": note_index})
            elif mention.match == UNRECOGNISED:
                result.unrecognised.append(claim)
            else:
                result.mentions.append(claim)

        for token in rates:
            item = {"rate": token.value, "quote": token.text, "cue": token.cue,
                    "sentence": _sentence_text(text, spans, token.start), "note_index": note_index}
            if token.role == ASK:
                result.customer_asks.append(item)
            elif token.role == OURS and not any(o["quote"] == token.text for o in result.our_rates):
                result.our_rates.append(item)
            elif token.role == UNATTRIBUTED:
                result.unattributed_rates.append(item)

        if with_signals:
            result.customer_signals, result.signals_method = self._customer_signals(text)
        return result

    # ------------------------------------------------------------- internals
    def _attach_rates(self, text: str, spans, found: list[Mention], claims: list[CompetitorClaim],
                      rates: list[RateToken]) -> None:
        """Give each rate a role. A rate becomes a competitor quote only when a
        lender is named before it in the same sentence (or right after it, as in
        "9.5% from HDFC"), and only when its lead-in is not the customer's ask or
        our own rate."""
        for token in rates:
            s_start, s_end = _sentence_bounds(spans, token.start)
            before = [i for i, m in enumerate(found) if s_start <= m.start and m.end <= token.start]
            boundary = max([found[i].end for i in before] +
                           [r.end for r in rates if s_start <= r.end <= token.start and r is not token] +
                           [s_start])
            lead_in = text[boundary:token.number_start or token.start].lower()
            ask = next((c for c in self.cues.ask if c in lead_in), None)
            ours = next((c for c in self.cues.ours if c in lead_in), None)
            if ask:
                token.role, token.cue = ASK, ask.strip()
                continue
            if ours:
                token.role, token.cue = OURS, ours.strip()
                continue
            target = next((i for i in reversed(before) if claims[i].rate is None), None)
            if target is None:
                after = [i for i, m in enumerate(found)
                         if token.end <= m.start < s_end and claims[i].rate is None]
                if after and _FOLLOWING_LINK.match(text[token.end:found[after[0]].start] + " "):
                    target = after[0]
            if target is None:
                continue
            claims[target].rate, claims[target].rate_text = token.value, token.text
            claims[target].rate_span = (token.start, token.end)
            if found[target].match == SELF:
                token.role, token.cue = OURS, found[target].alias
            else:
                token.role = QUOTE

    def _classify_evidence(self, text: str, spans, found: list[Mention],
                           claims: list[CompetitorClaim]) -> None:
        """Evidence per mention from the words in its scope: from the start of
        its sentence (or the clause break after the previous mention) to the
        clause break before the next mention - plus a following fragment such as
        "Not verified." that has no subject of its own."""
        order = sorted(range(len(found)), key=lambda i: found[i].start)
        splits: dict[int, int] = {}
        for a, b in zip(order, order[1:]):
            between = text[found[a].end:found[b].start]
            last = None
            for last in _CLAUSE.finditer(between):
                pass
            splits[a] = found[a].end + (last.start() if last else len(between))

        for pos, i in enumerate(order):
            mention = found[i]
            s_start, s_end = _sentence_bounds(spans, mention.start)
            start = s_start
            if pos > 0:
                start = max(start, splits[order[pos - 1]])
            end = _extend_over_fragments(text, spans, s_end)
            if pos + 1 < len(order):
                end = min(end, splits[i])
            scope = text[start:end]
            evidence, cue = self._evidence(scope, mention.start - start)
            claims[i].evidence, claims[i].evidence_cue = evidence, cue

    def _evidence(self, scope: str, anchor: int) -> tuple[str, str | None]:
        for label, regex in ((UNDOCUMENTED, self.cues.undocumented),
                             (DOCUMENTED, self.cues.documented),
                             (UNDOCUMENTED, self.cues.hearsay)):
            if regex is None:
                continue
            hits = list(regex.finditer(scope))
            if label == DOCUMENTED and self.cues.undocumented is not None:
                # "no document shared" must not read as "document shared".
                negated = [(m.start(), m.end()) for m in self.cues.undocumented.finditer(scope)]
                hits = [h for h in hits if not any(h.start() < e and s < h.end() for s, e in negated)]
            if hits:
                nearest = min(hits, key=lambda h: abs(h.start() - anchor))
                return label, nearest.group(0)
        return UNKNOWN, None

    def _customer_signals(self, text: str) -> tuple[dict[str, dict[str, Any]], str]:
        doc = SourceDocument(text=text, source="sales_note", reliability="salesperson_estimate")
        method = "rule"
        extracted = []
        if self.signals_mode == "claude":
            extracted = self.agent.claude.extract(doc)
            method = "claude" if extracted else "rule (claude returned nothing)"
        if not extracted:
            extracted = self.agent.rule.extract(doc)
        out: dict[str, dict[str, Any]] = {}
        for signal in extracted:
            # In rule mode the expected rate comes from the rate-role parser above;
            # Claude's is kept because it is asked specifically for the customer's ask.
            allowed = CUSTOMER_SIGNALS + (("expected_rate",) if method == "claude" else ())
            if signal.name in allowed and signal.name not in out:
                out[signal.name] = {"value": signal.value, "quote": signal.quote,
                                    "extractor": signal.extractor}
        return out, method


def _sentence_bounds(spans, position: int) -> tuple[int, int]:
    for s, e in spans:
        if s <= position < e:
            return s, e
    return spans[-1] if spans else (0, position)


def _sentence_text(text: str, spans, position: int) -> str:
    s, e = _sentence_bounds(spans, position)
    return text[s:e].strip()


def _extend_over_fragments(text: str, spans, end: int) -> int:
    """A trailing fragment of five words or fewer ("Not verified.") belongs to
    the claim before it; a longer sentence is its own statement."""
    for s, e in spans:
        if s < end:
            continue
        if len(text[s:e].split()) <= 5:
            end = e
            continue
        break
    return end


# ------------------------------------------------------------- current loan
def extract_current(notes: list[str], context: dict[str, Any] | None = None, *,
                    extractor: NoteExtractor | None = None, allow_llm: bool = True) -> dict[str, Any]:
    """What the notes on the file being priced say, merged across notes.

    ``notes`` are in the order they were written; the last is the latest and
    wins a conflict, but every claim keeps its note index and quote. ``context``
    is the structured sales context (competitor_name, competitor_rate,
    competitor_evidence_verified); disagreements with it are flagged, never
    resolved silently in either direction.
    """
    extractor = extractor or NoteExtractor(allow_llm=allow_llm)
    context = dict(context or {})
    per_note = [extractor.extract(text, index) for index, text in enumerate(notes or [])]

    merged: dict[str, dict[str, Any]] = {}
    for note in per_note:
        for name, claim in note.competitors().items():
            entry = merged.setdefault(name, {
                "competitor": name, "lender_type": claim.lender_type, "rate": None,
                "rate_text": None, "rate_note_index": None, "evidence": UNKNOWN,
                "evidence_cue": None, "evidence_note_index": None, "claims": []})
            entry["claims"].append(claim.as_dict())
            if claim.rate is not None:
                entry.update(rate=claim.rate, rate_text=claim.rate_text,
                             rate_note_index=note.note_index, quote=claim.sentence)
            if claim.evidence != UNKNOWN:
                entry.update(evidence=claim.evidence, evidence_cue=claim.evidence_cue,
                             evidence_note_index=note.note_index)

    signals: dict[str, dict[str, Any]] = {}
    for note in per_note:
        for name, item in note.customer_signals.items():
            previous = signals.get(name)
            record = {**item, "note_index": note.note_index}
            if previous is not None and previous["value"] != item["value"]:
                record["superseded"] = previous.get("superseded", []) + [
                    {k: previous[k] for k in ("value", "quote", "note_index")}]
            signals[name] = record
    asks = [a for note in per_note for a in note.customer_asks]
    if asks:
        latest = asks[-1]
        signals.setdefault("expected_rate", {"value": latest["rate"], "quote": latest["sentence"],
                                             "extractor": "rule", "note_index": latest["note_index"]})

    methods = sorted({note.signals_method for note in per_note}) or ["rule"]
    return {
        "notes_read": len(per_note),
        "competitors": list(merged.values()),
        "unrecognised_lenders": [m.as_dict() for note in per_note for m in note.unrecognised],
        "customer_signals": signals,
        "customer_asks": asks,
        "our_rates": [r for note in per_note for r in note.our_rates],
        "unattributed_rates": [r for note in per_note for r in note.unattributed_rates],
        "conflicts": conflicts(merged, per_note, context, extractor.registry),
        "notes": [note.as_dict() for note in per_note],
        "method": {
            "competitors": "rule: registry aliases + rate patterns + evidence cues",
            "customer_signals": methods[0] if len(methods) == 1 else methods,
            "llm_enabled": bool(getattr(extractor.agent, "llm_enabled", False)),
            "registry_version": extractor.registry.version,
            "extractor_version": EXTRACTOR_VERSION,
        },
        "pricing_effect": "none - extracted claims are context for the reviewer, not price inputs",
    }


def conflicts(merged: dict[str, dict[str, Any]], per_note: list[NoteExtraction],
              context: dict[str, Any], registry: CompetitorRegistry) -> list[dict[str, Any]]:
    """Where the notes and the structured sales context disagree."""
    out: list[dict[str, Any]] = []
    raw_name = context.get("competitor_name")
    structured = registry.resolve(raw_name)
    s_rate = coerce("competitor_rate", context.get("competitor_rate"))
    s_verified = context.get("competitor_evidence_verified")

    if raw_name and structured is None:
        out.append({"type": "unrecognised_structured_competitor", "severity": "info",
                    "message": f"The application names '{raw_name}', which the competitor registry "
                               "does not recognise, so its quote cannot be benchmarked.",
                    "structured": {"competitor_name": raw_name}})

    entry = merged.get(structured) if structured else None
    if structured and entry is None and merged:
        out.append({"type": "structured_competitor_not_in_notes", "severity": "info",
                    "message": f"The structured record names {structured}, but the notes only mention "
                               f"{', '.join(sorted(merged))}.",
                    "structured": {"competitor_name": structured}})

    if entry is not None:
        if s_rate is not None and entry["rate"] is not None and abs(s_rate - entry["rate"]) >= 0.005:
            out.append({"type": "rate_mismatch", "severity": "warning",
                        "message": f"Structured competitor rate is {s_rate:.2f}% but note "
                                   f"{entry['rate_note_index'] + 1} quotes {structured} at {entry['rate']:.2f}% "
                                   f"(\"{entry['rate_text']}\").",
                        "structured": {"competitor_rate": s_rate},
                        "notes": {"rate": entry["rate"], "note_index": entry["rate_note_index"],
                                  "quote": entry.get("quote")}})
        if s_rate is None and entry["rate"] is not None:
            out.append({"type": "quote_not_captured", "severity": "info",
                        "message": f"Note {entry['rate_note_index'] + 1} quotes {structured} at "
                                   f"{entry['rate']:.2f}% but the structured competitor rate is empty.",
                        "notes": {"rate": entry["rate"], "note_index": entry["rate_note_index"]}})
        if s_verified is True and entry["evidence"] == UNDOCUMENTED:
            out.append({"type": "evidence_mismatch", "severity": "warning",
                        "message": f"The file marks the {structured} quote as verified, but note "
                                   f"{entry['evidence_note_index'] + 1} says \"{entry['evidence_cue']}\".",
                        "structured": {"competitor_evidence_verified": True},
                        "notes": {"evidence": UNDOCUMENTED, "cue": entry["evidence_cue"],
                                  "note_index": entry["evidence_note_index"]}})
        if s_verified is False and entry["evidence"] == DOCUMENTED:
            out.append({"type": "evidence_not_recorded", "severity": "warning",
                        "message": f"Note {entry['evidence_note_index'] + 1} cites a \"{entry['evidence_cue']}\" "
                                   f"for {structured}, but the file is not marked verified - check the "
                                   "document and update the flag.",
                        "structured": {"competitor_evidence_verified": False},
                        "notes": {"evidence": DOCUMENTED, "cue": entry["evidence_cue"],
                                  "note_index": entry["evidence_note_index"]}})

    for name, item in merged.items():
        rated = [c for c in item["claims"] if c.get("rate") is not None]
        if len({round(c["rate"], 2) for c in rated}) > 1:
            history = ", ".join(f"{c['rate']:.2f}% (note {c['note_index'] + 1})" for c in rated)
            out.append({"type": "notes_disagree", "severity": "info",
                        "message": f"{name} is quoted differently across notes: {history}. The latest is used.",
                        "competitor": name})
        if name != structured and item["rate"] is not None:
            if structured:
                message = (f"Notes also quote {name} at {item['rate']:.2f}% ({item['evidence']}); "
                           "the structured record has no slot for a second lender.")
            else:
                message = (f"Notes quote {name} at {item['rate']:.2f}% ({item['evidence']}); the structured "
                           "record does not name a recognised lender, so this quote is not on it.")
            out.append({"type": "additional_competitor", "severity": "info", "message": message,
                        "competitor": name})

    for note in per_note:
        for claim in note.unrecognised:
            out.append({"type": "unrecognised_lender", "severity": "info",
                        "message": f"Note {note.note_index + 1} names \"{claim.alias}\", which is not in the "
                                   "competitor registry - flagged for review, not mapped.",
                        "alias": claim.alias})
    return out


# ------------------------------------------------------------- historical
def _assert_no_latent(columns: Iterable[str]) -> None:
    leaked = [c for c in columns if str(c).startswith(LATENT_PREFIX)]
    if leaked:
        raise AssertionError(f"extractor input carries latent truth columns: {leaked}")


def extract_history(notes: pd.DataFrame, *, extractor: NoteExtractor | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the deterministic reader over a notes table.

    Returns ``(signals, mentions)``: one row per note, and one row per lender
    mention. The input is narrowed to ``INPUT_COLUMNS`` first, so latent truth
    columns cannot influence a single extracted value.
    """
    frame = notes[[c for c in INPUT_COLUMNS if c in notes.columns]].copy()
    _assert_no_latent(frame.columns)
    extractor = extractor or NoteExtractor(allow_llm=False)
    version = f"{EXTRACTOR_VERSION}+{config_fingerprint()}"

    signal_rows, mention_rows = [], []
    meta_cols = [c for c in INPUT_COLUMNS if c in frame.columns and c != "note_text"]
    for row in frame.itertuples(index=False):
        meta = {c: getattr(row, c) for c in meta_cols}
        note = extractor.extract(getattr(row, "note_text", ""), with_signals=True)
        per_lender = note.competitors()
        signals = note.customer_signals
        signal_rows.append({
            **meta,
            "n_mentions": len(note.mentions),
            "competitors": "|".join(per_lender),
            "rates": "|".join("" if c.rate is None else f"{c.rate:g}" for c in per_lender.values()),
            "evidence": "|".join(c.evidence for c in per_lender.values()),
            "evidence_cues": "|".join(c.evidence_cue or "" for c in per_lender.values()),
            "match_types": "|".join(c.match for c in per_lender.values()),
            "unrecognised": "|".join(c.alias for c in note.unrecognised),
            "customer_ask_rate": note.customer_asks[-1]["rate"] if note.customer_asks else None,
            "our_rate": note.our_rates[-1]["rate"] if note.our_rates else None,
            "unattributed_rates": "|".join(f"{r['rate']:g}" for r in note.unattributed_rates),
            **{name: (signals.get(name) or {}).get("value") for name in
               ("customer_urgency", "rate_sensitivity", "customer_intent", "required_amount")},
            "signals_method": note.signals_method,
            "extractor_version": version,
        })
        for index, claim in enumerate([*note.mentions, *note.unrecognised]):
            mention_rows.append({
                **meta, "mention_index": index,
                "competitor": claim.competitor or "", "lender_type": claim.lender_type or "",
                "alias": claim.alias, "match": claim.match, "fuzzy_ratio": claim.fuzzy_ratio,
                "rate": claim.rate, "rate_text": claim.rate_text,
                "evidence": claim.evidence, "evidence_cue": claim.evidence_cue,
                "start": claim.span[0], "end": claim.span[1],
            })
    mention_columns = [*meta_cols, "mention_index", "competitor", "lender_type", "alias", "match",
                       "fuzzy_ratio", "rate", "rate_text", "evidence", "evidence_cue", "start", "end"]
    return _normalise(pd.DataFrame(signal_rows)), _normalise(pd.DataFrame(mention_rows, columns=mention_columns))


def notes_path():
    return resolve(model_config()["data"]["notes_path"])


def load_history(rebuild: bool = False, source=None, cache_dir=None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-note signals and per-mention tables for the whole notes file, cached.

    The cache is rebuilt when the notes file is newer than it, or when the
    extractor version or registry content changed - a cached table built with
    yesterday's alias list would quietly disagree with today's live extraction.
    """
    directory = resolve(cache_dir) if cache_dir else PROCESSED_DIR
    signals_path, mentions_path = directory / SIGNALS_FILE, directory / MENTIONS_FILE
    source = resolve(source) if source else notes_path()
    version = f"{EXTRACTOR_VERSION}+{config_fingerprint()}"

    if not rebuild and signals_path.exists() and mentions_path.exists():
        built = min(signals_path.stat().st_mtime, mentions_path.stat().st_mtime)
        if not source.exists() or built >= source.stat().st_mtime:
            cached = _normalise(pd.read_csv(signals_path, dtype=_TEXT_COLUMNS))
            if len(cached) and str(cached["extractor_version"].iloc[0]) == version:
                return cached, _normalise(pd.read_csv(mentions_path, dtype=_TEXT_COLUMNS))
            log.info("note signals cache was built by %s; rebuilding for %s",
                     cached["extractor_version"].iloc[0] if len(cached) else "nothing", version)

    if not source.exists():
        raise FileNotFoundError(f"{source} not found; the historical extraction needs the notes file")
    signals, mentions = extract_history(pd.read_csv(source))
    directory.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: the API may read the cache while a CLI run rewrites it,
    # and a half-written CSV would load as a smaller, silently wrong history.
    for frame, path in ((mentions, mentions_path), (signals, signals_path)):
        partial = path.with_name(f".{path.stem}.{os.getpid()}.tmp.csv")      # gitignored like the cache
        frame.to_csv(partial, index=False)
        os.replace(partial, path)
    log.info("wrote %s note signals and %s mentions -> %s", len(signals), len(mentions), directory)
    return signals, mentions


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Text columns are '' when empty, fresh or read back from the cache alike."""
    for column in _TEXT_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    return frame


# Pipe-joined and free-text columns must stay strings when read back.
_TEXT_COLUMNS = {c: str for c in ("competitors", "rates", "evidence", "evidence_cues", "match_types",
                                  "unrecognised", "unattributed_rates", "competitor", "lender_type",
                                  "alias", "match", "rate_text", "evidence_cue")}
