"""Competitor registry: canonical lenders, their aliases, and lender-shaped unknowns.

Branches do not write "LIC HF". They write "LICHFL", "LIC Housing", "lic", and
occasionally "Shubam Housing". The registry turns those spans into one canonical
name so quotes can be counted per lender - and it is deliberately unwilling to
guess. A name it does not know is reported as *unrecognised*, with the span, for
an analyst to add or reject. Mapping an unknown "Xyz Housing" onto the nearest
known lender would silently merge two lenders' price books, which is worse than
missing the mention.

Matching is three-tier, and every mention records which tier fired:

* ``exact``      an alias from ``config/competitors.yaml`` (case-insensitive)
* ``ambiguous``  a bare word that is also a person's name or a document
                 ("Shubham", "Aadhaar", "LIC") - accepted only in a sentence about
                 lending and never beside its other sense ("Aadhaar card")
* ``fuzzy``      a near-spelling of a registry name (difflib ratio above the
                 configured floor), reported with its score
* ``unrecognised`` lender-shaped ("<Name> Housing Finance") but unknown

Self-names (Aavas) are matched too, so a rate written next to our own name is
read as ours and never as a competitor quote.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from ..paths import CONFIG_DIR

CONFIG_FILE = CONFIG_DIR / "competitors.yaml"

EXACT, AMBIGUOUS, FUZZY, UNRECOGNISED, SELF = "exact", "ambiguous", "fuzzy", "unrecognised", "self"

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|\n+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")


def load_config() -> dict[str, Any]:
    # Through the shared loader, so the registry is listed with every other config
    # file and cached the same way.
    from ..config import load

    return load("competitors")


def config_fingerprint() -> str:
    """Short content hash: a registry edit must invalidate cached extractions even
    when the file's mtime does not move (a checkout, a copy)."""
    return hashlib.sha1(CONFIG_FILE.read_bytes()).hexdigest()[:8]


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of sentences. Decimal points never split (no space
    follows them) and neither does "p.a. de raha hai" (lower case follows)."""
    spans, start = [], 0
    for match in _SENTENCE.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if e > s]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _alias_pattern(alias: str) -> str:
    words = [re.escape(w) for w in alias.split()]
    return r"\s+".join(words)


@dataclass(frozen=True)
class Lender:
    name: str
    type: str
    aliases: tuple[str, ...]
    ambiguous: tuple[str, ...] = ()
    not_when_followed_by: tuple[str, ...] = ()
    note: str | None = None


@dataclass
class Mention:
    """One lender named in one piece of text, with the span as written."""

    competitor: str | None          # canonical name; None when unrecognised
    lender_type: str | None
    alias: str                      # exactly as written in the note
    start: int
    end: int
    match: str                      # exact | ambiguous | fuzzy | unrecognised | self
    fuzzy_ratio: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_competitor(self) -> bool:
        return self.match not in (SELF, UNRECOGNISED)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("extra", None)
        return out


class CompetitorRegistry:
    """Alias index over ``config/competitors.yaml``."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config if config is not None else load_config()
        self.config = cfg
        self.version = str(cfg.get("competitors_config_version", "unversioned"))
        self.lenders: dict[str, Lender] = {}
        self._lookup: dict[str, tuple[str, str]] = {}      # normalised alias -> (canonical, kind)
        for name, spec in (cfg.get("lenders") or {}).items():
            lender = Lender(
                name=str(name), type=str(spec.get("type", "unknown")),
                aliases=tuple(str(a) for a in spec.get("aliases") or ()),
                ambiguous=tuple(str(a) for a in spec.get("ambiguous") or ()),
                not_when_followed_by=tuple(_norm(w) for w in spec.get("not_when_followed_by") or ()),
                note=spec.get("note"))
            self.lenders[lender.name] = lender
            self._lookup.setdefault(_norm(lender.name), (lender.name, EXACT))
            for alias in lender.aliases:
                self._lookup[_norm(alias)] = (lender.name, EXACT)
            for alias in lender.ambiguous:
                self._lookup[_norm(alias)] = (lender.name, AMBIGUOUS)
        self.self_names = [str(s) for s in cfg.get("self_names") or ()]
        for alias in self.self_names:
            self._lookup[_norm(alias)] = ("__self__", SELF)

        # One alternation, longest alias first, so "LIC Housing Finance" wins over
        # "LIC" at the same position.
        alternatives = sorted(self._lookup, key=len, reverse=True)
        self._alias_re = re.compile(
            r"(?<![A-Za-z0-9])(" + "|".join(_alias_pattern(a) for a in alternatives) + r")(?![A-Za-z0-9])",
            re.I)

        suffixes = sorted((str(s) for s in cfg.get("lender_suffixes") or ()), key=len, reverse=True)
        self._suffixes = [_norm(s) for s in suffixes]
        self._shape_re = re.compile(
            r"(?P<head>(?:[A-Z][A-Za-z&'-]*\.?\s+){1,3}?)(?P<suffix>(?i:"
            + "|".join(_alias_pattern(s) for s in suffixes) + r"))(?![A-Za-z0-9])")
        self._generic = {_norm(w) for w in cfg.get("generic_heads") or ()}
        self._context = [_norm(c) for c in cfg.get("lending_context_cues") or ()]
        fuzzy = cfg.get("fuzzy") or {}
        self.min_ratio = float(fuzzy.get("min_ratio", 0.86))
        self.min_length = int(fuzzy.get("min_length", 5))
        self.min_head_ratio = float(fuzzy.get("min_head_ratio", 0.80))
        self._heads = self._build_heads()
        self._fuzzy_memo: dict[tuple[str, str, bool], tuple[str, float] | None] = {}

    # ------------------------------------------------------------------ lookup
    def resolve(self, name: str | None) -> str | None:
        """Canonical name for an alias or canonical name; None when unknown.

        Exact only - used for structured fields and API paths, where a guess
        would be a silent relabelling."""
        if not name:
            return None
        hit = self._lookup.get(_norm(str(name)))
        if hit is None or hit[1] == SELF:
            return None
        return hit[0]

    def lender_type(self, name: str | None) -> str | None:
        lender = self.lenders.get(name or "")
        return lender.type if lender else None

    def is_self(self, name: str | None) -> bool:
        hit = self._lookup.get(_norm(str(name or "")))
        return bool(hit and hit[1] == SELF)

    # --------------------------------------------------------------- matching
    def find(self, text: str) -> list[Mention]:
        """Every lender (and self) mention in ``text``, in reading order."""
        if not text:
            return []
        spans = sentences(text)
        mentions: list[Mention] = []
        taken: list[tuple[int, int]] = []

        for match in self._alias_re.finditer(text):
            written = match.group(1)
            canonical, kind = self._lookup[_norm(written)]
            start, end = match.start(1), match.end(1)
            if kind == AMBIGUOUS and not self._ambiguous_ok(text, start, end, canonical, spans):
                continue
            if kind == SELF:
                mentions.append(Mention(None, None, written, start, end, SELF))
            else:
                mentions.append(Mention(canonical, self.lenders[canonical].type, written,
                                        start, end, kind))
            taken.append((start, end))

        # Lender-shaped phrases the alias table did not cover.
        for match in self._shape_re.finditer(text):
            end = match.end()
            # Strip generic leading words: "Customer Xyz Housing" -> "Xyz Housing".
            words = [(w.start() + match.start("head"), w.group(0))
                     for w in re.finditer(r"\S+", match.group("head"))]
            while words and _norm(words[0][1].rstrip(".")) in self._generic:
                words = words[1:]
            if not words:
                continue
            start = words[0][0]
            if _overlaps(start, end, taken):
                continue
            head = " ".join(w.rstrip(".") for _, w in words)
            written = text[start:end]
            mention = self._fuzzy(head, written, start, end, suffix=match.group("suffix"))
            if mention is None:
                if not self._lending_sentence(text, start, spans):
                    continue
                mention = Mention(None, None, written, start, end, UNRECOGNISED)
            mentions.append(mention)
            taken.append((start, end))

        # Bare near-spellings ("comparing with Shubam") - only in a lending sentence.
        for match in _TOKEN.finditer(text):
            token = match.group(0)
            if len(token) < self.min_length or not token[0].isupper():
                continue
            start, end = match.start(), match.end()
            if _overlaps(start, end, taken) or _norm(token) in self._generic:
                continue
            if _norm(token) in self._lookup:
                continue
            mention = self._fuzzy(token, token, start, end, suffix=None, single=True)
            if mention is not None and self._lending_sentence(text, start, spans):
                mentions.append(mention)
                taken.append((start, end))

        return sorted(mentions, key=lambda m: m.start)

    # ------------------------------------------------------------- internals
    def _build_heads(self) -> list[tuple[str, str, str]]:
        """(name head, full alias, canonical) for fuzzy comparison. The head is
        the alias without its lender suffix: "Shubham Housing" -> "shubham"."""
        heads = []
        for canonical, lender in self.lenders.items():
            for alias in (*lender.aliases, *lender.ambiguous):
                full = _norm(alias)
                head = full
                for suffix in self._suffixes:
                    if head.endswith(" " + suffix):
                        head = head[: -len(suffix) - 1].strip()
                        break
                if len(head) >= 3:
                    heads.append((head, full, canonical))
        return heads

    def _fuzzy(self, head: str, written: str, start: int, end: int, suffix: str | None,
               single: bool = False) -> Mention | None:
        key = _norm(head)
        if len(key.replace(" ", "")) < self.min_length:
            return None
        memo = (key, _norm(written), single)
        if memo not in self._fuzzy_memo:
            self._fuzzy_memo[memo] = self._best_head(key, memo[1], single)
        best = self._fuzzy_memo[memo]
        if best is None:
            return None
        canonical, ratio = best
        return Mention(canonical, self.lenders[canonical].type, written, start, end, FUZZY,
                       fuzzy_ratio=ratio)

    def _best_head(self, key: str, full: str, single: bool) -> tuple[str, float] | None:
        """Best registry name for a near-spelling, or None.

        A bare word must itself be close to a registry name ("Shubam"). A
        lender-shaped phrase must be close as a whole *and* in its name part:
        "Shubh Housing" passes, "Muthoot Fincorp" does not - same group name,
        different lender, so it goes to review instead."""
        best: tuple[float, str] | None = None
        for alias_head, alias_full, canonical in self._heads:
            if single and " " in alias_head:
                continue
            head = SequenceMatcher(None, key, alias_head)
            floor = self.min_ratio if single else self.min_head_ratio
            if head.real_quick_ratio() < floor or head.quick_ratio() < floor or head.ratio() < floor:
                continue
            score = head.ratio() if single else SequenceMatcher(None, full, alias_full).ratio()
            if score < self.min_ratio:
                continue
            if best is None or score > best[0]:
                best = (score, canonical)
        return None if best is None else (best[1], round(best[0], 3))

    def _sentence_of(self, start: int, spans: list[tuple[int, int]]) -> tuple[int, int]:
        for s, e in spans:
            if s <= start < e:
                return s, e
        return spans[-1] if spans else (0, start)

    def _lending_sentence(self, text: str, start: int, spans: list[tuple[int, int]]) -> bool:
        s, e = self._sentence_of(start, spans)
        sentence = text[s:e].lower()
        if re.search(r"\d{1,2}(?:\.\d{1,3})?\s*(?:%|pc\b|percent)|@\s*\d|\broi\b", sentence):
            return True
        return any(cue in sentence for cue in self._context)

    def _ambiguous_ok(self, text: str, start: int, end: int, canonical: str,
                      spans: list[tuple[int, int]]) -> bool:
        following = _norm(text[end:end + 30])
        for word in self.lenders[canonical].not_when_followed_by:
            if following.startswith(word):
                return False
        return self._lending_sentence(text, start, spans)

    # -------------------------------------------------------------- reporting
    def listing(self) -> list[dict[str, Any]]:
        return [{"name": l.name, "type": l.type, "aliases": list(l.aliases),
                 "ambiguous": list(l.ambiguous), "note": l.note}
                for l in self.lenders.values()]


def _overlaps(start: int, end: int, taken: list[tuple[int, int]]) -> bool:
    return any(start < e and s < end for s, e in taken)


@lru_cache(maxsize=1)
def default_registry() -> CompetitorRegistry:
    return CompetitorRegistry()


def reset() -> None:
    from ..config import load

    load.cache_clear()
    default_registry.cache_clear()
