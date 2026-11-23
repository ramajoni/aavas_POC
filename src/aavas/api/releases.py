"""Release notes, for the application's Release notes page.

    GET /releases             every released version, newest first, with its highlights
    GET /releases/{version}   the full notes of one version

The notes are the project's own files: docs/releases/<version>.md (0.6.0 onward) and the
sections of CHANGELOG.md (every version, 0.1.0 onward). On the server the application runs
from a clone of the release tag, so both sit at the repo root there too. A version with no
notes file is served from its CHANGELOG section. Staff roles only: a borrower gets 403.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .. import __version__
from ..paths import ROOT
from .security import Principal, current_principal

router = APIRouter(prefix="/releases", tags=["release notes"])

BORROWER = "borrower"
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")   # used with fullmatch: ASCII digits only, no trailing newline
SECTION = re.compile(r"^## \[(?P<v>[^\]]+)\](?: - (?P<d>\d{4}-\d{2}-\d{2}))?[ \t]*$", re.M)
TITLE_DATE = re.compile(r"^# Release \S+ \((\d{4}-\d{2}-\d{2})\)", re.M)

_lock = threading.Lock()
_cache: dict[str, Any] = {"key": None, "data": None}


def _changelog() -> Path:
    return ROOT / "CHANGELOG.md"


def _notes_dir() -> Path:
    return ROOT / "docs" / "releases"


def _key(value: str) -> tuple[int, int, int]:
    a, b, c = value.split(".")
    return int(a), int(b), int(c)


def _plain(text: str) -> str:
    """One line of plain text: markdown marks removed, links reduced to their words."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _summary(markdown: str) -> str:
    """The first item's bold lead, else its first sentence (or the first paragraph's)."""
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("|"):
            continue
        line = re.sub(r"^(?:[-*+]|\d+\.)\s+", "", line)
        lead = re.match(r"\*\*(.+?)\*\*", line)
        if lead:
            return _plain(lead.group(1)).rstrip(".:")
        text = _plain(line)
        sentence = re.match(r"(.+?[.!?])(?:\s|$)", text)
        out = sentence.group(1) if sentence else text
        return out if len(out) <= 160 else out[:157].rstrip() + "..."
    return ""


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}[ \t]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return match.group(1).strip() if match else ""


def _changelog_sections(text: str) -> dict[str, dict[str, str | None]]:
    found: dict[str, dict[str, str | None]] = {}
    marks = list(SECTION.finditer(text))
    for i, m in enumerate(marks):
        if not VERSION.match(m.group("v")):
            continue                                            # [Unreleased]
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        found[m.group("v")] = {"date": m.group("d"), "body": text[m.end():end].strip()}
    return found


def _signature() -> tuple:
    files = [_changelog(), *sorted(_notes_dir().glob("*.md"))] if _notes_dir().is_dir() else [_changelog()]
    out = []
    for f in files:
        try:
            out.append((f.name, f.stat().st_mtime_ns))
        except OSError:
            out.append((f.name, None))
    return tuple(out)


def _build() -> dict[str, dict[str, Any]]:
    try:
        sections = _changelog_sections(_changelog().read_text(encoding="utf-8"))
    except OSError:
        sections = {}
    notes: dict[str, str] = {}
    if _notes_dir().is_dir():
        for f in _notes_dir().glob("*.md"):
            if VERSION.match(f.stem):
                notes[f.stem] = f.read_text(encoding="utf-8")
    entries: dict[str, dict[str, Any]] = {}
    for version in set(sections) | set(notes):
        section = sections.get(version) or {}
        full = notes.get(version)
        if full is not None:
            highlights = _section(full, "Highlights") or str(section.get("body") or "")
            title = TITLE_DATE.search(full)
            date = section.get("date") or (title.group(1) if title else None)
            markdown = full
        else:
            highlights = markdown = str(section.get("body") or "")
            date = section.get("date")
        entries[version] = {"version": version, "date": date, "has_notes": full is not None,
                            "highlights": highlights, "summary": _summary(highlights),
                            "markdown": markdown}
    return entries


def _entries() -> dict[str, dict[str, Any]]:
    key = _signature()
    with _lock:
        if _cache["key"] != key:
            _cache.update(key=key, data=_build())
        return _cache["data"]


def _staff(principal: Principal = Depends(current_principal)) -> Principal:
    """Every signed-in role except the borrower."""
    roles = set(principal.roles)
    if not roles or roles <= {BORROWER}:
        raise HTTPException(403, "release notes are for staff")
    return principal


@router.get("", summary="Released versions, newest first")
def list_releases(_: Principal = Depends(_staff)) -> dict[str, Any]:
    entries = _entries()
    items = [{"version": e["version"], "date": e["date"], "current": e["version"] == __version__,
              "has_notes": e["has_notes"], "highlights": e["highlights"], "summary": e["summary"]}
             for e in sorted(entries.values(), key=lambda e: _key(e["version"]), reverse=True)]
    return {"current": __version__, "releases": items}


@router.get("/{version}", summary="The full notes of one version")
def get_release(version: str, _: Principal = Depends(_staff)) -> dict[str, Any]:
    if not VERSION.fullmatch(version):
        raise HTTPException(404, "no such release")
    entry = _entries().get(version)
    if entry is None:
        raise HTTPException(404, "no such release")
    # The version is a key into what was read from docs/releases, never a path; the
    # check below is the stated guarantee that nothing outside that folder is served.
    root = _notes_dir().resolve()
    if entry["has_notes"] and not (root / f"{version}.md").resolve().is_relative_to(root):
        raise HTTPException(404, "no such release")
    return {"version": version, "date": entry["date"], "current": version == __version__,
            "has_notes": entry["has_notes"], "markdown": entry["markdown"]}
