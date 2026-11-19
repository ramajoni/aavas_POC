#!/usr/bin/env python3
"""Application versioning and release notes.

The version lives in one place, ``src/aavas/__init__.py`` (``__version__``); every
priced decision is stamped with it. This script keeps pyproject.toml and the web
application's package files in step with it, and keeps CHANGELOG.md and the
release notes in docs/releases/ in step with the version.

    python scripts/release.py check            # everything agrees (also run by the tests)
    python scripts/release.py minor            # cut the next minor release from [Unreleased]
    python scripts/release.py patch --date 2026-11-20
    python scripts/release.py versions         # the config file versions a release ships with

Semantic versioning, pre-1.0: a change to pricing behaviour, the API or the data
contract is a minor release; new features are a minor release; fixes only, a patch.
Cutting a release edits files only. Commit, then tag ``v<version>``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "src" / "aavas" / "__init__.py"
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE = ROOT / "application" / "package.json"
LOCK = ROOT / "application" / "package-lock.json"
CHANGELOG = ROOT / "CHANGELOG.md"
NOTES = ROOT / "docs" / "releases"

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
SECTION = re.compile(r"^## \[(?P<v>[^\]]+)\](?: - (?P<d>\d{4}-\d{2}-\d{2}))?\s*$", re.M)
INIT_RE = re.compile(r'^__version__ = "([^"]+)"', re.M)
PYPROJECT_RE = re.compile(r'(\[project\][^\[]*?^version = ")([^"]+)(")', re.M | re.S)


def _key(v: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in v.split("."))  # type: ignore[return-value]


def current() -> str:
    return INIT_RE.search(INIT.read_text(encoding="utf-8")).group(1)


def found_versions() -> dict[str, str]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    return {
        "src/aavas/__init__.py": current(),
        "pyproject.toml": PYPROJECT_RE.search(PYPROJECT.read_text(encoding="utf-8")).group(2),
        "application/package.json": json.loads(PACKAGE.read_text(encoding="utf-8"))["version"],
        "application/package-lock.json": lock["version"],
        "application/package-lock.json (packages)": lock["packages"][""]["version"],
    }


def sections(text: str) -> list[tuple[str, str | None]]:
    return [(m.group("v"), m.group("d")) for m in SECTION.finditer(text)]


def check() -> list[str]:
    """Every way the version and the notes can drift apart; empty when they agree."""
    problems = []
    version = current()
    if not SEMVER.match(version):
        problems.append(f"__version__ {version!r} is not MAJOR.MINOR.PATCH")
    for where, v in found_versions().items():
        if v != version:
            problems.append(f"{where} says {v}, src/aavas/__init__.py says {version}")
    if not CHANGELOG.exists():
        return problems + ["CHANGELOG.md is missing"]
    found = sections(CHANGELOG.read_text(encoding="utf-8"))
    if not found or found[0][0] != "Unreleased":
        problems.append("CHANGELOG.md must start with an [Unreleased] section")
    released = [(v, d) for v, d in found if v != "Unreleased"]
    if not released or released[0][0] != version:
        problems.append(f"the newest CHANGELOG.md release is {released[0][0] if released else 'none'}, "
                        f"not {version}")
    for v, d in released:
        if not SEMVER.match(v) or not d:
            problems.append(f"CHANGELOG.md section [{v}] needs a MAJOR.MINOR.PATCH version and a date")
    keys = [_key(v) for v, _ in released if SEMVER.match(v)]
    if keys != sorted(keys, reverse=True) or len(set(keys)) != len(keys):
        problems.append("CHANGELOG.md releases must be unique and newest first")
    if not (NOTES / f"{version}.md").exists():
        problems.append(f"docs/releases/{version}.md (the release notes for {version}) is missing")
    return problems


def config_versions() -> dict[str, str]:
    """The version each configuration file declares, for the release notes."""
    out = {}
    for path in sorted((ROOT / "config").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        key = next((k for k in data if isinstance(k, str) and (k == "version" or k.endswith("_version"))), None)
        if key:
            out[path.name] = str(data[key])
    return out


def next_version(version: str, part: str) -> str:
    major, minor, patch = _key(version)
    return {"major": f"{major + 1}.0.0", "minor": f"{major}.{minor + 1}.0", "patch": f"{major}.{minor}.{patch + 1}"}[part]


def _set(new: str) -> None:
    INIT.write_text(INIT_RE.sub(f'__version__ = "{new}"', INIT.read_text(encoding="utf-8"), count=1), encoding="utf-8")
    PYPROJECT.write_text(PYPROJECT_RE.sub(lambda m: f"{m.group(1)}{new}{m.group(3)}",
                                          PYPROJECT.read_text(encoding="utf-8"), count=1), encoding="utf-8")
    for path in (PACKAGE, LOCK):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = new
        if path is LOCK:
            data["packages"][""]["version"] = new
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cut(part: str, on: str) -> str:
    problems = check()
    if problems:
        raise SystemExit("fix these before cutting a release:\n  " + "\n  ".join(problems))
    text = CHANGELOG.read_text(encoding="utf-8")
    head = re.search(r"^## \[Unreleased\]\s*$", text, re.M)
    nxt = SECTION.search(text, head.end())
    body = text[head.end():nxt.start() if nxt else len(text)].strip()
    if not body:
        raise SystemExit("[Unreleased] is empty: there is nothing to release")
    new = next_version(current(), part)
    text = text[:head.end()] + f"\n\n## [{new}] - {on}\n\n{body}\n\n" + text[nxt.start() if nxt else len(text):]
    CHANGELOG.write_text(text, encoding="utf-8")
    _set(new)
    NOTES.mkdir(parents=True, exist_ok=True)
    configs = "\n".join(f"| {name} | {v} |" for name, v in config_versions().items())
    (NOTES / f"{new}.md").write_text(
        f"# Release {new} ({on})\n\n<!-- Write for the team: what changed for them, what to do, what is not done. -->\n\n"
        f"## Highlights\n\n## Changes\n\n{body}\n\n## Upgrade steps\n\n## Known issues\n\n"
        f"## Configuration versions in this release\n\n| File | Version |\n|---|---|\n{configs}\n", encoding="utf-8")
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Application versioning and release notes")
    parser.add_argument("action", choices=["check", "versions", "major", "minor", "patch"])
    parser.add_argument("--date", default=date.today().isoformat(), help="release date, YYYY-MM-DD")
    args = parser.parse_args(argv)
    if args.action == "check":
        problems = check()
        print(f"version {current()}: " + ("consistent" if not problems else "\n  " + "\n  ".join(problems)))
        return 1 if problems else 0
    if args.action == "versions":
        for name, v in config_versions().items():
            print(f"{name:<28} {v}")
        return 0
    new = cut(args.action, args.date)
    print(f"released {new}: CHANGELOG.md dated, versions synced, docs/releases/{new}.md started.\n"
          f"Finish the notes, then: git commit -am 'Release {new}' && git tag v{new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
