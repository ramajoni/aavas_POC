"""Application versioning: one version everywhere, and release notes for it.

What must hold: src/aavas/__init__.py, pyproject.toml and the web application's package
files carry the same MAJOR.MINOR.PATCH; CHANGELOG.md opens with [Unreleased], its newest
release is that version, and every release is dated and in order; docs/releases has the
notes for it; the version is what decisions and /health are stamped with; and cutting a
release moves [Unreleased] under the new version and syncs every file.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest

import aavas
from aavas.config import versions

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release", ROOT / "scripts" / "release.py")
release = importlib.util.module_from_spec(spec)
sys.modules["release"] = release
spec.loader.exec_module(release)


def test_the_version_and_its_release_notes_agree() -> None:
    assert release.check() == []


def test_decisions_are_stamped_with_the_package_version() -> None:
    assert versions()["app_version"] == aavas.__version__ == release.current()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A copy of every file a release touches, so cutting one here changes nothing real."""
    files = {"INIT": "src/aavas/__init__.py", "PYPROJECT": "pyproject.toml", "PACKAGE": "application/package.json",
             "LOCK": "application/package-lock.json", "CHANGELOG": "CHANGELOG.md"}
    for attr, rel in files.items():
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest)
        monkeypatch.setattr(release, attr, dest)
    notes = tmp_path / "docs" / "releases"
    notes.mkdir(parents=True)
    shutil.copy(ROOT / "docs" / "releases" / f"{release.current()}.md", notes)
    monkeypatch.setattr(release, "NOTES", notes)
    return tmp_path


def test_an_empty_unreleased_section_cannot_be_released(sandbox) -> None:
    text = release.CHANGELOG.read_text(encoding="utf-8")
    emptied = re.sub(r"(## \[Unreleased\]\s*\n).*?(?=^## \[)", r"\1\n", text, count=1, flags=re.S | re.M)
    release.CHANGELOG.write_text(emptied, encoding="utf-8")         # the sandbox copy, not the real file
    with pytest.raises(SystemExit):
        release.cut("patch", "2030-01-01")


def test_cutting_a_release_moves_unreleased_and_syncs_every_file(sandbox) -> None:
    before = release.current()
    text = release.CHANGELOG.read_text(encoding="utf-8")
    release.CHANGELOG.write_text(text.replace("## [Unreleased]\n", "## [Unreleased]\n\n### Fixed\n- A test entry.\n", 1),
                                 encoding="utf-8")
    new = release.cut("minor", "2030-01-01")
    assert new == release.next_version(before, "minor")
    assert set(release.found_versions().values()) == {new}
    found = release.sections(release.CHANGELOG.read_text(encoding="utf-8"))
    assert found[0] == ("Unreleased", None) and found[1] == (new, "2030-01-01") and found[2][0] == before
    notes = (release.NOTES / f"{new}.md").read_text(encoding="utf-8")
    assert "A test entry." in notes and "pricing_policy.yaml" in notes
    assert release.check() == []


def test_drift_between_the_files_is_caught(sandbox) -> None:
    release.PACKAGE.write_text(release.PACKAGE.read_text(encoding="utf-8").replace(
        f'"version": "{release.current()}"', '"version": "9.9.9"', 1), encoding="utf-8")
    assert any("application/package.json" in p for p in release.check())
