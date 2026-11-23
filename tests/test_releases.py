"""The Release notes endpoints: what they list, who may read them, and that the version in
the URL can never become a path."""

from __future__ import annotations

import re
from urllib.parse import quote

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from aavas import __version__
from aavas.api import access, releases
from aavas.api.main import app
from aavas.api.security import Principal
from aavas.paths import ROOT

PASSWORDS = {"rahul": "Rahul@2026", "rohit": "Rohit@2026", "meera": "Meera@2026",
             "arjun": "Arjun@2026", "admin": "Admin@2026"}
STAFF = ("rohit", "meera", "arjun", "admin")
NOTES = ROOT / "docs" / "releases"


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def sign_in(client: TestClient, username: str) -> dict[str, str]:
    r = client.post("/auth/login", json={"username": username, "password": PASSWORDS[username]})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def semver(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


@pytest.fixture
def listing(client) -> dict:
    r = client.get("/releases", headers=sign_in(client, "meera"))
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ the list

def test_the_list_is_newest_first_by_semantic_version(listing) -> None:
    versions = [r["version"] for r in listing["releases"]]
    assert versions == sorted(versions, key=semver, reverse=True)
    # 0.10.x sorts after 0.9.x: a string sort would get this wrong
    if "0.10.0" in versions and "0.9.6" in versions:
        assert versions.index("0.10.0") < versions.index("0.9.6")
    assert "Unreleased" not in " ".join(versions)


def test_only_the_running_version_is_current(listing) -> None:
    assert listing["current"] == __version__
    current = [r["version"] for r in listing["releases"] if r["current"]]
    assert current in ([__version__], [])       # [] while the version has no section yet


def test_each_item_carries_highlights_and_a_plain_summary(listing) -> None:
    for r in listing["releases"]:
        assert re.fullmatch(r"\d+\.\d+\.\d+", r["version"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["date"]), r
        assert r["highlights"].strip(), r["version"]
        assert r["summary"] and "\n" not in r["summary"] and len(r["summary"]) <= 160
        assert not re.search(r"[*`]|\]\(", r["summary"]), r["summary"]
        assert isinstance(r["has_notes"], bool)


def test_highlights_come_from_the_notes_file_when_there_is_one(listing) -> None:
    item = next(r for r in listing["releases"] if r["version"] == "0.9.4")
    text = (NOTES / "0.9.4.md").read_text(encoding="utf-8")
    assert item["has_notes"] is True
    assert item["highlights"] == releases._section(text, "Highlights")
    assert "## Changes" not in item["highlights"]
    bold = re.search(r"\*\*(.+?)\*\*", item["highlights"]).group(1).rstrip(".:")
    assert item["summary"] == releases._plain(bold)


def test_a_version_with_no_notes_file_is_served_from_the_changelog(client) -> None:
    assert not (NOTES / "0.5.0.md").exists()
    headers = sign_in(client, "arjun")
    item = next(r for r in client.get("/releases", headers=headers).json()["releases"] if r["version"] == "0.5.0")
    assert item["has_notes"] is False and item["date"] == "2026-11-18"
    body = client.get("/releases/0.5.0", headers=headers)
    assert body.status_code == 200
    one = body.json()
    assert one["has_notes"] is False and one["version"] == "0.5.0"
    assert "HL-LAP-POLICY-2026.14" in one["markdown"]
    assert "## [0.4.1]" not in one["markdown"] and "## [0.5.0]" not in one["markdown"]


def test_the_full_notes_are_the_whole_file(client) -> None:
    r = client.get("/releases/0.9.4", headers=sign_in(client, "meera")).json()
    assert r["markdown"] == (NOTES / "0.9.4.md").read_text(encoding="utf-8")
    assert r["has_notes"] is True


def test_every_notes_file_appears_in_the_list(listing) -> None:
    listed = {r["version"] for r in listing["releases"]}
    files = {f.stem for f in NOTES.glob("*.md")}
    assert files and files <= listed, files - listed


def test_the_changelog_and_a_notes_file_build_the_list_in_isolation(tmp_path, monkeypatch) -> None:
    (tmp_path / "docs" / "releases").mkdir(parents=True)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n- not yet\n\n"
        "## [1.2.0] - 2030-01-02\n\n### Added\n- **Big thing.** Works [well](http://x.test) now.\n\n"
        "## [1.10.0] - 2030-02-03\n\n- Plain sentence one. Sentence two.\n", encoding="utf-8")
    (tmp_path / "docs" / "releases" / "1.2.0.md").write_text(
        "# Release 1.2.0 (2030-01-02)\n\n## Highlights\n\n- **Lead words.** More.\n\n## Changes\n\n- x\n", encoding="utf-8")
    monkeypatch.setattr(releases, "ROOT", tmp_path)
    saved = dict(releases._cache)
    releases._cache["key"] = None
    try:
        entries = releases._entries()
        assert set(entries) == {"1.2.0", "1.10.0"}
        assert entries["1.2.0"]["summary"] == "Lead words" and entries["1.2.0"]["has_notes"]
        assert entries["1.10.0"]["summary"] == "Plain sentence one."
        assert entries["1.10.0"]["markdown"].startswith("- Plain sentence one")
    finally:
        releases._cache.update(saved)


# ---------------------------------------------------------- unknown and unsafe

def test_an_unknown_version_is_404(client) -> None:
    headers = sign_in(client, "meera")
    assert client.get("/releases/99.99.99", headers=headers).status_code == 404
    assert client.get("/releases/0.0.1", headers=headers).status_code == 404


@pytest.mark.parametrize("bad", ["../x", "1.2", "1.2.3/../..", "%2e%2e", "..", "1.2.3.4", "1.2.x",
                                 "0.9.4%2f..%2f..%2fCHANGELOG", "0.9.4%00", " 0.9.4", "0.9.4\n"])
def test_a_version_that_is_not_x_y_z_never_reaches_the_filesystem(client, monkeypatch, bad) -> None:
    def touched(*_a, **_k):
        raise AssertionError("the filesystem was read for a version that is not x.y.z")
    monkeypatch.setattr(releases, "_entries", touched)
    # sent as raw as a client can: the test client would otherwise collapse "/.." itself
    sent = quote(bad, safe="%").replace("..", "%2e%2e")
    r = client.get(f"/releases/{sent}", headers=sign_in(client, "meera"))
    # "../x" is normalised to /x before it is sent: no route, so the access rules refuse it
    assert r.status_code in (403, 404, 405), (bad, r.status_code)
    # the handler itself, called as the router would with the decoded text
    with pytest.raises(HTTPException) as caught:
        releases.get_release(bad.replace("%2e", "."), Principal(name="t", roles=["area_manager"]))
    assert caught.value.status_code == 404


def test_the_version_check_is_strict() -> None:
    assert releases.VERSION.fullmatch("0.10.1")
    for bad in ("0.10", "0.10.1\n", "v0.10.1", "0.10.1/", "٠.١.٢"):
        assert not releases.VERSION.fullmatch(bad), bad


# ----------------------------------------------------------------- who may read

def test_a_borrower_gets_403(client) -> None:
    headers = sign_in(client, "rahul")
    assert client.get("/releases", headers=headers).status_code == 403
    assert client.get("/releases/0.9.4", headers=headers).status_code == 403


def test_the_borrower_is_refused_by_the_endpoint_itself_too() -> None:
    with pytest.raises(HTTPException) as caught:
        releases._staff(Principal(name="rahul", roles=["borrower"], authenticated=True))
    assert caught.value.status_code == 403


def test_no_sign_in_is_401(client) -> None:
    assert client.get("/releases").status_code == 401


@pytest.mark.parametrize("who", STAFF)
def test_every_staff_role_gets_200(client, who) -> None:
    headers = sign_in(client, who)
    assert client.get("/releases", headers=headers).status_code == 200
    assert client.get("/releases/0.9.4", headers=headers).status_code == 200
