"""Borrowers registering online.

What must hold: a registered account is always a borrower, whatever the request
says; it cannot take a staff username or another borrower's; weak or malformed
details are refused; and the new borrower is signed in and limited to what a
borrower may do.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.api import access
from aavas.api.main import app
from aavas.governance import store


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def details(username: str, **extra) -> dict:
    return {"username": username, "password": "Secret123", "name": "Priya Menon",
            "mobile": "98765 43210", "email": "priya@example.com", **extra}


def test_a_borrower_registers_and_is_signed_in(client) -> None:
    response = client.post("/auth/register", json=details("priya.menon"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["user"]["role"] == "borrower" and body["user"]["landing"] == "offer"
    assert body["user"]["name"] == "Priya Menon" and body["user"]["mobile"] == "9876543210"
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['token']}"}).json()
    assert me["user"]["username"] == "priya.menon"
    login = client.post("/auth/login", json={"username": "Priya.Menon", "password": "Secret123"})
    assert login.status_code == 200
    stored = store.fetch_one("SELECT password FROM borrower_accounts WHERE username = 'priya.menon'")
    assert stored["password"].startswith("pbkdf2_sha256$") and "Secret123" not in stored["password"]


def test_registration_can_never_make_staff(client) -> None:
    body = client.post("/auth/register", json=details("sneaky1", role="administrator",
                                                       governance_roles=["model_risk"])).json()
    assert body["user"]["role"] == "borrower" and body["user"]["roles"] == ["borrower"]
    headers = {"Authorization": f"Bearer {body['token']}"}
    assert client.get("/config", headers=headers).status_code == 403
    assert client.get("/applications", headers=headers).json() == {"applications": []}


@pytest.mark.parametrize("username", ["admin", "rohit", "borrower", "administrator", "root"])
def test_staff_and_reserved_names_cannot_be_taken(client, username) -> None:
    assert client.post("/auth/register", json=details(username)).status_code == 409


def test_a_taken_username_is_refused(client) -> None:
    assert client.post("/auth/register", json=details("twice.user")).status_code == 200
    again = client.post("/auth/register", json=details("twice.user", password="Other1234"))
    assert again.status_code == 409
    login = client.post("/auth/login", json={"username": "twice.user", "password": "Other1234"})
    assert login.status_code == 401, "the second registration must not replace the first password"


@pytest.mark.parametrize("change, message", [
    ({"password": "short1"}, "at least 8"),
    ({"password": "lettersonly"}, "a letter and a digit"),
    ({"username": "x"}, "at least 3 characters"),
    ({"username": "has space"}, "try has.space"),
    ({"username": "priya#menon"}, "not #"),
    ({"username": "not@an-email"}, "email address is not valid"),
    ({"mobile": "12345"}, "mobile"),
    ({"email": "not-an-email"}, "email"),
    ({"name": " "}, "full name"),
])
def test_bad_details_are_refused(client, change, message) -> None:
    response = client.post("/auth/register", json={**details("valid.name"), **change})
    assert response.status_code == 422 and message in response.json()["detail"], response.text


def test_registered_borrowers_are_not_shown_on_the_sign_in_screen(client) -> None:
    client.post("/auth/register", json=details("hidden.borrower"))
    names = [p["username"] for p in client.get("/auth/personas").json()["personas"]]
    assert "hidden.borrower" not in names


def test_registration_can_be_switched_off(client, monkeypatch) -> None:
    real = access.config
    monkeypatch.setattr(access, "config", lambda: {**real(), "registration": {"borrowers": False}})
    assert client.get("/auth/registration").json() == {"borrowers": False}
    assert client.post("/auth/register", json=details("closed.door")).status_code == 403


def test_an_email_address_can_be_the_username(client) -> None:
    response = client.post("/auth/register", json=details("Priya.Menon@Example.com"))
    assert response.status_code == 200, response.text
    assert response.json()["user"]["username"] == "priya.menon@example.com"
    login = client.post("/auth/login", json={"username": "priya.menon@example.com", "password": "Secret123"})
    assert login.status_code == 200
