"""Logins a relationship manager issues with an assisted application.

What must hold: entering an application for a customer gives that customer a
borrower login (mobile number as username, one-time temporary password) and the
file becomes theirs to follow; a customer who already has an account is linked,
not given a second one; a temporary password opens one door only - setting your
own; the password is never stored or audited in clear; and only the RM issues or
resets a login.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.api import access
from aavas.api.main import app
from aavas.governance import store

PASSWORDS = {"rohit": "Rohit@2026", "meera": "Meera@2026", "rahul": "Rahul@2026"}


@pytest.fixture
def client(monkeypatch, bundle) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def sign_in(client, username, password=None) -> dict[str, str]:
    body = client.post("/auth/login", json={"username": username,
                                             "password": password or PASSWORDS[username]}).json()
    return {"Authorization": f"Bearer {body['token']}"}


def walk_in(loan_id: str, name: str, mobile: str) -> dict:
    return {"loan_id": loan_id, "name": name, "mobile": mobile, "email": "", "segment": "salaried_formal",
            "city": "Jaipur", "channel": "branch_walkin", "property_type": "resale_flat",
            "declared_monthly_income": 60000, "requested_loan_amount": 1500000,
            "requested_tenure_years": 20, "applicant_age": 33, "declared_obligations": 0,
            "estimated_property_value": 2500000, "bureau_score": 760, "sales_context": {}}


def enter(client, rohit, loan_id, name, mobile):
    a = walk_in(loan_id, name, mobile)
    quote = {k: v for k, v in a.items() if k not in ("name", "mobile", "email")}
    assert client.post("/pricing/pre-assessment", json=quote, headers=rohit).status_code == 200
    saved = client.post("/applications", json={"loan_id": loan_id, "applicant": a}, headers=rohit)
    assert saved.status_code == 200, saved.text
    return saved.json()


def test_entering_an_application_issues_the_customer_a_login(client) -> None:
    rohit = sign_in(client, "rohit")
    saved = enter(client, rohit, "LOGIN-1", "Sunita Kulkarni", "98220 11111")
    login = saved["customer_login"]
    assert login["created"] and login["username"] == "9822011111"
    password = login["temporary_password"]
    assert len(password) == 10 and any(c.isdigit() for c in password) and any(c.isalpha() for c in password)
    assert saved["created_by"] == "rohit" and saved["source"] == "assisted"

    stored = store.fetch_one("SELECT password FROM borrower_accounts WHERE username = '9822011111'")
    assert password not in stored["password"], "the temporary password is never stored in clear"
    audit_text = str(store.query("SELECT detail FROM audit_log WHERE entity_id = '9822011111'"))
    assert password not in audit_text and "issued_customer_login" in str(
        store.query("SELECT action FROM audit_log WHERE entity_id = '9822011111'"))

    listed = client.get("/applications", headers=rohit).json()["applications"]
    assert next(r for r in listed if r["loan_id"] == "LOGIN-1")["customer_login"] == "9822011111"


def test_a_temporary_password_only_lets_the_customer_set_their_own(client) -> None:
    login = enter(client, sign_in(client, "rohit"), "LOGIN-2", "Imran Shaikh", "9822022222")["customer_login"]
    first = client.post("/auth/login", json={"username": login["username"], "password": login["temporary_password"]})
    assert first.status_code == 200 and first.json()["user"]["must_change_password"] is True
    temp = {"Authorization": f"Bearer {first.json()['token']}"}
    assert client.get("/applications", headers=temp).status_code == 403, "nothing else until the password is set"
    assert client.get("/auth/me", headers=temp).status_code == 200

    assert client.post("/auth/password", headers=temp, json={"current_password": "wrong1234",
                                                             "new_password": "MyOwn2026"}).status_code == 403
    assert client.post("/auth/password", headers=temp, json={"current_password": login["temporary_password"],
                                                             "new_password": "short"}).status_code == 422
    done = client.post("/auth/password", headers=temp, json={"current_password": login["temporary_password"],
                                                             "new_password": "MyOwn2026"})
    assert done.status_code == 200 and done.json()["user"]["must_change_password"] is False

    mine = client.get("/applications", headers=temp).json()["applications"]
    assert [r["loan_id"] for r in mine] == ["LOGIN-2"], "the file is the customer's to follow"
    assert client.post("/auth/login", json={"username": login["username"],
                                            "password": login["temporary_password"]}).status_code == 401
    assert client.post("/auth/login", json={"username": login["username"],
                                            "password": "MyOwn2026"}).status_code == 200


def test_a_customer_with_an_account_is_linked_not_duplicated(client) -> None:
    rohit = sign_in(client, "rohit")
    first = enter(client, rohit, "LOGIN-3A", "Kavya Rao", "9822033333")["customer_login"]
    second = enter(client, rohit, "LOGIN-3B", "Kavya Rao", "9822033333")["customer_login"]
    assert second == {"username": first["username"], "temporary_password": None, "created": False}
    assert store.fetch_one("SELECT COUNT(*) AS n FROM borrower_accounts WHERE mobile = '9822033333'")["n"] == 1


def test_the_rm_can_reset_a_lost_password_and_no_one_else_can(client) -> None:
    rohit = sign_in(client, "rohit")
    login = enter(client, rohit, "LOGIN-4", "Arun Pillai", "9822044444")["customer_login"]
    reset = client.post("/applications/LOGIN-4/customer-login", headers=rohit)
    assert reset.status_code == 200 and reset.json()["reset"] and reset.json()["username"] == login["username"]
    assert reset.json()["temporary_password"] != login["temporary_password"]
    again = client.post("/auth/login", json={"username": login["username"],
                                             "password": reset.json()["temporary_password"]}).json()
    assert again["user"]["must_change_password"] is True
    assert client.post("/applications/LOGIN-4/customer-login", headers=sign_in(client, "meera")).status_code == 403
    assert client.post("/applications/LOGIN-4/customer-login", headers=sign_in(client, "rahul")).status_code == 403
    assert client.post("/applications/NO-SUCH/customer-login", headers=rohit).status_code == 404


def test_a_login_needs_a_mobile_number(client) -> None:
    rohit = sign_in(client, "rohit")
    a = walk_in("LOGIN-5", "No Phone", "12345")
    quote = {k: v for k, v in a.items() if k not in ("name", "mobile", "email")}
    client.post("/pricing/pre-assessment", json=quote, headers=rohit)
    refused = client.post("/applications", json={"loan_id": "LOGIN-5", "applicant": a}, headers=rohit)
    assert refused.status_code == 422 and "mobile" in refused.json()["detail"]


def test_staff_cannot_change_their_password_here(client) -> None:
    rohit = sign_in(client, "rohit")
    response = client.post("/auth/password", headers=rohit,
                           json={"current_password": "Rohit@2026", "new_password": "Another2026"})
    assert response.status_code == 400
