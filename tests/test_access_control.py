"""Sign-in and role-based access.

These test the control, not the happy path: a route reachable without signing
in, a role that reaches another role's data, a token that survives tampering or
sign-out, or an approver the caller can name themselves would each pass a naive
test and defeat the control.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest
import yaml
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from aavas.api import access, security
from aavas.api.main import app

PASSWORDS = {"rahul": "Rahul@2026", "rohit": "Rohit@2026", "meera": "Meera@2026",
             "arjun": "Arjun@2026", "admin": "Admin@2026"}

# Routes with no rule on purpose: only a role with all_routes reaches them.
ADMINISTRATOR_ONLY = {"/connectors", "/connectors/{name}",
                      # Configuration writes: GET /config is shared, the writes are not.
                      "/config/{name}/preview", "/config/{name}/history/{change_id}/restore"}
ADMINISTRATOR_ONLY_WRITES = {("PUT", "/config/{name}")}


@pytest.fixture
def session_mode(monkeypatch):
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield
    access.reset_state()


@pytest.fixture
def client(session_mode) -> TestClient:
    return TestClient(app)


def sign_in(client: TestClient, username: str) -> dict[str, str]:
    response = client.post("/auth/login",
                           json={"username": username, "password": PASSWORDS[username]})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


# ----------------------------------------------------------------- the mode

def test_sign_in_is_required_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AAVAS_AUTH_MODE", raising=False)
    assert security.mode() == security.SESSION
    assert security.status()["enforced"] is True


def test_an_unrecognised_mode_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "off-please")
    assert security.mode() == security.SESSION


# --------------------------------------------------------------- signing in

def test_nothing_but_the_public_routes_answer_without_a_session(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/model-cards").status_code == 401
    assert client.get("/governance").status_code == 401
    assert client.post("/pricing/pre-assessment", json={}).status_code == 401
    assert client.get("/auth/me").status_code == 401


def test_a_forged_host_header_cannot_make_a_route_look_public(client) -> None:
    """Starlette builds request.url partly from the Host header, while the router
    dispatches on the scope path. The check has to read what the router reads, or
    'Host: x/health?' turns every route into /health."""
    forged = {"Host": "x/health?"}
    assert client.get("/connectors", headers=forged).status_code == 401
    assert client.get("/model-cards", headers=forged).status_code == 401
    borrower = {**sign_in(client, "rahul"), **forged}
    assert client.post("/pricing/override", json={}, headers=borrower).status_code == 403


def test_the_sign_in_screen_lists_the_four_personas_and_nothing_secret(client) -> None:
    personas = client.get("/auth/personas").json()["personas"]
    assert [p["username"] for p in personas] == ["rahul", "rohit", "meera", "arjun"]
    assert all("password" not in p for p in personas)


def test_a_wrong_password_reads_exactly_like_an_unknown_user(client: TestClient) -> None:
    wrong = client.post("/auth/login", json={"username": "rohit", "password": "guess"})
    unknown = client.post("/auth/login", json={"username": "nobody", "password": "guess"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]


def test_signing_in_returns_the_role_and_what_it_may_open(client: TestClient) -> None:
    body = client.post("/auth/login",
                       json={"username": "Rohit", "password": PASSWORDS["rohit"]}).json()
    assert body["user"]["name"] == "Rohit Sharma"
    assert body["user"]["role"] == "relationship_manager"
    assert body["user"]["landing"] == "applications"
    assert "governance" not in body["user"]["pages"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['token']}"}).json()
    assert me["user"]["username"] == "rohit"


def test_repeated_failures_lock_the_account_even_for_the_right_password(client) -> None:
    for _ in range(5):
        client.post("/auth/login", json={"username": "meera", "password": "nope"})
    locked = client.post("/auth/login",
                         json={"username": "meera", "password": PASSWORDS["meera"]})
    assert locked.status_code == 401


def test_parallel_guesses_cannot_outrun_the_lockout(session_mode) -> None:
    """The password check is slow on purpose. If the lock were checked before it
    and the failure counted after, thirty guesses sent at once would all be tried."""
    from concurrent.futures import ThreadPoolExecutor

    def guess(_):
        try:
            access.authenticate("meera", "wrong")
        except access.LoginRejected:
            pass

    with ThreadPoolExecutor(max_workers=30) as pool:
        list(pool.map(guess, range(30)))
    assert len(access._failures["meera"]) == 5
    with pytest.raises(access.LoginRejected):
        access.authenticate("meera", PASSWORDS["meera"])


def test_unknown_usernames_leave_nothing_behind(session_mode) -> None:
    for i in range(20):
        with pytest.raises(access.LoginRejected):
            access.authenticate(f"ghost-{i}", "x")
    assert not any(name.startswith("ghost-") for name in access._failures)


# ------------------------------------------------------------------- tokens

def test_a_tampered_token_is_rejected(client: TestClient) -> None:
    token = sign_in(client, "rahul")["Authorization"].split()[1]
    body, signature = token.split(".")
    forged = access._b64(access._unb64(body).replace(b'"rahul"', b'"admin"'))
    assert client.get("/connectors",
                      headers={"Authorization": f"Bearer {forged}.{signature}"}).status_code == 401


@pytest.mark.parametrize("token", ["garbage", "a.b.c", "é.é", "eyJzdWIiOiJhZG1pbiJ9.x"])
def test_a_malformed_token_is_a_bad_token_not_a_broken_service(client, token) -> None:
    raw = f"Bearer {token}".encode("latin-1")       # as it arrives off the wire
    response = client.get("/governance", headers={"Authorization": raw})
    assert response.status_code == 401


def test_signing_out_ends_the_session(client: TestClient) -> None:
    headers = sign_in(client, "arjun")
    assert client.get("/governance", headers=headers).status_code == 200
    assert client.post("/auth/logout", headers=headers).status_code == 200
    assert client.get("/governance", headers=headers).status_code == 401


def test_an_expired_token_is_rejected(client: TestClient) -> None:
    """Correctly signed, so only the expiry can be what rejects it."""
    body = access._b64(json.dumps({"sub": "arjun", "iat": 1, "exp": 2, "jti": "old"}).encode())
    headers = {"Authorization": f"Bearer {body}.{access._sign(body)}"}
    assert client.get("/governance", headers=headers).status_code == 401


def test_a_removed_user_loses_access_at_the_next_request(client, monkeypatch, tmp_path) -> None:
    copy = tmp_path / "access.yaml"
    shutil.copy(access.access_file(), copy)
    monkeypatch.setenv("AAVAS_ACCESS_FILE", str(copy))
    headers = sign_in(client, "arjun")
    assert client.get("/governance", headers=headers).status_code == 200

    data = yaml.safe_load(copy.read_text())
    del data["users"]["arjun"]
    copy.write_text(yaml.safe_dump(data))
    os.utime(copy, (copy.stat().st_atime, copy.stat().st_mtime + 5))
    assert client.get("/governance", headers=headers).status_code == 401


# -------------------------------------------------------------------- roles

@pytest.mark.parametrize("username, method, path, expected", [
    ("rahul", "GET", "/model-cards", 403),
    ("rahul", "GET", "/governance", 403),
    ("rahul", "GET", "/competitors", 403),
    ("rahul", "GET", "/customers/ANY/behaviour", 403),
    ("arjun", "GET", "/customers/ANY/behaviour", 403),
    ("rohit", "GET", "/model-cards", 200),
    ("rohit", "GET", "/competitors", 403),
    ("rohit", "GET", "/notes/summary", 403),
    ("meera", "GET", "/competitors", 200),
    ("rohit", "GET", "/governance", 403),
    ("rohit", "GET", "/config", 403),
    ("meera", "GET", "/pricing/override/reasons", 200),
    ("meera", "GET", "/xai", 403),
    ("arjun", "GET", "/governance", 200),
    ("arjun", "GET", "/config", 200),
    ("arjun", "GET", "/connectors", 403),
    ("arjun", "GET", "/pricing/override/reasons", 403),
    ("admin", "GET", "/connectors", 200),
    ("admin", "GET", "/governance", 200),
])
def test_each_role_reaches_its_own_surfaces_only(client, username, method, path, expected):
    response = client.request(method, path, headers=sign_in(client, username))
    assert response.status_code == expected, response.text


def test_a_borrower_cannot_price_a_verified_file(client: TestClient) -> None:
    response = client.post("/pricing/final-assessment", json={},
                           headers=sign_in(client, "rahul"))
    assert response.status_code == 403
    assert "Borrower" in response.json()["detail"]


def test_a_route_with_no_rule_is_denied_to_every_role_but_the_administrator() -> None:
    assert not access.decide("GET", "/a-route-added-tomorrow", ["model_validation"]).allowed
    assert access.decide("GET", "/a-route-added-tomorrow", ["administrator"]).allowed


def test_every_route_is_classified() -> None:
    """A new endpoint must be given a rule (or listed here as administrator-only)
    on purpose - deny by default keeps it closed, this keeps it from being forgotten."""
    unclassified = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue
        probe = route.path.replace("{", "x").replace("}", "")
        for method in route.methods - {"HEAD"}:
            if access.is_public(method, probe) or route.path in ADMINISTRATOR_ONLY \
                    or (method, route.path) in ADMINISTRATOR_ONLY_WRITES:
                continue
            if access._best(access.config().get("routes") or [], method, probe) is None:
                unclassified.append(f"{method} {route.path}")
    assert not unclassified, f"no access rule for: {unclassified}"


# ------------------------------------------------------- the human decision

def test_the_override_approver_is_the_person_signed_in(client, bundle) -> None:
    """The request names a pricing-committee approver; the record must carry the
    RM who signed in, at the RM's own authority."""
    headers = sign_in(client, "rohit")
    payload = {
        "loan_id": "ACCESS-TEST-001", "segment": "salaried_formal", "city": "Jaipur",
        "channel": "branch_walkin", "property_type": "resale_flat",
        "declared_monthly_income": 55000, "requested_loan_amount": 1500000,
        "requested_tenure_years": 20, "applicant_age": 36, "declared_obligations": 4000,
        "estimated_property_value": 2100000, "bureau_score": 742,
        "verified_monthly_income": 53000, "appraised_property_value": 2050000,
        "sanctioned_loan_amount": 1450000, "tenure_years": 20, "bureau_vintage_months": 96,
        "delinquency_count_24m": 0, "banking_behaviour_score": 71.0, "deviation_count": 1,
        "legal_flag": 0, "technical_flag": 0}
    priced = client.post("/pricing/final-assessment", json=payload, headers=headers).json()
    card_rate = priced["policy"]["card_rate"]

    body = client.post("/pricing/override", headers=headers, json={
        "loan_id": payload["loan_id"], "human_rate": round(card_rate - 0.30, 2),
        "reason": "competitor_lower_rate", "approved_by": "someone:pricing_committee",
        "evidence_verified": True}).json()
    assert body["approved_by"] == "rohit:sales_officer"
    assert body["within_authority"] is False
