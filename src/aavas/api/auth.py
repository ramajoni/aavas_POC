"""POST /auth/login, /auth/logout, /auth/register, /auth/password; GET /auth/me, /auth/personas.

Sign-in for the lending workspace and the OS console. The token is a bearer
token the browser sends on every call; ``access`` holds the users, the roles and
the route rules it is checked against.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from . import access, security

router = APIRouter(prefix="/auth", tags=["sign-in"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class RegisterRequest(BaseModel):
    """A borrower's own account. There is no role field: it is always borrower."""
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)
    name: str = Field(..., min_length=1, max_length=120)
    mobile: str = Field(..., min_length=1, max_length=20)
    email: str | None = Field(None, max_length=120)


@router.post("/login", summary="Sign in and receive a bearer token")
def login(request: LoginRequest) -> dict[str, Any]:
    try:
        user = access.authenticate(request.username, request.password)
    except access.LoginRejected as exc:
        raise HTTPException(401, str(exc)) from exc
    token, expires_at = access.issue_token(user)
    return {"token": token, "token_type": "bearer", "expires_at": expires_at,
            "user": access.profile(user)}


@router.post("/register", summary="Create a borrower account and sign in")
def register(request: RegisterRequest) -> dict[str, Any]:
    from ..governance import audit

    try:
        user = access.register(request.username, request.password, request.name,
                               request.mobile, request.email)
    except access.RegistrationRejected as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    audit.record(user.username, "registered_borrower", "user", user.username)
    token, expires_at = access.issue_token(user)
    return {"token": token, "token_type": "bearer", "expires_at": expires_at,
            "user": access.profile(user)}


class PasswordChange(BaseModel):
    current_password: str = Field(..., max_length=128)
    new_password: str = Field(..., max_length=128)


@router.post("/password", summary="Set your own password (a borrower)")
def change_password(body: PasswordChange, request: Request) -> dict[str, Any]:
    from ..governance import audit

    token = security.bearer_token(request)
    current = access.user_for_token(token) if token else None
    if current is None:
        raise HTTPException(401, "sign in to continue", headers={"WWW-Authenticate": "Bearer"})
    try:
        user = access.change_password(current.username, body.current_password, body.new_password)
    except access.RegistrationRejected as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    audit.record(user.username, "changed_password", "user", user.username)
    return {"user": access.profile(user)}


@router.get("/registration", summary="Whether borrowers may register online")
def registration() -> dict[str, Any]:
    return {"borrowers": access.registration_open()}


@router.post("/logout", summary="End this session")
def logout(request: Request) -> dict[str, Any]:
    token = security.bearer_token(request)
    if token:
        access.revoke(token)
    return {"signed_out": True}


@router.get("/me", summary="Who is signed in, and what their role may open")
def me(request: Request) -> dict[str, Any]:
    # Read from the token in every mode: the browser always signs in, even when
    # AAVAS_AUTH_MODE=disabled leaves the API itself unenforced.
    token = security.bearer_token(request)
    user = access.user_for_token(token) if token else None
    if user is None:
        raise HTTPException(401, "sign in to continue", headers={"WWW-Authenticate": "Bearer"})
    return {"user": access.profile(user), "auth_mode": security.mode()}


@router.get("/personas", summary="The people shown on the sign-in screen")
def personas() -> dict[str, Any]:
    return {"personas": [
        {"username": u.username, "name": u.name, "initials": u.initials, "title": u.title,
         "role": u.role, "description": u.description,
         "console": bool(access.role_config(u.role).get("console"))}
        for u in access.users() if u.login_card]}
