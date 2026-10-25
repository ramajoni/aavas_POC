"""POST /auth/login, /auth/logout; GET /auth/me, /auth/personas.

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


@router.post("/login", summary="Sign in and receive a bearer token")
def login(request: LoginRequest) -> dict[str, Any]:
    try:
        user = access.authenticate(request.username, request.password)
    except access.LoginRejected as exc:
        raise HTTPException(401, str(exc)) from exc
    token, expires_at = access.issue_token(user)
    return {"token": token, "token_type": "bearer", "expires_at": expires_at,
            "user": access.profile(user)}


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
