"""Who is calling, and what they are allowed to do.

The API had no authentication at all: the approver on an override was a string
the caller supplied, and the authority check was satisfied by whatever the
request said. This adds identity and role enforcement without pretending to be
more than it is.

**Three modes.**

``session`` (the default) requires sign-in. ``POST /auth/login`` exchanges a
username and password from ``config/access.yaml`` for a bearer token, and every
route is checked against that file's role rules before it runs (see ``access``).
The workspace in the browser cannot be reached without signing in.

``disabled`` keeps the POC runnable with no setup, for scripts and tests. Every caller is
an unauthenticated principal carrying every role, and *every surface that reports
a governance action says so*. That last part is the whole design: a signature from
an unauthenticated caller is a name in a text field, and a system that rendered it
identically to a verified one would be the more dangerous failure. ``/health``
reports the mode, approvals record ``authenticated: false`` against every
signature, and the control tower shows the estate as unsecured.

``apikey`` reads ``AAVAS_API_KEYS`` and enforces roles on every write. It is a
deployment-local scheme, not an identity provider: it exists so the enforcement
path is real and tested. Production needs SSO with roles from the directory, and
this module is the seam where that goes - ``principal_from_request`` is the only
function that would change.

Nothing here is a substitute for a network boundary. An API key in an environment
variable protects against the wrong colleague, not against an attacker on the
host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import Depends, Header, HTTPException, Request

from ..config import env, load
from ..logging_utils import get_logger

log = get_logger(__name__)

DISABLED, APIKEY, SESSION = "disabled", "apikey", "session"
ANONYMOUS = "anonymous"


@dataclass
class Principal:
    name: str
    roles: list[str] = field(default_factory=list)
    authenticated: bool = False
    mode: str = DISABLED
    display_name: str | None = None
    rate_authority: str | None = None
    # A login the branch issued, still on its temporary password.
    must_change_password: bool = False

    def has(self, role: str) -> bool:
        return role in self.roles

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "display_name": self.display_name or self.name,
                "roles": sorted(self.roles),
                "authenticated": self.authenticated, "auth_mode": self.mode,
                # Carried into every audit entry a signature produces. An approval
                # signed by an unverified caller must never read like a verified one.
                "identity_verified": self.authenticated}


def mode() -> str:
    """The configured mode. Anything unrecognised means session: a typo in the
    variable must fail closed, not open."""
    value = (env("AAVAS_AUTH_MODE", SESSION) or SESSION).strip().lower()
    return value if value in (DISABLED, APIKEY, SESSION) else SESSION


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization") or ""
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def session_principal(token: str | None) -> Principal | None:
    from . import access

    found = access.user_for_token(token) if token else None
    if found is None:
        return None
    return Principal(name=found.username, roles=found.roles, authenticated=True,
                     mode=SESSION, display_name=found.name,
                     rate_authority=found.rate_authority,
                     must_change_password=found.must_change_password)


def known_roles() -> list[str]:
    try:
        return sorted(load("governance").get("roles") or {})
    except Exception:                                           # noqa: BLE001
        return []


def _key_table() -> dict[str, Principal]:
    """``AAVAS_API_KEYS`` as ``key:name:role|role,key:name:role``.

    Parsed on every call rather than cached: a key revoked in the environment has
    to stop working at the next request, not at the next restart.
    """
    raw = env("AAVAS_API_KEYS", "") or ""
    table: dict[str, Principal] = {}
    valid = set(known_roles())
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) < 3:
            log.warning("ignoring malformed AAVAS_API_KEYS entry (expected key:name:roles)")
            continue
        key, name, roles = parts[0], parts[1], parts[2]
        granted = [r for r in roles.split("|") if r]
        unknown = [r for r in granted if valid and r not in valid]
        if unknown:
            log.warning("api key for %s grants unknown role(s) %s; they are dropped", name, unknown)
            granted = [r for r in granted if r not in unknown]
        table[key] = Principal(name=name, roles=granted, authenticated=True, mode=APIKEY)
    return table


def principal_from_request(request: Request, api_key: str | None) -> Principal:
    current = mode()
    if current == SESSION:
        # The access middleware has already checked the route and stored who it is.
        found = getattr(request.state, "principal", None) or session_principal(
            bearer_token(request))
        if found is None:
            raise HTTPException(401, "sign in to continue",
                                headers={"WWW-Authenticate": "Bearer"})
        return found
    if current == DISABLED:
        # Every role, and authenticated=False. The roles are granted so the POC
        # runs; the flag is what every downstream surface actually reports.
        return Principal(name=ANONYMOUS, roles=known_roles(), authenticated=False, mode=DISABLED)

    if not api_key:
        raise HTTPException(401, "authentication is enabled; supply an X-API-Key header")
    found = _key_table().get(api_key)
    if not found:
        # Deliberately the same message as a missing key: distinguishing them
        # tells a caller whether a key exists.
        raise HTTPException(401, "authentication is enabled; supply an X-API-Key header")
    return found


def current_principal(request: Request,
                      x_api_key: str | None = Header(None, alias="X-API-Key")) -> Principal:
    return principal_from_request(request, x_api_key)


def require_roles(*roles: str, any_of: bool = True) -> Callable[..., Principal]:
    """Dependency enforcing that the caller holds the named role(s).

    With auth disabled this passes, because the anonymous principal holds every
    role. That is intentional and is the reason the mode is reported everywhere
    rather than only here.
    """
    wanted = set(roles)

    def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        held = set(principal.roles)
        ok = bool(held & wanted) if any_of else wanted <= held
        if not ok:
            raise HTTPException(
                403, f"this action needs {'one of ' if any_of else 'all of '}"
                     f"{sorted(wanted)}; you hold {sorted(held)}")
        return principal

    return dependency


def status() -> dict[str, Any]:
    current = mode()
    if current == SESSION:
        from . import access

        configured = len(access.users())
        return {
            "mode": current, "enforced": True, "principals_configured": configured,
            "roles": known_roles(),
            "warning": (
                f"Sign-in is enforced over {configured} user(s) in config/access.yaml, and every "
                f"route is checked against that file's role rules. This is a deployment-local "
                f"user table, not an identity provider; production needs SSO with roles from "
                f"the directory."),
        }
    configured = len(_key_table()) if current == APIKEY else 0
    return {
        "mode": current,
        "enforced": current != DISABLED,
        "principals_configured": configured,
        "roles": known_roles(),
        "warning": (
            "Authentication is disabled. Every caller is anonymous and holds every role, so "
            "role checks pass and approval signatures are names in a text field rather than "
            "verified identities. Set AAVAS_AUTH_MODE=apikey with AAVAS_API_KEYS before any "
            "deployment carrying real applications."
            if current == DISABLED else
            f"API-key authentication is enforced over {configured} principal(s). This is a "
            f"deployment-local scheme, not an identity provider; production needs SSO with "
            f"roles from the directory."),
    }
