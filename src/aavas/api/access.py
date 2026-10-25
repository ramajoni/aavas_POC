"""Sign-in and role-based access: who the users are, and what each role may reach.

Everything lives in ``config/access.yaml``: the roles, the users that hold them,
and the route rules. The file is re-read whenever it changes on disk, so a user
removed or a role narrowed takes effect at the next request, not the next restart.

**Sessions.** A successful sign-in returns a bearer token: a small JSON payload
and an HMAC-SHA256 signature over it, keyed by ``AAVAS_SESSION_SECRET``. The
token carries the username only, never the roles; roles are looked up from the
file on every request for the same reason the file is re-read. With no secret
set, one is generated per process: sessions then end on every restart, which is
the safe failure for a POC and is logged once.

**Route rules.** Among the rules whose method and path match, the one with the
longest path wins. A route no rule matches is open only to a role marked
``all_routes`` - deny by default, so a new endpoint is never public by accident.

This is a deployment-local scheme for synthetic personas. Production replaces
the user table with SSO and roles from the directory; ``security.principal_from_request``
remains the only seam that changes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import env
from ..logging_utils import get_logger
from ..paths import CONFIG_DIR

log = get_logger(__name__)

HASH_SCHEME = "pbkdf2_sha256"
HASH_ITERATIONS = 200_000

_lock = threading.Lock()
_cache: dict[str, Any] = {"path": None, "mtime": None, "data": None}
_revoked: dict[str, float] = {}           # token id -> expiry, for sign-out
_failures: dict[str, list[float]] = {}    # username -> recent failure times
_secret: bytes | None = None


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# ------------------------------------------------------------------ the file

def access_file() -> Path:
    return Path(env("AAVAS_ACCESS_FILE") or CONFIG_DIR / "access.yaml")


def config() -> dict[str, Any]:
    path = access_file()
    mtime = path.stat().st_mtime
    with _lock:
        if _cache["path"] != path or _cache["mtime"] != mtime:
            with open(path, "r", encoding="utf-8") as fh:
                _cache.update(path=path, mtime=mtime, data=yaml.safe_load(fh) or {})
        return _cache["data"]


@dataclass(frozen=True)
class User:
    username: str
    name: str
    role: str
    title: str = ""
    description: str = ""
    governance_roles: tuple[str, ...] = ()
    rate_authority: str | None = None
    login_card: bool = True

    @property
    def initials(self) -> str:
        return "".join(part[0] for part in self.name.split()[:2]).upper()

    @property
    def roles(self) -> list[str]:
        return [self.role, *self.governance_roles]


def role_config(role: str) -> dict[str, Any]:
    return (config().get("roles") or {}).get(role) or {}


def user(username: str) -> User | None:
    entry = (config().get("users") or {}).get(username)
    if not entry or entry.get("role") not in (config().get("roles") or {}):
        return None
    return User(
        username=username, name=entry.get("name", username), role=entry["role"],
        title=entry.get("title", ""), description=entry.get("description", ""),
        governance_roles=tuple(entry.get("governance_roles") or ()),
        rate_authority=role_config(entry["role"]).get("rate_authority"),
        login_card=entry.get("login_card", True))


def users() -> list[User]:
    return [u for name in (config().get("users") or {}) if (u := user(name))]


def profile(u: User) -> dict[str, Any]:
    """What the browser needs to draw the right workspace for this user."""
    role = role_config(u.role)
    return {
        "username": u.username, "name": u.name, "initials": u.initials,
        "title": u.title, "role": u.role, "role_label": role.get("label", u.role),
        "roles": u.roles, "rate_authority": u.rate_authority,
        "landing": role.get("landing"), "pages": role.get("pages") or [],
        "all_pages": bool(role.get("all_pages")), "console": bool(role.get("console")),
    }


# ----------------------------------------------------------------- passwords

def hash_password(password: str, *, salt: bytes | None = None,
                  iterations: int = HASH_ITERATIONS) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"{HASH_SCHEME}${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, digest = stored.split("$")
        if scheme != HASH_SCHEME:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), _unb64(salt), int(iterations))
        return hmac.compare_digest(actual, _unb64(digest))
    except (ValueError, TypeError):
        return False


# A real hash to check against when the username does not exist, so an unknown
# user costs the same time as a wrong password and the two cannot be told apart.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16), iterations=HASH_ITERATIONS)


class LoginRejected(Exception):
    """Wrong username or password, or the account is locked. One message for all."""


def authenticate(username: str, password: str) -> User:
    username = (username or "").strip().lower()
    settings = config().get("session") or {}
    limit = int(settings.get("max_failed_logins", 5))
    window = float(settings.get("lockout_minutes", 5)) * 60
    now = time.time()
    entry = (config().get("users") or {}).get(username) or {}
    found = user(username)

    # Count the attempt as a failure *before* the slow password check, under the
    # lock: checked first and counted after, guesses sent in parallel would all
    # pass the count. A success clears it. Unknown names are never stored, so the
    # table cannot be grown by inventing usernames.
    with _lock:
        recent = [t for t in _failures.get(username, []) if now - t < window]
        locked = len(recent) >= limit
        if found is not None and not locked:
            recent.append(now)
        if recent:
            _failures[username] = recent
        else:
            _failures.pop(username, None)

    ok = verify_password(password or "", entry.get("password") or _DUMMY_HASH)
    if locked or not ok or found is None:
        raise LoginRejected("Incorrect username or password, or too many attempts. "
                            "Try again in a few minutes.")
    with _lock:
        _failures.pop(username, None)
    return found


# -------------------------------------------------------------------- tokens

def _session_secret() -> bytes:
    global _secret
    configured = env("AAVAS_SESSION_SECRET")
    if configured:
        return configured.encode()
    if _secret is None:
        _secret = secrets.token_bytes(32)
        log.warning("AAVAS_SESSION_SECRET is not set; using a per-process secret, so every "
                    "restart signs everyone out")
    return _secret


def _sign(body: str) -> str:
    return _b64(hmac.new(_session_secret(), body.encode(), hashlib.sha256).digest())


def issue_token(u: User) -> tuple[str, int]:
    ttl = int(float((config().get("session") or {}).get("ttl_minutes", 480)) * 60)
    now = int(time.time())
    payload = {"sub": u.username, "iat": now, "exp": now + ttl, "jti": secrets.token_urlsafe(12)}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    return f"{body}.{_sign(body)}", payload["exp"]


def _claims(token: str) -> dict[str, Any] | None:
    try:
        body, signature = token.split(".")
    except ValueError:
        return None
    # Bytes, not str: compare_digest raises on non-ASCII text, and a garbage
    # token must read as a bad token, not as a broken service.
    if not hmac.compare_digest(signature.encode(), _sign(body).encode()):
        return None
    try:
        claims = json.loads(_unb64(body))
    except (ValueError, TypeError):
        return None
    if not isinstance(claims, dict) or claims.get("exp", 0) <= time.time():
        return None
    with _lock:
        if claims.get("jti") in _revoked:
            return None
    return claims


def user_for_token(token: str) -> User | None:
    claims = _claims(token)
    return user(str(claims.get("sub"))) if claims else None


def revoke(token: str) -> None:
    claims = _claims(token)
    if not claims:
        return
    now = time.time()
    with _lock:
        for jti, expiry in list(_revoked.items()):
            if expiry <= now:
                del _revoked[jti]
        _revoked[claims["jti"]] = claims["exp"]


# --------------------------------------------------------------- route rules

@dataclass(frozen=True)
class Decision:
    allowed: bool
    public: bool = False
    rule: str | None = None
    roles: tuple[str, ...] = field(default_factory=tuple)


def _matches(rule: dict[str, Any], method: str, path: str) -> bool:
    methods = [m.upper() for m in rule.get("methods") or []]
    if methods and method not in methods:
        return False
    target = rule["path"]
    if rule.get("exact"):
        return path == target
    return path == target or path.startswith(target.rstrip("/") + "/")


def _best(rules: list[dict[str, Any]], method: str, path: str) -> dict[str, Any] | None:
    hits = [r for r in rules if _matches(r, method, path)]
    return max(hits, key=lambda r: len(r["path"])) if hits else None


def is_public(method: str, path: str) -> bool:
    method = "GET" if method == "HEAD" else method
    return _best(config().get("public") or [], method, path) is not None


def decide(method: str, path: str, roles: list[str]) -> Decision:
    """May a caller holding ``roles`` make this request?"""
    method = "GET" if method == "HEAD" else method
    if is_public(method, path):
        return Decision(allowed=True, public=True, rule="public")
    if any(role_config(r).get("all_routes") for r in roles):
        return Decision(allowed=True, rule="all_routes")
    rule = _best(config().get("routes") or [], method, path)
    if rule is None:
        return Decision(allowed=False, rule=None)
    allowed = rule.get("roles") or []
    if allowed == "authenticated":
        return Decision(allowed=True, rule=rule["path"])
    return Decision(allowed=bool(set(roles) & set(allowed)), rule=rule["path"],
                    roles=tuple(allowed))


# ------------------------------------------------------------------- helpers

def reset_state() -> None:
    """Forget sign-outs and failed attempts. For tests."""
    with _lock:
        _revoked.clear()
        _failures.clear()
        _cache.update(path=None, mtime=None, data=None)


if __name__ == "__main__":                                       # pragma: no cover
    import getpass

    print(hash_password(os.environ.get("PASSWORD") or getpass.getpass("Password: ")))
