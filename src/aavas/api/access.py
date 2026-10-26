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

**Registered borrowers.** Borrowers may also create their own account
(``register``). Those accounts live in the ``borrower_accounts`` table, never in
the file, and can only ever hold the borrower role: a registration form is not a
way to become staff. A username in the file always wins over the table.

**Logins issued by a relationship manager.** When an RM enters an application
for a customer (``issue_customer_login``), the customer gets a borrower account:
the mobile number as username, and a one-time temporary password the RM hands
over. Until the customer sets their own password the account can do nothing
else (``must_change_password``; enforced in ``api/main.py``).

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

import re

from ..config import env
from ..governance import store
from ..logging_utils import get_logger
from ..paths import CONFIG_DIR

log = get_logger(__name__)

HASH_SCHEME = "pbkdf2_sha256"
HASH_ITERATIONS = 200_000

_lock = threading.Lock()
_register_lock = threading.Lock()
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
    mobile: str = ""
    email: str = ""
    must_change_password: bool = False

    @property
    def initials(self) -> str:
        return "".join(part[0] for part in self.name.split()[:2]).upper()

    @property
    def roles(self) -> list[str]:
        return [self.role, *self.governance_roles]


def role_config(role: str) -> dict[str, Any]:
    return (config().get("roles") or {}).get(role) or {}


BORROWER = "borrower"


def _registered(username: str) -> dict[str, Any] | None:
    try:
        return store.fetch_one("SELECT * FROM borrower_accounts WHERE username = ?", (username,))
    except Exception:                                           # noqa: BLE001
        return None


def _issued(username: str) -> dict[str, Any] | None:
    try:
        return store.fetch_one("SELECT * FROM customer_logins WHERE username = ?", (username,))
    except Exception:                                           # noqa: BLE001
        return None


def is_borrower_account(username: str | None) -> bool:
    return bool(username) and username not in (config().get("users") or {}) \
        and _registered(username) is not None


def user(username: str) -> User | None:
    entry = (config().get("users") or {}).get(username)
    if entry is None:
        row = _registered(username)
        if row is None or BORROWER not in (config().get("roles") or {}):
            return None
        issued = _issued(username)
        return User(username=username, name=row["name"], role=BORROWER, title="Borrower",
                    description="Login issued by the branch" if issued else "Registered online",
                    login_card=False, mobile=row["mobile"] or "", email=row["email"] or "",
                    must_change_password=bool(issued) and not issued.get("password_changed_at"))
    if entry.get("role") not in (config().get("roles") or {}):
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
        "mobile": u.mobile, "email": u.email, "must_change_password": u.must_change_password,
        "limits": {k: role[k] for k in ("offer_below_recommendation_bps",
                                        "approve_below_recommendation_bps") if k in role},
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
    entry = (config().get("users") or {}).get(username) or _registered(username) or {}
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


# -------------------------------------------------------------- registration

# A plain username, or an email address used as one.
USERNAME = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
USERNAME_CHARS = re.compile(r"[^a-z0-9._@+-]")
MOBILE = re.compile(r"^[6-9]\d{9}$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
RESERVED = {"admin", "administrator", "root", "system", "support", "staff", "aavas", "anonymous"}


def suggest_username(text: str) -> str:
    """'Priya Menon' -> 'priya.menon': what a person probably meant."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return ".".join(words)[:64]


def username_problem(username: str) -> str | None:
    """Why ``username`` cannot be used, in words the person can act on, or None."""
    if not username:
        return "enter a username"
    if " " in username:
        return f"a username cannot contain spaces: try {suggest_username(username)}"
    if "@" in username:
        return None if EMAIL.match(username) and len(username) <= 64 and not USERNAME_CHARS.search(username) \
            else "that email address is not valid as a username"
    if len(username) < 3:
        return "a username needs at least 3 characters"
    if len(username) > 64:
        return "a username can have at most 64 characters"
    bad = sorted(set(USERNAME_CHARS.findall(username)) | ({"+"} & set(username)))
    if bad:
        return (f"a username can only use letters, digits, dots, dashes and underscores "
                f"(not {' '.join(bad)}), or be your email address")
    if not USERNAME.match(username):
        return "a username must start with a letter or a digit"
    return None


class RegistrationRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def registration_open() -> bool:
    return bool((config().get("registration") or {}).get("borrowers", False))


def register(username: str, password: str, name: str, mobile: str, email: str | None) -> User:
    """Create a borrower account. Only ever the borrower role, whatever is sent."""
    if not registration_open():
        raise RegistrationRejected(403, "online registration is closed")
    username = (username or "").strip().lower()
    name = " ".join((name or "").split())
    mobile = re.sub(r"[\s-]", "", mobile or "")
    mobile = mobile[3:] if mobile.startswith("+91") else mobile
    email = (email or "").strip().lower()
    problem = username_problem(username)
    if problem:
        raise RegistrationRejected(422, problem)
    if not 2 <= len(name) <= 80:
        raise RegistrationRejected(422, "enter your full name")
    if not MOBILE.match(mobile):
        raise RegistrationRejected(422, "enter a 10-digit Indian mobile number")
    if email and not EMAIL.match(email):
        raise RegistrationRejected(422, "enter a valid email address, or leave it empty")
    _check_password(password, username)
    roles = set(config().get("roles") or {})
    staff = set(config().get("users") or {})
    # Its own lock: config() takes _lock, and a threading.Lock cannot be taken twice.
    with _register_lock:
        if username in RESERVED or username in roles or username in staff \
                or _registered(username) is not None:
            raise RegistrationRejected(409, "that username is taken; choose another")
        store.insert("borrower_accounts", {
            "username": username, "name": name, "mobile": mobile, "email": email or None,
            "password": hash_password(password), "created_at": store.now()})
    found = user(username)
    if found is None:                                           # pragma: no cover
        raise RegistrationRejected(500, "the account could not be created")
    return found


def _check_password(password: str, username: str) -> None:
    if len(password or "") < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise RegistrationRejected(422, "use a password of at least 8 characters with a letter and a digit")
    if password.lower() == username:
        raise RegistrationRejected(422, "the password cannot be the username")


# ------------------------------------------------- logins issued by the branch

# No 0/O or 1/l/I: the RM reads this out or writes it down.
_TEMP_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"


def temporary_password() -> str:
    while True:
        candidate = "".join(secrets.choice(_TEMP_CHARS) for _ in range(10))
        if re.search(r"[A-Za-z]", candidate) and re.search(r"\d", candidate):
            return candidate


def _account_by_mobile(mobile: str) -> dict[str, Any] | None:
    try:
        return store.fetch_one("SELECT * FROM borrower_accounts WHERE mobile = ? ORDER BY created_at LIMIT 1",
                               (mobile,))
    except Exception:                                           # noqa: BLE001
        return None


def issue_customer_login(name: str, mobile: str, email: str | None, *, issued_by: str,
                         loan_id: str | None) -> tuple[User, str | None, bool]:
    """A borrower login for a customer an RM is serving: (user, temporary password, created).

    A customer who already has an account - registered online or issued before -
    keeps it: the file is linked to it and no password changes (None). Otherwise
    the account is created with the mobile number as username (or the name, if
    that number is somehow taken by a username) and a temporary password, which is
    returned once and never stored in clear.
    """
    name = " ".join((name or "").split())
    mobile = re.sub(r"[\s-]", "", mobile or "")
    mobile = mobile[3:] if mobile.startswith("+91") else mobile
    email = (email or "").strip().lower()
    if not 2 <= len(name) <= 80:
        raise RegistrationRejected(422, "the customer's full name is needed to create their login")
    if not MOBILE.match(mobile):
        raise RegistrationRejected(422, "a 10-digit mobile number is needed to create the customer's login")
    if email and not EMAIL.match(email):
        email = ""
    staff = set(config().get("users") or {}) | set(config().get("roles") or {}) | RESERVED
    with _register_lock:
        existing = _account_by_mobile(mobile)
        if existing is not None:
            found = user(existing["username"])
            if found is not None:
                return found, None, False
        username = mobile
        base, n = suggest_username(name) or "customer", 1
        while username in staff or _registered(username) is not None or username_problem(username):
            n += 1
            username = f"{base}{n}"
        password = temporary_password()
        now = store.now()
        store.insert("borrower_accounts", {
            "username": username, "name": name, "mobile": mobile, "email": email or None,
            "password": hash_password(password), "created_at": now})
        store.insert("customer_logins", {"username": username, "issued_by": issued_by,
                                         "loan_id": loan_id, "issued_at": now,
                                         "password_changed_at": None})
    found = user(username)
    if found is None:                                           # pragma: no cover
        raise RegistrationRejected(500, "the login could not be created")
    return found, password, True


def reset_customer_password(username: str, *, issued_by: str, loan_id: str | None) -> str:
    """A new temporary password for a borrower who has lost theirs; they set their own again."""
    if not is_borrower_account(username):
        raise RegistrationRejected(404, "this file has no customer login")
    password = temporary_password()
    with _register_lock:
        with store.connect() as conn:
            conn.execute("UPDATE borrower_accounts SET password = ? WHERE username = ?",
                         (hash_password(password), username))
        previous = _issued(username) or {}
        store.insert("customer_logins", {"username": username, "issued_by": issued_by,
                                         "loan_id": loan_id or previous.get("loan_id"),
                                         "issued_at": store.now(), "password_changed_at": None})
    with _lock:
        _failures.pop(username, None)
    return password


def change_password(username: str, current: str, new: str) -> User:
    """A borrower sets their own password. Staff passwords live in the access file."""
    if not is_borrower_account(username):
        raise RegistrationRejected(400, "staff passwords are set by the administrator")
    row = _registered(username) or {}
    if not verify_password(current or "", row.get("password") or _DUMMY_HASH):
        raise RegistrationRejected(403, "the current password is not right")
    _check_password(new, username)
    if current == new:
        raise RegistrationRejected(422, "choose a password different from the temporary one")
    with _register_lock:
        with store.connect() as conn:
            conn.execute("UPDATE borrower_accounts SET password = ? WHERE username = ?",
                         (hash_password(new), username))
            conn.execute("UPDATE customer_logins SET password_changed_at = ? WHERE username = ?",
                         (store.now(), username))
    found = user(username)
    if found is None:                                           # pragma: no cover
        raise RegistrationRejected(500, "the password could not be changed")
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
