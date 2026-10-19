"""Prompts as registered, versioned objects.

A prompt is model input. An unversioned prompt on a priced path means a decision
cannot say what the model was actually asked - which is the same defect as an
unversioned model, and easier to miss because a prompt looks like a comment.

The id is derived from the content, for the reason the agent manifest's is: an
edited prompt gets a new id whether or not anyone remembered to bump one.

Nothing here is on the priced path today - the extraction agent is dormant. That
is exactly why it is worth having now: wiring that agent in without this would put
an unversioned prompt in front of a customer's own words.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..logging_utils import get_logger
from . import store

log = get_logger(__name__)


def _digest(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def prompt_id(name: str, body: str) -> str:
    return f"PRM-{name}-{_digest(body)[:8]}"


def register(name: str, body: str, *, role: str = "system", used_by: str = "",
             actor: str = "startup") -> str:
    identifier = prompt_id(name, body)
    if store.fetch_one("SELECT prompt_id FROM prompts WHERE prompt_id = ?", (identifier,)):
        return identifier
    store.insert("prompts", {
        "prompt_id": identifier, "name": name, "content_hash": _digest(body),
        "registered_at": store.now(), "role": role, "used_by": used_by, "body": body,
    })
    from . import audit
    audit.record(actor, "register_prompt", "prompt", identifier,
                 {"name": name, "chars": len(body), "used_by": used_by})
    log.info("registered prompt %s (%s chars)", identifier, len(body))
    return identifier


def register_declared(actor: str = "startup") -> list[str]:
    """Register every prompt the code declares.

    Declared rather than scanned, following the roster: a scan would find string
    constants and tell you nothing about which of them a model ever sees.
    """
    registered = []
    try:
        from ..acceptance import extraction_agent

        registered.append(register(
            "acceptance_extraction", extraction_agent.SYSTEM_PROMPT, role="system",
            used_by="aavas.acceptance.extraction_agent (dormant)", actor=actor))
    except Exception as exc:                                    # noqa: BLE001
        log.warning("could not register the extraction prompt: %s", exc)
    return registered


def get(identifier: str) -> dict[str, Any] | None:
    return store.fetch_one("SELECT * FROM prompts WHERE prompt_id = ?", (identifier,))


def registered(limit: int = 50) -> list[dict[str, Any]]:
    rows = store.query("SELECT prompt_id, name, content_hash, registered_at, role, used_by, "
                       "LENGTH(body) AS chars FROM prompts ORDER BY registered_at DESC LIMIT ?",
                       (limit,))
    return rows


def current_ids() -> dict[str, str]:
    """name -> id for the prompts the code declares right now."""
    out = {}
    try:
        from ..acceptance import extraction_agent
        out["acceptance_extraction"] = prompt_id("acceptance_extraction",
                                                 extraction_agent.SYSTEM_PROMPT)
    except Exception:                                           # noqa: BLE001
        pass
    return out
