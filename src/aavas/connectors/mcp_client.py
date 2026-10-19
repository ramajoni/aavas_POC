"""MCP transport for the connectors.

One client for every external system: one place to configure a server, one audit
path, one place to revoke. WhatsApp, Gmail, Outlook, Salesforce and Oracle
Fusion all arrive through here rather than through five bespoke SDK
integrations.

The repository rule for optional dependencies applies, and applies hard: the
``mcp`` package may be absent, the server may be unreachable, the call may time
out. Every one of those degrades to "unavailable", never to an exception that
takes a pricing request down - and never to a silent empty result that would let
an acceptance score claim it saw a source it could not read. ``call()`` returns
a result that says which of those happened.

``Exception`` is caught rather than ``ImportError`` for the reason the shap
probe does: a library built against a different dependency raises
``AttributeError``, and a broken optional dependency must degrade exactly like a
missing one.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ..logging_utils import get_logger

log = get_logger(__name__)

PROBE_TIMEOUT = 0.4


@dataclass
class McpResult:
    """What came back, or why nothing did."""

    connector: str
    server: str
    tool: str
    available: bool
    documents: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"connector": self.connector, "server": self.server, "tool": self.tool,
                "available": self.available, "documents": len(self.documents),
                "reason": self.reason}


def _client_available() -> tuple[bool, str | None]:
    try:
        import mcp                                              # noqa: F401
    except Exception as exc:                                    # noqa: BLE001
        return False, (f"the `mcp` client library is not usable in this interpreter "
                       f"({type(exc).__name__}). Install it to enable connectors.")
    return True, None


def reachable(endpoint: str | None, timeout: float = PROBE_TIMEOUT) -> bool | None:
    """Is something listening where this server is configured to be.

    Returns None when no endpoint is configured at all - "not configured" and
    "configured but down" are different findings and the caller reports them
    differently.
    """
    if not endpoint:
        return None
    parts = urlsplit(endpoint if "//" in endpoint else f"//{endpoint}")
    host, port = parts.hostname, parts.port
    if not host or not port:
        return None
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe(connectors) -> dict[str, bool]:
    """Reachability per connector, for the ones that name an endpoint."""
    out: dict[str, bool] = {}
    for connector in connectors:
        state = reachable(connector.transport.get("endpoint"))
        if state is not None:
            out[connector.name] = state
    return out


def call(connector, context: dict[str, Any] | None = None) -> McpResult:
    """Fetch from one connector's MCP server.

    Not implemented against a live server here, and it says so rather than
    pretending: this repository ships no MCP endpoint, and a connector that
    quietly returned nothing would let an acceptance score record a source it
    never read. Wiring this to a real client is a contained change - build the
    session from ``connector.transport`` and map the tool result onto
    ``documents``.
    """
    transport = connector.transport or {}
    server = str(transport.get("server", "?"))
    tool = str(transport.get("tool", "?"))
    result = McpResult(connector=connector.name, server=server, tool=tool, available=False)

    ok, why = _client_available()
    if not ok:
        result.reason = why
        return result
    if not transport.get("endpoint"):
        result.reason = (f"no endpoint configured for MCP server {server!r}; set "
                         f"connectors.{connector.name}.transport.endpoint")
        return result

    state = reachable(transport["endpoint"])
    if state is False:
        result.reason = f"MCP server {server!r} is not reachable at {transport['endpoint']}"
        return result

    result.reason = ("transport is configured but the live MCP session is not implemented "
                     "in this build; no documents were read")
    log.info("connector %s: %s", connector.name, result.reason)
    return result
