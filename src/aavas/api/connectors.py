"""External-system connectors, and whether the engine may read them.

Five systems declared in ``config/connectors.yaml`` - WhatsApp, Gmail, Outlook,
Salesforce and Oracle Fusion - reached over MCP. This endpoint answers the only
question an operator actually has about them: *is this one running, and if not,
which gate is holding it*.

Three gates, reported separately because they are fixed in different ways: a
switch is flipped in config, an unreachable server is an infrastructure problem,
and a missing consent record is neither.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..connectors import mcp_client
from ..connectors.registry import all_connectors, consent_backend, status
from ..logging_utils import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/connectors", tags=["connectors"])


def _reachability() -> dict[str, bool]:
    return mcp_client.probe(all_connectors().values())


@router.get("", summary="Every external system, and which gate is holding it")
def index() -> dict[str, Any]:
    state = status(_reachability())
    client_ok, client_reason = mcp_client._client_available()
    return {
        **state,
        "transport": {"kind": "mcp", "client_available": client_ok,
                      "reason": client_reason},
        "note": ("Nothing is enabled. A connector that reads a customer's own words stays "
                 "blocked until a consent store exists, whatever its switch says - that is "
                 "the intended failure, not a gap in the config."),
    }


@router.get("/{name}", summary="One connector")
def detail(name: str) -> dict[str, Any]:
    connector = all_connectors().get(name)
    if connector is None:
        known = sorted(all_connectors())
        raise HTTPException(404, f"unknown connector {name!r}; expected one of {known}")
    reachable = mcp_client.reachable(connector.transport.get("endpoint"))
    return connector.as_dict(consent_backend(), reachable)
