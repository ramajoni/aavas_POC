"""External-system connectors: WhatsApp, Gmail, Outlook, Salesforce, Oracle Fusion.

Declared in ``config/connectors.yaml``, reached over MCP, and gated on consent
before a customer's own words are read. See :mod:`registry` for the gates.

``all_connectors`` rather than ``registry``: the latter would shadow the
submodule of that name, which is a trap for anyone importing it.
"""

from . import mcp_client, registry
from .mcp_client import McpResult, call, probe, reachable
from .registry import Connector, all_connectors, consent_backend, for_layer, status, usable

__all__ = ["Connector", "McpResult", "all_connectors", "call", "consent_backend",
           "for_layer", "mcp_client", "probe", "reachable", "registry", "status", "usable"]
