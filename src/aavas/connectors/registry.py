"""The external systems this engine reads from, and whether it may.

Five systems across three layers, declared in ``config/connectors.yaml``. The
registry's job is not to fetch anything - it is to answer, for each connector,
*may this run right now and why not*. Three gates, in order:

1. **enabled** - a switch a business owner sets.
2. **transport** - is the MCP server actually reachable.
3. **consent** - for a connector that reads a customer's own words, is consent
   on record for this customer and this purpose.

The third is the one that matters. There is no consent store in this repository,
so every communication connector is blocked whatever its switch says. That is
the correct behaviour and it is stated rather than implied: a pricing engine
that reads a customer's WhatsApp because a YAML flag was flipped has a consent
problem, not a configuration problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import load
from ..logging_utils import get_logger

log = get_logger(__name__)

# Where a connector's output is allowed to land. Anything else is a bug in the
# config, not a new capability.
LAYERS = {"acceptance_signals", "application_record", "economics", "outcomes"}


@dataclass
class Connector:
    name: str
    system: str
    supplies: str
    consumes: str
    reliability: str
    description: str
    enabled: bool = False
    consent_required: bool = False
    personal_data: bool = False
    transport: dict[str, Any] = field(default_factory=dict)
    fields: list[str] = field(default_factory=list)
    proposes: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------- gates
    def blocked_reason(self, consent_backend: str, reachable: bool | None = None) -> str | None:
        """Why this connector may not run, or None if it may."""
        if not self.enabled:
            return "switched off in config/connectors.yaml"
        if self.consumes not in LAYERS:
            return f"declares an unknown layer {self.consumes!r}"
        if self.consent_required and consent_backend in ("none", None, ""):
            return ("reads a customer's own words and no consent store is configured "
                    "(connectors.consent.backend is 'none')")
        if reachable is False:
            server = self.transport.get("server", "?")
            return f"MCP server {server!r} is not reachable from the API host"
        return None

    def as_dict(self, consent_backend: str = "none",
                reachable: bool | None = None) -> dict[str, Any]:
        blocked = self.blocked_reason(consent_backend, reachable)
        return {
            "name": self.name, "system": self.system, "supplies": self.supplies,
            "consumes": self.consumes, "reliability": self.reliability,
            "description": self.description, "enabled": self.enabled,
            "consent_required": self.consent_required, "personal_data": self.personal_data,
            "transport": self.transport, "fields": self.fields, "proposes": self.proposes,
            "reachable": reachable,
            "status": "blocked" if blocked else "ready",
            "blocked_reason": blocked,
        }


def _config() -> dict[str, Any]:
    return load("connectors")


def consent_backend() -> str:
    return str((_config().get("consent") or {}).get("backend") or "none")


def defaults() -> dict[str, Any]:
    return dict(_config().get("defaults") or {})


def all_connectors() -> dict[str, Connector]:
    base = defaults()
    out: dict[str, Connector] = {}
    for name, settings in (_config().get("connectors") or {}).items():
        out[name] = Connector(
            name=name,
            system=settings.get("system", name),
            supplies=settings.get("supplies", "unknown"),
            consumes=settings.get("consumes", "unknown"),
            reliability=settings.get("reliability", "salesperson_estimate"),
            description=settings.get("description", ""),
            enabled=bool(settings.get("enabled", False)),
            consent_required=bool(settings.get("consent_required", False)),
            personal_data=bool(settings.get("personal_data", False)),
            transport={**{"kind": base.get("transport", "mcp")},
                       **(settings.get("transport") or {})},
            fields=list(settings.get("fields") or []),
            proposes=list(settings.get("proposes") or []),
            settings={**base, **settings},
        )
    return out


def for_layer(layer: str) -> list[Connector]:
    """Connectors whose output is allowed to land in one layer.

    The acceptance agents call this rather than iterating every connector: a
    Salesforce record is not an acceptance signal, and an engine that let it
    become one would be extracting intent from a typed column.
    """
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}; expected one of {sorted(LAYERS)}")
    return [c for c in all_connectors().values() if c.consumes == layer]


def usable(layer: str, reachability: dict[str, bool] | None = None) -> list[Connector]:
    """The connectors in one layer that may actually run right now."""
    backend = consent_backend()
    out = []
    for connector in for_layer(layer):
        reachable = (reachability or {}).get(connector.name)
        if connector.blocked_reason(backend, reachable) is None:
            out.append(connector)
    return out


def status(reachability: dict[str, bool] | None = None) -> dict[str, Any]:
    backend = consent_backend()
    rows = [c.as_dict(backend, (reachability or {}).get(name))
            for name, c in sorted(all_connectors().items())]
    return {
        "connectors": rows,
        "version": _config().get("connectors_config_version"),
        "consent": {**(_config().get("consent") or {}),
                    "configured": backend not in ("none", None, "")},
        "by_layer": {layer: [r["name"] for r in rows if r["consumes"] == layer]
                     for layer in sorted(LAYERS)},
        "ready": [r["name"] for r in rows if r["status"] == "ready"],
        "blocked": [r["name"] for r in rows if r["status"] == "blocked"],
    }
