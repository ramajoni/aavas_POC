"""Model registry and the loaded bundle the agents share.

Two responsibilities:

* record which model version is champion, with its card, so a decision made last
  quarter can be reproduced,
* load the artefacts once per process instead of once per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..logging_utils import get_logger
from ..models import base as model_base
from . import store

log = get_logger(__name__)

MODEL_NAMES = ["stage1", "stage2", "conversion", "prepayment", "risk", "override"]


# ------------------------------------------------------------------- registry


def register(card: model_base.ModelCard, role: str = "champion") -> None:
    store.insert("model_registry", {
        "name": card.name, "version": card.version, "role": role,
        "registered_at": store.now(), "card": card.as_dict(),
    })
    log.info("registered %s %s as %s", card.name, card.version, role)


def registered() -> list[dict[str, Any]]:
    rows = store.query("SELECT * FROM model_registry ORDER BY name, registered_at DESC")
    for row in rows:
        row["card"] = store.decode(row.get("card"))
    return rows


def champion(name: str) -> dict[str, Any] | None:
    row = store.fetch_one(
        "SELECT * FROM model_registry WHERE name = ? AND role = 'champion' ORDER BY registered_at DESC LIMIT 1",
        (name,))
    if row:
        row["card"] = store.decode(row.get("card"))
    return row


def promote(name: str, version: str, actor: str = "governance") -> None:
    from . import audit

    with store.connect() as conn:
        conn.execute("UPDATE model_registry SET role = 'retired' WHERE name = ? AND role = 'champion'", (name,))
        conn.execute("UPDATE model_registry SET role = 'champion' WHERE name = ? AND version = ?", (name, version))
    audit.record(actor, "promote_model", "model", f"{name}:{version}", {"new_role": "champion"})
    log.info("promoted %s %s to champion", name, version)


# --------------------------------------------------------------------- bundle


@dataclass
class ModelBundle:
    stage1: Any = None
    stage2: Any = None
    conversion: Any = None
    prepayment: Any = None
    risk: Any = None
    override: Any = None
    missing: list[str] = None            # type: ignore[assignment]

    def versions(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in MODEL_NAMES:
            model = getattr(self, name, None)
            card = getattr(model, "card", None)
            if card is not None:
                out[name] = card.version
        return out

    def cards(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name in MODEL_NAMES:
            card = getattr(getattr(self, name, None), "card", None)
            if card is not None:
                out[name] = card.as_dict()
        return out

    @property
    def ready(self) -> bool:
        return not self.missing


_BUNDLE: ModelBundle | None = None


def load_bundle(refresh: bool = False) -> ModelBundle:
    """Load every champion artefact. Missing models are reported, not fatal, so a
    partially trained environment still serves what it can."""
    global _BUNDLE
    if _BUNDLE is not None and not refresh:
        return _BUNDLE

    from ..models.conversion.predict import ConversionModel
    from ..models.override.predict import OverrideModel
    from ..models.prepayment.predict import PrepaymentModel
    from ..models.risk.predict import RiskModel
    from ..models.stage1.predict import Stage1PricingModel
    from ..models.stage2.predict import Stage2PricingModel

    loaders = {
        "stage1": Stage1PricingModel, "stage2": Stage2PricingModel,
        "conversion": ConversionModel, "prepayment": PrepaymentModel,
        "risk": RiskModel, "override": OverrideModel,
    }
    bundle = ModelBundle(missing=[])
    for name, cls in loaders.items():
        try:
            setattr(bundle, name, cls.load())
        except FileNotFoundError:
            bundle.missing.append(name)
            log.warning("model artefact missing: %s", name)
    _BUNDLE = bundle
    return bundle


def clear_cache() -> None:
    global _BUNDLE
    _BUNDLE = None
