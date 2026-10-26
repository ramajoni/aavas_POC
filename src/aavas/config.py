"""Config access.

All tunable business and model behaviour lives in ``config/*.yaml``. Nothing in
``src`` may hard-code a rate, spread, cost or threshold that a business user
would reasonably want to change.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

from .paths import CONFIG_DIR

_FILES = {
    "model": "model_config.yaml",
    "policy": "pricing_policy.yaml",
    "features": "feature_config.yaml",
    "federated": "federated_config.yaml",
    "acceptance": "acceptance.yaml",
    "automl": "automl.yaml",
    "connectors": "connectors.yaml",
    "governance": "governance.yaml",
    "competitors": "competitors.yaml",
    "intake": "intake.yaml",
    "market_research": "market_research.yaml",
}


def config_path(name: str):
    """Where one config file lives. Read at call time so a test can point it elsewhere."""
    if name not in _FILES:
        raise KeyError(f"unknown config {name!r}; expected one of {sorted(_FILES)}")
    return CONFIG_DIR / _FILES[name]


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """Load and cache one config file by logical name."""
    with open(config_path(name), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def model_config() -> dict[str, Any]:
    return load("model")


def policy() -> dict[str, Any]:
    return load("policy")


def feature_config() -> dict[str, Any]:
    return load("features")


def federated_config() -> dict[str, Any]:
    return load("federated")


def acceptance_config() -> dict[str, Any]:
    return load("acceptance")


def automl_config() -> dict[str, Any]:
    return load("automl")


def connectors_config() -> dict[str, Any]:
    return load("connectors")


def reload() -> None:
    """Drop the cache (used by tests and by the /admin config reload path)."""
    load.cache_clear()


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def seed() -> int:
    """Global RNG seed: env wins so a run can be varied without editing config."""
    return int(os.environ.get("AAVAS_SEED", model_config().get("seed", 42)))


def versions() -> dict[str, str]:
    """Version stamp attached to every decision for audit and reproducibility."""
    from . import __version__

    # Imported lazily: governance imports config, so a module-level import here
    # would close the cycle. The manifest id is derived from code and cached, so
    # this costs a dict lookup rather than a query.
    from .governance import agent_registry

    return {
        "app_version": __version__,
        "policy_version": policy()["policy_version"],
        "feature_config_version": feature_config()["feature_config_version"],
        "model_config_version": model_config()["model_config_version"],
        # Which roster priced this, as distinct from which rules it priced under.
        "agent_manifest": agent_registry.current_id(),
    }
