"""Logging + optional MLflow tracking, both safe to use when nothing is installed."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from .config import env_bool, model_config

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=os.environ.get("AAVAS_LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def tracking_enabled() -> bool:
    cfg = model_config().get("tracking", {})
    if not env_bool("AAVAS_TRACKING_ENABLED", bool(cfg.get("enabled", False))):
        return False
    try:
        import mlflow  # noqa: F401
    except Exception:
        get_logger(__name__).warning("tracking requested but mlflow is unavailable")
        return False
    return True


@contextmanager
def run(name: str, params: dict[str, Any] | None = None) -> Iterator[Any]:
    """MLflow run if tracking is on and mlflow importable, else a no-op."""
    if not tracking_enabled():
        yield None
        return

    import mlflow

    cfg = model_config()["tracking"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["uri"]))
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=name) as active:
        if params:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        yield active


def log_metrics(active: Any, metrics: dict[str, float]) -> None:
    if active is None:
        return
    import mlflow

    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
