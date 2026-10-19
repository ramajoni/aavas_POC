"""MLflow connection for the AutoML search.

Extends the repo's existing no-op-capable helpers in ``logging_utils`` with what
a search needs: one parent run for the search, a nested child run per candidate,
and tags that make a run answerable later - which algorithm, which family, was it
monotone, did the fairness gate pass.

The rule from the rest of the repository holds. Tracking is off by default, and
when it is off (or mlflow is missing, or the tracking server is unreachable)
every call here is a no-op. A search must never fail because nobody started an
MLflow server.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from ..config import env, env_bool, model_config
from ..logging_utils import get_logger

log = get_logger(__name__)


def enabled() -> bool:
    cfg = model_config().get("tracking", {})
    if not env_bool("AAVAS_TRACKING_ENABLED", bool(cfg.get("enabled", False))):
        return False
    try:
        import mlflow  # noqa: F401
    except Exception:  # noqa: BLE001
        log.warning("automl: tracking requested but mlflow is unavailable")
        return False
    return True


def _mlflow():
    import mlflow

    cfg = model_config().get("tracking", {})
    mlflow.set_tracking_uri(env("MLFLOW_TRACKING_URI") or cfg.get("uri", "file:./mlruns"))
    mlflow.set_experiment(cfg.get("experiment", "aavas-pricing-poc"))
    return mlflow


class Tracker:
    """Thin wrapper so the search body never branches on whether tracking is on."""

    def __init__(self, run_name: str, params: dict[str, Any] | None = None,
                 tags: dict[str, Any] | None = None):
        self.run_name = run_name
        self.params = params or {}
        self.tags = tags or {}
        self.active = False
        self._mlflow = None
        self._run = None

    def __enter__(self) -> "Tracker":
        if not enabled():
            return self
        try:
            self._mlflow = _mlflow()
            self._run = self._mlflow.start_run(run_name=self.run_name)
            self._run.__enter__()
            self.active = True
            self.log_params(self.params)
            self.set_tags(self.tags)
        except Exception as exc:  # noqa: BLE001 - never fail a search over telemetry
            log.warning("automl: could not start MLflow run (%s); continuing untracked", exc)
            self.active = False
        return self

    def __exit__(self, *exc) -> bool:
        if self.active and self._run is not None:
            try:
                self._run.__exit__(*exc)
            except Exception:  # noqa: BLE001
                pass
        self.active = False
        return False

    # -------------------------------------------------------------------- log
    def log_params(self, params: dict[str, Any]) -> None:
        if not self.active or not params:
            return
        try:
            self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        except Exception as exc:  # noqa: BLE001
            log.debug("automl: log_params failed (%s)", exc)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.active or not metrics:
            return
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float)) and v == v}
        try:
            self._mlflow.log_metrics(clean, step=step)
        except Exception as exc:  # noqa: BLE001
            log.debug("automl: log_metrics failed (%s)", exc)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if not self.active or not tags:
            return
        try:
            self._mlflow.set_tags({k: str(v)[:250] for k, v in tags.items()})
        except Exception as exc:  # noqa: BLE001
            log.debug("automl: set_tags failed (%s)", exc)

    def log_dict(self, payload: dict[str, Any], filename: str) -> None:
        """Persist a JSON artefact - the leaderboard, the fairness report."""
        if not self.active:
            return
        try:
            self._mlflow.log_dict(payload, filename)
        except Exception as exc:  # noqa: BLE001
            log.debug("automl: log_dict failed (%s)", exc)

    @contextmanager
    def child(self, run_name: str, params: dict[str, Any] | None = None,
              tags: dict[str, Any] | None = None) -> Iterator["Tracker"]:
        """A nested run for one candidate, so the search reads as a tree."""
        if not self.active:
            yield Tracker(run_name)          # inert, same interface
            return
        try:
            nested = self._mlflow.start_run(run_name=run_name, nested=True)
        except Exception as exc:  # noqa: BLE001
            log.debug("automl: nested run failed (%s)", exc)
            yield Tracker(run_name)
            return

        holder = Tracker(run_name)
        holder._mlflow = self._mlflow
        holder.active = True
        with nested:
            holder.log_params(params or {})
            holder.set_tags(tags or {})
            try:
                yield holder
            finally:
                holder.active = False


def status() -> dict[str, Any]:
    """What `make automl-status` prints: is tracking on, and pointing where."""
    cfg = model_config().get("tracking", {})
    uri = env("MLFLOW_TRACKING_URI") or cfg.get("uri", "file:./mlruns")
    try:
        import mlflow

        version = mlflow.__version__
        installed = True
    except Exception:  # noqa: BLE001
        version, installed = None, False
    return {
        "mlflow_installed": installed,
        "mlflow_version": version,
        "tracking_enabled": enabled(),
        "tracking_uri": uri,
        "experiment": cfg.get("experiment", "aavas-pricing-poc"),
        "env_override": os.environ.get("AAVAS_TRACKING_ENABLED"),
    }
