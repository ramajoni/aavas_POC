"""Federated client.

Phase 2 of the plan. What is federated here is deliberately narrow: a small
linear *regional rate adjustment* fitted on the residual between the central
Stage-2 recommendation and what the region actually sanctions. Gradient-boosted
trees do not average across clients in any meaningful way; a parameter vector
does, so that is what crosses the wire.

The interface mirrors ``flwr.client.NumPyClient`` (``get_parameters`` / ``fit`` /
``evaluate``) so moving from this in-process simulation to real Flower transport
is a wiring change rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import federated_config, seed
from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class RegionalDataset:
    """A region's private data. Held out locally too: personalization has to be
    judged out of sample, otherwise a thin region's overfitted local model always
    looks better than the federated one and the comparison is meaningless."""

    region: str
    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.y)

    @property
    def has_holdout(self) -> bool:
        return self.y_test is not None and len(self.y_test) > 0


class RegionalAdjustmentClient:
    """One region. Holds its own data and never shares it - only parameters."""

    def __init__(self, dataset: RegionalDataset, learning_rate: float | None = None,
                 l2: float | None = None):
        cfg = federated_config()
        learning_rate = cfg.get("learning_rate", 0.3) if learning_rate is None else learning_rate
        l2 = cfg.get("l2", 1e-3) if l2 is None else l2
        return self.__init_impl(dataset, learning_rate, l2)

    def __init_impl(self, dataset: RegionalDataset, learning_rate: float, l2: float):
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.l2 = l2
        n_features = dataset.X.shape[1]
        rng = np.random.default_rng(seed())
        self.parameters = np.zeros(n_features + 1)      # weights + bias
        self.local_parameters = self.parameters.copy()
        self._rng = rng

    # ---------------------------------------------------------- flower-shaped
    def get_parameters(self, config: dict | None = None) -> np.ndarray:
        return self.parameters.copy()

    def set_parameters(self, parameters: np.ndarray) -> None:
        self.parameters = np.asarray(parameters, dtype=float).copy()

    def fit(self, parameters: np.ndarray, config: dict | None = None) -> tuple[np.ndarray, int, dict]:
        """Local training from the received global parameters."""
        config = config or {}
        epochs = int(config.get("local_epochs", federated_config()["local_epochs"]))
        self.set_parameters(parameters)

        X, y = self.dataset.X, self.dataset.y
        n = len(y)
        if n == 0:
            return self.get_parameters(), 0, {"skipped": 1.0}

        design = np.hstack([X, np.ones((n, 1))])
        proximal = config.get("proximal_mu", 0.0)
        global_parameters = np.asarray(parameters, dtype=float)

        for _ in range(epochs):
            residual = design @ self.parameters - y
            gradient = design.T @ residual / n + self.l2 * self.parameters
            if proximal:
                # FedProx: keep local training from wandering away from global.
                gradient = gradient + proximal * (self.parameters - global_parameters)
            self.parameters = self.parameters - self.learning_rate * gradient

        self.local_parameters = self.parameters.copy()
        loss = self.evaluate(self.parameters, on="train")[0]
        return self.get_parameters(), n, {"train_mae_bps": loss}

    def evaluate(self, parameters: np.ndarray, config: dict | None = None,
                 on: str = "test") -> tuple[float, int, dict]:
        """MAE in basis points. Defaults to the region's holdout, not its training
        rows - a local model fitted on 400 rows will always win on those 400 rows."""
        if on == "test" and self.dataset.has_holdout:
            X, y = self.dataset.X_test, self.dataset.y_test
        else:
            X, y = self.dataset.X, self.dataset.y
        if len(y) == 0:
            return float("nan"), 0, {}
        design = np.hstack([X, np.ones((len(y), 1))])
        predictions = design @ np.asarray(parameters, dtype=float)
        mae_bps = float(np.mean(np.abs(predictions - y)) * 100)
        return mae_bps, len(y), {"mae_bps": mae_bps}


def build_clients(df: pd.DataFrame, feature_columns: list[str], target_column: str,
                  partition_key: str | None = None) -> list[RegionalAdjustmentClient]:
    cfg = federated_config()
    key = partition_key or cfg["partition_key"]
    minimum = cfg["min_rows_per_client"]

    holdout = float(cfg.get("holdout_fraction", 0.25))
    rng = np.random.default_rng(seed())

    clients: list[RegionalAdjustmentClient] = []
    for region, group in df.groupby(key):
        if len(group) < minimum:
            log.info("skipping %s: %s rows is below the %s row minimum", region, len(group), minimum)
            continue
        X = group[feature_columns].to_numpy(dtype=float)
        # Standardising locally keeps one region's scale from dominating the average.
        X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        y = group[target_column].to_numpy(dtype=float)

        indices = rng.permutation(len(y))
        cut = max(int(len(y) * (1 - holdout)), 1)
        train_idx, test_idx = indices[:cut], indices[cut:]
        dataset = RegionalDataset(str(region), X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        clients.append(RegionalAdjustmentClient(dataset))
    return clients
