"""Regional personalization.

A single global pricing model is the wrong default for a lender operating across
states whose competitive conditions genuinely differ. A single local model per
state is the other failure: thin regions overfit their own noise.

The served parameters are a blend, with the weight on the global model falling as
a region accumulates its own evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import federated_config
from ..logging_utils import get_logger
from .server import FederatedRun

log = get_logger(__name__)


@dataclass
class PersonalizedModel:
    region: str
    parameters: np.ndarray
    alpha: float                 # weight on the global model
    n_rows: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        design = np.hstack([np.asarray(X, dtype=float), np.ones((len(X), 1))])
        return design @ self.parameters

    def as_dict(self) -> dict:
        return {"region": self.region, "alpha_global": round(self.alpha, 3), "n_rows": self.n_rows,
                "parameters": [round(float(p), 5) for p in self.parameters]}


def blend(run: FederatedRun, region: str) -> PersonalizedModel:
    cfg = federated_config()["personalization"]
    local = run.client_parameters[region]
    n_rows = run.client_sizes[region]

    if not cfg["enabled"]:
        return PersonalizedModel(region, run.global_parameters.copy(), 1.0, n_rows)

    alpha = cfg["alpha"]
    if cfg.get("adaptive_alpha"):
        # A region with plenty of its own evidence leans local; a thin one leans global.
        share = min(n_rows / cfg["rows_for_full_local"], 1.0)
        alpha = float(np.clip(cfg["alpha"] * (1 - share) + 0.15 * share, 0.15, 1.0))

    parameters = alpha * run.global_parameters + (1 - alpha) * local
    return PersonalizedModel(region, parameters, alpha, n_rows)


def personalize_all(run: FederatedRun) -> dict[str, PersonalizedModel]:
    return {region: blend(run, region) for region in run.client_parameters}


def compare(run: FederatedRun, clients) -> pd.DataFrame:
    """Global vs local vs personalized, per region, on each region's holdout.

    This is the table that decides whether federated learning is worth its
    operational cost. If ``personalized_vs_global_bps`` is ~0 everywhere, a single
    central model is sufficient and federation is buying nothing but complexity.
    """
    personalized = personalize_all(run)
    rows = []
    for client in clients:
        region = client.dataset.region
        rows.append({
            "region": region,
            "n_rows": len(client.dataset),
            "global_mae_bps": round(client.evaluate(run.global_parameters)[0], 2),
            "local_mae_bps": round(client.evaluate(client.local_parameters)[0], 2),
            "personalized_mae_bps": round(client.evaluate(personalized[region].parameters)[0], 2),
            "alpha_global": round(personalized[region].alpha, 3),
        })
    table = pd.DataFrame(rows)
    table["personalized_vs_global_bps"] = (table["personalized_mae_bps"] - table["global_mae_bps"]).round(2)
    return table
