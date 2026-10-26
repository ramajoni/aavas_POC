"""Federated server - runs rounds over the regional clients.

In-process simulation today. With Flower installed, the same loop becomes
``fl.server.start_server`` with a strategy wrapping ``aggregate``; the client
contract already matches ``NumPyClient``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import federated_config
from ..logging_utils import get_logger
from . import strategy as strategies
from .client import RegionalAdjustmentClient

log = get_logger(__name__)


@dataclass
class FederatedRun:
    global_parameters: np.ndarray
    history: list[dict] = field(default_factory=list)
    client_parameters: dict[str, np.ndarray] = field(default_factory=dict)
    client_sizes: dict[str, int] = field(default_factory=dict)
    # Which strategy actually ran, and whether the proximal term was active. A
    # run that cannot say which of the two it was is not a comparable result.
    strategy: str = "fedavg"
    proximal_mu: float = 0.0
    run_id: str | None = None

    def summary(self) -> dict:
        return {
            "rounds": len(self.history),
            "run_id": self.run_id,
            "strategy": self.strategy,
            "proximal_mu": self.proximal_mu,
            "regions": list(self.client_parameters),
            "client_sizes": self.client_sizes,
            "final_global_mae_bps": self.history[-1]["global_mae_bps"] if self.history else None,
            "history": self.history,
        }


def run(clients: list[RegionalAdjustmentClient], rounds: int | None = None,
        strategy_name: str | None = None, persist: bool = True) -> FederatedRun:
    """``persist=False`` computes the same run (and the same id) without recording
    it: a screen showing the evidence should not write a round each time it opens."""
    cfg = federated_config()
    rounds = rounds or cfg["rounds"]
    # Resolve the strategy once. FedProx differs from FedAvg in the *client*
    # objective, not the aggregation, so the name has to reach both the
    # aggregator and the fit config - reading the argument for one and the config
    # for the other let run(strategy_name="fedprox") aggregate as FedProx and
    # train without the proximal term, which is plain FedAvg wearing the name.
    strategy_name = strategy_name or cfg["strategy"]
    aggregate = strategies.get(strategy_name)

    if len(clients) < cfg["min_clients_per_round"]:
        raise ValueError(
            f"{len(clients)} clients is below the configured minimum of {cfg['min_clients_per_round']}")

    global_parameters = np.zeros_like(clients[0].get_parameters())
    result = FederatedRun(global_parameters=global_parameters)
    fit_config = {"local_epochs": cfg["local_epochs"],
                  "proximal_mu": (cfg.get("proximal_mu", 0.01)
                                  if strategy_name == "fedprox" else 0.0)}

    for round_index in range(1, rounds + 1):
        updates = []
        for client in clients:
            parameters, n, _ = client.fit(global_parameters, fit_config)
            updates.append((parameters, n))
        global_parameters = aggregate(updates)

        losses = [(client.evaluate(global_parameters)[0], len(client.dataset)) for client in clients]
        total = sum(n for _, n in losses) or 1
        global_mae = sum(loss * n for loss, n in losses) / total
        result.history.append({
            "round": round_index,
            "global_mae_bps": round(float(global_mae), 3),
            "per_region_mae_bps": {c.dataset.region: round(float(l), 3) for c, (l, _) in zip(clients, losses)},
        })

    result.global_parameters = global_parameters
    result.client_parameters = {c.dataset.region: c.local_parameters.copy() for c in clients}
    result.client_sizes = {c.dataset.region: len(c.dataset) for c in clients}
    result.strategy = strategy_name
    result.proximal_mu = fit_config["proximal_mu"]
    result.run_id = _persist(result) if persist else run_id_for(result)
    log.info("federated run complete: %s rounds, strategy %s, final global MAE %.2f bps (%s)",
             rounds, strategy_name, result.history[-1]["global_mae_bps"], result.run_id)
    return result


def run_id_for(result: "FederatedRun") -> str:
    """A content hash of what each client trained: the same data and settings give the same id."""
    import hashlib
    import json

    blob = json.dumps({r: p.tolist() for r, p in result.client_parameters.items()},
                      sort_keys=True).encode()
    return f"FED-{hashlib.sha256(blob).hexdigest()[:12]}"


def _persist(result: "FederatedRun") -> str | None:
    """Record the round so a decision can name the branch model that was active.

    Until this existed, federated learning had an implementation and no
    persistence: no round, participant or regional adjustment was on file, so the
    provenance question "which branch model priced this" had nothing to resolve
    against. Degrades rather than failing - a training run must not die because
    the audit store is unwritable.
    """
    try:
        from ..governance import audit, store

        run_id = run_id_for(result)
        digest = run_id.removeprefix("FED-")
        store.insert("federated_rounds", {
            "run_id": run_id, "created_at": store.now(), "strategy": result.strategy,
            "proximal_mu": result.proximal_mu, "rounds": len(result.history),
            "participants": len(result.client_parameters),
            "global_version": f"FEDG-{digest[:8]}",
            "final_mae_bps": result.history[-1]["global_mae_bps"] if result.history else None,
            "client_sizes": result.client_sizes,
            "per_region": result.history[-1].get("per_region_mae_bps") if result.history else {},
            "global_parameters": list(map(float, result.global_parameters)),
        })
        audit.record("federated_server", "record_federated_round", "federated", run_id,
                     {"strategy": result.strategy, "participants": len(result.client_parameters)})
        return run_id
    except Exception as exc:                                    # noqa: BLE001
        log.warning("federated round not persisted: %s", exc)
        return None
