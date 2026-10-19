"""Aggregation strategies. FedAvg by default, sample-weighted."""

from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)


def fedavg(updates: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """Sample-weighted average - the standard FedAvg aggregation."""
    valid = [(p, n) for p, n in updates if n > 0]
    if not valid:
        raise ValueError("no client updates to aggregate")
    total = sum(n for _, n in valid)
    return sum(p * (n / total) for p, n in valid)


def uniform_average(updates: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """Equal weight per region.

    Worth having as an option: sample-weighted averaging lets the largest state
    set the price everywhere, which is precisely the failure mode regional
    personalization exists to avoid.
    """
    valid = [p for p, n in updates if n > 0]
    if not valid:
        raise ValueError("no client updates to aggregate")
    return sum(valid) / len(valid)


def trimmed_mean(updates: list[tuple[np.ndarray, int]], trim: float = 0.1) -> np.ndarray:
    """Robust aggregation - drops the extreme coordinate values before averaging."""
    valid = np.array([p for p, n in updates if n > 0])
    if len(valid) == 0:
        raise ValueError("no client updates to aggregate")
    k = int(len(valid) * trim)
    if k == 0 or len(valid) <= 2 * k:
        return valid.mean(axis=0)
    ordered = np.sort(valid, axis=0)
    return ordered[k:len(ordered) - k].mean(axis=0)


STRATEGIES = {"fedavg": fedavg, "fedavg_weighted": fedavg, "uniform": uniform_average,
              "trimmed_mean": trimmed_mean, "fedprox": fedavg}


def get(name: str):
    if name not in STRATEGIES:
        raise KeyError(f"unknown federated strategy {name!r}; expected one of {sorted(STRATEGIES)}")
    return STRATEGIES[name]
