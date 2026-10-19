"""Hyperparameter optimisation - every search strategy in the catalogue.

Grid, random, Bayesian/TPE, Hyperband, successive halving, and an evolutionary
searcher that also covers the population-based / NAS row.

**Splits are chronological by default.** The gap-doc finding at section 8.4 was
that the existing Optuna objective in `models/stage2/train.py` tunes against a
random KFold, in a repo whose loader states plainly that a random split lets the
model peek at the future. That mistake is not repeated here: `cv` builds a
`TimeSeriesSplit` unless a caller explicitly asks otherwise.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

STRATEGIES = ["grid", "random", "bayesian", "tpe", "hyperband",
              "successive_halving", "evolutionary"]


@dataclass
class Trial:
    params: dict[str, Any]
    score: float
    resource: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class HPOResult:
    strategy: str
    best_params: dict[str, Any]
    best_score: float
    trials: list[Trial]
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "best_params": self.best_params,
                "best_score": round(float(self.best_score), 6),
                "n_trials": len(self.trials), "note": self.note}


def cv(n_splits: int = 3, chronological: bool = True):
    """Chronological by default - see the module docstring."""
    if chronological:
        from sklearn.model_selection import TimeSeriesSplit

        return TimeSeriesSplit(n_splits=n_splits)
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(n_splits=n_splits, shuffle=False)


# ------------------------------------------------------------------- sampling
def _sample(spec: tuple, rng: random.Random) -> Any:
    kind = spec[0]
    if kind == "int":
        return rng.randint(int(spec[1]), int(spec[2]))
    if kind == "float":
        return rng.uniform(float(spec[1]), float(spec[2]))
    if kind == "log":
        return math.exp(rng.uniform(math.log(float(spec[1])), math.log(float(spec[2]))))
    if kind == "choice":
        return rng.choice(list(spec[1]))
    raise ValueError(f"unknown search-space spec {spec!r}")


def _grid_values(spec: tuple, points: int) -> list[Any]:
    kind = spec[0]
    if kind == "choice":
        return list(spec[1])
    if kind == "int":
        lo, hi = int(spec[1]), int(spec[2])
        return sorted({int(round(v)) for v in np.linspace(lo, hi, points)})
    if kind == "log":
        return list(np.exp(np.linspace(math.log(spec[1]), math.log(spec[2]), points)))
    return list(np.linspace(float(spec[1]), float(spec[2]), points))


# ----------------------------------------------------------------- strategies
def _grid(space, objective, budget, rng) -> list[Trial]:
    if not space:
        return [Trial({}, objective({}))]
    points = max(2, int(round(budget ** (1 / max(len(space), 1)))))
    keys = list(space)
    combos = list(itertools.product(*[_grid_values(space[k], points) for k in keys]))[:budget]
    return [Trial(dict(zip(keys, combo)), objective(dict(zip(keys, combo)))) for combo in combos]


def _random(space, objective, budget, rng) -> list[Trial]:
    out = []
    for _ in range(budget):
        params = {k: _sample(v, rng) for k, v in space.items()}
        out.append(Trial(params, objective(params)))
    return out


def _optuna(space, objective, budget, sampler_name: str) -> list[Trial] | None:
    """Bayesian / TPE via Optuna when installed."""
    try:
        import optuna
    except Exception:  # noqa: BLE001
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)
    if sampler_name == "bayesian":
        try:
            sampler = optuna.integration.BoTorchSampler()
        except Exception:  # noqa: BLE001 - GP backend optional; TPE is the fallback
            sampler = optuna.samplers.TPESampler(seed=42)

    def _objective(trial):
        params = {}
        for key, spec in space.items():
            kind = spec[0]
            if kind == "int":
                params[key] = trial.suggest_int(key, int(spec[1]), int(spec[2]))
            elif kind == "log":
                params[key] = trial.suggest_float(key, float(spec[1]), float(spec[2]), log=True)
            elif kind == "choice":
                params[key] = trial.suggest_categorical(key, list(spec[1]))
            else:
                params[key] = trial.suggest_float(key, float(spec[1]), float(spec[2]))
        return objective(params)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_objective, n_trials=budget, show_progress_bar=False)
    return [Trial(t.params, float(t.value)) for t in study.trials if t.value is not None]


def _successive_halving(space, objective, n_configs, min_resource, factor, rng):
    """One bracket: evaluate cheaply, keep the top 1/factor, spend more on them."""
    def evaluate(params, resource):
        try:
            return objective(params, resource=resource)
        except TypeError:          # objective ignores the resource budget
            return objective(params)

    population = [{k: _sample(v, rng) for k, v in space.items()} for _ in range(n_configs)]
    resource = min_resource
    trials: list[Trial] = []

    while population:
        scored = [(p, evaluate(p, min(resource, 1.0))) for p in population]
        trials.extend(Trial(p, s, resource=min(resource, 1.0)) for p, s in scored)
        if len(scored) <= 1 or resource >= 1.0:
            break
        scored.sort(key=lambda kv: kv[1], reverse=True)
        population = [p for p, _ in scored[: max(1, len(scored) // factor)]]
        resource *= factor
    return trials


def _hyperband(space, objective, budget, rng, factor: int = 3):
    """Several halving brackets, trading breadth against depth.

    Bracket 0 is many configurations on a small slice; the last bracket is few
    configurations at full resource. Hyperband hedges across that trade-off
    instead of guessing which end is right.
    """
    max_brackets = max(1, int(math.log(max(budget, factor), factor)))
    trials: list[Trial] = []
    per_bracket = max(2, budget // max_brackets)

    for bracket in range(max_brackets):
        n_configs = max(2, per_bracket // (factor ** bracket) or 2)
        min_resource = 1.0 / (factor ** (max_brackets - 1 - bracket))
        trials.extend(_successive_halving(space, objective, n_configs,
                                          min(min_resource, 1.0), factor, rng))
    return trials


def _evolutionary(space, objective, budget, rng, *, population_size: int = 8,
                  mutation: float = 0.25) -> list[Trial]:
    """Evolutionary / population-based search.

    Covers the NAS row of the catalogue. Architecture search over a fixed set of
    tabular learners is hyperparameter search over a structured space; calling it
    NAS does not change what it does.
    """
    population = [{k: _sample(v, rng) for k, v in space.items()} for _ in range(population_size)]
    trials: list[Trial] = []
    evaluations = 0

    while evaluations < budget and population:
        scored = []
        for params in population:
            if evaluations >= budget:
                break
            score = objective(params)
            evaluations += 1
            scored.append((params, score))
            trials.append(Trial(params, score, meta={"generation": len(trials) // population_size}))
        if not scored or evaluations >= budget:
            break
        scored.sort(key=lambda kv: kv[1], reverse=True)
        survivors = [p for p, _ in scored[: max(2, len(scored) // 2)]]
        children = []
        while len(children) + len(survivors) < population_size:
            mother, father = rng.choice(survivors), rng.choice(survivors)
            child = {k: (mother[k] if rng.random() < 0.5 else father[k]) for k in space}
            for key, spec in space.items():
                if rng.random() < mutation:
                    child[key] = _sample(spec, rng)
            children.append(child)
        population = survivors + children
    return trials


# ---------------------------------------------------------------------- entry
def optimize(space: dict[str, tuple], objective: Callable[..., float], *,
             strategy: str = "random", budget: int = 20, seed: int = 42) -> HPOResult:
    """Maximise ``objective`` over ``space`` using one strategy."""
    rng = random.Random(seed)
    note = ""

    if not space:
        return HPOResult(strategy, {}, float(objective({})), [Trial({}, 0.0)],
                         "no search space declared; default params used")

    if strategy == "grid":
        trials = _grid(space, objective, budget, rng)
    elif strategy == "random":
        trials = _random(space, objective, budget, rng)
    elif strategy in {"bayesian", "tpe"}:
        trials = _optuna(space, objective, budget, strategy)
        if trials is None:
            note = "optuna unavailable; fell back to random search"
            log.warning("automl: %s", note)
            trials = _random(space, objective, budget, rng)
    elif strategy == "successive_halving":
        trials = _successive_halving(space, objective, max(budget, 4), 1.0 / 9, 3, rng)
    elif strategy == "hyperband":
        trials = _hyperband(space, objective, budget, rng)
    elif strategy == "evolutionary":
        trials = _evolutionary(space, objective, budget, rng)
    else:
        raise ValueError(f"unknown HPO strategy {strategy!r}; expected one of {STRATEGIES}")

    best = max(trials, key=lambda t: t.score)
    return HPOResult(strategy, best.params, best.score, trials, note)
