"""The agentic AutoML search.

Implements the requested flow end to end:

    Algorithm Selection -> HPO -> Imbalance Handling -> Calibration -> Ensemble
    -> Fairness Check -> Profit/PD Optimisation -> Champion Selection

Two properties are structural rather than configurable.

**Contract-scoped candidates.** The model suite is not homogeneous. `conversion`
and `prepayment` are monotone-constrained in price, so for those contracts the
search only ever sees monotone-capable learners, and an ensemble that would mix a
monotone with a non-monotone member is refused rather than silently unblended.
`stage1`, `stage2`, `risk` and `override` carry no such requirement and get the
full catalogue.

**Chronological splits throughout.** Out-of-time validation is the default, not an
option, for the reason `ingestion/loader.py` states: a random split lets the model
peek at the future.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..config import load
from ..logging_utils import get_logger
from ..kernel import fairness
from . import (calibration, ensembles, hpo, imbalance, locality,
               registry, selection)
from .tracking import Tracker

log = get_logger(__name__)

# Contracts whose learner must be able to carry a monotone constraint.
MONOTONE_CONTRACTS = {"conversion", "prepayment"}


@dataclass
class SearchConfig:
    contract: str = "generic"
    hpo_strategy: str = "random"
    hpo_budget: int = 12
    imbalance_strategy: str = "auto"
    calibration_methods: Sequence[str] = ("platt", "isotonic", "beta")
    ensemble_methods: Sequence[str] = ("voting_soft", "weighted")
    ensemble_top_k: int = 3
    families: list[str] | None = None
    include: list[str] | None = None
    exclude: list[str] | None = None
    interpretable_only: bool = False
    weights: dict[str, float] | None = None
    gates: dict[str, Any] | None = None
    seed: int = 42

    @property
    def monotone_required(self) -> bool:
        return self.contract in MONOTONE_CONTRACTS

    @classmethod
    def from_yaml(cls, contract: str = "generic") -> "SearchConfig":
        cfg = load("automl")
        base = dict(cfg.get("defaults", {}))
        base.update(cfg.get("contracts", {}).get(contract, {}))
        base["contract"] = contract
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in base.items() if k in known})


@dataclass
class SearchResult:
    contract: str
    selection: selection.Selection
    steps: list[dict[str, Any]] = field(default_factory=list)
    candidates_considered: int = 0
    candidates_skipped: list[dict[str, str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"contract": self.contract, "elapsed_seconds": round(self.elapsed_seconds, 2),
                "candidates_considered": self.candidates_considered,
                "candidates_skipped": self.candidates_skipped,
                "steps": self.steps, **self.selection.as_dict()}


class AutoMLSearch:
    def __init__(self, config: SearchConfig | None = None):
        self.cfg = config or SearchConfig()

    # ------------------------------------------------------------------- run
    def run(self, X_train, y_train, X_valid, y_valid, *,
            groups_valid=None, units_valid=None, profit_fn: Callable | None = None,
            profit_scale: float | None = None) -> SearchResult:
        started = time.time()
        result = SearchResult(contract=self.cfg.contract, selection=None)  # type: ignore[arg-type]

        with Tracker(f"automl-search::{self.cfg.contract}",
                     params={"contract": self.cfg.contract,
                             "hpo_strategy": self.cfg.hpo_strategy,
                             "hpo_budget": self.cfg.hpo_budget,
                             "monotone_required": self.cfg.monotone_required},
                     tags={"stage": "automl", "contract": self.cfg.contract}) as tracker:

            # -------------------------------------------- 1. algorithm selection
            pool = self._algorithm_selection(result, n_rows=len(y_train))
            result.candidates_considered = len(pool)
            result.steps.append({"step": "algorithm_selection", "selected": [a.name for a in pool],
                                 "monotone_required": self.cfg.monotone_required})

            # ----------------------------------------------- 2. imbalance handling
            strategy = self.cfg.imbalance_strategy
            diagnosis = imbalance.diagnose(y_train)
            if strategy == "auto":
                strategy = diagnosis["recommended_strategy"]
            resampled = imbalance.apply(X_train, y_train, strategy, seed=self.cfg.seed)
            result.steps.append({"step": "imbalance", "diagnosis": diagnosis,
                                 **resampled.as_dict()})
            tracker.log_metrics({"train_positive_rate": diagnosis["positive_rate"]})

            # ------------------------------------ 3. HPO + 4. calibration per model
            # Chronological split of the validation window: the calibrator is fitted
            # on the earlier half and every metric is measured on the later half.
            # Fitting and scoring a calibrator on the same rows reports an ECE of
            # roughly zero for any model, which is not a finding.
            folds = _calibration_split(X_valid, y_valid, groups_valid, units_valid)
            result.steps.append({"step": "validation_split",
                                 "calibration_rows": int(len(folds["y_cal"])),
                                 "evaluation_rows": int(len(folds["y_eval"]))})

            candidates: list[selection.Candidate] = []
            for algo in pool:
                candidate = self._fit_candidate(
                    algo, resampled, folds, profit_fn, tracker)
                if candidate is not None:
                    candidates.append(candidate)

            # ------------------------------------------------------ 5. ensembles
            candidates.extend(self._build_ensembles(
                candidates, resampled, folds, profit_fn, tracker))

            # ------------------------- 6. fairness, 7. profit, 8. champion selection
            result.selection = selection.select(
                candidates, weights=self.cfg.weights, gates=self._gates(),
                profit_scale=profit_scale)
            result.steps.append({"step": "champion_selection",
                                 "champion": result.selection.champion.name
                                 if result.selection.champion else None,
                                 "referred": len(result.selection.referred_to_review)})

            result.elapsed_seconds = time.time() - started
            tracker.log_dict(result.as_dict(), "automl_search.json")
            if result.selection.champion is not None:
                tracker.set_tags({"champion": result.selection.champion.name})
                tracker.log_metrics({f"champion_{k}": v for k, v in
                                     result.selection.champion.metrics.items()
                                     if isinstance(v, (int, float))})
        return result

    # ------------------------------------------------------------------ steps
    def _algorithm_selection(self, result: SearchResult,
                             n_rows: int | None = None) -> list[registry.Algorithm]:
        pool = registry.candidates(
            monotone_required=self.cfg.monotone_required,
            interpretable_only=self.cfg.interpretable_only,
            families=self.cfg.families, exclude=self.cfg.exclude, n_rows=n_rows)
        if self.cfg.include:
            wanted = set(self.cfg.include)
            pool = [a for a in pool if a.name in wanted]

        for algo in registry.ALGORITHMS:
            if algo in pool:
                continue
            reason = algo.unavailable_reason()
            if reason is None and self.cfg.monotone_required and not algo.monotone:
                reason = "not monotone-capable; this contract constrains price"
            if reason is None and n_rows is not None and algo.max_rows is not None \
                    and n_rows > algo.max_rows:
                reason = f"impractical above {algo.max_rows:,} rows ({n_rows:,} given)"
            if reason:
                result.candidates_skipped.append({"name": algo.name, "reason": reason})
        return pool

    def _gates(self) -> dict[str, Any]:
        gates = dict(self.cfg.gates or {})
        gates.setdefault("require_monotone", self.cfg.monotone_required)
        return gates

    def _fit_candidate(self, algo, resampled, folds, profit_fn, tracker) -> selection.Candidate | None:
        with tracker.child(f"candidate::{algo.name}",
                           params={"algorithm": algo.name, "family": algo.family},
                           tags={"family": algo.family, "monotone": algo.monotone}) as child:
            try:
                def objective(params, resource: float = 1.0):
                    n = len(resampled.y)
                    take = max(int(n * resource), 50) if resource < 1.0 else n
                    X_s = _slice(resampled.X, slice(0, take))
                    y_s = np.asarray(resampled.y)[:take]
                    model = algo.instantiate(**{**params, **resampled.fit_params})
                    model.fit(X_s, y_s)
                    return float(selection.evaluate_predictions(
                        folds["y_cal"], _proba(model, folds["X_cal"])).get("auc", float("nan")))

                tuned = hpo.optimize(algo.search_space, objective,
                                     strategy=self.cfg.hpo_strategy,
                                     budget=self.cfg.hpo_budget, seed=self.cfg.seed)

                model = algo.instantiate(**{**tuned.best_params, **resampled.fit_params})
                model.fit(resampled.X, resampled.y)

                metrics, calibrated, p_eval = self._calibrate_and_score(
                    model, resampled, folds, profit_fn)

                fair = (fairness.evaluate(folds["y_eval"], p_eval, folds["groups_eval"])
                        if folds["groups_eval"] is not None else None)
                local = (locality.evaluate(folds["y_eval"], p_eval, folds["units_eval"])
                         if folds["units_eval"] is not None else None)

                child.log_params(tuned.best_params)
                child.log_metrics(metrics)
                if fair is not None:
                    child.set_tags({"fairness_passed": fair.get("passed")})
                if local is not None:
                    child.set_tags({"locality_passed": local.get("passed")})
                    child.log_metrics({"local_auc_spread":
                                       (local.get("variance", {}).get("auc") or {}).get("spread",
                                                                                        float("nan"))})

                notes = [f"hpo={tuned.strategy}", f"calibration={calibrated.method}"]
                if tuned.note:
                    notes.append(tuned.note)
                if resampled.applied:
                    notes.append(f"imbalance={resampled.strategy}")

                return selection.Candidate(
                    name=algo.name, family=algo.family, params=tuned.best_params,
                    metrics=metrics, monotone=algo.monotone,
                    interpretable=algo.interpretable, fairness=fair, locality=local,
                    notes=notes, estimator=model)
            except Exception as exc:  # noqa: BLE001 - one bad learner must not stop the search
                log.warning("automl: candidate %s failed (%s)", algo.name, exc)
                return None

    def _calibrate_and_score(self, model, resampled, folds, profit_fn):
        """Fit the calibrator on the earlier slice, measure everything on the later one."""
        p_cal = _proba(model, folds["X_cal"])
        calibrated = calibration.best(p_cal, folds["y_cal"], list(self.cfg.calibration_methods))
        if resampled.recalibration_required and calibrated.method == "none":
            calibrated = calibration.fit(p_cal, folds["y_cal"], "isotonic")

        p_eval = np.asarray(calibrated.predict(_proba(model, folds["X_eval"])), dtype=float)
        metrics = selection.evaluate_predictions(folds["y_eval"], p_eval, profit_fn=profit_fn)

        # Population stability: the model's own score distribution in-sample versus
        # out-of-time. Comparing raw with calibrated scores would only measure the
        # calibrator, which is meant to move the distribution.
        p_train = np.asarray(calibrated.predict(_proba(model, resampled.X)), dtype=float)
        metrics["psi"] = selection.psi(p_train, p_eval)
        metrics.update({f"calib_{k}": v for k, v in calibrated.metrics.items()})
        return metrics, calibrated, p_eval

    def _build_ensembles(self, candidates, resampled, folds, profit_fn,
                         tracker) -> list[selection.Candidate]:
        if len(candidates) < 2:
            return []
        ranked = sorted(candidates, key=lambda c: c.metrics.get("auc", 0.0), reverse=True)
        members = [c.name for c in ranked[: self.cfg.ensemble_top_k]
                   if c.name in registry.BY_NAME]
        if len(members) < 2:
            return []

        out: list[selection.Candidate] = []
        for method in self.cfg.ensemble_methods:
            try:
                spec = ensembles.build(method, members,
                                       monotone_required=self.cfg.monotone_required)
                spec.estimator.fit(resampled.X, resampled.y)
                metrics, calibrated, p_eval = self._calibrate_and_score(
                    spec.estimator, resampled, folds, profit_fn)
                fair = (fairness.evaluate(folds["y_eval"], p_eval, folds["groups_eval"])
                        if folds["groups_eval"] is not None else None)
                local = (locality.evaluate(folds["y_eval"], p_eval, folds["units_eval"])
                         if folds["units_eval"] is not None else None)

                with tracker.child(f"ensemble::{method}",
                                   params={"members": ",".join(members)},
                                   tags={"family": "ensemble"}) as child:
                    child.log_metrics(metrics)

                out.append(selection.Candidate(
                    name=f"ensemble_{method}", family="ensemble", params={"members": members},
                    metrics=metrics, monotone=spec.monotone, fairness=fair, locality=local,
                    notes=[f"members={members}", f"calibration={calibrated.method}"],
                    estimator=spec.estimator))
            except ValueError as exc:
                # The monotone refusal lands here. It is a finding, not a failure.
                log.info("automl: ensemble %s not built - %s", method, exc)
            except Exception as exc:  # noqa: BLE001
                log.warning("automl: ensemble %s failed (%s)", method, exc)
        return out


def _slice(X, sl):
    return X.iloc[sl] if hasattr(X, "iloc") else np.asarray(X)[sl]


def _calibration_split(X_valid, y_valid, groups_valid, units_valid=None,
                       fraction: float = 0.5) -> dict[str, Any]:
    """Split the validation window chronologically into calibrate / evaluate."""
    n = len(y_valid)
    cut = max(int(n * fraction), 1)
    y = np.asarray(y_valid)
    groups = None if groups_valid is None else np.asarray(groups_valid)
    units = None if units_valid is None else np.asarray(units_valid)
    return {
        "X_cal": _slice(X_valid, slice(0, cut)), "y_cal": y[:cut],
        "X_eval": _slice(X_valid, slice(cut, n)), "y_eval": y[cut:],
        "groups_eval": None if groups is None else groups[cut:],
        "units_eval": None if units is None else units[cut:],
    }


def _proba(model, X) -> np.ndarray:
    """Probability for the positive class, whatever the estimator exposes."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))   # squashed; calibration fixes the scale
    return np.asarray(model.predict(X), dtype=float)
