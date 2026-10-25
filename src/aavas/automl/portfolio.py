"""Model-portfolio selection: the AutoML search behind the five deck models.

The deck says each model was chosen by AutoML, out of time, on "best validation
score, then explainability". This module is that claim, made checkable:

* one catalogue of algorithm families per task (``regression``, ``quantile``,
  ``binary``), each with a small random-search space, built lazily so a missing
  library is a *skipped* family with a reason rather than an ImportError;
* every trial fitted on the earlier part of the training window and scored on
  the later part - the trainers' own holdout (``loader.train_test_split``, the
  last 20%, roughly the last four months) is never handed to this module at all;
* every trial recorded - family, params, validation metrics, fit seconds - so
  the leaderboard is evidence rather than a summary of it;
* the deck's rule applied literally: the best validation score wins, unless a
  more explainable family is within ``selection.tolerance_rel`` of it (see
  ``EXPLAINABILITY``), and the report says which rule fired.

The selection only matters because the trainers consume it. ``build`` hands a
trainer a freshly constructed estimator for the selected family and params, with
the contract's monotone constraints and the quantile ``alpha`` applied, and falls
back to the trainer's own learner when nothing was selected or the selected
library is missing. ``make train`` therefore works before any search has run,
and after one it fits exactly what the search chose.

Interface used by the trainers (and by the LGD / uplift workstream), kept stable:

* ``SelectionData``  - what a trainer hands the search for one model.
* ``build(name, task, spec, fallback, alpha=None)`` - the estimator a trainer fits.
* ``selected(name)`` - the recorded selection for one model, or None.

Deliberately *not* AutoGluon, Optuna or any other framework: none is installed
here, and a report claiming one would be false. The ``framework`` field of every
report names this module and the library versions that actually ran.

Imports only the shared kernel (config, ingestion, logging, paths), like the rest
of ``aavas.automl`` - a feature spec arrives as an argument, duck-typed.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import model_config, seed as global_seed
from ..logging_utils import get_logger
from ..paths import REPORTS_DIR, ROOT

log = get_logger(__name__)

TASKS = ("regression", "quantile", "binary")
REPORT_DIR = REPORTS_DIR / "automl" / "portfolio"
SELECTION_FILE = REPORT_DIR / "selection.json"

# Explainability classes, most explainable first. Used for one thing only: to
# break a near-tie on validation score (the deck's "then explainability").
#
#   linear         Coefficients are the model. The per-loan attribution is exact
#                  (coefficient x distance from the training mean) and the global
#                  weights can be read straight off the fit.
#   boosted_trees  Shallow additive trees. Exact TreeSHAP per loan - native in
#                  XGBoost, LightGBM and CatBoost, shap.TreeExplainer for sklearn
#                  - and they carry monotone constraints, so the shape of a price
#                  effect is guaranteed rather than inspected.
#   bagged_trees   Hundreds of deep, independently grown trees. TreeSHAP is still
#                  exact, but the attribution is spread over interactions no one
#                  can follow, explaining one loan is slow, and the artefact is
#                  large - so they rank last.
#
# Within a class there is no ranking: the better validation score wins.
EXPLAINABILITY = ("linear", "boosted_trees", "bagged_trees")


# ===================================================================== contract
@dataclass
class SelectionData:
    """Fit / validation frames for one model, split out of time.

    ``task`` is one of TASKS. ``metric`` names the primary validation metric the
    search ranks on. ``quantiles`` is set only for a quantile task.

    Optional additions (defaults keep older callers working):

    * ``target``    - the target column, for the report.
    * ``incumbent`` - ``{"family": ..., "params": {...}}``: the learner the
      trainer fits today. It becomes trial 0 of its family, so the search can
      never report a winner that loses to what is already deployed without
      that being visible on the leaderboard.
    * ``scale`` / ``units`` - how a target difference is reported: a rate in %
      is x100 -> bps; a severity is percentage points. ``units`` is inferred
      from ``metric`` when it ends in ``_bps`` / ``_pts``. ``scale`` left None
      is inferred too: 100 for a rate or a 0-1 fraction, 1 for a target already
      in points (any |y| above 1.5 on a ``pts`` metric).
    """

    name: str
    task: str
    X_fit: pd.DataFrame
    y_fit: Any
    X_valid: pd.DataFrame
    y_valid: Any
    spec: Any = None
    metric: str = ""
    quantiles: list[float] | None = None
    window: dict[str, Any] = field(default_factory=dict)
    target: str = ""
    incumbent: dict[str, Any] | None = None
    scale: float | None = None
    units: str = ""

    def unit(self) -> str:
        if self.units:
            return self.units
        for suffix in ("bps", "pts"):
            if self.metric.endswith(f"_{suffix}"):
                return suffix
        return "bps"

    def factor(self) -> float:
        if self.scale is not None:
            return float(self.scale)
        if self.task == "binary":
            return 1.0
        if self.unit() == "pts":
            y = np.abs(np.asarray(self.y_fit, dtype=float))
            if y.size and np.nanmax(y) > 1.5:
                return 1.0
        return 100.0


# =================================================================== catalogue
@dataclass
class Family:
    """One algorithm family for one task.

    ``monotone`` names how the family carries a monotone constraint - the keyword
    its estimator takes - or None when it cannot. ``space`` is the random-search
    space; ``defaults`` is trial 0 when the family is not the incumbent.
    """

    name: str
    task: str
    label: str
    explainability: str
    make: Callable[..., Any]
    library: str = "sklearn"
    monotone: str | None = None
    no_monotone_reason: str = ""
    # Set when the family can carry monotone constraints but cannot then be
    # explained exactly; such a family is excluded from a monotone contract.
    monotone_unexplainable: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)
    space: dict[str, tuple] = field(default_factory=dict)

    @property
    def tier(self) -> int:
        return EXPLAINABILITY.index(self.explainability) + 1

    def available(self) -> str | None:
        """None when the family can run, else the reason it cannot."""
        try:
            importlib.import_module(self.library)
        except Exception as exc:                       # noqa: BLE001 - broken == missing
            return f"{self.library} is not importable here ({type(exc).__name__}: {exc})"
        return None

    def construct(self, params: dict[str, Any], *, monotone: list[int] | None = None,
                  alpha: float | None = None, categorical: tuple[int, ...] = (),
                  random_state: int | None = None, n_jobs: int | None = None):
        if self.task == "quantile" and alpha is None:
            raise ValueError(f"{self.name}: a quantile family needs alpha")
        if monotone is not None and not any(monotone):
            monotone = None
        if monotone is not None and self.monotone is None:
            raise ValueError(f"{self.name} cannot carry monotone constraints: "
                             f"{self.no_monotone_reason}")
        return self.make(dict(params), monotone=monotone, alpha=alpha, categorical=categorical,
                         random_state=global_seed() if random_state is None else random_state,
                         n_jobs=_n_jobs() if n_jobs is None else n_jobs)


def _n_jobs() -> int:
    return int((model_config().get("selection") or {}).get("n_jobs", 4))


def _thread_cap(n: int):
    """Cap native thread pools (OpenMP, BLAS) for the search.

    HistGradientBoosting has no n_jobs: it opens one OpenMP thread per core, and
    on a shared, oversubscribed machine OpenMP's barrier spinning turned a
    ten-second family into a quarter of an hour. The cap makes ``n_jobs`` true
    for every family. Without threadpoolctl the search simply runs uncapped.
    """
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=n)
    except Exception:                                            # noqa: BLE001
        from contextlib import nullcontext

        return nullcontext()


# ------------------------------------------------------------- linear models
class _LinearDesign:
    """Standardised design over a feature contract.

    Numerics are z-scored; categorical columns (ordinal codes from the feature
    transformer) are one-hot encoded, because a coefficient on an ordinal city
    code would be a straw man rather than a baseline. ``exact_contributions``
    maps the one-hot columns back onto their original feature, so the
    attribution is per contract feature like every other family's.
    """

    def _fit_design(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        categorical = set(int(i) for i in (self.categorical or ()))
        self.n_features_in_ = X.shape[1]
        self.blocks_: list[tuple[str, Any]] = []
        owner: list[int] = []
        for j in range(X.shape[1]):
            if j in categorical:
                levels = np.unique(X[:, j])
                self.blocks_.append(("onehot", levels))
                owner += [j] * len(levels)
            else:
                sd = float(X[:, j].std()) or 1.0
                self.blocks_.append(("scale", (float(X[:, j].mean()), sd)))
                owner.append(j)
        self.owner_ = np.asarray(owner, dtype=int)
        Z = self._design(X)
        self.design_mean_ = Z.mean(axis=0)
        return Z

    def _design(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        parts = []
        for j, (kind, prm) in enumerate(self.blocks_):
            if kind == "onehot":
                parts.append((X[:, [j]] == prm[None, :]).astype(float))
            else:
                parts.append(((X[:, j] - prm[0]) / prm[1])[:, None])
        return np.hstack(parts)

    def _weights(self) -> tuple[np.ndarray, float]:
        coef = np.ravel(self.model_.coef_)
        return coef, float(np.ravel(self.model_.intercept_)[0])

    def exact_contributions(self, X) -> tuple[np.ndarray, float]:
        """(per-feature contributions, base value) in the linear predictor's units.

        Interventional SHAP for a linear model against the training mean, which
        is exact: base + contributions reproduces the prediction (the log-odds,
        for the classifier) to floating-point precision.
        """
        coef, intercept = self._weights()
        Z = self._design(X)
        columns = (Z - self.design_mean_) * coef
        out = np.zeros((Z.shape[0], self.n_features_in_))
        for k, j in enumerate(self.owner_):
            out[:, j] += columns[:, k]
        return out, intercept + float(self.design_mean_ @ coef)


def _sk_bases():
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

    return BaseEstimator, ClassifierMixin, RegressorMixin


_BaseEstimator, _ClassifierMixin, _RegressorMixin = _sk_bases()


class LinearRegressor(_LinearDesign, _RegressorMixin, _BaseEstimator):
    """Ridge regression on the standardised design."""

    def __init__(self, alpha: float = 1.0, categorical: tuple[int, ...] = ()):
        self.alpha = alpha
        self.categorical = categorical

    def fit(self, X, y):
        from sklearn.linear_model import Ridge

        self.model_ = Ridge(alpha=self.alpha).fit(self._fit_design(X), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.model_.predict(self._design(X))


class LinearClassifier(_LinearDesign, _ClassifierMixin, _BaseEstimator):
    """L2 logistic regression on the standardised design."""

    def __init__(self, C: float = 1.0, categorical: tuple[int, ...] = ()):
        self.C = C
        self.categorical = categorical

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        self.model_ = LogisticRegression(C=self.C, max_iter=2000).fit(
            self._fit_design(X), np.asarray(y).astype(int))
        self.classes_ = self.model_.classes_
        return self

    def decision_function(self, X):
        return self.model_.decision_function(self._design(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._design(X))

    def predict(self, X):
        return self.model_.predict(self._design(X))


# ---------------------------------------------------------------- builders
def _sk(path: str, monotone_kw: str | None = None, **fixed):
    def make(params, *, monotone, alpha, categorical, random_state, n_jobs):
        module_name, cls_name = path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), cls_name)
        kwargs = {**fixed, **params}
        if "random_state" in cls().get_params():
            kwargs.setdefault("random_state", random_state)
        if "n_jobs" in cls().get_params():
            kwargs.setdefault("n_jobs", n_jobs)
        if monotone is not None and monotone_kw:
            kwargs[monotone_kw] = list(monotone)
        if kwargs.pop("_quantile", False):
            kwargs["quantile" if cls_name.startswith("Hist") else "alpha"] = alpha
        return cls(**kwargs)
    return make


def _linear(kind: str):
    def make(params, *, monotone, alpha, categorical, random_state, n_jobs):
        cls = LinearRegressor if kind == "regression" else LinearClassifier
        return cls(categorical=tuple(categorical), **params)
    return make


def _xgb(kind: str):
    def make(params, *, monotone, alpha, categorical, random_state, n_jobs):
        import xgboost

        kwargs = {"random_state": random_state, "n_jobs": n_jobs, "verbosity": 0, **params}
        if monotone is not None:
            kwargs["monotone_constraints"] = tuple(int(m) for m in monotone)
        if kind == "binary":
            return xgboost.XGBClassifier(eval_metric="logloss", **kwargs)
        if kind == "quantile":
            kwargs.update(objective="reg:quantileerror", quantile_alpha=alpha)
        return xgboost.XGBRegressor(**kwargs)
    return make


def _lgbm(kind: str):
    def make(params, *, monotone, alpha, categorical, random_state, n_jobs):
        import lightgbm

        kwargs = {"random_state": random_state, "n_jobs": n_jobs, "verbose": -1,
                  "subsample_freq": 1, **params}
        if monotone is not None:
            kwargs["monotone_constraints"] = [int(m) for m in monotone]
        if kind == "binary":
            return lightgbm.LGBMClassifier(**kwargs)
        if kind == "quantile":
            kwargs.update(objective="quantile", alpha=alpha)
        return lightgbm.LGBMRegressor(**kwargs)
    return make


def _cat(kind: str):
    def make(params, *, monotone, alpha, categorical, random_state, n_jobs):
        import catboost

        kwargs = {"random_seed": random_state, "thread_count": n_jobs, "verbose": 0,
                  "allow_writing_files": False, **params}
        if monotone is not None:
            # The string form: CatBoost rewrites a list in place, which sklearn's
            # clone() - and so isotonic calibration - then refuses.
            kwargs["monotone_constraints"] = "(" + ",".join(str(int(m)) for m in monotone) + ")"
        if kind == "binary":
            return catboost.CatBoostClassifier(**kwargs)
        if kind == "quantile":
            kwargs["loss_function"] = f"Quantile:alpha={alpha}"
        return catboost.CatBoostRegressor(**kwargs)
    return make


# Search spaces: ("int", lo, hi) | ("float", lo, hi) | ("log", lo, hi) | ("choice", [...])
_FOREST = {"n_estimators": ("int", 150, 400), "max_depth": ("int", 6, 18),
           "min_samples_leaf": ("int", 1, 30), "max_features": ("choice", [0.33, 0.5, 0.7, 1.0])}
_FOREST_DEFAULTS = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5,
                    "max_features": 0.5}
_HGB = {"max_iter": ("int", 150, 500), "learning_rate": ("log", 0.02, 0.2),
        "max_depth": ("int", 3, 8), "min_samples_leaf": ("int", 10, 80),
        "l2_regularization": ("log", 0.01, 10.0)}
_HGB_DEFAULTS = {"max_iter": 300, "learning_rate": 0.06, "max_depth": 5, "min_samples_leaf": 40,
                 "l2_regularization": 1.0}
_XGB = {"n_estimators": ("int", 200, 700), "max_depth": ("int", 3, 8),
        "learning_rate": ("log", 0.02, 0.2), "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0), "reg_lambda": ("log", 0.1, 10.0),
        "min_child_weight": ("log", 1.0, 20.0)}
_XGB_DEFAULTS = {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9,
                 "colsample_bytree": 0.9, "reg_lambda": 1.0}
_LGBM = {"n_estimators": ("int", 200, 700), "num_leaves": ("int", 15, 127),
         "learning_rate": ("log", 0.02, 0.2), "subsample": ("float", 0.6, 1.0),
         "colsample_bytree": ("float", 0.6, 1.0), "reg_lambda": ("log", 0.01, 10.0),
         "min_child_samples": ("int", 10, 80)}
_LGBM_DEFAULTS = {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.05, "subsample": 0.9,
                  "colsample_bytree": 0.9, "min_child_samples": 20}
_CAT = {"iterations": ("int", 300, 900), "depth": ("int", 4, 8),
        "learning_rate": ("log", 0.03, 0.2), "l2_leaf_reg": ("log", 1.0, 10.0)}
_CAT_DEFAULTS = {"iterations": 600, "depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 3.0}
_GBR = {"n_estimators": ("int", 150, 400), "max_depth": ("int", 2, 5),
        "learning_rate": ("log", 0.03, 0.15), "min_samples_leaf": ("int", 20, 80),
        "subsample": ("float", 0.7, 1.0)}
_GBR_DEFAULTS = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
                 "min_samples_leaf": 40, "subsample": 0.9}

# Measured, not assumed: with monotone constraints, CatBoost's ShapValues depend
# on the pool they are computed over (the expected value moved from 0.6821 to
# 0.6851 log-odds between a one-row and a 300-row pool of conversion files; the
# unconstrained model's did not move). A per-loan explanation would then change
# with whatever else was scored alongside it.
_CATBOOST_MONOTONE = ("with monotone constraints CatBoost's SHAP values depend on the batch they "
                      "are computed over, so a per-loan explanation would not be reproducible")

_NO_LINEAR_MONOTONE = ("sklearn's Ridge / LogisticRegression take no sign constraint on a "
                       "coefficient, so the contract's monotone direction could not be guaranteed")


def _catalogue() -> dict[str, list[Family]]:
    """Built on call, not at import: nothing here imports a learner library."""
    reg, clf = "regression", "binary"
    tree = "boosted_trees"
    return {
        reg: [
            Family("ridge", reg, "Ridge regression", "linear", _linear(reg),
                   no_monotone_reason=_NO_LINEAR_MONOTONE,
                   defaults={"alpha": 1.0}, space={"alpha": ("log", 0.01, 100.0)}),
            Family("random_forest", reg, "Random forest", "bagged_trees",
                   _sk("sklearn.ensemble.RandomForestRegressor", "monotonic_cst"),
                   monotone="monotonic_cst", defaults=_FOREST_DEFAULTS, space=_FOREST),
            Family("extra_trees", reg, "Extra trees", "bagged_trees",
                   _sk("sklearn.ensemble.ExtraTreesRegressor", "monotonic_cst"),
                   monotone="monotonic_cst", defaults=_FOREST_DEFAULTS, space=_FOREST),
            Family("hist_gradient_boosting", reg, "HistGradientBoosting", tree,
                   _sk("sklearn.ensemble.HistGradientBoostingRegressor", "monotonic_cst"),
                   monotone="monotonic_cst", defaults=_HGB_DEFAULTS, space=_HGB),
            Family("xgboost", reg, "XGBoost", tree, _xgb(reg), library="xgboost",
                   monotone="monotone_constraints", defaults=_XGB_DEFAULTS, space=_XGB),
            Family("lightgbm", reg, "LightGBM", tree, _lgbm(reg), library="lightgbm",
                   monotone="monotone_constraints", defaults=_LGBM_DEFAULTS, space=_LGBM),
            Family("catboost", reg, "CatBoost", tree, _cat(reg), library="catboost",
                   monotone="monotone_constraints", monotone_unexplainable=_CATBOOST_MONOTONE,
                   defaults=_CAT_DEFAULTS, space=_CAT),
        ],
        clf: [
            Family("logistic_regression", clf, "Logistic regression", "linear", _linear(clf),
                   no_monotone_reason=_NO_LINEAR_MONOTONE,
                   defaults={"C": 1.0}, space={"C": ("log", 0.001, 10.0)}),
            Family("random_forest", clf, "Random forest", "bagged_trees",
                   _sk("sklearn.ensemble.RandomForestClassifier", "monotonic_cst"),
                   monotone="monotonic_cst", defaults=_FOREST_DEFAULTS, space=_FOREST),
            Family("extra_trees", clf, "Extra trees", "bagged_trees",
                   _sk("sklearn.ensemble.ExtraTreesClassifier", "monotonic_cst"),
                   monotone="monotonic_cst", defaults=_FOREST_DEFAULTS, space=_FOREST),
            Family("hist_gradient_boosting", clf, "HistGradientBoosting", tree,
                   _sk("sklearn.ensemble.HistGradientBoostingClassifier", "monotonic_cst"),
                   monotone="monotonic_cst", defaults=_HGB_DEFAULTS, space=_HGB),
            Family("xgboost", clf, "XGBoost", tree, _xgb(clf), library="xgboost",
                   monotone="monotone_constraints", defaults=_XGB_DEFAULTS, space=_XGB),
            Family("lightgbm", clf, "LightGBM", tree, _lgbm(clf), library="lightgbm",
                   monotone="monotone_constraints", defaults=_LGBM_DEFAULTS, space=_LGBM),
            Family("catboost", clf, "CatBoost", tree, _cat(clf), library="catboost",
                   monotone="monotone_constraints", monotone_unexplainable=_CATBOOST_MONOTONE,
                   defaults=_CAT_DEFAULTS, space=_CAT),
        ],
        "quantile": [
            # The incumbent Stage-1 learner. sklearn's GradientBoostingRegressor has
            # no monotonic_cst (only the Hist variant does).
            Family("gradient_boosting_quantile", "quantile", "Gradient boosting (quantile)", tree,
                   _sk("sklearn.ensemble.GradientBoostingRegressor", None, loss="quantile",
                       _quantile=True),
                   no_monotone_reason="sklearn's GradientBoostingRegressor has no monotonic_cst",
                   defaults=_GBR_DEFAULTS, space=_GBR),
            Family("hist_gradient_boosting_quantile", "quantile", "HistGradientBoosting (quantile)",
                   tree, _sk("sklearn.ensemble.HistGradientBoostingRegressor", "monotonic_cst",
                             loss="quantile", _quantile=True),
                   monotone="monotonic_cst", defaults=_HGB_DEFAULTS, space=_HGB),
            Family("lightgbm_quantile", "quantile", "LightGBM (quantile)", tree,
                   _lgbm("quantile"), library="lightgbm",
                   no_monotone_reason="LightGBM refuses monotone constraints with the quantile "
                                      "objective",
                   defaults=_LGBM_DEFAULTS, space=_LGBM),
            Family("xgboost_quantile", "quantile", "XGBoost (quantile)", tree, _xgb("quantile"),
                   library="xgboost", monotone="monotone_constraints",
                   defaults=_XGB_DEFAULTS, space=_XGB),
            Family("catboost_quantile", "quantile", "CatBoost (quantile)", tree, _cat("quantile"),
                   library="catboost", monotone="monotone_constraints",
                   monotone_unexplainable=_CATBOOST_MONOTONE,
                   defaults=_CAT_DEFAULTS, space=_CAT),
        ],
    }


def families(task: str) -> list[Family]:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    return _catalogue()[task]


def family(task: str, name: str) -> Family | None:
    return next((f for f in families(task) if f.name == name), None)


def label(name: str) -> str:
    """Display label for a family name, whatever its task. A fallback learner's
    config name ("hist_gradient_boosting_classifier") maps onto its family."""
    for candidate in (name, name.removesuffix("_classifier").removesuffix("_regressor")):
        for task in TASKS:
            found = family(task, candidate)
            if found is not None:
                return found.label
    return name.replace("_", " ")


# ====================================================================== config
def settings() -> dict[str, Any]:
    cfg = dict(model_config().get("selection") or {})
    cfg.setdefault("budget_per_family", 8)
    cfg.setdefault("budget_overrides", {})
    cfg.setdefault("fast_budget", 2)
    cfg.setdefault("valid_size", 0.25)
    cfg.setdefault("tolerance_rel", 0.01)
    return cfg


def budget_for(family_name: str, budget: int | None = None) -> int:
    if budget is not None:
        return max(1, int(budget))
    cfg = settings()
    return max(1, int((cfg["budget_overrides"] or {}).get(family_name, cfg["budget_per_family"])))


# ===================================================================== windows
def split(dev_df: pd.DataFrame, valid_size: float | None = None,
          time_column: str | None = None, holdout_df: pd.DataFrame | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit / validation slices of a training window, chronologically.

    The later ``valid_size`` share validates. Every row sharing the boundary date
    is moved to the validation side: ``loader.train_test_split`` cuts by row
    position and a busy day otherwise lands on both sides, which for a split a
    champion is *selected* on is a day of the future leaking into the fit (the
    same rule as ``scripts/run_automl.py``'s ``_clean_boundary``).

    The same position cut puts the holdout's first date on both sides of the
    trainer's split too. Given ``holdout_df``, the training rows dated on or
    after the holdout's first day are left out of the search altogether, so the
    search never sees a date the holdout covers.
    """
    from ..ingestion import loader

    time_column = time_column or model_config()["data"]["time_column"]
    valid_size = settings()["valid_size"] if valid_size is None else valid_size
    if (holdout_df is not None and not holdout_df.empty and time_column in dev_df.columns
            and time_column in holdout_df.columns):
        dev_df = dev_df[dev_df[time_column] < holdout_df[time_column].min()]
    fit_df, valid_df = loader.train_test_split(dev_df, test_size=valid_size, strategy="time")
    if time_column not in fit_df.columns or valid_df.empty:
        return fit_df, valid_df
    boundary = valid_df[time_column].min()
    straddling = fit_df[time_column] >= boundary
    if straddling.any():
        valid_df = pd.concat([fit_df[straddling], valid_df]).sort_values(time_column, kind="stable")
        fit_df = fit_df[~straddling]
    return fit_df.copy(), valid_df.copy()


def window(fit_df: pd.DataFrame, valid_df: pd.DataFrame, holdout_df: pd.DataFrame | None = None,
           time_column: str | None = None) -> dict[str, Any]:
    """Dates and row counts of the three windows, for the report."""
    time_column = time_column or model_config()["data"]["time_column"]

    def span(frame: pd.DataFrame | None) -> dict[str, Any]:
        if frame is None or frame.empty or time_column not in frame.columns:
            return {"rows": 0 if frame is None else int(len(frame)), "from": None, "to": None}
        dates = pd.to_datetime(frame[time_column])
        return {"rows": int(len(frame)), "from": str(dates.min().date()),
                "to": str(dates.max().date())}

    out = {"fit": span(fit_df), "valid": span(valid_df)}
    if holdout_df is not None:
        out["holdout"] = {**span(holdout_df), "shown_to_search": False}
    return out


# ===================================================================== metrics
def _regression(y, pred, scale: float, unit: str) -> dict[str, float]:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    err = y - pred
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {f"mae_{unit}": float(np.abs(err).mean() * scale),
            f"rmse_{unit}": float(np.sqrt((err ** 2).mean()) * scale),
            "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")}


def _pinball(y, pred, q: float) -> float:
    diff = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _quantile(y, preds: dict[float, np.ndarray], scale: float, unit: str) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    qs = sorted(preds)
    out = {f"pinball_q{int(round(q * 100))}_{unit}": _pinball(y, preds[q], q) * scale for q in qs}
    out[f"pinball_{unit}"] = float(np.mean([_pinball(y, preds[q], q) for q in qs]) * scale)
    # Quantile models are fitted independently and can cross; the band the
    # Stage-1 scorer shows is the min/max of the outer pair, so that is scored.
    low = np.minimum(preds[qs[0]], preds[qs[-1]])
    high = np.maximum(preds[qs[0]], preds[qs[-1]])
    out["coverage_pct"] = float(((y >= low) & (y <= high)).mean() * 100)
    out["nominal_coverage_pct"] = round((qs[-1] - qs[0]) * 100, 1)
    out[f"mean_width_{unit}"] = float((high - low).mean() * scale)
    if len(qs) % 2 == 1:
        out[f"median_mae_{unit}"] = float(np.abs(y - preds[qs[len(qs) // 2]]).mean() * scale)
    return out


def _binary(y, proba) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    from .calibration import expected_calibration_error

    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return {"auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "ece": float(expected_calibration_error(p, y)),
            "calibration_gap": float(abs(p.mean() - y.mean())),
            "base_rate": float(y.mean())}


def primary_metric(data: SelectionData) -> tuple[str, bool]:
    """(metric key, higher_is_better) the search ranks on."""
    unit = data.unit()
    default = {"regression": f"mae_{unit}", "quantile": f"pinball_{unit}", "binary": "auc"}[data.task]
    known = {"regression": {f"mae_{unit}", f"rmse_{unit}", "r2"},
             "quantile": {f"pinball_{unit}", f"mean_width_{unit}"},
             "binary": {"auc", "log_loss", "brier", "ece"}}[data.task]
    key = data.metric if data.metric in known else default
    return key, key in {"auc", "r2"}


# ====================================================================== search
def _sample(spec: tuple, rng: np.random.Generator) -> Any:
    kind = spec[0]
    if kind == "int":
        return int(rng.integers(int(spec[1]), int(spec[2]) + 1))
    if kind == "float":
        return round(float(rng.uniform(spec[1], spec[2])), 4)
    if kind == "log":
        value = math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2])))
        return float(f"{value:.4g}")
    if kind == "choice":
        options = list(spec[1])
        return options[int(rng.integers(len(options)))]
    raise ValueError(f"unknown search-space spec {spec!r}")


def _trial_params(fam: Family, budget: int, seed: int,
                  incumbent: dict[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
    """Trial 0 is the incumbent (if this is its family) or the family defaults;
    the rest are seeded random draws. The seed is per family, so adding a family
    to the catalogue does not change another family's draws."""
    rng = np.random.default_rng([seed, zlib.crc32(fam.name.encode())])
    if incumbent and incumbent.get("family") == fam.name:
        first = ("incumbent", dict(incumbent.get("params") or {}))
    else:
        first = ("defaults", dict(fam.defaults))
    trials = [first]
    seen = {json.dumps(first[1], sort_keys=True, default=str)}
    attempts = 0
    while len(trials) < budget and attempts < budget * 20:
        attempts += 1
        params = {k: _sample(v, rng) for k, v in fam.space.items()}
        key = json.dumps(params, sort_keys=True, default=str)
        if key in seen:
            continue                     # a 1-D space runs out of new points quickly
        seen.add(key)
        trials.append(("random", params))
    return trials


def _categorical_positions(spec, columns: list[str]) -> tuple[int, ...]:
    names = set(getattr(spec, "categorical", None) or [])
    return tuple(i for i, c in enumerate(columns) if c in names)


def _monotone(spec, columns: list[str]) -> list[int] | None:
    monotone = dict(getattr(spec, "monotone", None) or {})
    if not monotone:
        return None
    vector = [int(monotone.get(c, 0)) for c in columns]
    return vector if any(vector) else None


def _run_trial(fam: Family, params: dict, data: SelectionData, *, monotone, categorical,
               seed: int, n_jobs: int) -> tuple[dict[str, float], float]:
    unit, scale = data.unit(), data.factor()
    started = time.perf_counter()
    if data.task == "quantile":
        preds = {}
        for q in data.quantiles or []:
            est = fam.construct(params, monotone=monotone, alpha=q, categorical=categorical,
                                random_state=seed, n_jobs=n_jobs)
            est.fit(data.X_fit, np.asarray(data.y_fit, dtype=float))
            preds[q] = est.predict(data.X_valid)
        seconds = time.perf_counter() - started
        return _quantile(data.y_valid, preds, scale, unit), seconds
    est = fam.construct(params, monotone=monotone, categorical=categorical,
                        random_state=seed, n_jobs=n_jobs)
    if data.task == "binary":
        est.fit(data.X_fit, np.asarray(data.y_fit).astype(int))
        seconds = time.perf_counter() - started
        return _binary(data.y_valid, est.predict_proba(data.X_valid)[:, 1]), seconds
    est.fit(data.X_fit, np.asarray(data.y_fit, dtype=float))
    seconds = time.perf_counter() - started
    return _regression(data.y_valid, est.predict(data.X_valid), scale, unit), seconds


def _round(metrics: dict[str, float]) -> dict[str, float]:
    return {k: (round(float(v), 6) if isinstance(v, (int, float)) and np.isfinite(v) else None)
            for k, v in metrics.items()}


def _rank_key(row: dict[str, Any], metric: str, higher: bool) -> float:
    """Sort key, best first; a trial without the metric sorts last either way."""
    value = (row.get("metrics") or {}).get(metric)
    if value is None:
        return np.inf
    return -float(value) if higher else float(value)


def choose(best: list[dict[str, Any]], metric: str, higher_is_better: bool,
           tolerance_rel: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """The deck's rule: best validation score, then explainability.

    ``best`` holds the best trial of each family. Every family whose score is
    within ``tolerance_rel`` (relative) of the overall best counts as tied; among
    the tied, the most explainable class wins, and within a class the better
    score. Returns (winner, how the rule resolved).
    """
    scored = [b for b in best if b["metrics"].get(metric) is not None]
    if not scored:
        raise RuntimeError("no family produced a usable validation score")
    sign = -1.0 if higher_is_better else 1.0
    top = min(scored, key=lambda b: sign * b["metrics"][metric])
    top_value = top["metrics"][metric]
    margin = abs(top_value) * tolerance_rel
    tied = [b for b in scored
            if (top_value - b["metrics"][metric] if higher_is_better
                else b["metrics"][metric] - top_value) <= margin + 1e-12]
    winner = min(tied, key=lambda b: (b["explainability_tier"], sign * b["metrics"][metric]))
    resolved = {
        "best_score_family": top["family"], "best_score": top_value,
        "tolerance_abs": round(margin, 6),
        "within_tolerance": sorted(b["family"] for b in tied),
        "decided_by": ("validation score" if winner["family"] == top["family"]
                       else "explainability (within tolerance of the best score)"),
    }
    return winner, resolved


def search(data: SelectionData, *, budget: int | None = None, fast: bool = False,
           only: list[str] | None = None, seed: int | None = None) -> dict[str, Any]:
    """Random search over every available family for one model; returns the report.

    ``budget`` (trials per family) overrides config; ``fast`` uses
    ``selection.fast_budget``. ``only`` restricts the families (tests use it).
    Nothing is written - see ``record``.
    """
    if data.task not in TASKS:
        raise ValueError(f"unknown task {data.task!r}; expected one of {TASKS}")
    if data.task == "quantile" and not data.quantiles:
        raise ValueError(f"{data.name}: a quantile task needs quantiles")
    cfg = settings()
    seed = global_seed() if seed is None else int(seed)
    if fast and budget is None:
        budget = int(cfg["fast_budget"])
    n_jobs = _n_jobs()
    columns = list(data.X_fit.columns)
    monotone = _monotone(data.spec, columns)
    categorical = _categorical_positions(data.spec, columns)
    metric, higher = primary_metric(data)

    started = time.perf_counter()
    considered, skipped, leaderboard, libraries = [], [], [], {}
    # Every fit runs under the thread cap; see _thread_cap.
    with _thread_cap(n_jobs):
        for fam in families(data.task):
            if only and fam.name not in only:
                continue
            missing = fam.available()
            if missing:
                skipped.append({"family": fam.name, "kind": "unavailable", "reason": missing})
                log.info("[%s] skipping %s: %s", data.name, fam.name, missing)
                continue
            if monotone is not None and (fam.monotone is None or fam.monotone_unexplainable):
                constrained = ", ".join(c for c, m in zip(columns, monotone) if m)
                why = fam.no_monotone_reason if fam.monotone is None else fam.monotone_unexplainable
                skipped.append({"family": fam.name, "kind": "excluded",
                                "reason": f"the contract carries monotone constraints ({constrained}) "
                                          f"and {why}"})
                log.info("[%s] excluding %s: %s", data.name, fam.name, why)
                continue
            considered.append(fam.name)
            libraries[fam.library] = _version(fam.library)
            for i, (origin, params) in enumerate(_trial_params(fam, budget_for(fam.name, budget),
                                                               seed, data.incumbent)):
                row = {"family": fam.name, "trial": i, "origin": origin, "params": params,
                       "explainability": fam.explainability, "explainability_tier": fam.tier}
                try:
                    metrics, seconds = _run_trial(fam, params, data, monotone=monotone,
                                                  categorical=categorical, seed=seed, n_jobs=n_jobs)
                    row.update(status="ok", metrics=_round(metrics), fit_seconds=round(seconds, 3))
                except Exception as exc:                    # noqa: BLE001 - recorded, not raised
                    row.update(status="failed", metrics={}, fit_seconds=None,
                               error=f"{type(exc).__name__}: {exc}"[:300])
                    log.warning("[%s] %s trial %s failed: %s", data.name, fam.name, i, exc)
                leaderboard.append(row)
            ok = [r for r in leaderboard if r["family"] == fam.name and r["status"] == "ok"
                  and r["metrics"].get(metric) is not None]
            if ok:
                best = min(ok, key=lambda r: _rank_key(r, metric, higher))
                log.info("[%s] %-32s best %s %.4f over %s trial(s)", data.name, fam.name, metric,
                         best["metrics"][metric], len(ok))

    best_per_family = []
    for name in considered:
        ok = [r for r in leaderboard if r["family"] == name and r["status"] == "ok"
              and r["metrics"].get(metric) is not None]
        if ok:
            best_per_family.append(min(ok, key=lambda r: _rank_key(r, metric, higher)))
    best_per_family.sort(key=lambda r: _rank_key(r, metric, higher))
    winner, resolved = choose(best_per_family, metric, higher, float(cfg["tolerance_rel"]))
    leaderboard.sort(key=lambda r: (r["status"] != "ok", _rank_key(r, metric, higher)))
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank

    return {
        "model": data.name,
        "task": data.task,
        "target": data.target,
        "framework": framework(libraries),
        "metric": metric,
        "higher_is_better": higher,
        "units": data.unit(),
        "quantiles": data.quantiles,
        "features": columns,
        "monotone_constraints": ({c: m for c, m in zip(columns, monotone) if m}
                                 if monotone else {}),
        "families_considered": considered,
        "families_skipped": skipped,
        "budget_per_family": {name: budget_for(name, budget) for name in considered},
        "trials": len(leaderboard),
        "trials_failed": sum(1 for r in leaderboard if r["status"] != "ok"),
        "leaderboard": leaderboard,
        "best_per_family": [{k: r[k] for k in ("family", "trial", "origin", "params", "metrics",
                                              "fit_seconds", "explainability")}
                            for r in best_per_family],
        "incumbent": data.incumbent,
        "selected": {"family": winner["family"], "label": label(winner["family"]),
                     "params": winner["params"], "trial": winner["trial"],
                     "origin": winner["origin"], "explainability": winner["explainability"],
                     "validation": winner["metrics"]},
        "rule": rule_text(metric, higher, float(cfg["tolerance_rel"]), resolved),
        "windows": data.window,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "generated_at": _now(),
        "notes": _notes(data),
    }


def rule_text(metric: str, higher: bool, tolerance_rel: float,
              resolved: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "statement": "Best validation score, then explainability.",
        "primary": f"{'highest' if higher else 'lowest'} out-of-time validation {metric}",
        "tolerance_rel": tolerance_rel,
        "tie_break": (f"families within {tolerance_rel:.1%} (relative) of the best score are "
                      "tied; among them the most explainable class wins, then the better score"),
        "explainability_ranking": list(EXPLAINABILITY),
        **(resolved or {}),
    }


def _notes(data: SelectionData) -> list[str]:
    notes = ["The trainers' holdout (the last 20% by application date) was never passed to "
             "this search; validation is the later part of the training window."]
    if data.task == "binary":
        notes.append("Classifiers are ranked on AUC, which calibration cannot change. log_loss, "
                     "brier and ece are for the raw learner: the trainer calibrates the selected "
                     "model isotonically afterwards, so deployed calibration is better than "
                     "recorded here.")
    if data.task == "quantile":
        notes.append("Each trial fits one model per quantile with the same params; pinball loss "
                     "is the mean across the quantiles.")
    return notes


def _version(library: str) -> str | None:
    try:
        return getattr(importlib.import_module(library), "__version__", None)
    except Exception:                                            # noqa: BLE001
        return None


def framework(libraries: dict[str, str | None]) -> str:
    used = ", ".join(f"{'scikit-learn' if lib == 'sklearn' else lib} {ver or '?'}"
                     for lib, ver in sorted(libraries.items()))
    return f"aavas.automl.portfolio random search ({used or 'no library ran'})"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ===================================================================== records
def report_path(name: str) -> Path:
    return REPORT_DIR / f"{name}.json"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Any) -> None:
    """Atomic: a reader never sees half a selection file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def record(report: dict[str, Any], directory: Path | None = None) -> Path:
    """Write ``<model>.json`` and point ``selection.json`` at it.

    The selection file is read-modify-written under a lock: several models (and
    the LGD / uplift workstream) record into it, and a lost update would silently
    put a model back on its fallback learner.
    """
    directory = Path(directory) if directory else REPORT_DIR
    name = report["model"]
    out = directory / f"{name}.json"
    _write_json(out, report)
    selected_ = report.get("selected") or {}
    metric = report.get("metric")
    entry = {"family": selected_.get("family"), "task": report.get("task"),
             "params": selected_.get("params"), "metric": metric,
             "value": (selected_.get("validation") or {}).get(metric),
             "selected_at": report.get("generated_at") or _now(),
             "report": _relative(out)}
    with _locked(directory / ".selection.lock"):
        current = _read(directory / SELECTION_FILE.name) or {}
        current[name] = entry
        _write_json(directory / SELECTION_FILE.name, current)
    log.info("[%s] selected %s (%s=%s) -> %s", name, entry["family"], metric, entry["value"],
             _relative(out))
    return out


class _locked:
    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            import fcntl

            fcntl.flock(self.fh, fcntl.LOCK_EX)
        except ImportError:                                     # not POSIX: best effort
            pass
        return self

    def __exit__(self, *exc):
        try:
            import fcntl

            fcntl.flock(self.fh, fcntl.LOCK_UN)
        except ImportError:
            pass
        self.fh.close()
        return False


def selections(directory: Path | None = None) -> dict[str, Any]:
    return _read((Path(directory) if directory else REPORT_DIR) / SELECTION_FILE.name) or {}


def selected(name: str) -> dict[str, Any] | None:
    """The recorded selection for one model, or None."""
    entry = selections().get(name)
    return entry if isinstance(entry, dict) and entry.get("family") else None


def load_report(name: str) -> dict[str, Any] | None:
    return _read(report_path(name))


# ======================================================================= build
def build(name: str, task: str, spec=None, fallback: Callable[[], tuple[Any, str]] | None = None,
          alpha: float | None = None, **kwargs) -> tuple[Any, str]:
    """(estimator, learner label) for a trainer to fit.

    The selected family and params when ``selection.json`` holds them for
    ``name``; ``fallback()`` - the trainer's own learner - when it does not, when
    the selection was made for another task, or when the selected family's
    library is unavailable. Which of those happened is logged, because a model
    card that names a learner nobody fitted is worse than none.
    """
    entry = selected(name)
    reason = None
    if entry is None:
        reason = "no selection recorded"
    elif entry.get("task") not in (None, task):
        reason = f"the recorded selection is for task {entry.get('task')!r}, not {task!r}"
    else:
        fam = family(task, entry["family"])
        if fam is None:
            reason = f"family {entry['family']!r} is not in the {task} catalogue"
        else:
            missing = fam.available()
            if missing:
                reason = f"selected family {fam.name} is unavailable: {missing}"
            else:
                columns = list(getattr(spec, "columns", None) or [])
                monotone = _monotone(spec, columns) if spec is not None else None
                if monotone is not None and (fam.monotone is None or fam.monotone_unexplainable):
                    reason = (f"selected family {fam.name} cannot carry this contract's monotone "
                              f"constraints and stay exactly explainable")
                else:
                    estimator = fam.construct(entry.get("params") or {}, monotone=monotone,
                                              alpha=alpha,
                                              categorical=_categorical_positions(spec, columns),
                                              **kwargs)
                    log.info("[%s] fitting the selected %s (%s)", name, fam.name,
                             entry.get("report"))
                    return estimator, fam.name
    if fallback is None:
        raise RuntimeError(f"{name}: {reason} and no fallback given")
    estimator, learner = fallback()
    log.info("[%s] %s; fitting the trainer's own learner %s", name, reason, learner)
    return estimator, learner


def provenance(name: str, learner: str) -> dict[str, Any]:
    """What a model card records about the selection behind its learner."""
    entry = selected(name)
    if entry is None:
        return {"status": "no_selection",
                "detail": f"no selection recorded in {_relative(SELECTION_FILE)}; the configured "
                          "fallback learner was fitted"}
    if entry.get("family") != learner:
        return {"status": "fallback", "selected_family": entry.get("family"),
                "report": entry.get("report"),
                "detail": f"{entry.get('family')} was selected but {learner} was fitted "
                          "(see the training log for why)"}
    return {"status": "selected", **entry}


__all__ = ["TASKS", "EXPLAINABILITY", "SelectionData", "Family", "LinearRegressor",
           "LinearClassifier", "families", "family", "label", "settings", "budget_for", "split",
           "window", "primary_metric", "search", "choose", "record", "selections", "selected",
           "load_report", "build", "provenance", "report_path", "framework", "REPORT_DIR",
           "SELECTION_FILE"]
