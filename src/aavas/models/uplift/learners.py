"""The uplift learners: what a discount does to acceptance, adjusted for who gets one.

Every learner answers the same question in the same units - the probability of
acceptance a discount of ``d`` bps adds for a file, against leaving the
recommendation alone - so the search can rank them on one metric and the served
model can draw one curve over ``d``. They differ in how they deal with the fact
that discounts are not handed out at random (see ``train.py`` for the design):

* ``dose``  - a single classifier on (context, propensity score, discount) with
  a monotone +1 constraint on the discount, trained with stabilised inverse-
  propensity weights. The curve over ``d`` is fitted, not assumed, and can never
  bend down. Feeding the propensity score in as a covariate is the Bayesian
  causal forest remedy for regularisation-induced confounding: a shrunk S-learner
  otherwise lets the selection signal masquerade as an effect.
* ``rdose`` - the R-learner with a continuous treatment: residualise acceptance
  and discount size on context (cross-fitted), regress one residual on the other.
  Neyman-orthogonal, so first-stage errors enter only at second order. The effect
  is linear in ``d`` by construction.
* ``t`` / ``x`` / ``dr`` - the binary meta-learners (discounted vs not). They
  estimate the effect of *the discount a file like this typically gets*; divided
  by that typical size (a regression of discount size on context among discounted
  files) it becomes a per-bps slope, so their curve is linear in ``d`` too.

All probabilities, all clipped so a discount never *lowers* acceptance: an
estimate below zero is noise around "buys nothing", and is reported as such.

The algorithm families are the portfolio's (``automl.portfolio``): one catalogue,
one parameter space per family, the same monotone plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ...automl import portfolio
from ...config import feature_config, model_config, seed
from .. import base

LEARNERS = ("dose", "rdose", "t", "x", "dr")
LABELS = {
    "dose": "Monotone dose-response learner (propensity-adjusted)",
    "rdose": "R-learner, continuous discount",
    "t": "T-learner",
    "x": "X-learner",
    "dr": "DR-learner",
}
CURVES = {"dose": "fitted, monotone in discount", "rdose": "linear in discount",
          "t": "linear in discount (per-bps slope)", "x": "linear in discount (per-bps slope)",
          "dr": "linear in discount (per-bps slope)"}

PROPENSITY = "_propensity"
MIN_RESIDUAL_BPS = 0.5


def dose_contract() -> tuple[str, int]:
    """(column, monotone sign) of the treatment, from the uplift contract.

    The constraint lives with the contract (``feature_config.uplift.dose``), not
    here: a bigger discount never lowers acceptance."""
    block = (feature_config().get("uplift") or {}).get("dose") or {}
    return str(block.get("column", "discount_bps")), int(block.get("monotone", 1))


DOSE = dose_contract()[0]


# ------------------------------------------------------------------ families


def family_unavailable(name: str) -> str | None:
    """None when ``name`` can serve every role an uplift learner needs, else why not."""
    clf, reg = portfolio.family("binary", name), portfolio.family("regression", name)
    if clf is None or reg is None:
        return f"{name} is not in both the binary and regression catalogues of automl.portfolio"
    if clf.monotone is None:
        return f"cannot carry the monotone discount constraint: {clf.no_monotone_reason}"
    return clf.available() or reg.available()


def _n_jobs() -> int:
    """Threads per fit. A learner fits many small models in a row, and on a shared
    machine more OpenMP threads than free cores makes every one of them slower."""
    return int((model_config().get("uplift") or {}).get("n_jobs", 4))


def classifier(name: str, params: dict, monotone: list[int] | None = None):
    return portfolio.family("binary", name).construct(dict(params), monotone=monotone,
                                                      n_jobs=_n_jobs())


def regressor(name: str, params: dict):
    return portfolio.family("regression", name).construct(dict(params), n_jobs=_n_jobs())


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _folds(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold

    return list(KFold(max(2, k), shuffle=True, random_state=seed()).split(np.arange(n)))


def propensity(X: pd.DataFrame, treated: np.ndarray, params: dict, folds: int,
               clip: tuple[float, float]) -> tuple[Any, np.ndarray]:
    """P(discounted | context): a model for new rows, and out-of-fold scores for
    the rows it was fitted on (in-sample scores are overconfident exactly where
    the inverse weights are largest)."""
    treated = np.asarray(treated).astype(int)
    oof = np.zeros(len(X))
    for fit_idx, out_idx in _folds(len(X), folds):
        model = base.hgb_classifier(params).fit(X.iloc[fit_idx], treated[fit_idx])
        oof[out_idx] = _proba(model, X.iloc[out_idx])
    full = base.hgb_classifier(params).fit(X, treated)
    return full, np.clip(oof, *clip)


def ipw(treated: np.ndarray, e: np.ndarray, stabilised: bool = True) -> np.ndarray:
    treated = np.asarray(treated).astype(int)
    share = treated.mean() if stabilised else 1.0
    return np.where(treated == 1, share / e, (1 - share if stabilised else 1.0) / (1 - e))


# ------------------------------------------------------------------ learners


class Learner:
    """Common shape: ``fit`` on (X, treated, dose, y, e); ``uplift`` and
    ``baseline`` in probability units; ``uplift`` >= 0 and non-decreasing in dose."""

    name = ""

    def __init__(self, family: str, params: dict, *, folds: int = 3):
        self.family = family
        self.params = dict(params)
        self.folds = folds

    def fit(self, X, treated, dose, y, e) -> "Learner":
        raise NotImplementedError

    def uplift(self, X, dose, e) -> np.ndarray:
        raise NotImplementedError

    def baseline(self, X, e) -> np.ndarray:
        raise NotImplementedError

    def raw_effect(self, X, e) -> np.ndarray | None:
        """The unclipped per-bps slope, for learners whose curve is linear."""
        return None


class _DoseScale:
    """The discount a file like this typically gets when it is discounted.

    Turns a binary effect ("what the usual discount bought") into a per-bps slope.
    Clipped to the inner 90% of observed discounts so a tiny predicted size
    cannot blow the slope up."""

    def __init__(self, family: str, params: dict):
        self.model = regressor(family, params)
        self.bounds = (1.0, 100.0)

    def fit(self, X, dose) -> "_DoseScale":
        dose = np.asarray(dose, dtype=float)
        self.model.fit(X, dose)
        self.bounds = (float(np.percentile(dose, 5)), float(np.percentile(dose, 95)))
        return self

    def predict(self, X) -> np.ndarray:
        return np.clip(self.model.predict(X), max(self.bounds[0], 1.0), self.bounds[1])


class _BinaryLearner(Learner):
    """Shared plumbing for T / X / DR: an effect at the typical discount, a
    no-discount acceptance model, and the dose scale."""

    def _effect(self, X, e) -> np.ndarray:
        raise NotImplementedError

    def _fit_common(self, X, treated, dose, y):
        t = np.asarray(treated).astype(int) == 1
        y = np.asarray(y).astype(int)
        self.m0_ = classifier(self.family, self.params).fit(X[~t], y[~t])
        self.scale_ = _DoseScale(self.family, self.params).fit(X[t], np.asarray(dose)[t])
        return t, y

    def raw_effect(self, X, e) -> np.ndarray:
        return self._effect(X, e) / self.scale_.predict(X)

    def uplift(self, X, dose, e) -> np.ndarray:
        slope = np.maximum(self.raw_effect(X, e), 0.0)
        return slope * np.broadcast_to(np.asarray(dose, dtype=float), slope.shape)

    def baseline(self, X, e) -> np.ndarray:
        return _proba(self.m0_, X)


class TLearner(_BinaryLearner):
    name = "t"

    def fit(self, X, treated, dose, y, e):
        t, y = self._fit_common(X, treated, dose, y)
        self.m1_ = classifier(self.family, self.params).fit(X[t], y[t])
        return self

    def _effect(self, X, e):
        return _proba(self.m1_, X) - _proba(self.m0_, X)


class XLearner(_BinaryLearner):
    name = "x"

    def fit(self, X, treated, dose, y, e):
        t, y = self._fit_common(X, treated, dose, y)
        self.m1_ = classifier(self.family, self.params).fit(X[t], y[t])
        imputed_treated = y[t] - _proba(self.m0_, X[t])
        imputed_control = _proba(self.m1_, X[~t]) - y[~t]
        self.r1_ = regressor(self.family, self.params).fit(X[t], imputed_treated)
        self.r0_ = regressor(self.family, self.params).fit(X[~t], imputed_control)
        return self

    def _effect(self, X, e):
        # Kunzel et al.: weight the control-side estimate by the propensity.
        return e * self.r0_.predict(X) + (1 - e) * self.r1_.predict(X)


class DRLearner(_BinaryLearner):
    name = "dr"

    def fit(self, X, treated, dose, y, e):
        t, y = self._fit_common(X, treated, dose, y)
        e = np.asarray(e, dtype=float)
        mu0, mu1 = np.zeros(len(X)), np.zeros(len(X))
        for fit_idx, out_idx in _folds(len(X), self.folds):
            tf, yf = t[fit_idx], y[fit_idx]
            Xf = X.iloc[fit_idx]
            mu0[out_idx] = _proba(classifier(self.family, self.params).fit(Xf[~tf], yf[~tf]),
                                  X.iloc[out_idx])
            mu1[out_idx] = _proba(classifier(self.family, self.params).fit(Xf[tf], yf[tf]),
                                  X.iloc[out_idx])
        tt = t.astype(float)
        pseudo = mu1 - mu0 + tt * (y - mu1) / e - (1 - tt) * (y - mu0) / (1 - e)
        self.final_ = regressor(self.family, self.params).fit(X, pseudo)
        return self

    def _effect(self, X, e):
        return self.final_.predict(X)


class DoseLearner(Learner):
    """S-learner over (context, propensity, discount), monotone +1 in discount."""

    name = "dose"

    def _design(self, X, e, dose) -> pd.DataFrame:
        Z = X.copy()
        Z[PROPENSITY] = np.asarray(e, dtype=float)
        Z[DOSE] = np.broadcast_to(np.asarray(dose, dtype=float), (len(X),)).copy()
        return Z

    def fit(self, X, treated, dose, y, e):
        monotone = [0] * X.shape[1] + [0, dose_contract()[1]]
        self.model_ = classifier(self.family, self.params, monotone=monotone)
        Z = self._design(X, e, np.where(np.asarray(treated).astype(int) == 1, dose, 0.0))
        self.model_.fit(Z, np.asarray(y).astype(int), sample_weight=ipw(treated, e))
        return self

    def acceptance(self, X, dose, e) -> np.ndarray:
        return _proba(self.model_, self._design(X, e, dose))

    def uplift(self, X, dose, e) -> np.ndarray:
        return np.maximum(self.acceptance(X, dose, e) - self.acceptance(X, 0.0, e), 0.0)

    def baseline(self, X, e) -> np.ndarray:
        return self.acceptance(X, 0.0, e)


class RDoseLearner(Learner):
    """R-learner, continuous treatment: (y - m(x)) = theta(x) * (d - g(x)) + noise."""

    name = "rdose"

    def fit(self, X, treated, dose, y, e):
        d = np.where(np.asarray(treated).astype(int) == 1, np.asarray(dose, dtype=float), 0.0)
        y = np.asarray(y).astype(int)
        m, g = np.zeros(len(X)), np.zeros(len(X))
        for fit_idx, out_idx in _folds(len(X), self.folds):
            Xf = X.iloc[fit_idx]
            m[out_idx] = _proba(classifier(self.family, self.params).fit(Xf, y[fit_idx]),
                                X.iloc[out_idx])
            g[out_idx] = regressor(self.family, self.params).fit(Xf, d[fit_idx]).predict(
                X.iloc[out_idx])
        res_d, res_y = d - g, y - m
        keep = np.abs(res_d) > MIN_RESIDUAL_BPS
        self.theta_ = regressor(self.family, self.params).fit(
            X[keep], res_y[keep] / res_d[keep], sample_weight=res_d[keep] ** 2)
        self.m_ = classifier(self.family, self.params).fit(X, y)
        self.g_ = regressor(self.family, self.params).fit(X, d)
        return self

    def raw_effect(self, X, e) -> np.ndarray:
        return self.theta_.predict(X)

    def uplift(self, X, dose, e) -> np.ndarray:
        slope = np.maximum(self.raw_effect(X, e), 0.0)
        return slope * np.broadcast_to(np.asarray(dose, dtype=float), slope.shape)

    def baseline(self, X, e) -> np.ndarray:
        slope = np.maximum(self.raw_effect(X, e), 0.0)
        return np.clip(_proba(self.m_, X) - slope * np.maximum(self.g_.predict(X), 0.0), 0.0, 1.0)


_CLASSES = {"dose": DoseLearner, "rdose": RDoseLearner, "t": TLearner, "x": XLearner,
            "dr": DRLearner}


def make(learner: str, family: str, params: dict, *, folds: int = 3) -> Learner:
    if learner not in _CLASSES:
        raise ValueError(f"unknown uplift learner {learner!r}; expected one of {LEARNERS}")
    return _CLASSES[learner](family, params, folds=folds)


# ------------------------------------------------------------- served object


@dataclass
class UpliftEstimator:
    """Everything needed to score a file: the contract transformer, the
    propensity model, the selected learner and its bootstrap bag.

    ``members`` are the same learner refitted on bootstrap resamples of the
    training rows. The served estimate is the median over the full-data fit and
    its refits, and the interval is quantiles of the same set, so the interval
    always brackets the estimate. Every member is monotone in the discount, and
    an order statistic of monotone curves is monotone - so the estimate and both
    edges of the band are too, and the band never crosses itself.
    """

    transformer: Any
    propensity_model: Any
    learner: Learner
    members: list[Learner] = field(default_factory=list)
    clip: tuple[float, float] = (0.03, 0.97)
    interval: tuple[float, float] = (0.10, 0.90)
    reference_discount_bps: float = 10.0
    support: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.learner.name}/{self.learner.family}"

    def design(self, built: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = self.transformer.transform(built)
        return X, np.clip(_proba(self.propensity_model, X), *self.clip)

    @property
    def ensemble(self) -> list[Learner]:
        return [self.learner, *self.members]

    def _draws(self, X, e, dose) -> np.ndarray:
        return np.vstack([m.uplift(X, dose, e) for m in self.ensemble])

    def uplift(self, X, e, dose) -> np.ndarray:
        return np.median(self._draws(X, e, dose), axis=0)

    def band(self, X, e, dose) -> tuple[np.ndarray, np.ndarray]:
        low, high = np.quantile(self._draws(X, e, dose), self.interval, axis=0)
        return low, high

    def baseline(self, X, e) -> np.ndarray:
        return np.median(np.vstack([m.baseline(X, e) for m in self.ensemble]), axis=0)
