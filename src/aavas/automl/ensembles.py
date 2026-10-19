"""Ensembles - bagging, voting, stacking, blending, weighted.

One constraint governs this module and it comes from the optimizer, not from
taste: **a blend of a monotone and a non-monotone model is not monotone.**
``price_optimizer`` searches a grid over price, and that search is only safe
while the demand curve cannot bend the wrong way.

So ``build`` refuses to combine members that would break the guarantee when the
contract requires it. The refusal is explicit rather than a silent filter,
because a quietly dropped member is how a monotone guarantee is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..logging_utils import get_logger
from . import registry

log = get_logger(__name__)

METHODS = ["voting_soft", "voting_hard", "bagging", "stacking", "blending", "weighted"]


class WeightedEnsemble:
    """Probability average with fixed weights. Monotone if every member is."""

    def __init__(self, members: Sequence[Any], weights: Sequence[float] | None = None):
        self.members = list(members)
        n = len(self.members)
        w = np.asarray(weights if weights is not None else [1.0] * n, dtype=float)
        self.weights = (w / w.sum()).tolist()

    def fit(self, X, y):
        for member in self.members:
            member.fit(X, y)
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict_proba(self, X):
        stacked = np.stack([m.predict_proba(X)[:, 1] for m in self.members])
        blended = np.average(stacked, axis=0, weights=self.weights)
        return np.column_stack([1 - blended, blended])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class BlendingEnsemble:
    """Holdout blending: members fit on part A, the meta-learner on part B.

    Distinct from stacking, which uses cross-validated out-of-fold predictions.
    Cheaper, and with a time-ordered split it keeps the chronological discipline
    the rest of this repo insists on.
    """

    def __init__(self, members: Sequence[Any], holdout: float = 0.25, meta=None):
        self.members = list(members)
        self.holdout = holdout
        self.meta = meta

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y)
        n = len(y)
        cut = int(n * (1 - self.holdout))
        X_a, X_b = _slice(X, slice(0, cut)), _slice(X, slice(cut, n))
        y_a, y_b = y[:cut], y[cut:]

        for member in self.members:
            member.fit(X_a, y_a)
        meta_features = np.column_stack([m.predict_proba(X_b)[:, 1] for m in self.members])
        self.meta_ = self.meta or LogisticRegression(max_iter=1000)
        self.meta_.fit(meta_features, y_b)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack([m.predict_proba(X)[:, 1] for m in self.members])
        return self.meta_.predict_proba(meta_features)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _slice(X, sl):
    return X.iloc[sl] if hasattr(X, "iloc") else np.asarray(X)[sl]


@dataclass
class EnsembleSpec:
    method: str
    members: list[str]
    monotone: bool
    estimator: Any


def build(method: str, member_names: Sequence[str], *, monotone_required: bool = False,
          weights: Sequence[float] | None = None, params: dict[str, Any] | None = None) -> EnsembleSpec:
    """Assemble one ensemble from registered algorithm names."""
    params = params or {}
    algos = [registry.get(name) for name in member_names]

    unavailable = [a.name for a in algos if not a.available()]
    if unavailable:
        raise RuntimeError(f"ensemble members unavailable: {unavailable}")

    non_monotone = [a.name for a in algos if not a.monotone]
    if monotone_required and non_monotone:
        raise ValueError(
            f"{method}: members {non_monotone} are not monotone-capable, and this contract "
            "requires monotonicity in price. A blend of monotone and non-monotone members "
            "is not monotone.")

    members = [a.instantiate(**params.get(a.name, {})) for a in algos]
    monotone = all(a.monotone for a in algos)

    if method in {"voting_soft", "voting_hard"}:
        from sklearn.ensemble import VotingClassifier

        estimator = VotingClassifier(
            estimators=list(zip(member_names, members)),
            voting="soft" if method == "voting_soft" else "hard")
    elif method == "bagging":
        from sklearn.ensemble import BaggingClassifier

        estimator = BaggingClassifier(estimator=members[0], n_estimators=10, n_jobs=-1)
        monotone = algos[0].monotone
    elif method == "stacking":
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression

        estimator = StackingClassifier(
            estimators=list(zip(member_names, members)),
            final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1)
    elif method == "blending":
        estimator = BlendingEnsemble(members)
    elif method == "weighted":
        estimator = WeightedEnsemble(members, weights)
    else:
        raise ValueError(f"unknown ensemble method {method!r}; expected one of {METHODS}")

    return EnsembleSpec(method=method, members=list(member_names),
                        monotone=monotone, estimator=estimator)
