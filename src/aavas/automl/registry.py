"""Every algorithm in the AutoML catalogue, with what it can and cannot do.

The registry is deliberately complete: each family from the catalogue has an
entry, including the ones that cannot run here. An algorithm that is absent from
a registry is invisible; an algorithm that is present and reports
``unavailable: no sequence data in the schema`` has told you something.

Three capability flags do the real work downstream:

* ``monotone``    - can carry a monotone constraint. The conversion and prepayment
  contracts require this, because ``price_optimizer`` searches a grid over price
  and a demand curve that bends the wrong way makes that search unsafe.
* ``interpretable`` - can yield a per-feature attribution a reason code can be
  written from.
* ``needs``       - the data shape it requires: tabular, sequence or graph.

Nothing here is imported at module load. A missing library is reported, never
raised, which is the same rule the rest of the repository follows.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

TABULAR, SEQUENCE, GRAPH = "tabular", "sequence", "graph"


@dataclass
class Algorithm:
    name: str
    family: str
    build: Callable[..., Any] | None = None
    requires: str | None = None
    monotone: bool = False
    interpretable: bool = False
    calibratable: bool = True
    needs: str = TABULAR
    notes: str = ""
    # Above this many training rows the learner is impractical rather than wrong.
    # None means no limit.
    max_rows: int | None = None
    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- availability
    def available(self) -> bool:
        if self.build is None:
            return False
        if self.requires is None:
            return True
        try:
            importlib.import_module(self.requires)
            return True
        except Exception:  # noqa: BLE001 - a broken build degrades like a missing one
            return False

    def unavailable_reason(self) -> str | None:
        """Why this cannot run, stated so a reader knows whether it is fixable.

        The three kinds are genuinely different and were previously flattened into
        one sentence each. A missing package is an install; a missing adapter is
        work someone could do; a missing *data shape* is not fixable at all
        without inventing structure the schema does not have - and saying that
        plainly is more useful than implying somebody forgot.
        """
        if self.needs == SEQUENCE:
            return ("no sequence data in the schema: a loan carries one application date and "
                    "scalar outcomes, not a repayment or balance path. This becomes available "
                    "when the schema carries per-period rows per loan, not before - and "
                    "manufacturing a sequence from scalars would be inventing the data")
        if self.needs == GRAPH:
            return ("no graph data in the schema: there are no modelled relationships between "
                    "applicants. The only edges available are branch and sales officer, which "
                    "are identity columns held out of the risk features on purpose - a graph "
                    "over them would price a customer on who served them")
        if self.build is None:
            return f"no adapter implemented (would need `{self.requires}`)" if self.requires \
                else "no adapter implemented"
        if not self.available():
            return f"requires `pip install {self.requires}`"
        return None

    def instantiate(self, **params):
        if self.build is None:
            raise RuntimeError(f"{self.name}: no adapter implemented")
        merged = {**self.default_params, **params}
        return self.build(**merged)


# ============================================================ WoE scorecard
class WoEScorecard:
    """Weight-of-Evidence logistic scorecard - the artefact a regulator reads.

    Quantile-bins each numeric feature, replaces the value with the log-odds
    ratio of that bin, and fits a logistic regression on the result. Monotone in
    every transformed feature by construction, and every coefficient is a
    readable points-per-bin contribution.

    Implemented natively because it needs no dependency and it is the one model
    on this list that a credit committee can audit line by line.
    """

    def __init__(self, n_bins: int = 10, C: float = 1.0, min_bin_fraction: float = 0.02):
        self.n_bins = n_bins
        self.C = C
        self.min_bin_fraction = min_bin_fraction

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"n_bins": self.n_bins, "C": self.C, "min_bin_fraction": self.min_bin_fraction}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------- fit
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n_features = X.shape[1]
        self.edges_: list[np.ndarray] = []
        self.woe_: list[np.ndarray] = []

        total_pos = max(int(y.sum()), 1)
        total_neg = max(int(len(y) - y.sum()), 1)
        floor = max(int(self.min_bin_fraction * len(y)), 1)

        for j in range(n_features):
            column = X[:, j]
            quantiles = np.unique(np.nanquantile(column, np.linspace(0, 1, self.n_bins + 1)))
            edges = quantiles[1:-1] if len(quantiles) > 2 else np.array([np.nanmedian(column)])
            idx = np.digitize(column, edges)
            woe = np.zeros(len(edges) + 1)
            for b in range(len(edges) + 1):
                mask = idx == b
                pos = int(y[mask].sum())
                neg = int(mask.sum() - pos)
                # Laplace smoothing: an empty bin must not produce an infinite weight.
                pos_rate = (pos + 0.5) / (total_pos + floor)
                neg_rate = (neg + 0.5) / (total_neg + floor)
                woe[b] = float(np.log(pos_rate / neg_rate))
            self.edges_.append(edges)
            self.woe_.append(woe)

        self.model_ = LogisticRegression(C=self.C, max_iter=1000)
        self.model_.fit(self._transform(X), y)
        self.classes_ = self.model_.classes_
        return self

    def _transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            idx = np.digitize(X[:, j], self.edges_[j])
            out[:, j] = self.woe_[j][np.clip(idx, 0, len(self.woe_[j]) - 1)]
        return out

    def predict_proba(self, X):
        return self.model_.predict_proba(self._transform(X))

    def predict(self, X):
        return self.model_.predict(self._transform(X))

    def scorecard(self, feature_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Points table: what each bin of each feature contributes."""
        names = feature_names or [f"f{j}" for j in range(len(self.edges_))]
        rows = []
        for j, name in enumerate(names):
            coef = float(self.model_.coef_[0][j])
            for b, woe in enumerate(self.woe_[j]):
                lower = -np.inf if b == 0 else float(self.edges_[j][b - 1])
                upper = np.inf if b == len(self.edges_[j]) else float(self.edges_[j][b])
                rows.append({"feature": name, "bin": b, "lower": lower, "upper": upper,
                             "woe": round(float(woe), 4), "coefficient": round(coef, 4),
                             "points": round(float(woe) * coef, 4)})
        return rows


# ================================================================== builders
def _sk(path: str, **fixed):
    """Lazy sklearn builder - keeps import cost off module load."""
    def build(**params):
        module_name, cls_name = path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), cls_name)
        return cls(**{**fixed, **params})
    return build


def _xgb(**fixed):
    def build(**params):
        from xgboost import XGBClassifier

        return XGBClassifier(**{"eval_metric": "logloss", "verbosity": 0, **fixed, **params})
    return build


def _lgbm(**fixed):
    def build(**params):
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**{"verbose": -1, **fixed, **params})
    return build


def _cat(**fixed):
    def build(**params):
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**{"verbose": 0, "allow_writing_files": False, **fixed, **params})
    return build


def _ebm(**fixed):
    def build(**params):
        from interpret.glassbox import ExplainableBoostingClassifier

        return ExplainableBoostingClassifier(**{**fixed, **params})
    return build


def _tabpfn(**fixed):
    def build(**params):
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(**{**fixed, **params})
    return build


def _imb(path: str, **fixed):
    def build(**params):
        module_name, cls_name = path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), cls_name)
        return cls(**{**fixed, **params})
    return build


# =================================================================== catalogue
ALGORITHMS: list[Algorithm] = [
    # ---------------------------------------------------- linear / scorecard
    Algorithm("logistic_regression", "linear_scorecard",
              _sk("sklearn.linear_model.LogisticRegression", max_iter=1000),
              interpretable=True, notes="Baseline. Coefficients read directly.",
              search_space={"C": ("log", 0.01, 10.0)}),
    Algorithm("ridge_logistic", "linear_scorecard",
              _sk("sklearn.linear_model.LogisticRegression", penalty="l2", max_iter=1000),
              interpretable=True, search_space={"C": ("log", 0.01, 10.0)}),
    Algorithm("lasso_logistic", "linear_scorecard",
              _sk("sklearn.linear_model.LogisticRegression", penalty="l1",
                  solver="liblinear", max_iter=1000),
              interpretable=True, notes="Embedded feature selection via L1.",
              search_space={"C": ("log", 0.01, 10.0)}),
    Algorithm("elastic_net_logistic", "linear_scorecard",
              _sk("sklearn.linear_model.LogisticRegression", penalty="elasticnet",
                  solver="saga", l1_ratio=0.5, max_iter=5000),
              interpretable=True, search_space={"C": ("log", 0.01, 10.0),
                                                "l1_ratio": ("float", 0.1, 0.9)}),
    Algorithm("woe_scorecard", "linear_scorecard", lambda **p: WoEScorecard(**p),
              monotone=True, interpretable=True,
              notes="Native implementation. Monotone in WoE space by construction.",
              search_space={"n_bins": ("int", 5, 20), "C": ("log", 0.05, 5.0)}),

    # ------------------------------------------------------------------ trees
    Algorithm("decision_tree", "tree", _sk("sklearn.tree.DecisionTreeClassifier"),
              interpretable=True, search_space={"max_depth": ("int", 2, 12)}),
    Algorithm("random_forest", "tree", _sk("sklearn.ensemble.RandomForestClassifier", n_jobs=-1),
              search_space={"n_estimators": ("int", 100, 500), "max_depth": ("int", 3, 16)}),
    Algorithm("extra_trees", "tree", _sk("sklearn.ensemble.ExtraTreesClassifier", n_jobs=-1),
              search_space={"n_estimators": ("int", 100, 500), "max_depth": ("int", 3, 16)}),

    # -------------------------------------------------------------- boosting
    Algorithm("hist_gradient_boosting", "boosting",
              _sk("sklearn.ensemble.HistGradientBoostingClassifier"),
              monotone=True, notes="The incumbent learner in this repo.",
              search_space={"learning_rate": ("log", 0.02, 0.3), "max_depth": ("int", 3, 10)}),
    Algorithm("gradient_boosting", "boosting",
              _sk("sklearn.ensemble.GradientBoostingClassifier"),
              search_space={"learning_rate": ("log", 0.02, 0.3), "max_depth": ("int", 2, 6)}),
    Algorithm("xgboost", "boosting", _xgb(), requires="xgboost", monotone=True,
              search_space={"learning_rate": ("log", 0.02, 0.3), "max_depth": ("int", 3, 10)}),
    Algorithm("lightgbm", "boosting", _lgbm(), requires="lightgbm", monotone=True,
              search_space={"learning_rate": ("log", 0.02, 0.3), "num_leaves": ("int", 15, 255)}),
    Algorithm("catboost", "boosting", _cat(), requires="catboost", monotone=True,
              search_space={"learning_rate": ("log", 0.02, 0.3), "depth": ("int", 3, 10)}),

    # --------------------------------------------------------- interpretable
    Algorithm("ebm", "interpretable", _ebm(), requires="interpret",
              monotone=True, interpretable=True,
              notes="Glass-box GA2M. The highest-value addition to this suite.",
              search_space={"max_bins": ("int", 128, 512)}),
    Algorithm("gam", "interpretable", _ebm(interactions=0), requires="interpret",
              monotone=True, interpretable=True,
              notes="EBM with interactions disabled is an additive GAM."),

    # --------------------------------------------------------- kernel / margin
    Algorithm("svm_rbf", "kernel", _sk("sklearn.svm.SVC", probability=True),
              calibratable=True, max_rows=5000,
              notes="Kernel SVM is superlinear in rows; skipped on larger training sets.",
              search_space={"C": ("log", 0.1, 10.0)}),
    Algorithm("linear_svm", "kernel", _sk("sklearn.svm.LinearSVC"), calibratable=True,
              notes="No predict_proba; must be wrapped in calibration.",
              search_space={"C": ("log", 0.01, 10.0)}),

    # ------------------------------------------------------------- instance
    Algorithm("knn", "instance", _sk("sklearn.neighbors.KNeighborsClassifier", n_jobs=-1),
              max_rows=20000,
              notes="Weak on mixed-type tabular; included for completeness.",
              search_space={"n_neighbors": ("int", 5, 75)}),

    # ---------------------------------------------------------- probabilistic
    Algorithm("gaussian_nb", "probabilistic", _sk("sklearn.naive_bayes.GaussianNB")),
    Algorithm("bernoulli_nb", "probabilistic", _sk("sklearn.naive_bayes.BernoulliNB")),
    Algorithm("bayesian_logistic", "probabilistic",
              _sk("sklearn.linear_model.LogisticRegression", penalty="l2", max_iter=1000),
              interpretable=True,
              notes="L2 logistic is the MAP estimate under a Gaussian prior - a Laplace "
                    "approximation, not full posterior inference.",
              search_space={"C": ("log", 0.01, 10.0)}),

    # -------------------------------------------------------- neural tabular
    Algorithm("mlp", "neural_tabular",
              _sk("sklearn.neural_network.MLPClassifier", max_iter=500),
              search_space={"alpha": ("log", 1e-5, 1e-1)}),
    Algorithm("tabnet", "neural_tabular", None, requires="pytorch_tabnet",
              notes="No monotone constraints; disqualified for conversion and prepayment."),
    Algorithm("ft_transformer", "neural_tabular", None, requires="rtdl",
              notes="No monotone constraints."),
    Algorithm("saint", "neural_tabular", None, requires="saint",
              notes="No monotone constraints; no maintained package."),

    # ----------------------------------------------------- foundation models
    Algorithm("tabpfn", "foundation", _tabpfn(), requires="tabpfn", max_rows=10000,
              notes="Sample-size ceiling; no monotone constraints. Research only."),
    Algorithm("tabicl", "foundation", None, requires="tabicl",
              notes="No monotone constraints. Research only."),

    # ----------------------------------------------------------- sequential
    Algorithm("lstm", "sequential", None, needs=SEQUENCE),
    Algorithm("gru", "sequential", None, needs=SEQUENCE),
    Algorithm("tcn", "sequential", None, needs=SEQUENCE),
    Algorithm("transformer", "sequential", None, needs=SEQUENCE),

    # ---------------------------------------------------------------- graph
    Algorithm("gcn", "graph", None, needs=GRAPH),
    Algorithm("graphsage", "graph", None, needs=GRAPH),
    Algorithm("gat", "graph", None, needs=GRAPH),

    # --------------------------------------------------- imbalanced learning
    Algorithm("balanced_random_forest", "imbalanced",
              _imb("imblearn.ensemble.BalancedRandomForestClassifier", n_jobs=-1),
              requires="imblearn",
              notes="Resampling shifts the base rate; recalibrate before pricing."),
    Algorithm("easy_ensemble", "imbalanced",
              _imb("imblearn.ensemble.EasyEnsembleClassifier"), requires="imblearn",
              max_rows=50000,
              notes="Resampling shifts the base rate; recalibrate before pricing."),

    # --------------------------------------------------------- cost sensitive
    Algorithm("weighted_logistic", "cost_sensitive",
              _sk("sklearn.linear_model.LogisticRegression",
                  class_weight="balanced", max_iter=1000),
              interpretable=True, notes="Class weights distort calibration; recalibrate.",
              search_space={"C": ("log", 0.01, 10.0)}),
    Algorithm("cost_sensitive_random_forest", "cost_sensitive",
              _sk("sklearn.ensemble.RandomForestClassifier",
                  class_weight="balanced", n_jobs=-1),
              search_space={"n_estimators": ("int", 100, 400)}),
    Algorithm("cost_sensitive_xgboost", "cost_sensitive", _xgb(scale_pos_weight=3.0),
              requires="xgboost", monotone=True),
    Algorithm("cost_sensitive_lightgbm", "cost_sensitive", _lgbm(class_weight="balanced"),
              requires="lightgbm", monotone=True),
    Algorithm("cost_sensitive_catboost", "cost_sensitive", _cat(auto_class_weights="Balanced"),
              requires="catboost", monotone=True),
]

BY_NAME: dict[str, Algorithm] = {a.name: a for a in ALGORITHMS}
FAMILIES: list[str] = sorted({a.family for a in ALGORITHMS})


# ==================================================================== queries
def get(name: str) -> Algorithm:
    if name not in BY_NAME:
        raise KeyError(f"unknown algorithm {name!r}; {len(BY_NAME)} registered")
    return BY_NAME[name]


def candidates(*, monotone_required: bool = False, interpretable_only: bool = False,
               families: list[str] | None = None, available_only: bool = True,
               exclude: list[str] | None = None, n_rows: int | None = None) -> list[Algorithm]:
    """Filter the catalogue down to what may legally and practically run."""
    excluded = set(exclude or [])
    out = []
    for algo in ALGORITHMS:
        if algo.name in excluded:
            continue
        if n_rows is not None and algo.max_rows is not None and n_rows > algo.max_rows:
            continue
        if families and algo.family not in families:
            continue
        if monotone_required and not algo.monotone:
            continue
        if interpretable_only and not algo.interpretable:
            continue
        if available_only and (algo.needs != TABULAR or not algo.available()):
            continue
        out.append(algo)
    return out


def inventory() -> list[dict[str, Any]]:
    """Full catalogue with availability - what `make automl-inventory` prints."""
    return [{
        "name": a.name, "family": a.family, "monotone": a.monotone,
        "interpretable": a.interpretable, "needs": a.needs,
        "available": a.needs == TABULAR and a.available(),
        "reason": a.unavailable_reason(), "requires": a.requires, "notes": a.notes,
    } for a in ALGORITHMS]


def summary() -> dict[str, Any]:
    inv = inventory()
    return {
        "registered": len(inv),
        "available": sum(1 for r in inv if r["available"]),
        "monotone_capable": sum(1 for r in inv if r["monotone"] and r["available"]),
        "families": len(FAMILIES),
        "unavailable_by_reason": _count_reasons(inv),
    }


def _count_reasons(inv: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in inv:
        if not row["available"]:
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
