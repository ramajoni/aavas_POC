"""Attribution for pricing models.

Every family the AutoML portfolio can select is explained with exact SHAP values,
computed by the engine that owns the model wherever one exists:

* XGBoost  - ``Booster.predict(DMatrix, pred_contribs=True)``
* LightGBM - ``predict(..., pred_contrib=True)``
* CatBoost - ``get_feature_importance(type="ShapValues")``
* linear   - coefficient x distance from the training mean, which *is* the
             (interventional) SHAP value of a linear model - no library needed
* sklearn trees (HistGradientBoosting, GradientBoosting, forests) -
             ``shap.TreeExplainer``

The native engines exist for a reason beyond speed: shap 0.49 cannot parse the
``base_score`` XGBoost 3.x writes (``could not convert string to float:
'[1.12689085E1]'``), which silently sent the Stage-2 rate model to the ablation
fallback. Each engine returns a bias column, which is the base value, and a
self-check at construction confirms ``base value + sum(contributions)`` equals
the model's raw output on the reference row; an engine that does not reconcile is
not used.

Units follow the model's own output: the rate for a regressor; log-odds for a
boosted or linear classifier (the raw margin, before calibration); probability
for a forest classifier, whose trees average leaf probabilities. ``units`` says
which.

The last resort is a baseline ablation explainer: hold the file at the
population-median customer, put one real feature value back at a time, and
measure how the prediction moves, rescaled so the reason codes add up to the
number on the screen. It is not SHAP - it ignores interactions - and when it runs
``method()`` says "ablation" and ``fallback_reason`` says why. A pricing screen
that shows nothing because a library is missing is worse than one that shows an
honest approximation and names the method that produced it.
"""

from __future__ import annotations

import os
from contextlib import redirect_stderr
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import model_config
from ..features.feature_pipeline import display_value, reason_label
from ..logging_utils import get_logger
from ..models import base as model_base

log = get_logger(__name__)

# |base + sum(contributions) - raw output| allowed at the self-check. XGBoost
# accumulates in float32, so exact-to-1e-12 is not on offer; 1e-4 of a rate % is
# a hundredth of a basis point.
RECONCILE_TOLERANCE = 1e-4

_FOREST_CLASSES = {"RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier"}


@dataclass
class Attribution:
    feature: str
    value: object
    contribution: float          # in the model's output units (rate % or probability)

    @property
    def bps(self) -> float:
        return round(self.contribution * 100, 1)


class PricingExplainer:
    """Wraps a fitted model plus its transformer and explains one row at a time."""

    def __init__(self, model, transformer, backend: str | None = None, is_classifier: bool = False):
        self.model = model
        self.transformer = transformer
        self.is_classifier = is_classifier
        # Calibration wrappers are reached through to the fitted learner: the
        # attribution explains the learner's margin, which isotonic calibration
        # then maps monotonically onto a probability.
        self.estimator = model_base.base_estimator(model)
        self.baseline_ = transformer.baseline_row()
        self.engine = "ablation"
        self.fallback_reason: str | None = None
        self._units = "probability" if is_classifier else "output"
        self._shap = None
        self._linear_reference: np.ndarray | None = None
        self.backend = self._resolve_backend(backend)
        reference = self.baseline_.to_frame().T.astype(float)
        self._base_value: float | None = None
        if self.backend == "shap":
            self._base_value = self._setup(reference)
            if self._base_value is None:
                self.backend = "ablation"
                self.engine = "ablation"
                self._units = "probability" if is_classifier else "output"
                log.warning("exact attribution unavailable for %s (%s); using the ablation "
                            "explainer", type(self.estimator).__name__, self.fallback_reason)
        self.baseline_prediction_ = float(self._predict(reference)[0])

    # ------------------------------------------------------------------ backend
    def _resolve_backend(self, backend: str | None) -> str:
        choice = backend or model_config()["xai"].get("backend", "auto")
        if choice == "ablation":
            self.fallback_reason = "xai.backend is set to ablation"
            return "ablation"
        return "shap"

    def _import_shap(self):
        try:
            # Broad catch on purpose: a shap built against a different numpy ABI
            # raises AttributeError, not ImportError, and a broken dependency
            # should degrade exactly like a missing one rather than take the
            # pricing screen down with it. Its import chatter is redirected so a
            # broken optional dependency cannot spray a traceback into a
            # customer-facing log.
            with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
                import shap
            return shap
        except Exception as exc:                      # noqa: BLE001
            self.fallback_reason = f"shap unusable ({type(exc).__name__}: {exc})"
            return None

    def _setup(self, reference: pd.DataFrame) -> float | None:
        """Pick the exact engine for this estimator; return the base value, or None."""
        est = self.estimator
        library = type(est).__module__.split(".")[0]
        try:
            if library == "xgboost" and hasattr(est, "get_booster"):
                self.engine = "xgboost-native-treeshap"
                self._units = "log_odds" if self.is_classifier else "output"
            elif library == "lightgbm" and hasattr(est, "booster_"):
                self.engine = "lightgbm-native-treeshap"
                self._units = "log_odds" if self.is_classifier else "output"
            elif library == "catboost":
                self.engine = "catboost-native-shap"
                self._units = "log_odds" if self.is_classifier else "output"
            elif hasattr(est, "exact_contributions"):
                self.engine = "linear-exact"
                self._units = "log_odds" if self.is_classifier else "output"
            elif _is_plain_linear(est):
                # A linear model with no stored training mean: exact against the
                # transformer's median customer instead.
                self.engine = "linear-exact"
                self._units = "log_odds" if self.is_classifier else "output"
                self._linear_reference = reference.to_numpy(dtype=float)[0]
            else:
                shap = self._import_shap()
                if shap is None:
                    return None
                self._shap = shap.TreeExplainer(est)
                self.engine = "shap-treeexplainer"
                forest = type(est).__name__ in _FOREST_CLASSES
                self._units = ("probability" if forest else "log_odds") if self.is_classifier \
                    else "output"
            values, base = self._attribute(reference)
            raw = self.raw_output(reference)
            if self.engine == "catboost-native-shap":
                # CatBoost's expected value can depend on the batch it is computed
                # over (it does with monotone constraints). Score the reference
                # row again beside a different row: an explanation that changes
                # with its neighbours is not one to show a customer.
                other = reference.copy()
                other.iloc[0] = 0.0
                _, paired = self._attribute(pd.concat([reference, other], ignore_index=True))
                drift = abs(float(paired[0]) - float(np.ravel(base)[0]))
                if drift > RECONCILE_TOLERANCE:
                    self.fallback_reason = (f"{self.engine} is batch-dependent for this model "
                                            f"(expected value moved {drift:.3g})")
                    return None
            gap = float(np.max(np.abs(np.ravel(base) + values.sum(axis=1) - raw)))
            limit = RECONCILE_TOLERANCE * max(1.0, float(np.max(np.abs(raw))))
            if not np.isfinite(gap) or gap > limit:
                self.fallback_reason = (f"{self.engine} did not reconcile: base + contributions "
                                        f"is {gap:.3g} away from the model output")
                return None
            return float(np.ravel(base)[0])
        except Exception as exc:                      # noqa: BLE001 - any engine failure falls back
            self.fallback_reason = f"{self.engine} failed ({type(exc).__name__}: {exc})"
            return None

    # ------------------------------------------------------------- the engines
    def _attribute(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(contributions (n, features), base value per row) from the exact engine."""
        est = self.estimator
        if self.engine == "xgboost-native-treeshap":
            import xgboost

            out = est.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)
            out = out[:, -1, :] if out.ndim == 3 else out
            return np.asarray(out[:, :-1], dtype=float), np.asarray(out[:, -1], dtype=float)
        if self.engine == "lightgbm-native-treeshap":
            out = np.asarray(est.predict(X, pred_contrib=True), dtype=float)
            return out[:, :-1], out[:, -1]
        if self.engine == "catboost-native-shap":
            from catboost import Pool

            out = np.asarray(est.get_feature_importance(Pool(X), type="ShapValues"), dtype=float)
            out = out[:, -1, :] if out.ndim == 3 else out
            return out[:, :-1], out[:, -1]
        if self.engine == "linear-exact":
            if self._linear_reference is None:
                values, base = est.exact_contributions(X)
                return np.asarray(values, dtype=float), np.full(len(X), float(base))
            coef = np.ravel(est.coef_)
            values = (X.to_numpy(dtype=float) - self._linear_reference) * coef
            base = float(np.ravel(est.intercept_)[0]) + float(self._linear_reference @ coef)
            return values, np.full(len(X), base)
        if self.engine == "shap-treeexplainer":
            values = self._shap.shap_values(X)
            if isinstance(values, list):
                values = values[-1]
            values = np.asarray(values, dtype=float)
            if values.ndim == 3:                       # (n, features, classes)
                values = values[..., -1]
            base = float(np.ravel(self._shap.expected_value)[-1])
            return values, np.full(len(X), base)
        raise RuntimeError(f"no exact engine for {type(est).__name__}")

    def raw_output(self, X: pd.DataFrame) -> np.ndarray:
        """The model output the attributions sum to, computed independently of them:
        the margin for a boosted or linear classifier, the probability for a forest
        or the ablation fallback, the prediction for a regressor."""
        est = self.estimator
        if self.backend != "shap" or self.engine == "ablation":
            return self._predict(X)
        if self.engine == "xgboost-native-treeshap":
            import xgboost

            return np.asarray(est.get_booster().predict(xgboost.DMatrix(X), output_margin=True),
                              dtype=float)
        if self.engine == "lightgbm-native-treeshap":
            return np.asarray(est.predict(X, raw_score=True), dtype=float)
        if self.engine == "catboost-native-shap":
            return np.asarray(est.predict(X, prediction_type="RawFormulaVal"), dtype=float)
        if not self.is_classifier:
            return np.asarray(est.predict(X), dtype=float)
        if self._units == "probability":
            return np.asarray(est.predict_proba(X)[:, 1], dtype=float)
        return np.asarray(est.decision_function(X), dtype=float)

    # --------------------------------------------------------------- reference
    @property
    def base_value(self) -> float:
        """Where the attributions start: base value + attributions = prediction.

        SHAP starts from the model's average output over its training data (the
        engine's bias column); the ablation fallback starts from the
        population-median customer.
        """
        if self.backend == "shap" and self._base_value is not None:
            return self._base_value
        return self.baseline_prediction_

    @property
    def units(self) -> str:
        """Unit of the attributions: "output" for a regressor; for a classifier
        "log_odds" (boosted / linear margin) or "probability" (forests, ablation)."""
        return self._units

    # --------------------------------------------------------------- prediction
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.is_classifier:
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    # ---------------------------------------------------------------- explain
    def explain(self, application: pd.DataFrame, build=None) -> list[Attribution]:
        """Attributions for a single application, largest magnitude first."""
        row = application.iloc[[0]] if len(application) > 1 else application
        frame = build(row) if build else row
        X = self.transformer.transform(frame)
        contributions = self.contributions(X)[0]

        attributions = [
            Attribution(feature=col, value=self._display_value(frame, col, X, i),
                        contribution=float(contributions[i]))
            for i, col in enumerate(X.columns)
        ]
        attributions.sort(key=lambda a: abs(a.contribution), reverse=True)
        return attributions

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-feature contributions for an already-transformed frame, (n, features)."""
        if self.backend == "shap" and self._base_value is not None:
            return self._attribute(X)[0]
        return self._ablation_values(X)

    def _ablation_values(self, X: pd.DataFrame, chunk: int = 256) -> np.ndarray:
        """One-at-a-time contribution against the median customer, rescaled to the
        total gap so the reason codes reconcile with the headline number."""
        baseline = self.baseline_.reindex(X.columns).to_numpy(dtype=float)
        n_features = X.shape[1]
        out = np.zeros((len(X), n_features))
        values = X.to_numpy(dtype=float)
        for start in range(0, len(X), chunk):
            block = values[start:start + chunk]
            # Row r, feature i: the median customer with feature i set to row r's value.
            grid = np.repeat(baseline[None, :], len(block) * n_features, axis=0)
            rows = np.repeat(np.arange(len(block)), n_features)
            cols = np.tile(np.arange(n_features), len(block))
            grid[np.arange(len(grid)), cols] = block[rows, cols]
            moved = (self._predict(pd.DataFrame(grid, columns=X.columns))
                     - self.baseline_prediction_).reshape(len(block), n_features)
            total_gap = self._predict(pd.DataFrame(block, columns=X.columns)) \
                - self.baseline_prediction_
            moved_sum = moved.sum(axis=1)
            scale = np.divide(total_gap, moved_sum, out=np.ones_like(total_gap),
                              where=np.abs(moved_sum) > 1e-9)
            out[start:start + len(block)] = moved * scale[:, None]
        return out

    def _display_value(self, frame: pd.DataFrame, column: str, X: pd.DataFrame, index: int):
        if column in self.transformer.spec.categorical:
            return self.transformer.decode_category(column, X.iloc[0, index])
        raw = frame[column].iloc[0] if column in frame.columns else X.iloc[0, index]
        value = float(raw) if isinstance(raw, (int, float, np.floating, np.integer)) else raw
        # A sentinel code (bureau_score -1) is shown as what it means.
        return display_value(column, value)

    # --------------------------------------------------------------- reporting
    def attribution_frame(self, application: pd.DataFrame, build=None) -> pd.DataFrame:
        attributions = self.explain(application, build)
        return pd.DataFrame([
            {"feature": a.feature, "value": a.value, "contribution": round(a.contribution, 5), "bps": a.bps}
            for a in attributions
        ])

    def method(self) -> str:
        return self.backend

    def describe(self) -> dict[str, Any]:
        """Which method ran, which engine computed it, in which units - and why
        the fallback ran, when it did."""
        return {"method": self.method(), "engine": self.engine, "units": self.units,
                "fallback_reason": self.fallback_reason if self.backend != "shap" else None}


def _is_plain_linear(est) -> bool:
    return (type(est).__module__.startswith("sklearn.linear_model")
            and hasattr(est, "coef_") and hasattr(est, "intercept_")
            and np.ravel(est.coef_).ndim == 1 and len(np.ravel(est.intercept_)) == 1)


# ------------------------------------------------------------- global weights
def global_weights(explainer: PricingExplainer, X: pd.DataFrame,
                   top: int | None = None) -> dict[str, Any]:
    """Each feature's share of mean |SHAP| over the rows of ``X``.

    Shares, not raw magnitudes, because the units differ between models (rate %
    for the anchor, log-odds for PD) and a share is the one number comparable
    across them. ``X`` is the model's transformed holdout - a ranking computed on
    the rows a model was fitted on measures memorisation.
    """
    values = explainer.contributions(X)
    mean_abs = np.abs(values).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    rows = [{"feature": col, "label": reason_label(col)["label"],
             "mean_abs": round(float(m), 6), "share_pct": round(100 * float(m) / total, 2)}
            for col, m in zip(X.columns, mean_abs)]
    rows.sort(key=lambda r: r["mean_abs"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {**explainer.describe(), "rows": int(len(X)),
            "features": rows[:top] if top else rows}
