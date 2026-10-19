"""Attribution for pricing models.

SHAP is used when it is installed. When it is not, the fallback is a baseline
ablation explainer: hold the file at the population-median customer, put one
real feature value back at a time, and measure how the prediction moves. The
attributions are then rescaled to sum to the total distance from the baseline
prediction, so the reason codes always add up to the number on the screen.

The fallback is not SHAP - it ignores interactions and is order-independent only
because it uses a single reference point. That trade is deliberate: a pricing
screen that shows nothing because a library is missing is worse than one that
shows an honest approximation and says which method produced it.
"""

from __future__ import annotations

import os
from contextlib import redirect_stderr
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import model_config
from ..logging_utils import get_logger
from ..models import base as model_base

log = get_logger(__name__)


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
        self.baseline_ = transformer.baseline_row()
        self.backend = self._resolve_backend(backend)
        self._shap = None
        if self.backend == "shap":
            self._shap = self._build_shap()
            if self._shap is None:
                self.backend = "ablation"
        self.baseline_prediction_ = float(self._predict(self.baseline_.to_frame().T)[0])

    # ------------------------------------------------------------------ backend
    def _resolve_backend(self, backend: str | None) -> str:
        choice = backend or model_config()["xai"].get("backend", "auto")
        if choice == "ablation":
            return "ablation"
        try:
            # Broad catch on purpose: a shap built against a different numpy ABI
            # raises AttributeError, not ImportError, and a broken dependency
            # should degrade exactly like a missing one rather than take the
            # pricing screen down with it. Its import chatter is redirected so a
            # broken optional dependency cannot spray a traceback into a
            # customer-facing log.
            with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
                import shap  # noqa: F401
        except Exception as exc:                      # noqa: BLE001
            log.warning("shap unusable (%s: %s); using the ablation explainer",
                        type(exc).__name__, exc)
            return "ablation"
        return "shap"

    def _build_shap(self):
        try:
            import shap

            return shap.TreeExplainer(model_base.base_estimator(self.model))
        except Exception as exc:                      # noqa: BLE001 - any shap failure falls back
            log.warning("shap explainer unavailable (%s); using the ablation explainer", exc)
            return None

    # --------------------------------------------------------------- reference
    @property
    def base_value(self) -> float:
        """Where the attributions start: base value + attributions = prediction.

        SHAP starts from the model's average output over its training data; the
        ablation fallback starts from the population-median customer.
        """
        if self.backend == "shap" and self._shap is not None:
            return float(np.ravel(self._shap.expected_value)[-1])
        return self.baseline_prediction_

    @property
    def units(self) -> str:
        """Unit of the attributions. TreeSHAP explains a classifier's raw margin,
        which is log-odds; the ablation fallback moves the probability itself."""
        if not self.is_classifier:
            return "output"
        return "log_odds" if self.backend == "shap" and self._shap is not None else "probability"

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
        prediction = float(self._predict(X)[0])

        if self.backend == "shap" and self._shap is not None:
            contributions = self._shap_values(X)
        else:
            contributions = self._ablation_values(X, prediction)

        attributions = [
            Attribution(feature=col, value=self._display_value(frame, col, X, i),
                        contribution=float(contributions[i]))
            for i, col in enumerate(X.columns)
        ]
        attributions.sort(key=lambda a: abs(a.contribution), reverse=True)
        return attributions

    def _shap_values(self, X: pd.DataFrame) -> np.ndarray:
        values = self._shap.shap_values(X)
        values = np.asarray(values)
        if values.ndim == 3:                       # (n, features, classes)
            values = values[..., -1]
        return values[0]

    def _ablation_values(self, X: pd.DataFrame, prediction: float) -> np.ndarray:
        """One-at-a-time contribution against the median customer, rescaled to the
        total gap so the reason codes reconcile with the headline number."""
        baseline = self.baseline_.copy()
        counterfactuals = pd.DataFrame(
            np.repeat(baseline.to_numpy().reshape(1, -1), len(X.columns), axis=0),
            columns=X.columns,
        )
        for i, col in enumerate(X.columns):
            counterfactuals.iat[i, i] = X.iloc[0, i]
        moved = self._predict(counterfactuals) - self.baseline_prediction_

        total_gap = prediction - self.baseline_prediction_
        moved_sum = moved.sum()
        if abs(moved_sum) > 1e-9:
            moved = moved * (total_gap / moved_sum)
        return moved

    def _display_value(self, frame: pd.DataFrame, column: str, X: pd.DataFrame, index: int):
        if column in self.transformer.spec.categorical:
            return self.transformer.decode_category(column, X.iloc[0, index])
        raw = frame[column].iloc[0] if column in frame.columns else X.iloc[0, index]
        return float(raw) if isinstance(raw, (int, float, np.floating, np.integer)) else raw

    # --------------------------------------------------------------- reporting
    def attribution_frame(self, application: pd.DataFrame, build=None) -> pd.DataFrame:
        attributions = self.explain(application, build)
        return pd.DataFrame([
            {"feature": a.feature, "value": a.value, "contribution": round(a.contribution, 5), "bps": a.bps}
            for a in attributions
        ])

    def method(self) -> str:
        return self.backend
