"""LGD scoring: loss severity for a file, as a fraction of exposure at default.

Clipped to the configured range here and floored by the consumer
(``RiskModel``, ``risk.lgd_floor``): the floor is a policy on top of the model,
not part of it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...config import model_config
from ...features import stage2_features
from ...features.feature_pipeline import FeatureTransformer
from .. import base

NAME = "lgd"


class LGDModel:
    def __init__(self, model, transformer: FeatureTransformer,
                 clip: tuple[float, float] | list[float] | None = None,
                 evaluation: dict[str, Any] | None = None):
        self.model = model
        self.transformer = transformer
        self.clip = tuple(clip or model_config()[NAME].get("clip", [0.0, 0.95]))
        self.evaluation = evaluation or {}
        self.card: base.ModelCard | None = None

    @classmethod
    def load(cls) -> "LGDModel":
        payload, card = base.load(NAME)
        obj = cls(payload["model"], payload["transformer"], payload.get("clip"),
                  payload.get("evaluation"))
        obj.card = card
        return obj

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Expected LGD (a fraction) for each row, before any policy floor."""
        X = self.transformer.transform(stage2_features.build(df))
        return np.clip(self.model.predict(X), self.clip[0], self.clip[1])

    def predict_points(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict(df) * 100.0

    @property
    def importance(self) -> dict[str, Any]:
        """Global mean |attribution| per feature, in LGD points, with its method."""
        return self.evaluation.get("importance", {})

    @property
    def calibration(self) -> dict[str, Any]:
        """Holdout calibration by predicted decile, for the model and the table."""
        return self.evaluation.get("calibration", {})
