"""Override-risk scoring for a live recommendation."""

from __future__ import annotations

import pandas as pd

from ...features import sales_features
from .. import base

NAME = "override"


class OverrideModel:
    def __init__(self, classifier, regressor, transformer):
        self.classifier = classifier
        self.regressor = regressor
        self.transformer = transformer
        self.card: base.ModelCard | None = None

    @classmethod
    def load(cls) -> "OverrideModel":
        payload, card = base.load(NAME)
        obj = cls(payload["classifier"], payload["regressor"], payload["transformer"])
        obj.card = card
        return obj

    def predict(self, df: pd.DataFrame, recommended_rate: float | None = None) -> pd.DataFrame:
        scored = df.copy()
        if recommended_rate is not None:
            scored["ai_recommended_rate"] = recommended_rate
        scored = sales_features.build(scored, reference_rate_col="ai_recommended_rate")
        X = self.transformer.transform(scored)
        probability = self.classifier.predict_proba(X)[:, 1]
        magnitude = self.regressor.predict(X)
        return pd.DataFrame({
            "override_probability": probability.round(4),
            "expected_override_bps": magnitude.round(1),
            # What a human is likely to actually offer if they intervene.
            "anticipated_human_rate": (scored["ai_recommended_rate"].to_numpy() + magnitude / 100).round(2),
        }, index=df.index)
