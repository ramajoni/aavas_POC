"""Stage-2 scoring: an anchor rate plus the uncertainty around it."""

from __future__ import annotations

import pandas as pd

from ...features import stage2_features
from ...ingestion.schemas import RateBand
from .. import base

NAME = "stage2"


class Stage2PricingModel:
    def __init__(self, model, transformer, residual_std: float):
        self.model = model
        self.transformer = transformer
        self.residual_std = residual_std
        self.card: base.ModelCard | None = None

    @classmethod
    def load(cls) -> "Stage2PricingModel":
        payload, card = base.load(NAME)
        obj = cls(payload["model"], payload["transformer"], payload["residual_std"])
        obj.card = card
        return obj

    def predict_frame(self, df: pd.DataFrame) -> pd.Series:
        X = self.transformer.transform(stage2_features.build(df))
        return pd.Series(self.model.predict(X), index=df.index).round(4)

    def predict(self, application) -> dict:
        df = application if isinstance(application, pd.DataFrame) else pd.DataFrame([application])
        rate = float(self.predict_frame(df).iloc[0])
        # +/- 1 residual sigma: the search window handed to the optimizer, which
        # then narrows it further using policy and yield.
        band = RateBand(round(rate - self.residual_std, 2), round(rate + self.residual_std, 2))
        return {
            "anchor_rate": round(rate, 2),
            "search_band": band.as_dict(),
            "residual_std_bps": round(self.residual_std * 100, 1),
            "model_version": self.card.version if self.card else None,
        }

    def features_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transformer.transform(stage2_features.build(df))
