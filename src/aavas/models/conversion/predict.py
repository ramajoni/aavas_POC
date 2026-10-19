"""Conversion scoring, including the price-response curve the optimizer walks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...features.feature_pipeline import FeatureTransformer
from .. import base

NAME = "conversion"


class ConversionModel:
    def __init__(self, model, transformer: FeatureTransformer, support: dict | None = None):
        self.model = model
        self.transformer = transformer
        self.support = support or {}
        self.card: base.ModelCard | None = None

    @classmethod
    def load(cls) -> "ConversionModel":
        payload, card = base.load(NAME)
        obj = cls(payload["model"], payload["transformer"], payload.get("support"))
        obj.card = card
        return obj

    def supported_rate_band(self, df: pd.DataFrame) -> tuple[float, float] | None:
        """Rate range this model can actually speak to for this file.

        Bounded by the rates seen in training and by the competitor gap seen in
        training, whichever binds first. Outside it the model returns a flat
        extrapolation that the optimizer must not mistake for a demand curve.
        """
        if not self.support:
            return None
        low, high = self.support["offered_rate"]
        gap_low, gap_high = self.support["competitor_gap_bps"]
        competitor = pd.to_numeric(df.get("competitor_rate"), errors="coerce")
        if competitor is not None and len(competitor) and pd.notna(competitor.iloc[0]):
            reference = float(competitor.iloc[0])
            low = max(low, reference + gap_low / 100.0)
            high = min(high, reference + gap_high / 100.0)
        return (round(float(low), 2), round(float(high), 2))

    def predict_at_rate(self, df: pd.DataFrame, rate: float | np.ndarray) -> np.ndarray:
        """P(disbursed) if this customer is offered this rate."""
        scored = df.copy()
        scored["offered_rate"] = rate
        if "competitor_rate" in scored.columns:
            comp = pd.to_numeric(scored["competitor_rate"], errors="coerce")
            scored["competitor_gap_bps"] = ((scored["offered_rate"] - comp) * 100).fillna(0.0)
        return self.model.predict_proba(self.transformer.transform(scored))[:, 1]

    def predict_over_rates(self, application: pd.DataFrame, rates: np.ndarray) -> np.ndarray:
        """Conversion at every candidate rate, in a single scoring pass."""
        expanded = base.expand_over_rates(application, rates)
        return self.model.predict_proba(self.transformer.transform(expanded))[:, 1]

    def demand_curve(self, application: pd.DataFrame, rates: np.ndarray) -> pd.DataFrame:
        """The whole curve for one application - what the sales screen plots."""
        return pd.DataFrame({"rate": np.asarray(rates, dtype=float),
                             "conversion_probability": self.predict_over_rates(application, rates)})

    def elasticity(self, application: pd.DataFrame, rate: float, bump_bps: float = 25.0) -> float:
        """Percentage points of conversion lost per ``bump_bps`` of extra price."""
        p0, p1 = self.predict_over_rates(application, np.array([rate, rate + bump_bps / 100.0]))
        return round(float(p0 - p1) * 100, 3)
