"""Stage-1 scoring: an indicative band, eligibility and a confidence level."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import policy
from ...features import stage1_features
from ...ingestion.schemas import RateBand
from .. import base

NAME = "stage1"


class Stage1PricingModel:
    def __init__(self, models: dict, transformer, quantiles: list[float]):
        self.models = models
        self.transformer = transformer
        self.quantiles = quantiles
        self.card: base.ModelCard | None = None

    @classmethod
    def load(cls) -> "Stage1PricingModel":
        payload, card = base.load(NAME)
        obj = cls(payload["models"], payload["transformer"], payload["quantiles"])
        obj.card = card
        return obj

    # ------------------------------------------------------------------ scoring
    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        pol = policy()
        s1 = pol["stage1"]
        built = stage1_features.build(df)
        X = self.transformer.transform(built)

        q_lo, q_mid, q_hi = (str(q) for q in self.quantiles)
        low = self.models[q_lo].predict(X)
        mid = self.models[q_mid].predict(X)
        high = self.models[q_hi].predict(X)

        # Quantile models are fitted independently and can cross on thin slices.
        low, high = np.minimum(low, high), np.maximum(low, high)
        mid = np.clip(mid, low, high)

        low, high = _enforce_band_width(low, high, s1)
        floor, ceiling = pol["reference"]["card_rate_floor"], pol["reference"]["card_rate_ceiling"]
        low = np.clip(_round_to(low, s1["round_to_bps"]), floor, ceiling)
        high = np.clip(_round_to(high, s1["round_to_bps"]), floor, ceiling)
        mid = np.clip(mid, low, high)

        width_bps = (high - low) * 100
        out = pd.DataFrame({
            "min_rate": np.round(low, 2),
            "max_rate": np.round(high, 2),
            "indicative_rate": np.round(mid, 2),
            "band_width_bps": np.round(width_bps, 1),
            "confidence": stage1_features.confidence(built, width_bps).to_numpy(),
        }, index=df.index)
        out["eligible_amount_max"] = stage1_features.eligibility(built, out["min_rate"], pol)
        out["eligible_amount_min"] = stage1_features.eligibility(built, out["max_rate"], pol)
        out["eligibility_binding"] = stage1_features.eligibility_binding_constraint(built, out["min_rate"], pol)
        return out

    def predict(self, application: dict | pd.Series | pd.DataFrame) -> dict:
        """Single-application scoring, shaped for the pre-assessment API."""
        df = _as_frame(application)
        row = self.predict_frame(df).iloc[0]
        band = RateBand(float(row["min_rate"]), float(row["max_rate"]))
        return {
            "band": band.as_dict(),
            "indicative_rate": float(row["indicative_rate"]),
            "confidence": float(row["confidence"]),
            "eligible_amount": {
                "min": float(row["eligible_amount_min"]),
                "max": float(row["eligible_amount_max"]),
                "binding_constraint": str(row["eligibility_binding"]),
            },
            "model_version": self.card.version if self.card else None,
        }


def _enforce_band_width(low: np.ndarray, high: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """A band that is too tight is falsely precise; one that is too wide is not a
    quote at all. Both get pulled back to the policy envelope around the midpoint."""
    mid = (low + high) / 2
    half_min = cfg["min_band_width_bps"] / 200.0
    half_max = cfg["max_band_width_bps"] / 200.0
    half = np.clip((high - low) / 2, half_min, half_max)
    return mid - half, mid + half


def _round_to(values: np.ndarray, bps: int) -> np.ndarray:
    step = bps / 100.0
    return np.round(values / step) * step


def _as_frame(application) -> pd.DataFrame:
    if isinstance(application, pd.DataFrame):
        return application
    if isinstance(application, pd.Series):
        return application.to_frame().T
    return pd.DataFrame([application])
