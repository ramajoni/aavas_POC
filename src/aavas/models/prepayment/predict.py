"""Prepayment scoring and the expected-life translation the economics need."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import policy
from .. import base

NAME = "prepayment"


class PrepaymentModel:
    def __init__(self, model, transformer, horizon_months: int):
        self.model = model
        self.transformer = transformer
        self.horizon_months = horizon_months
        self.card: base.ModelCard | None = None

    @classmethod
    def load(cls) -> "PrepaymentModel":
        payload, card = base.load(NAME)
        obj = cls(payload["model"], payload["transformer"], payload["horizon_months"])
        obj.card = card
        return obj

    def predict_at_rate(self, df: pd.DataFrame, rate: float | np.ndarray) -> np.ndarray:
        scored = df.copy()
        scored["offered_rate"] = rate
        if "competitor_rate" in scored.columns:
            comp = pd.to_numeric(scored["competitor_rate"], errors="coerce")
            scored["competitor_gap_bps"] = ((scored["offered_rate"] - comp) * 100).fillna(0.0)
        return self.model.predict_proba(self.transformer.transform(scored))[:, 1]

    def predict_over_rates(self, application: pd.DataFrame, rates: np.ndarray) -> np.ndarray:
        """BT-out probability at every candidate rate, in one scoring pass."""
        expanded = base.expand_over_rates(application, rates)
        return self.model.predict_proba(self.transformer.transform(expanded))[:, 1]

    def life_from_probability(self, p_bt: np.ndarray) -> np.ndarray:
        """Behavioural loan life implied by a BT-out probability."""
        pol = policy()["economics"]
        p_bt = np.clip(np.asarray(p_bt, dtype=float), 1e-4, 0.95)
        annual_hazard = 1 - (1 - p_bt) ** (12.0 / self.horizon_months)
        return np.clip(1.0 / np.maximum(annual_hazard, 1e-3), 0.5, pol["behavioural_horizon_years"])

    def expected_life_years(self, df: pd.DataFrame, rate: float) -> np.ndarray:
        """Convert a 24-month BT probability into a behavioural loan life.

        The horizon probability is treated as a constant annual hazard and
        projected forward, capped at the policy behavioural horizon. Crude, but
        transparent and monotone - a survival model is the phase-2 upgrade.
        """
        pol = policy()["economics"]
        p_bt = np.clip(self.predict_at_rate(df, rate), 1e-4, 0.95)
        annual_hazard = 1 - (1 - p_bt) ** (12.0 / self.horizon_months)
        expected_life = 1.0 / np.maximum(annual_hazard, 1e-3)
        return np.clip(expected_life, 0.5, pol["behavioural_horizon_years"])
