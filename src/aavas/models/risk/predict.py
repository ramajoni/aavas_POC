"""Risk scoring: PD, grade and annualised expected loss in percentage points.

Unit discipline matters here. The classifier is fitted on a lifetime-to-date
default flag, so ``predict_proba`` is a *lifetime* PD. Pricing subtracts an
*annual* expected loss from an annual spread, and the grade cut-offs are annual
too. Everything below converts once, explicitly, rather than letting the two
units meet somewhere in the optimizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import model_config
from ...features import stage2_features
from .. import base

NAME = "risk"


class RiskModel:
    def __init__(self, model, transformer, grade_cutoffs: dict[str, float],
                 pd_horizon_years: float = 4.0):
        self.model = model
        self.transformer = transformer
        self.grade_cutoffs = grade_cutoffs
        self.pd_horizon_years = pd_horizon_years
        self.card: base.ModelCard | None = None

    @classmethod
    def load(cls) -> "RiskModel":
        payload, card = base.load(NAME)
        obj = cls(payload["model"], payload["transformer"], payload["grade_cutoffs"],
                  payload.get("pd_horizon_years", model_config()["risk"]["pd_horizon_years"]))
        obj.card = card
        return obj

    def lifetime_pd(self, df: pd.DataFrame) -> np.ndarray:
        X = self.transformer.transform(stage2_features.build(df))
        return self.model.predict_proba(X)[:, 1]

    def pd(self, df: pd.DataFrame) -> np.ndarray:
        """Annualised PD - the unit the grade cut-offs and expected loss use."""
        return np.clip(self.lifetime_pd(df) / self.pd_horizon_years, 1e-6, 1.0)

    def grade(self, pd_values: np.ndarray) -> np.ndarray:
        order = sorted(self.grade_cutoffs.items(), key=lambda kv: kv[1])
        grades = np.full(len(pd_values), order[-1][0], dtype=object)
        for name, cutoff in reversed(order):
            grades = np.where(pd_values <= cutoff, name, grades)
        return grades

    def assess(self, df: pd.DataFrame) -> pd.DataFrame:
        lifetime = self.lifetime_pd(df)
        pd_values = np.clip(lifetime / self.pd_horizon_years, 1e-6, 1.0)
        lgd = stage2_features.lgd(stage2_features.build(df)).to_numpy()
        return pd.DataFrame({
            "pd": np.round(pd_values, 5),
            "lifetime_pd": np.round(lifetime, 5),
            "lgd": np.round(lgd, 4),
            "risk_grade": self.grade(pd_values),
            # Annualised expected loss as percentage points of outstanding, which
            # is the unit the yield calculator subtracts from the spread.
            "expected_loss_pct": np.round(pd_values * lgd * 100, 4),
        }, index=df.index)
