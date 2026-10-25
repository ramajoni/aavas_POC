"""Risk scoring: PD, LGD, grade and annualised expected loss in percentage points.

Unit discipline matters here. The classifier is fitted on a lifetime-to-date
default flag, so ``predict_proba`` is a *lifetime* PD. Pricing subtracts an
*annual* expected loss from an annual spread, and the grade cut-offs are annual
too. Everything below converts once, explicitly, rather than letting the two
units meet somewhere in the optimizer.

LGD comes from the fitted LGD model (``models/lgd``) when its artefact exists,
and from the policy table (``stage2_features.lgd``) when it does not - or when
the frame lacks a column the model needs. Either way it is floored at
``risk.lgd_floor``, and every row says which source priced it (``lgd_source``),
because "expected loss 0.42%" means something different when half of it is a
table lookup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import model_config
from ...features import stage2_features
from ...logging_utils import get_logger
from .. import base

log = get_logger(__name__)
NAME = "risk"

LGD_MODEL = "model"
LGD_TABLE = "policy_table"


class RiskModel:
    def __init__(self, model, transformer, grade_cutoffs: dict[str, float],
                 pd_horizon_years: float = 4.0, lgd_model=None,
                 lgd_floor: float | None = None):
        self.model = model
        self.transformer = transformer
        self.grade_cutoffs = grade_cutoffs
        self.pd_horizon_years = pd_horizon_years
        self.lgd_model = lgd_model
        self.lgd_floor = (float(model_config()[NAME].get("lgd_floor", 0.0))
                          if lgd_floor is None else float(lgd_floor))
        self.card: base.ModelCard | None = None

    @classmethod
    def load(cls) -> "RiskModel":
        payload, card = base.load(NAME)
        obj = cls(payload["model"], payload["transformer"], payload["grade_cutoffs"],
                  payload.get("pd_horizon_years", model_config()["risk"]["pd_horizon_years"]),
                  lgd_model=load_lgd())
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

    def lgd(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(LGD after the floor, its source per row, the policy table's value).

        The table is always computed: it is the fallback, and showing it beside
        the model's number is how a reviewer sees what the model changed.
        """
        built = stage2_features.build(df)
        table = stage2_features.lgd(built).to_numpy(float)
        values, source = table, LGD_TABLE
        if self.lgd_model is not None:
            try:
                values, source = np.asarray(self.lgd_model.predict(built), dtype=float), LGD_MODEL
            except KeyError as exc:
                # A frame without the collateral fields the model needs is priced
                # on the table, and says so, rather than failing the decision.
                log.warning("LGD model could not score this frame (%s); using the policy table", exc)
        return (np.maximum(values, self.lgd_floor), np.full(len(values), source, dtype=object),
                table)

    def assess(self, df: pd.DataFrame) -> pd.DataFrame:
        lifetime = self.lifetime_pd(df)
        pd_values = np.clip(lifetime / self.pd_horizon_years, 1e-6, 1.0)
        lgd, source, table = self.lgd(df)
        return pd.DataFrame({
            "pd": np.round(pd_values, 5),
            "lifetime_pd": np.round(lifetime, 5),
            "lgd": np.round(lgd, 4),
            "lgd_source": source,
            "lgd_policy_table": np.round(np.maximum(table, self.lgd_floor), 4),
            "risk_grade": self.grade(pd_values),
            # Annualised expected loss as percentage points of outstanding, which
            # is the unit the yield calculator subtracts from the spread.
            "expected_loss_pct": np.round(pd_values * lgd * 100, 4),
        }, index=df.index)


def load_lgd():
    """The LGD artefact when it exists; None (-> the policy table) otherwise.

    A broken LGD artefact must not take PD - and so every decision - down with
    it, so anything other than "not trained" is logged and degrades the same way.
    """
    if not base.exists("lgd"):
        return None
    from ..lgd.predict import LGDModel

    try:
        return LGDModel.load()
    except Exception as exc:                                        # noqa: BLE001 - degrade, say so
        log.warning("LGD artefact present but unloadable (%s); risk will use the policy table", exc)
        return None
