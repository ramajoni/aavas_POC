"""Pricing-uplift scoring: the per-file curve, and the assessment an RM sees.

``curve`` answers "what would each discount buy?"; ``assess`` answers the
question on the table - "this discount, on this file: how many points of
acceptance, how sure are we, and is it worth it?". The value half is computed
with the same yield calculator the optimiser prices with, and only when the
inputs it needs (expected loss, and expected life from the prepayment model or
the decision on record) are actually available. Otherwise it is omitted with the
reason, rather than estimated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...config import model_config
from ...features.feature_pipeline import display_value, reason_label
from ...logging_utils import get_logger
from ...optimization import yield_calculator as yc
from .. import base
from . import learners as lr
from .train import features

log = get_logger(__name__)
NAME = "uplift"
POINTS = 100.0


class UpliftModel:
    def __init__(self, estimator: lr.UpliftEstimator, surrogate=None,
                 evaluation: dict[str, Any] | None = None):
        self.estimator = estimator
        self.surrogate = surrogate
        self.evaluation = evaluation or {}
        self.card: base.ModelCard | None = None

    @classmethod
    def load(cls) -> "UpliftModel":
        payload, card = base.load(NAME)
        obj = cls(payload["estimator"], payload.get("surrogate"), payload.get("evaluation"))
        obj.card = card
        return obj

    # ------------------------------------------------------------- plumbing
    @property
    def model(self):
        """The explainable face of this model: the surrogate of the uplift at the
        reference discount, in acceptance points. ``(model, transformer)`` is the
        pair generic explainers (``xai.shap_explainer.PricingExplainer``) take."""
        return self.surrogate

    @property
    def transformer(self):
        return self.estimator.transformer

    @property
    def method(self) -> str:
        learner = self.estimator.learner
        from ...automl import portfolio

        return (f"{lr.LABELS[learner.name]} on {portfolio.label(learner.family)}; selection "
                f"adjusted by a propensity model of who gets discounted; curve "
                f"{lr.CURVES[learner.name]}")

    def _prepare(self, application: pd.DataFrame, recommended_rate: float | None):
        frame = application.copy()
        if recommended_rate is not None:
            frame["ai_recommended_rate"] = float(recommended_rate)
        if "ai_recommended_rate" not in frame.columns or frame["ai_recommended_rate"].isna().any():
            raise ValueError("the uplift model needs the recommended rate the discount is measured "
                             "from (ai_recommended_rate or recommended_rate)")
        built = features(frame)
        X, e = self.estimator.design(built)
        return built, X, e

    def label(self, uplift_pts: float) -> str:
        cuts = model_config()[NAME]["labels"]
        if uplift_pts < float(cuts["little_below_pts"]):
            return "discount buys little"
        if uplift_pts < float(cuts["moves_from_pts"]):
            return "discount buys a modest lift"
        return "discount moves acceptance"

    # ---------------------------------------------------------------- curve
    def curve(self, application: pd.DataFrame, discounts_bps=None, *,
              recommended_rate: float | None = None) -> pd.DataFrame:
        """Uplift (acceptance points) at each discount, one row per file x discount."""
        grid = [float(d) for d in (discounts_bps if discounts_bps is not None
                                   else model_config()[NAME]["curve_bps"])]
        built, X, e = self._prepare(application, recommended_rate)
        p0 = self.estimator.baseline(X, e)
        top = self.estimator.support.get("max_bps", np.inf)
        rows = []
        for d in grid:
            u = self.estimator.uplift(X, e, d)
            low, high = self.estimator.band(X, e, d)
            for i, idx in enumerate(application.index):
                rows.append({"row": idx, "discount_bps": d,
                             "proposed_rate": round(float(built["ai_recommended_rate"].iloc[i]) - d / 100, 4),
                             "uplift_pts": round(float(u[i] * POINTS), 3),
                             "uplift_low_pts": round(float(low[i] * POINTS), 3),
                             "uplift_high_pts": round(float(high[i] * POINTS), 3),
                             "acceptance_without": round(float(p0[i]), 4),
                             "acceptance_with": round(float(min(p0[i] + u[i], 1.0)), 4),
                             "beyond_support": bool(d > top)})
        return pd.DataFrame(rows)

    # --------------------------------------------------------------- assess
    def assess(self, application: pd.DataFrame, proposed_discount_bps: float, *,
               recommended_rate: float | None = None, expected_loss_pct: float | None = None,
               prepayment_model=None, expected_life_years: float | None = None) -> dict[str, Any]:
        """The assessment of one proposed discount on one file (the first row)."""
        if proposed_discount_bps < 0:
            raise ValueError("the uplift model covers discounts only: it was learned from discount "
                             "overrides against none, and premiums were excluded")
        application = application.iloc[[0]]
        built, X, e = self._prepare(application, recommended_rate)
        d = float(proposed_discount_bps)
        rec = float(built["ai_recommended_rate"].iloc[0])
        p0 = float(self.estimator.baseline(X, e)[0])
        u = float(self.estimator.uplift(X, e, d)[0])
        low, high = (float(v[0]) for v in self.estimator.band(X, e, d))
        cuts = model_config()[NAME]["labels"]
        support = self.estimator.support
        out = {
            "discount_bps": round(d, 1),
            "recommended_rate": round(rec, 2),
            "proposed_rate": round(rec - d / 100, 2),
            "uplift_pts": round(u * POINTS, 2),
            "interval_pts": [round(low * POINTS, 2), round(high * POINTS, 2)],
            "interval": (f"{self.estimator.interval[0]:.0%}-{self.estimator.interval[1]:.0%} range "
                         f"of the full-data fit and {len(self.estimator.members)} bootstrap refits"),
            "label": self.label(u * POINTS),
            "evidence": ("the whole interval clears the 'buys little' line"
                         if low * POINTS >= float(cuts["little_below_pts"]) else
                         "the interval reaches the 'buys little' line; the lift may be negligible"),
            "acceptance_without_discount": round(p0, 4),
            "acceptance_with_discount": round(min(p0 + u, 1.0), 4),
            "per_10bps_pts": round(u * POINTS / d * 10, 2) if d > 0 else None,
            "support_bps": [round(support.get("min_bps", 0.0), 1), round(support.get("max_bps", 0.0), 1)],
            "beyond_support": bool(d > support.get("max_bps", np.inf)),
            "drivers": self.drivers(built, X),
            "value": self._value(built, rec, d, p0, u, (low, high), expected_loss_pct,
                                 prepayment_model, expected_life_years),
            "method": self.method,
        }
        if out["beyond_support"]:
            out["warning"] = (f"{d:.0f} bps is beyond the largest discounts the model learned from "
                              f"({support.get('max_bps', 0):.0f} bps): the number is extrapolation")
        return out

    # ---------------------------------------------------------------- value
    def _value(self, application: pd.DataFrame, rec: float, d: float, p0: float, u: float,
               band: tuple[float, float], expected_loss_pct, prepayment_model,
               expected_life_years) -> dict[str, Any]:
        """Expected value of the discount, from optimization.yield_calculator.

        Three prices of one file: the recommendation at today's acceptance, the
        discount at today's acceptance (what the discount costs on customers who
        would have taken the loan anyway), and the discount at the lifted
        acceptance. The difference of the first and last is the answer; the
        middle one splits it into cost and gain.
        """
        if expected_loss_pct is None:
            return {"available": False, "reason": "no expected loss for this file (no risk "
                                                  "assessment on record and no risk model)"}
        proposed = rec - d / 100.0
        rates = np.array([rec, proposed])
        bt = life = None
        if prepayment_model is not None:
            try:
                bt = np.asarray(prepayment_model.predict_over_rates(application, rates), dtype=float)
                life = np.asarray(prepayment_model.life_from_probability(bt), dtype=float)
                life_source = "prepayment model at each rate"
            except (KeyError, ValueError) as exc:
                log.warning("prepayment model could not score this file for the uplift value: %s", exc)
        if life is None and expected_life_years is not None:
            bt, life = np.zeros(2), np.full(2, float(expected_life_years))
            life_source = ("the decision's expected life, held fixed at both rates (the prepayment "
                           "model was not available for this file)")
        if life is None:
            return {"available": False, "reason": "no expected life: neither the prepayment model "
                                                  "nor a recorded decision is available"}
        row = application.iloc[0]
        amount = float(row.get("sanctioned_loan_amount") or row.get("requested_loan_amount") or 0.0)
        tenure = float(row.get("tenure_years") or row.get("requested_tenure_years") or 20.0)
        if amount <= 0:
            return {"available": False, "reason": "no loan amount on the file"}

        def ev(rate_i: int, p: float) -> yc.YieldResult:
            return yc.compute(rate=float(rates[rate_i]), conversion_probability=min(p, 1.0),
                              expected_loss_pct=float(expected_loss_pct),
                              expected_life_years=float(life[rate_i]), loan_amount=amount,
                              tenure_years=tenure, bt_out_probability=float(bt[rate_i]))

        base_ev, cost_only, with_lift = ev(0, p0), ev(1, p0), ev(1, p0 + u)
        effect = with_lift.expected_value_inr - base_ev.expected_value_inr
        # Sorted: when the loan loses money at the proposed rate, more acceptance
        # is more loss, and the high-uplift edge is the low-value one.
        low_ev, high_ev = sorted(ev(1, p0 + b).expected_value_inr - base_ev.expected_value_inr
                                 for b in band)
        per_point = cost_only.lifetime_value_pct / 100.0 * amount / 100.0
        breakeven = ((base_ev.expected_value_inr - cost_only.expected_value_inr) / per_point
                     if per_point > 0 else None)
        verdict = ("the discount pays for itself" if low_ev > 0 else
                   "the discount costs more than it buys" if high_ev < 0 else
                   "uncertain: the interval spans break-even")
        return {
            "available": True,
            "value_effect_inr": round(effect, 0),
            "value_effect_interval_inr": [round(low_ev, 0), round(high_ev, 0)],
            "margin_given_up_inr": round(cost_only.expected_value_inr - base_ev.expected_value_inr, 0),
            "acceptance_bought_inr": round(with_lift.expected_value_inr - cost_only.expected_value_inr, 0),
            "breakeven_uplift_pts": round(breakeven, 2) if breakeven is not None else None,
            **({} if breakeven is not None else {"breakeven_note": (
                "no uplift can pay for this discount: the loan's lifetime value is negative at "
                "the proposed rate, so each extra acceptance loses money")}),
            "verdict": verdict,
            "expected_value_at_recommended_inr": round(base_ev.expected_value_inr, 0),
            "expected_value_at_proposed_inr": round(with_lift.expected_value_inr, 0),
            "expected_loss_pct": round(float(expected_loss_pct), 4),
            "expected_life_years": [round(float(v), 2) for v in life],
            "life_source": life_source,
            "method": "optimization.yield_calculator.compute at both rates; acceptance from this "
                      "model (no-discount level plus the estimated uplift)",
        }

    # -------------------------------------------------------------- drivers
    def drivers(self, built: pd.DataFrame, X: pd.DataFrame, top: int = 3) -> list[dict[str, Any]]:
        """Why a discount buys more or less on this file than on the average one:
        TreeSHAP of the surrogate of the uplift at the reference discount. Empty
        when shap or the surrogate is unavailable - never a made-up driver."""
        if self.surrogate is None:
            return []
        try:
            import shap

            values = np.asarray(shap.TreeExplainer(self.surrogate).shap_values(X.iloc[[0]]))[0]
        except Exception as exc:                                    # noqa: BLE001 - optional
            log.info("uplift drivers unavailable: %s", exc)
            return []
        out = []
        for i in np.argsort(-np.abs(values))[:top]:
            feature = X.columns[i]
            raw = built[feature].iloc[0] if feature in built.columns else None
            if isinstance(raw, (np.generic,)):
                raw = raw.item()
            out.append({"feature": feature, "label": reason_label(feature)["label"],
                        "value": display_value(feature, raw),
                        "contribution_pts": round(float(values[i]), 2),
                        "effect": "discount buys more here" if values[i] > 0
                        else "discount buys less here"})
        return out
