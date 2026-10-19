"""Learning agent - what the platform does after the decision.

It answers the questions the business statement poses at the end of the journey:
did the override improve conversion, did it improve long-term value, was the
recommendation too aggressive, is competitor information mattering more than it
used to, and does anything need retraining.

It proposes; it does not act. Every adjustment it surfaces is advisory and
routed to the pricing committee, because a loop that silently re-prices
customers on last month's residuals is not governable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..governance import audit, decision_logger, store
from ..learning import drift_detection, outcome_capture, override_learning, retraining
from ..logging_utils import get_logger

log = get_logger(__name__)


class LearningAgent:
    def __init__(self, bundle=None):
        self.bundle = bundle

    # ------------------------------------------------------------- capture
    def capture_override(self, loan_id: str, ai_rate: float, human_rate: float, reason: str,
                         **kwargs: Any) -> dict:
        if not override_learning.validate_reason(reason):
            raise ValueError(f"'{reason}' is not an allowed override reason under the current policy")
        return override_learning.capture(loan_id, ai_rate, human_rate, reason, **kwargs)

    def capture_outcome(self, loan_id: str, **kwargs: Any) -> dict:
        return outcome_capture.capture(loan_id, **kwargs)

    # ------------------------------------------------------------ analysis
    def review_overrides(self, df: pd.DataFrame) -> dict:
        """The override review pack: naive read, adjusted read, and the gap."""
        if self.bundle is None or self.bundle.override is None:
            return {"available": False, "reason": "override model unavailable"}

        propensity = self.bundle.override.predict(df)["override_probability"].to_numpy()
        naive = override_learning.naive_comparison(df)
        table = override_learning.reason_league_table(df, propensity)
        overall = table[table["label"] == "all overrides"].iloc[0].to_dict()

        return {
            "available": True,
            "naive_comparison": naive,
            "adjusted_overall": overall,
            "confounding_gap_inr": round(float(overall["value_lift_inr"] - naive["value_lift_inr"]), 0),
            "reason_league_table": table.to_dict(orient="records"),
            "evidence_effect": override_learning.evidence_effect(df, propensity).to_dict(orient="records"),
            "suggested_adjustments": override_learning.suggested_adjustments(table),
            "method_note": (
                "Overrides are not randomly assigned, so the naive comparison is confounded: "
                "humans discount the files that were hardest to convert. Figures marked "
                "'adjusted' compare overridden and non-overridden files within strata of "
                "predicted override propensity. This removes the first-order selection effect "
                "but is not a randomised test."
            ),
        }

    def review_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
        return {
            "input_drift": drift_detection.feature_drift(reference, current).head(12).to_dict(orient="records"),
            "conversion_drift": drift_detection.conversion_drift(reference, current),
            "retraining": retraining.assess(reference, current).as_dict(),
        }

    def recommendation_quality(self, df: pd.DataFrame) -> dict:
        """Was the engine too aggressive or too soft, versus what humans did?"""
        if "ai_recommended_rate" not in df.columns:
            return {"available": False}
        overridden = df["was_overridden"].astype(bool)
        gap_bps = ((df["sanctioned_rate"] - df["ai_recommended_rate"]) * 100)
        return {
            "available": True,
            "override_rate_pct": round(100 * float(overridden.mean()), 2),
            "mean_human_adjustment_bps": round(float(gap_bps.mean()), 1),
            "share_priced_down_pct": round(100 * float((gap_bps < -1).mean()), 2),
            "share_priced_up_pct": round(100 * float((gap_bps > 1).mean()), 2),
            "interpretation": _aggressiveness(float(gap_bps.mean()), float(overridden.mean())),
        }

    def questions(self, df: pd.DataFrame) -> list[dict]:
        """The standing review questions, answered from the data rather than asked."""
        answers: list[dict] = []
        review = self.review_overrides(df)
        if review.get("available"):
            overall = review["adjusted_overall"]
            answers.append({
                "question": "Did overrides improve conversion?",
                "answer": f"{overall['s2d_lift_pts']:+.1f} points once selection is accounted for "
                          f"(naive read: {review['naive_comparison']['s2d_lift_pts']:+.1f}).",
            })
            answers.append({
                "question": "Did overrides improve long-term value?",
                "answer": f"Rs {overall['value_lift_inr']:,.0f} per file (+/- {overall['value_lift_se_inr']:,.0f}) - "
                          f"verdict: {overall['verdict']}.",
            })
            evidence = review.get("evidence_effect", [])
            if evidence:
                lines = "; ".join(f"{row['label']}: Rs {row['value_lift_inr']:,.0f} ({row['verdict']})"
                                  for row in evidence)
                answers.append({"question": "Did competitor information matter?", "answer": lines})

        quality = self.recommendation_quality(df)
        if quality.get("available"):
            answers.append({"question": "Was the recommendation too aggressive?",
                            "answer": quality["interpretation"]})

        if "city" in df.columns:
            by_city = (df[df["was_overridden"].astype(bool)]
                       .groupby("city")["override_bps"].agg(["size", "mean"])
                       .sort_values("mean"))
            if len(by_city):
                worst = by_city.index[0]
                answers.append({
                    "question": "Where is override pressure concentrated?",
                    "answer": f"{worst} shows the deepest average discount "
                              f"({by_city.iloc[0]['mean']:.1f} bps over {int(by_city.iloc[0]['size'])} files).",
                })
        return answers


def _aggressiveness(mean_gap_bps: float, override_rate: float) -> str:
    if mean_gap_bps < -5 and override_rate > 0.30:
        return ("Humans price below the recommendation on a third of files, which suggests the "
                "recommendation sits above what the market is clearing at.")
    if mean_gap_bps > 5:
        return "Humans price above the recommendation on average; the engine may be leaving margin on the table."
    return "Human decisions sit close to the recommendation on average."
