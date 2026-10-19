"""Override learning - did the human decision create or destroy value?

This is the hardest question in the brief and the easiest one to get wrong.

Overrides are not randomly assigned. Salespeople discount precisely the files
that were going to be hard to close, so a naive comparison of overridden versus
non-overridden files makes every discount look like it *reduced* conversion. On
this repo's own synthetic data the naive read says discounted files convert 3.5
points *worse* than untouched ones, which is the opposite of the truth.

The fix used here is propensity stratification: score every file with the
override model, bucket by how likely an override was, and compare overridden to
non-overridden files only *within* a bucket. It is not a randomised experiment -
the honest answer to "was the salesperson right?" needs one - but it removes the
first-order confound and makes the reason-level league table usable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import policy
from ..governance import audit, store
from ..logging_utils import get_logger

log = get_logger(__name__)

MIN_STRATUM_SIZE = 25
MIN_REASON_SAMPLE = 40


# ------------------------------------------------------------------- capture


def capture(loan_id: str, ai_rate: float, human_rate: float, reason: str,
            decision_id: str | None = None, reason_detail: str | None = None,
            approved_by: str | None = None, required_approval: str | None = None,
            within_authority: bool | None = None, evidence_verified: bool = False,
            sales_context: dict | None = None, actor: str = "sales") -> dict:
    """Record a human decision against the AI recommendation. Never overwrite the
    recommendation - both numbers have to survive for the loop to learn."""
    override_bps = round((human_rate - ai_rate) * 100, 1)
    record = {
        "override_id": f"OVR-{loan_id}-{abs(int(override_bps))}",
        "decision_id": decision_id,
        "loan_id": loan_id,
        "created_at": store.now(),
        "ai_rate": ai_rate,
        "human_rate": human_rate,
        "override_bps": override_bps,
        "reason": reason,
        "reason_detail": reason_detail,
        "evidence_verified": bool(evidence_verified),
        "approved_by": approved_by,
        "required_approval": required_approval,
        "within_authority": None if within_authority is None else bool(within_authority),
        "sales_context": sales_context or {},
        "payload": {"direction": "discount" if override_bps < 0 else "premium" if override_bps > 0 else "none"},
    }
    store.insert("overrides", record)
    audit.record(actor, "capture_override", "loan", loan_id,
                 {"ai_rate": ai_rate, "human_rate": human_rate, "override_bps": override_bps,
                  "reason": reason, "approved_by": approved_by})
    return record


def validate_reason(reason: str) -> bool:
    return reason in policy()["override"]["allowed_reasons"]


# ------------------------------------------------------------------ analysis


@dataclass
class OverrideEffect:
    label: str
    n_override: int
    n_control: int
    s2d_lift_pts: float
    value_lift_inr: float
    value_lift_se_inr: float
    mean_override_bps: float
    verdict: str

    def as_dict(self) -> dict:
        return {
            "label": self.label, "n_override": self.n_override, "n_control": self.n_control,
            "s2d_lift_pts": round(self.s2d_lift_pts, 2),
            "value_lift_inr": round(self.value_lift_inr, 0),
            "value_lift_se_inr": round(self.value_lift_se_inr, 0),
            "mean_override_bps": round(self.mean_override_bps, 1),
            "verdict": self.verdict,
        }


def _value_column(df: pd.DataFrame) -> pd.Series:
    """Value of the decision in rupees, counting a lost customer as zero."""
    if "realized_value_inr" in df.columns:
        return pd.to_numeric(df["realized_value_inr"], errors="coerce").fillna(0.0)
    from ..features.feature_pipeline import numeric_column

    value_pct = numeric_column(df, "realized_value_pct", 0.0)
    amount = numeric_column(df, "sanctioned_loan_amount", 0.0)
    return value_pct / 100.0 * amount


def naive_comparison(df: pd.DataFrame) -> dict:
    """The comparison a dashboard would show by default - kept so the difference
    from the adjusted read is visible rather than asserted."""
    value = _value_column(df)
    overridden = df["was_overridden"].astype(bool)
    return {
        "s2d_lift_pts": round(100 * (df.loc[overridden, "disbursed"].mean()
                                     - df.loc[~overridden, "disbursed"].mean()), 2),
        "value_lift_inr": round(value[overridden].mean() - value[~overridden].mean(), 0),
        "n_override": int(overridden.sum()),
        "n_control": int((~overridden).sum()),
    }


def propensity_adjusted(df: pd.DataFrame, propensity: np.ndarray, group_mask: np.ndarray | None = None,
                        n_strata: int = 10, label: str = "all overrides") -> OverrideEffect:
    """Compare overridden and non-overridden files inside propensity strata."""
    work = df.copy()
    work["_value"] = _value_column(work)
    work["_override"] = work["was_overridden"].astype(bool) if group_mask is None else group_mask
    work["_propensity"] = propensity

    # Controls are always the non-overridden files, whatever subgroup is studied.
    work["_control"] = ~work["was_overridden"].astype(bool)
    work = work[work["_override"] | work["_control"]]

    try:
        work["_stratum"] = pd.qcut(work["_propensity"], n_strata, labels=False, duplicates="drop")
    except ValueError:
        work["_stratum"] = 0

    weights, s2d_diffs, value_diffs, variances = [], [], [], []
    n_override = n_control = 0
    for _, stratum in work.groupby("_stratum"):
        treated = stratum[stratum["_override"]]
        control = stratum[stratum["_control"]]
        if len(treated) < 5 or len(control) < 5:
            continue
        weight = len(treated)
        weights.append(weight)
        s2d_diffs.append(treated["disbursed"].astype(float).mean() - control["disbursed"].astype(float).mean())
        value_diffs.append(treated["_value"].mean() - control["_value"].mean())
        variances.append(treated["_value"].var(ddof=1) / max(len(treated), 1)
                         + control["_value"].var(ddof=1) / max(len(control), 1))
        n_override += len(treated)
        n_control += len(control)

    if not weights or n_override < MIN_STRATUM_SIZE:
        return OverrideEffect(label, n_override, n_control, float("nan"), float("nan"),
                              float("nan"), float("nan"), "insufficient evidence")

    weights_arr = np.asarray(weights, dtype=float)
    share = weights_arr / weights_arr.sum()
    s2d_lift = float(np.dot(share, s2d_diffs) * 100)
    value_lift = float(np.dot(share, value_diffs))
    standard_error = float(np.sqrt(np.dot(share ** 2, variances)))
    mean_bps = float(work.loc[work["_override"], "override_bps"].mean())

    return OverrideEffect(label, n_override, n_control, s2d_lift, value_lift, standard_error,
                          mean_bps, _verdict(value_lift, standard_error))


def _verdict(value_lift: float, standard_error: float) -> str:
    if not np.isfinite(standard_error) or standard_error == 0:
        return "insufficient evidence"
    z = value_lift / standard_error
    if z > 1.96:
        return "value creating"
    if z < -1.96:
        return "value destroying"
    return "no measurable effect"


def reason_league_table(df: pd.DataFrame, propensity: np.ndarray) -> pd.DataFrame:
    """Which override reasons actually pay for themselves."""
    rows = [propensity_adjusted(df, propensity, label="all overrides").as_dict()]
    overridden = df["was_overridden"].astype(bool)
    for reason, group in df[overridden].groupby("override_reason"):
        if len(group) < MIN_REASON_SAMPLE:
            continue
        mask = (df["override_reason"] == reason).to_numpy() & overridden.to_numpy()
        effect = propensity_adjusted(df, propensity, group_mask=mask, label=str(reason))
        rows.append(effect.as_dict())

    table = pd.DataFrame(rows)
    return table.sort_values("value_lift_inr", ascending=False, na_position="last").reset_index(drop=True)


def evidence_effect(df: pd.DataFrame, propensity: np.ndarray) -> pd.DataFrame:
    """Does a competitor claim backed by evidence behave differently from one
    that is not? This is the check that keeps unverified market claims honest."""
    overridden = df["was_overridden"].astype(bool)
    competitor = df["override_reason"].isin(["competitor_lower_rate", "customer_threatening_bt"])
    rows = []
    for verified in (True, False):
        mask = (overridden & competitor & (df["competitor_evidence_verified"].astype(bool) == verified)).to_numpy()
        if mask.sum() < MIN_REASON_SAMPLE:
            continue
        label = f"competitor claim {'evidenced' if verified else 'unevidenced'}"
        rows.append(propensity_adjusted(df, propensity, group_mask=mask, label=label).as_dict())
    return pd.DataFrame(rows)


def suggested_adjustments(table: pd.DataFrame, max_bps: float = 15.0) -> list[dict]:
    """Translate the league table into bounded, reviewable policy suggestions.

    Deliberately conservative and advisory: the output is a proposal for the
    pricing committee, not an automatic change to anyone's rate.
    """
    suggestions = []
    for row in table.to_dict(orient="records"):
        if row["label"] == "all overrides" or row["verdict"] == "insufficient evidence":
            continue
        if row["verdict"] == "value creating" and row["mean_override_bps"] < 0:
            suggestions.append({
                "reason": row["label"],
                "proposal": "allow this discount earlier in the journey",
                "suggested_bps": round(max(row["mean_override_bps"], -max_bps), 1),
                "evidence": f"{row['n_override']} cases, value lift Rs {row['value_lift_inr']:,.0f} "
                            f"(+/- {row['value_lift_se_inr']:,.0f})",
                "requires": "pricing committee approval",
            })
        elif row["verdict"] == "value destroying":
            suggestions.append({
                "reason": row["label"],
                "proposal": "tighten authority or require evidence for this reason",
                "suggested_bps": 0.0,
                "evidence": f"{row['n_override']} cases, value change Rs {row['value_lift_inr']:,.0f} "
                            f"(+/- {row['value_lift_se_inr']:,.0f})",
                "requires": "pricing committee approval",
            })
    return suggestions
