"""Commercial-context features.

Deliberately isolated from customer-risk features. These move price for reasons
that have nothing to do with the customer's credit quality - competitor bids,
target pressure, negotiation - and governance needs to be able to answer
"how much of this customer's rate came from commercial pressure rather than
their own risk profile?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_pipeline import FeatureSpec, boolean_column, numeric_column, spec


def sales_spec() -> FeatureSpec:
    return spec("sales_context")


def build(df: pd.DataFrame, reference_rate_col: str = "ai_recommended_rate") -> pd.DataFrame:
    out = df.copy()

    if "competitor_rate" not in out.columns:
        out["competitor_rate"] = np.nan
    ref = numeric_column(out, reference_rate_col, np.nan)
    if ref.isna().all():
        ref = numeric_column(out, "policy_card_rate", np.nan)
    comp = pd.to_numeric(out["competitor_rate"], errors="coerce")
    # No competitor quote on the table means no gap, not a favourable gap.
    out["competitor_gap_bps"] = ((ref - comp) * 100).round(1).fillna(0.0)

    if "days_to_month_end" not in out.columns:
        if "application_date" in out.columns:
            dates = pd.to_datetime(out["application_date"], errors="coerce")
            out["days_to_month_end"] = (dates.dt.days_in_month - dates.dt.day).fillna(15).astype(int)
        else:
            out["days_to_month_end"] = 15
    if "month_end" not in out.columns:
        out["month_end"] = out["days_to_month_end"] <= 5

    for col, default in (("sales_target_pct", 100.0), ("branch_target_pct", 100.0),
                         ("prior_negotiation_rounds", 0)):
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    for col in ("customer_negotiating", "existing_relationship", "competitor_evidence_verified"):
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)

    return out


def pressure_index(df: pd.DataFrame) -> pd.Series:
    """A single readable 0-1 summary of commercial pressure on the file.

    Used in the decision record so a reviewer can see at a glance whether a
    recommendation was made under heavy target pressure.
    """
    gap = numeric_column(df, "competitor_gap_bps", 0.0).clip(-100, 150) / 150.0
    target = (100 - numeric_column(df, "sales_target_pct", 100.0)).clip(0, 60) / 60.0
    branch = (100 - numeric_column(df, "branch_target_pct", 100.0)).clip(0, 60) / 60.0
    month_end = boolean_column(df, "month_end").astype(float)
    negotiating = boolean_column(df, "customer_negotiating").astype(float)

    idx = 0.32 * gap.clip(0, 1) + 0.24 * target + 0.14 * branch + 0.14 * month_end + 0.16 * negotiating
    return idx.clip(0, 1).round(3)
