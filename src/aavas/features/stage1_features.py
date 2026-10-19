"""Pre-assessment feature builder.

Stage 1 runs when income is self-declared, the property is unvisited and there
may be no full bureau pull. Anything the file learns *after* that point is
forbidden here - ``assert_no_leakage`` enforces it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingestion.validator import assert_no_leakage
from .feature_pipeline import FeatureSpec, numeric_column, spec


def stage1_spec() -> FeatureSpec:
    s = spec("stage1")
    assert_no_leakage(s.columns, "stage1")
    return s


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the pre-assessment view. Tolerant by design: a branch enquiry may
    arrive with property value or obligations missing."""
    out = df.copy()

    if "estimated_property_value" not in out or out["estimated_property_value"].isna().all():
        # No property identified yet: assume a policy-typical LTV to keep the
        # eligibility maths meaningful rather than refusing to quote.
        out["estimated_property_value"] = out["requested_loan_amount"] / 0.70
    if "estimated_ltv" not in out.columns:
        out["estimated_ltv"] = out["requested_loan_amount"] / out["estimated_property_value"]
    out["estimated_ltv"] = pd.to_numeric(out["estimated_ltv"], errors="coerce").clip(0.1, 0.95).fillna(0.70)

    if "declared_obligations" not in out.columns:
        out["declared_obligations"] = 0.0
    out["declared_obligations"] = out["declared_obligations"].fillna(0.0)

    if "new_to_credit" not in out.columns:
        out["new_to_credit"] = out.get("bureau_score", pd.Series(-1, index=out.index)) < 0
    if "bureau_score" not in out.columns:
        out["bureau_score"] = -1
    out["bureau_score"] = out["bureau_score"].fillna(-1)

    if "sales_target_pct" not in out.columns:
        out["sales_target_pct"] = 100.0
    if "branch_target_pct" not in out.columns:
        out["branch_target_pct"] = 100.0
    if "competitor_rate" not in out.columns or out["competitor_rate"].isna().all():
        # No competitor quote on the table yet - neutral placeholder, and the
        # confidence score below reflects the thinner information.
        out["competitor_rate"] = np.nan

    return out


def eligibility(df: pd.DataFrame, rate: pd.Series | np.ndarray, policy_cfg: dict) -> pd.Series:
    """Indicative loan eligibility at a given rate, using the conservative
    pre-assessment FOIR cap rather than the underwriting cap."""
    foir_cap = policy_cfg["stage1"]["eligibility_foir_cap"]
    income = numeric_column(df, "declared_monthly_income", 0.0)
    obligations = numeric_column(df, "declared_obligations", 0.0)
    tenure = numeric_column(df, "requested_tenure_years", 20.0)

    max_emi = np.maximum(foir_cap * income - obligations, 0.0)
    r = np.asarray(rate, dtype=float) / 100.0 / 12.0
    n = tenure.to_numpy() * 12
    growth = (1 + r) ** n
    principal = np.where(r > 0, max_emi * (growth - 1) / (r * growth), max_emi * n)

    ltv_cap = numeric_column(df, "estimated_property_value", 0.0) * policy_cfg["constraints"]["max_ltv"]
    eligible = np.minimum(principal, ltv_cap)
    eligible = np.minimum(eligible, policy_cfg["constraints"]["max_loan_amount"])
    return pd.Series(np.round(eligible, -3), index=df.index)


def eligibility_binding_constraint(df: pd.DataFrame, rate: pd.Series | np.ndarray,
                                   policy_cfg: dict) -> pd.Series:
    """Which limit is actually capping the eligible amount.

    Worth surfacing: when the property value caps eligibility, quoting a better
    rate does not raise the amount, and the branch should be told that rather
    than left to wonder why the range collapsed to a point.
    """
    foir_cap = policy_cfg["stage1"]["eligibility_foir_cap"]
    income = numeric_column(df, "declared_monthly_income", 0.0)
    obligations = numeric_column(df, "declared_obligations", 0.0)
    tenure = numeric_column(df, "requested_tenure_years", 20.0)

    max_emi = np.maximum(foir_cap * income - obligations, 0.0)
    r = np.asarray(rate, dtype=float) / 100.0 / 12.0
    n = tenure.to_numpy() * 12
    growth = (1 + r) ** n
    principal = np.where(r > 0, max_emi * (growth - 1) / (r * growth), max_emi * n)
    ltv_cap = numeric_column(df, "estimated_property_value", 0.0) * policy_cfg["constraints"]["max_ltv"]
    product_cap = policy_cfg["constraints"]["max_loan_amount"]

    binding = np.where(principal <= np.minimum(ltv_cap, product_cap), "affordability (FOIR)",
                       np.where(ltv_cap <= product_cap, "property value (LTV cap)", "product maximum"))
    return pd.Series(binding, index=df.index)


def confidence(df: pd.DataFrame, band_width_bps: pd.Series | np.ndarray) -> pd.Series:
    """How much the Stage-1 band should be trusted.

    Two drivers: how wide the predicted interval is, and how much of the
    pre-assessment information is actually present on the file.
    """
    width = pd.Series(np.asarray(band_width_bps, dtype=float), index=df.index)
    width_score = np.clip(1.0 - (width - 25.0) / 125.0, 0.25, 1.0)

    known = pd.Series(0.0, index=df.index)
    known += np.where(df.get("bureau_score", pd.Series(-1, index=df.index)) > 0, 0.30, 0.0)
    known += np.where(df.get("competitor_rate", pd.Series(np.nan, index=df.index)).notna(), 0.15, 0.0)
    known += np.where(df.get("estimated_property_value", pd.Series(np.nan, index=df.index)).notna(), 0.25, 0.0)
    known += np.where(df.get("declared_obligations", pd.Series(np.nan, index=df.index)).notna(), 0.10, 0.0)
    known += np.where(~df.get("new_to_credit", pd.Series(False, index=df.index)).astype(bool), 0.20, 0.0)

    return (0.55 * width_score + 0.45 * known).clip(0.15, 0.95).round(3)
