"""Post-assessment feature builder: verified income, valuation, bureau depth,
banking behaviour, deviations and the legal/technical outcome."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import model_config, policy
from .feature_pipeline import FeatureSpec, first_available, numeric_column, spec


def stage2_spec() -> FeatureSpec:
    return spec("stage2")


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "ltv" not in out.columns:
        out["ltv"] = out["sanctioned_loan_amount"] / out["appraised_property_value"]
    out["ltv"] = pd.to_numeric(out["ltv"], errors="coerce").clip(0.05, 0.95)

    if "income_verification_haircut" not in out.columns and {"declared_monthly_income", "verified_monthly_income"} <= set(out.columns):
        declared = pd.to_numeric(out["declared_monthly_income"], errors="coerce")
        verified = pd.to_numeric(out["verified_monthly_income"], errors="coerce")
        out["income_verification_haircut"] = ((declared - verified) / declared.replace(0, np.nan)).clip(0, 1).fillna(0)

    if "foir" not in out.columns:
        out["foir"] = _foir(out)

    for col, default in (("deviation_count", 0), ("legal_flag", 0), ("technical_flag", 0),
                         ("delinquency_count_24m", 0), ("bureau_vintage_months", 0),
                         ("banking_behaviour_score", 50.0), ("sales_target_pct", 100.0),
                         ("branch_target_pct", 100.0)):
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    if "risk_grade" not in out.columns:
        out["risk_grade"] = "C"     # neutral until the risk agent grades the file
    if "new_to_credit" not in out.columns:
        out["new_to_credit"] = out.get("bureau_score", pd.Series(-1, index=out.index)) < 0

    return out


def _foir(df: pd.DataFrame) -> pd.Series:
    pol = policy()
    # Indicative servicing rate: the recommendation if we have one, else the card
    # rate, else a product-typical rate. FOIR has to be computable at enquiry.
    rate = first_available(df, ["ai_recommended_rate", "policy_card_rate"], 11.5)
    tenure = first_available(df, ["tenure_years", "requested_tenure_years"], 20.0)
    principal = first_available(df, ["sanctioned_loan_amount", "requested_loan_amount"], 0.0)
    income = first_available(df, ["verified_monthly_income", "declared_monthly_income"], 0.0).replace(0, np.nan)
    obligations = numeric_column(df, "declared_obligations", 0.0)

    r = rate / 100.0 / 12.0
    n = tenure * 12
    growth = (1 + r) ** n
    emi = principal * r * growth / (growth - 1)
    return ((emi + obligations) / income).clip(0.05, 1.5).fillna(pol["constraints"]["max_foir"])


def lgd(df: pd.DataFrame) -> pd.Series:
    """Policy LGD table, adjusted for LTV.

    No longer the primary source: the fitted LGD model (``models/lgd``) prices
    severity when its artefact exists. This table is its fallback when the
    artefact is missing, and the benchmark its card is judged against."""
    table = model_config()["risk"]["lgd_by_property_type"]
    if "property_type" in df.columns:
        base = df["property_type"].map(table).fillna(np.mean(list(table.values())))
    else:
        base = pd.Series(float(np.mean(list(table.values()))), index=df.index)
    return (base + (numeric_column(df, "ltv", 0.70) - 0.70) * 0.35).clip(0.10, 0.65)
