"""Data validation.

Two layers, deliberately separate:

* schema  - are the columns present and the right type
* policy  - does the row make business sense (LTV within policy, rate inside the
            permissible envelope, lifecycle ordering intact)

A policy failure is not a crash: bad rows are quarantined and reported, because a
pricing platform that refuses to score anything the moment one field is odd is
useless to a branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import policy
from ..logging_utils import get_logger
from . import schemas

log = get_logger(__name__)


@dataclass
class ValidationReport:
    n_rows: int = 0
    n_valid: int = 0
    missing_columns: list[str] = field(default_factory=list)
    issues: dict[str, int] = field(default_factory=dict)
    quarantined_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_columns and self.n_valid == self.n_rows

    def summary(self) -> str:
        parts = [f"{self.n_valid}/{self.n_rows} rows valid"]
        if self.missing_columns:
            parts.append(f"missing columns: {self.missing_columns}")
        if self.issues:
            parts.append("issues: " + ", ".join(f"{k}={v}" for k, v in sorted(self.issues.items())))
        return " | ".join(parts)

    def as_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "n_valid": self.n_valid, "ok": self.ok,
            "missing_columns": self.missing_columns, "issues": self.issues,
            "quarantined": len(self.quarantined_ids),
        }


def validate(df: pd.DataFrame, required: list[str] | None = None, strict: bool = False) -> ValidationReport:
    pol = policy()
    rep = ValidationReport(n_rows=len(df))
    required = required or schemas.IDENTITY_COLUMNS + schemas.PRE_ASSESSMENT_COLUMNS
    rep.missing_columns = [c for c in required if c not in df.columns]
    if rep.missing_columns and strict:
        raise ValueError(f"missing required columns: {rep.missing_columns}")

    bad = pd.Series(False, index=df.index)

    def flag(name: str, mask: pd.Series) -> None:
        nonlocal bad
        mask = mask.fillna(False)
        count = int(mask.sum())
        if count:
            rep.issues[name] = count
            bad = bad | mask

    con = pol["constraints"]
    ref = pol["reference"]
    if "ltv" in df:
        flag("ltv_above_policy", df["ltv"] > con["max_ltv"] + 1e-6)
        flag("ltv_non_positive", df["ltv"] <= 0)
    if "foir" in df:
        flag("foir_above_policy", df["foir"] > con["max_foir"] + 1e-6)
    if "tenure_years" in df:
        flag("tenure_above_policy", df["tenure_years"] > con["max_tenure_years"])
    if "sanctioned_loan_amount" in df:
        flag("amount_above_policy", df["sanctioned_loan_amount"] > con["max_loan_amount"])
    for col in ("policy_card_rate", "ai_recommended_rate", "sales_offered_rate", "sanctioned_rate"):
        if col in df:
            flag(f"{col}_outside_envelope",
                 (df[col] < ref["card_rate_floor"] - 1e-6) | (df[col] > ref["card_rate_ceiling"] + 1e-6))
    if {"stage1_min_rate", "stage1_max_rate"} <= set(df.columns):
        flag("stage1_band_inverted", df["stage1_min_rate"] > df["stage1_max_rate"])
    if {"disbursed", "realized_yield_pct"} <= set(df.columns):
        flag("yield_without_disbursal", (~df["disbursed"].astype(bool)) & df["realized_yield_pct"].notna())

    rep.n_valid = int((~bad).sum())
    if "loan_id" in df:
        rep.quarantined_ids = df.loc[bad, "loan_id"].astype(str).tolist()
    if not rep.ok:
        log.warning("validation: %s", rep.summary())
    return rep


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes that survive a CSV round-trip badly (bools, dates)."""
    out = df.copy()
    for col in schemas.BOOL_COLUMNS:
        if col in out.columns:
            out[col] = (
                out[col].map({True: True, False: False, "True": True, "False": False,
                              "true": True, "false": False, 1: True, 0: False})
                .fillna(False).astype(bool)
            )
    if "application_date" in out.columns:
        out["application_date"] = pd.to_datetime(out["application_date"], errors="coerce")
    for col in out.columns:
        if out[col].dtype == object and col not in schemas.BOOL_COLUMNS:
            continue
    return out


def quarantine(df: pd.DataFrame, report: ValidationReport) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into (clean, quarantined) using a report's ids."""
    if not report.quarantined_ids or "loan_id" not in df.columns:
        return df, df.iloc[0:0]
    mask = df["loan_id"].astype(str).isin(set(report.quarantined_ids))
    return df.loc[~mask].copy(), df.loc[mask].copy()


def assert_no_leakage(feature_columns: list[str], stage: str) -> None:
    """Guard against Stage-2-only knowledge reaching a Stage-1 model."""
    if stage != "stage1":
        return
    forbidden = set(schemas.POST_ASSESSMENT_COLUMNS) | set(schemas.OUTCOME_COLUMNS) | set(schemas.LATENT_COLUMNS)
    leaked = sorted(set(feature_columns) & forbidden)
    if leaked:
        raise ValueError(f"stage1 features leak post-assessment information: {leaked}")
