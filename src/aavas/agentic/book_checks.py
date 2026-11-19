"""Checks a dataset must pass before anything trains on it: every loan obeys the policy.

A model learns whatever its history shows. A booked rate above the matrix, or a loan
that fails the stressed affordability test, would teach it to break the policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import policy
from ..ingestion import validator
from ..optimization import constraints

RATE_COLUMNS = ["policy_card_rate", "ai_recommended_rate", "sales_offered_rate", "sanctioned_rate"]


def run(df: pd.DataFrame, wanted: list[str] | None = None) -> list[dict[str, Any]]:
    pol = policy()
    ref, con = pol["reference"], pol["constraints"]
    checks: dict[str, tuple[int, str]] = {}

    present = [c for c in RATE_COLUMNS if c in df]
    outside = sum(int(((df[c] < ref["card_rate_floor"] - 1e-9) | (df[c] > ref["card_rate_ceiling"] + 1e-9)).sum())
                  for c in present)
    checks["rates_inside_envelope"] = (outside, f"rates outside {ref['card_rate_floor']:.2f}-{ref['card_rate_ceiling']:.2f}%")

    above = int((df["sanctioned_rate"] > df["policy_card_rate"] + 1e-9).sum()) if {"sanctioned_rate", "policy_card_rate"} <= set(df) else 0
    checks["no_rate_above_card"] = (above, "sanctioned above the loan's card (matrix) rate")

    months = df["tenure_years"] * 12
    bad_tenure = int(((months < con.get("min_tenure_months", 0)) | (months > con.get("max_tenure_months", 10**6))).sum())
    checks["tenure_in_range"] = (bad_tenure, f"tenure outside {con.get('min_tenure_months')}-{con.get('max_tenure_months')} months")

    cap = constraints.grade_foir_cap(df)
    emi = constraints._emi(df["sanctioned_loan_amount"].to_numpy(float),
                           df["policy_card_rate"].to_numpy(float) + constraints.stress_add_pct(),
                           df["tenure_years"].to_numpy(float))
    stressed = (emi + df["declared_obligations"].to_numpy(float)) / np.maximum(df["verified_monthly_income"].to_numpy(float), 1.0)
    checks["stressed_affordability"] = (int((stressed > cap + 1e-6).sum()),
                                        f"fail the stressed test (card + {constraints.stress_add_pct():.2f}%, grade FOIR cap)")

    scored = (df["bureau_score"] >= 0) & ~df["new_to_credit"].astype(bool)
    checks["bureau_minimum"] = (int((scored & (df["bureau_score"] < con.get("min_credit_score", 0))).sum()),
                                f"scored applicants below bureau {con.get('min_credit_score')}")

    declined = [g for g, v in pol["risk_grade_loading_bps"].items() if v is None]
    checks["no_declined_grade"] = (int(df["risk_grade"].astype(str).isin(declined).sum()),
                                   f"loans in a grade with no offer ({', '.join(declined)})")

    report = validator.validate(validator.coerce_types(df.copy()), strict=False)
    checks["validator_clean"] = (len(df) - report.n_valid, "rows ingestion validation would quarantine")

    names = wanted or list(checks)
    return [{"check": n, "violations": checks[n][0], "what": checks[n][1], "ok": checks[n][0] == 0}
            for n in names if n in checks]


def profile(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "dates": [str(df["application_date"].min())[:10], str(df["application_date"].max())[:10]],
        "grades": {k: int(v) for k, v in df["risk_grade"].value_counts().sort_index().items()},
        "sanctioned_rate": {"min": round(float(df["sanctioned_rate"].min()), 2),
                            "median": round(float(df["sanctioned_rate"].median()), 2),
                            "max": round(float(df["sanctioned_rate"].max()), 2)},
        "tenure_months": [int(df["tenure_years"].min() * 12), int(df["tenure_years"].max() * 12)],
        "disbursed_pct": round(100 * float(df["disbursed"].mean()), 1),
    }
