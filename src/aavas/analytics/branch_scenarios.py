"""Scenario analysis for one lower-rate request: what the general model predicts at
each rate, and how this branch's own history differs from it.

**General prediction.** The acceptance model, trained on every branch's book, scored
on this file with only the offered rate changed - the curve the optimiser prices on.

**This branch.** The model is checked against what actually happened on the branch's
recent files: the model's out-of-time holdout, so it is not marking its own homework.
How often customers there took the loan, against how often the model said they
would. The gap is carried onto this file's curve as one shift in log-odds: the same
customer at the same price is that much more, or less, likely to accept here.

**Branch decisions.** The branch's past discounts for the RM's reason, against
comparable files left at the recommendation (the briefing's propensity-matched
comparison), beside what the model expects the requested discount to buy.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from . import portfolio

MIN_BRANCH_FILES = 60


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def shift_for(predicted: np.ndarray, actual_rate: float) -> float:
    """The log-odds shift that makes the model's average match what happened."""
    z = _logit(np.asarray(predicted, dtype=float))
    low, high = -6.0, 6.0
    for _ in range(60):
        mid = (low + high) / 2
        if _sigmoid(z + mid).mean() < actual_rate:
            low = mid
        else:
            high = mid
    return round((low + high) / 2, 4)


@lru_cache(maxsize=2)
def _holdout(trained_at: str | None) -> pd.DataFrame:
    """Holdout files with the model's prediction beside what happened."""
    from ..governance import model_registry
    from ..models.conversion import train as conversion_train

    model = model_registry.load_bundle().conversion
    _, holdout = conversion_train.split(portfolio.frame())
    holdout = holdout.copy()
    holdout["predicted"] = model.model.predict_proba(model.transformer.transform(holdout))[:, 1]
    holdout["took"] = holdout["disbursed"].astype(float)
    return holdout[["city", "predicted", "took", "application_date"]]


def calibration(city: str | None) -> dict[str, Any]:
    """The model against what happened, in this branch and across every branch."""
    from ..governance import model_registry

    card = getattr(model_registry.load_bundle().conversion, "card", None)
    rows = _holdout(getattr(card, "trained_at", None))

    def check(part: pd.DataFrame) -> dict[str, Any]:
        actual = float(part["took"].mean())
        return {"files": int(len(part)), "actual": round(actual, 4),
                "predicted": round(float(part["predicted"].mean()), 4),
                "gap_pts": round((actual - float(part["predicted"].mean())) * 100, 1),
                "shift": shift_for(part["predicted"].to_numpy(), actual),
                "from": str(pd.to_datetime(part["application_date"]).min().date()),
                "to": str(pd.to_datetime(part["application_date"]).max().date())}

    everyone = check(rows)
    here = rows[rows["city"] == city] if city else rows.iloc[0:0]
    branch = check(here) if len(here) >= MIN_BRANCH_FILES else None
    return {"all_branches": everyone, "branch": branch, "city": city,
            "note": None if branch else (f"Too few recent files in {city or 'this city'} "
                                         f"(under {MIN_BRANCH_FILES}) to check the model against; "
                                         "the all-branch check is used.")}


def _score(payload: dict[str, Any], rates: np.ndarray) -> pd.DataFrame:
    from ..governance import model_registry
    from ..optimization import price_optimizer

    bundle = model_registry.load_bundle()
    snap = payload.get("application_snapshot") or {}
    return price_optimizer.evaluate_scenarios(
        application=pd.DataFrame([snap]), rates=rates, conversion_model=bundle.conversion,
        prepayment_model=bundle.prepayment,
        expected_loss_pct=float((payload.get("risk") or {}).get("expected_loss_pct") or 0.0),
        loan_amount=float(snap.get("sanctioned_loan_amount") or snap.get("requested_loan_amount") or 0),
        tenure_years=float(snap.get("tenure_years") or snap.get("requested_tenure_years") or 20))


def _emi(amount: float, rate: float, years: float) -> float:
    r, n = rate / 1200, years * 12
    return amount * r * (1 + r) ** n / ((1 + r) ** n - 1) if r else amount / n


def scenarios(request: dict[str, Any], payload: dict[str, Any], *, ceiling: float | None,
              branch_effect: dict[str, Any] | None) -> dict[str, Any]:
    snap = payload.get("application_snapshot") or {}
    env = payload.get("envelope") or {}
    rec = float((payload.get("recommendation") or {}).get("recommended_rate") or request["recommended_rate"])
    asked = float(request["requested_rate"])
    loan = float(request.get("loan_amount") or snap.get("sanctioned_loan_amount") or snap.get("requested_loan_amount") or 0)
    years = float(snap.get("tenure_years") or snap.get("requested_tenure_years") or 20)
    city = snap.get("city")
    check = calibration(city)
    used = check["branch"] or check["all_branches"]
    shift = used["shift"] - check["all_branches"]["shift"]   # what is different about this branch

    lo = max(float(env.get("min_rate") or 0), min(asked, rec) - 1.0)
    hi = min(float(env.get("max_rate") or 99), max(rec, ceiling or rec) + 1.0)
    curve_rates = np.round(np.arange(lo, hi + 1e-9, 0.05), 2)
    # The table runs from just under the ask to the highest rate that may be offered
    # (at most half a point over the recommendation), in steps that keep it readable.
    top = max(rec, min(ceiling or rec, rec + 0.5))
    step = 0.10 if top - min(asked, rec) <= 1.5 else 0.25
    table_rates = np.round(np.arange(min(asked, rec) - 0.20, top + 1e-9, step), 2)
    table_rates = np.unique(np.concatenate([table_rates, [asked, rec]]))
    table_rates = table_rates[(table_rates >= lo - 1e-9) & (table_rates <= hi + 1e-9)]

    def rows(rates: np.ndarray) -> list[dict[str, Any]]:
        scored = _score(payload, rates)
        p = scored["conversion_probability"].to_numpy(float)
        p_branch = _sigmoid(_logit(p) + shift)
        booked = np.where(p > 0, scored["expected_value_inr"].to_numpy(float) / np.maximum(p, 1e-9), 0)
        emi_rec = _emi(loan, rec, years) if loan else None
        out = []
        for i, r in enumerate(scored["rate"].to_numpy(float)):
            emi = _emi(loan, r, years) if loan else None
            out.append({"rate": round(float(r), 2), "emi": round(emi) if emi else None,
                        "acceptance_model": round(float(p[i]), 4), "acceptance_branch": round(float(p_branch[i]), 4),
                        "value_model_inr": round(float(scored["expected_value_inr"].iloc[i])),
                        "value_branch_inr": round(float(p_branch[i] * booked[i])),
                        "net_margin_pct": round(float(scored["net_margin_pct"].iloc[i]), 3),
                        "interest_given_up_inr": round((emi_rec - emi) * years * 12) if emi and emi_rec else None,
                        "mark": "requested" if abs(r - asked) < 0.005 else "recommended" if abs(r - rec) < 0.005 else None})
        return out

    table = rows(table_rates)
    at = {row["mark"]: row for row in table if row["mark"]}
    req_row, rec_row = at.get("requested"), at.get("recommended")
    effect = branch_effect or {}
    return {
        "request_id": request.get("request_id"), "requested_rate": asked, "recommended_rate": rec,
        "ceiling_rate": ceiling, "breakeven_rate": (payload.get("risk") or {}).get("breakeven_rate"),
        "loan_amount": loan, "tenure_years": years, "city": city,
        "curve": [{k: r[k] for k in ("rate", "acceptance_model", "acceptance_branch")} for r in rows(curve_rates)],
        "table": table,
        "calibration": check, "branch_shift": round(shift, 4),
        "summary": {
            "model_at_requested": req_row and req_row["acceptance_model"],
            "model_at_recommended": rec_row and rec_row["acceptance_model"],
            "branch_at_requested": req_row and req_row["acceptance_branch"],
            "branch_at_recommended": rec_row and rec_row["acceptance_branch"],
            "discount_buys_model_pts": round((req_row["acceptance_model"] - rec_row["acceptance_model"]) * 100, 1)
            if req_row and rec_row else None,
            "discount_buys_branch_pts": round((req_row["acceptance_branch"] - rec_row["acceptance_branch"]) * 100, 1)
            if req_row and rec_row else None,
            "value_change_model_inr": (req_row["value_model_inr"] - rec_row["value_model_inr"]) if req_row and rec_row else None,
            "value_change_branch_inr": (req_row["value_branch_inr"] - rec_row["value_branch_inr"]) if req_row and rec_row else None,
        },
        "branch_history": {"label": effect.get("label"), "verdict": effect.get("verdict"),
                           "s2d_lift_pts": effect.get("s2d_lift_pts"), "value_lift_inr": effect.get("value_lift_inr"),
                           "files": effect.get("n_override")} if effect else None,
        "method": ("General: the acceptance model, trained on every branch, scored on this file with only the "
                   "rate changed. This branch: the same curve shifted by how far the branch's recent files "
                   "(the model's out-of-time holdout) beat or fell short of the model, beyond the gap every "
                   "branch shows. Expected value is the value of the loan if booked, times the chance it is."),
    }
