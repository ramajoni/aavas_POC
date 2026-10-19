"""Portfolio analytics: is the book actually earning what pricing assumed?

Everything here reads the lifecycle table once and caches it. These are committee
questions - "are we pricing Gujarat right?", "where is the discount leaking?" - so
the answers have to be sliceable, and they have to separate what we *decided* from
what we *realized*.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ..config import policy
from ..ingestion import loader
from ..logging_utils import get_logger

log = get_logger(__name__)

# Dimensions a committee actually slices by.
DIMENSIONS = ["segment", "risk_grade", "channel", "state", "city", "property_type", "override_reason"]


@lru_cache(maxsize=1)
def frame() -> pd.DataFrame:
    """The lifecycle table, loaded once per process."""
    df = loader.load_training_table()
    log.info("portfolio analytics loaded %s applications", len(df))
    return df


def clear_cache() -> None:
    frame.cache_clear()


def _value_inr(df: pd.DataFrame) -> pd.Series:
    """Realized lifetime value in rupees, counting a lost customer as zero."""
    if "realized_value_inr" in df.columns:
        return pd.to_numeric(df["realized_value_inr"], errors="coerce").fillna(0.0)
    return pd.to_numeric(df.get("realized_value_pct", 0), errors="coerce").fillna(0.0) / 100.0 * \
        pd.to_numeric(df.get("sanctioned_loan_amount", 0), errors="coerce").fillna(0.0)


def _safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


# ------------------------------------------------------------------- headline


def summary(df: pd.DataFrame | None = None) -> dict:
    """The numbers that belong above the fold on the admin screen."""
    df = frame() if df is None else df
    disbursed = df[df["disbursed"].astype(bool)]
    value = _value_inr(df)
    econ = policy()["economics"]

    return {
        "applications": int(len(df)),
        "disbursed": int(len(disbursed)),
        "conversion_rate": _safe(df["disbursed"].mean()),
        "disbursed_amount_inr": _safe(disbursed["sanctioned_loan_amount"].sum()),
        "mean_ticket_inr": _safe(disbursed["sanctioned_loan_amount"].mean()),
        "mean_sanctioned_rate": _safe(df["sanctioned_rate"].mean()),
        "mean_card_rate": _safe(df["policy_card_rate"].mean()),
        "mean_discount_bps": _safe(df["total_discount_bps"].mean()),
        "override_rate": _safe(df["was_overridden"].mean()),
        "bt_out_rate_24m": _safe(disbursed["bt_out_24m"].mean()),
        "delinquency_rate": _safe(disbursed["delinquent_12m"].mean()),
        "mean_realized_yield_pct": _safe(disbursed["realized_yield_pct"].mean()),
        "mean_realized_life_years": _safe(disbursed["realized_life_years"].mean()),
        "mean_value_per_application_inr": _safe(value.mean()),
        "total_value_inr": _safe(value.sum()),
        "funding_cost_pct": econ["funding_cost_pct"],
        "window": {
            "from": str(df["application_date"].min().date()),
            "to": str(df["application_date"].max().date()),
        },
    }


# ------------------------------------------------------------------ slicing


def by_dimension(dimension: str, df: pd.DataFrame | None = None, top: int = 25) -> list[dict]:
    """Portfolio performance cut by one dimension, ordered by volume."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"unknown dimension {dimension!r}; expected one of {DIMENSIONS}")
    df = frame() if df is None else df
    work = df.copy()
    work["_value"] = _value_inr(work)

    grouped = work.groupby(dimension, observed=True)
    rows = []
    for key, group in grouped:
        disbursed = group[group["disbursed"].astype(bool)]
        rows.append({
            "key": str(key),
            "applications": int(len(group)),
            "conversion_rate": _safe(group["disbursed"].mean()),
            "mean_card_rate": _safe(group["policy_card_rate"].mean()),
            "mean_sanctioned_rate": _safe(group["sanctioned_rate"].mean()),
            "mean_discount_bps": _safe(group["total_discount_bps"].mean()),
            "override_rate": _safe(group["was_overridden"].mean()),
            "bt_out_rate": _safe(disbursed["bt_out_24m"].mean()) if len(disbursed) else None,
            "mean_realized_yield_pct": _safe(disbursed["realized_yield_pct"].mean()) if len(disbursed) else None,
            "mean_value_inr": _safe(group["_value"].mean()),
            "total_value_inr": _safe(group["_value"].sum()),
            "disbursed_amount_inr": _safe(disbursed["sanctioned_loan_amount"].sum()),
        })
    rows.sort(key=lambda r: r["applications"], reverse=True)
    return rows[:top]


def rate_distribution(df: pd.DataFrame | None = None, bins: int = 18) -> list[dict]:
    """Where the book is actually priced, and how it converts at each price."""
    df = frame() if df is None else df
    edges = np.linspace(df["sanctioned_rate"].min(), df["sanctioned_rate"].max(), bins + 1)
    work = df.assign(_bucket=pd.cut(df["sanctioned_rate"], bins=edges, include_lowest=True))

    rows = []
    for bucket, group in work.groupby("_bucket", observed=True):
        rows.append({
            "rate_from": _safe(round(bucket.left, 2)),
            "rate_to": _safe(round(bucket.right, 2)),
            "applications": int(len(group)),
            "conversion_rate": _safe(group["disbursed"].mean()),
            "mean_value_inr": _safe(_value_inr(group).mean()),
        })
    return rows


def monthly_trend(df: pd.DataFrame | None = None) -> list[dict]:
    """Volume, price and conversion over time - the series that shows a market moving."""
    df = frame() if df is None else df
    work = df.copy()
    work["_value"] = _value_inr(work)

    rows = []
    for month, group in work.groupby("application_month", observed=True):
        disbursed = group[group["disbursed"].astype(bool)]
        rows.append({
            "month": str(month),
            "applications": int(len(group)),
            "conversion_rate": _safe(group["disbursed"].mean()),
            "mean_sanctioned_rate": _safe(group["sanctioned_rate"].mean()),
            "mean_card_rate": _safe(group["policy_card_rate"].mean()),
            "mean_competitor_rate": _safe(group["competitor_rate"].mean()),
            "mean_discount_bps": _safe(group["total_discount_bps"].mean()),
            "override_rate": _safe(group["was_overridden"].mean()),
            "mean_value_inr": _safe(group["_value"].mean()),
            "bt_out_rate": _safe(disbursed["bt_out_24m"].mean()) if len(disbursed) else None,
        })
    rows.sort(key=lambda r: r["month"])
    return rows


# ------------------------------------------------------------- decision flow


def decision_flow(df: pd.DataFrame | None = None) -> dict:
    """Recommended -> offered -> sanctioned -> realized.

    Every step between these is a decision someone made, and the gaps are the
    learning signal the business statement asks for.
    """
    df = frame() if df is None else df
    disbursed = df[df["disbursed"].astype(bool)]

    stages = [
        {"stage": "Policy card rate", "mean_rate": _safe(df["policy_card_rate"].mean()),
         "n": int(len(df)), "note": "what the grid allows"},
        {"stage": "AI recommended", "mean_rate": _safe(df["ai_recommended_rate"].mean()),
         "n": int(len(df)), "note": "what the engine proposed"},
        {"stage": "Sales offered", "mean_rate": _safe(df["sales_offered_rate"].mean()),
         "n": int(len(df)), "note": "what the human quoted"},
        {"stage": "Sanctioned", "mean_rate": _safe(df["sanctioned_rate"].mean()),
         "n": int(len(df)), "note": "what credit approved"},
        {"stage": "Disbursed", "mean_rate": _safe(disbursed["sanctioned_rate"].mean()),
         "n": int(len(disbursed)), "note": "what the customer accepted"},
    ]
    for previous, current in zip(stages, stages[1:]):
        current["delta_bps"] = _safe(round((current["mean_rate"] - previous["mean_rate"]) * 100, 1))

    return {
        "stages": stages,
        "human_adjustment_bps": _safe(round(
            ((df["sales_offered_rate"] - df["ai_recommended_rate"]) * 100).mean(), 1)),
        "credit_clawback_rate": _safe(df["credit_clawback"].mean()) if "credit_clawback" in df else None,
        "share_priced_down": _safe((df["sales_offered_rate"] < df["ai_recommended_rate"] - 0.005).mean()),
        "share_priced_up": _safe((df["sales_offered_rate"] > df["ai_recommended_rate"] + 0.005).mean()),
        "share_unchanged": _safe(
            (df["sales_offered_rate"].sub(df["ai_recommended_rate"]).abs() <= 0.005).mean()),
    }


def yield_bridge(df: pd.DataFrame | None = None) -> list[dict]:
    """Card rate down to realized value, step by step.

    A committee asking "where did the yield go?" needs the bridge, not a single
    realized number. Each step is a percentage-point movement on the mean file.
    """
    df = frame() if df is None else df
    disbursed = df[df["disbursed"].astype(bool)]
    econ = policy()["economics"]

    card = float(df["policy_card_rate"].mean())
    sanctioned = float(disbursed["sanctioned_rate"].mean())
    expected_loss = float((disbursed["true_pd"] * disbursed["true_lgd"] * 100).mean()) \
        if "true_pd" in disbursed.columns else 0.0

    steps = [
        {"label": "Policy card rate", "value": round(card, 3), "kind": "start"},
        {"label": "Commercial discount", "value": round(sanctioned - card, 3), "kind": "delta"},
        {"label": "Funding cost", "value": -econ["funding_cost_pct"], "kind": "delta"},
        {"label": "Servicing cost", "value": -econ["servicing_cost_pct"], "kind": "delta"},
        {"label": "Capital charge", "value": -econ["capital_charge_pct"], "kind": "delta"},
        {"label": "Expected credit loss", "value": -round(expected_loss, 3), "kind": "delta"},
    ]
    running = card
    for step in steps[1:]:
        running += step["value"]
    steps.append({"label": "Net annual margin", "value": round(running, 3), "kind": "total"})
    return steps


def competitive_position(df: pd.DataFrame | None = None) -> list[dict]:
    """How we price against the market, and what it costs us when we are above it."""
    df = frame() if df is None else df
    work = df.copy()
    work["gap_bps"] = ((work["sanctioned_rate"] - work["competitor_rate"]) * 100).round(0)
    edges = [-np.inf, -25, 0, 25, 50, 75, np.inf]
    labels = ["> 25 bps below", "0-25 bps below", "0-25 bps above",
              "25-50 bps above", "50-75 bps above", "> 75 bps above"]
    work["_band"] = pd.cut(work["gap_bps"], bins=edges, labels=labels)

    rows = []
    for band, group in work.groupby("_band", observed=True):
        disbursed = group[group["disbursed"].astype(bool)]
        rows.append({
            "position": str(band),
            "applications": int(len(group)),
            "conversion_rate": _safe(group["disbursed"].mean()),
            "mean_value_inr": _safe(_value_inr(group).mean()),
            "bt_out_rate": _safe(disbursed["bt_out_24m"].mean()) if len(disbursed) else None,
        })
    return rows
