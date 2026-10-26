"""The model cards in business terms: each customer segment's risk, its starting
rate, and how its customers respond to price.

**Segments.** From the loan book: how often a disbursed loan in the segment
defaulted (over the book's ~4-year window, and per year), how much was lost when
one did (LGD), the expected credit loss per year that follows, and the rate the
segment was typically sanctioned at - beside the policy's starting rate for it
(the card-rate floor plus the segment's base spread, before any risk loading).

**Acceptance curves.** For each segment, the share of its customers likely to take
the loan at each rate: the conversion model scored on the segment's most recent
files with only the offered rate changed, then averaged. This is the curve the
optimiser trades against margin.

**What a discount buys.** Read off the same curves: acceptance points won by
pricing 0.25% below the segment's typical rate.

Scoring the curves takes a few seconds, so they are computed once per trained
conversion artefact.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ..config import model_config, policy
from . import portfolio

SEGMENT_LABELS = {"salaried_formal": "Salaried", "salaried_informal": "Salaried (informal)",
                  "self_employed_professional": "Self-employed professional",
                  "self_employed_non_professional": "Self-employed"}
STEP = 0.1
FILES_PER_SEGMENT = 400


def _r(value: Any, digits: int = 4) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), digits)


def segments(df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    df = portfolio.frame() if df is None else df
    pol = policy()
    floor = float(pol["reference"]["card_rate_floor"])
    spreads = pol.get("base_spread_bps") or {}
    horizon = float(model_config()["risk"]["pd_horizon_years"])
    booked = df[df["disbursed"].astype(bool)]
    out = []
    for segment, rows in df.groupby("segment"):
        loans = booked[booked["segment"] == segment]
        defaulted = loans[loans["defaulted"].astype(bool)]
        lifetime = loans["defaulted"].astype(float).mean() if len(loans) else None
        annual = lifetime / horizon if lifetime is not None else None
        lgd = defaulted["realized_lgd"].mean() if len(defaulted) else None
        out.append({
            "segment": segment, "label": SEGMENT_LABELS.get(segment, segment.replace("_", " ")),
            "applications": int(len(rows)), "loans": int(len(loans)),
            "starting_rate": _r(floor + float(spreads.get(segment, 0)) / 100, 2),
            "typical_rate": _r(loans["sanctioned_rate"].median(), 2),
            "acceptance": _r(rows["disbursed"].astype(float).mean()),
            "default_rate": _r(lifetime), "default_rate_annual": _r(annual),
            "lgd": _r(lgd), "expected_loss_pct": _r(annual * lgd * 100, 3)
            if annual is not None and lgd is not None else None,
        })
    out.sort(key=lambda s: s["starting_rate"] or 0)
    return out


def _recent(df: pd.DataFrame) -> pd.DataFrame:
    """The latest files, where customers respond to price as they do now."""
    time_col = model_config()["data"]["time_column"]
    return df.sort_values(time_col).groupby("segment").tail(FILES_PER_SEGMENT)


@lru_cache(maxsize=2)
def _curves(trained_at: str | None, low: float, high: float) -> tuple[tuple[str, tuple], ...]:
    from ..governance import model_registry

    model = model_registry.load_bundle().conversion
    rates = np.round(np.arange(low, high + 1e-9, STEP), 2)
    out = []
    for segment, rows in _recent(portfolio.frame()).groupby("segment"):
        expanded = rows.loc[rows.index.repeat(len(rates))].reset_index(drop=True)
        expanded["offered_rate"] = np.tile(rates, len(rows))
        if "competitor_rate" in expanded.columns:
            competitor = pd.to_numeric(expanded["competitor_rate"], errors="coerce")
            expanded["competitor_gap_bps"] = ((expanded["offered_rate"] - competitor) * 100).fillna(0.0)
        p = model.model.predict_proba(model.transformer.transform(expanded))[:, 1]
        mean = p.reshape(len(rows), len(rates)).mean(axis=0)
        out.append((segment, tuple((float(r), round(float(a), 4)) for r, a in zip(rates, mean))))
    return tuple(out)


def acceptance_curves(low: float = 8.6, high: float = 14.0) -> dict[str, Any] | None:
    from ..governance import model_registry

    bundle = model_registry.load_bundle()
    if bundle.conversion is None:
        return None
    card = getattr(bundle.conversion, "card", None)
    curves = _curves(getattr(card, "trained_at", None), low, high)
    by_segment = {s["segment"]: s for s in segments()}
    series = []
    for segment, points in curves:
        info = by_segment.get(segment) or {}
        typical = info.get("typical_rate")
        series.append({"segment": segment, "label": SEGMENT_LABELS.get(segment, segment),
                       "typical_rate": typical,
                       "points": [{"rate": r, "acceptance": a} for r, a in points],
                       "discount_25bps_pts": _gain(points, typical, 0.25)})
    series.sort(key=lambda s: (by_segment.get(s["segment"]) or {}).get("starting_rate") or 0)
    return {"rates": [low, high], "files_per_segment": FILES_PER_SEGMENT, "series": series,
            "method": (f"The acceptance model scored on each segment's latest {FILES_PER_SEGMENT} files "
                       f"with only the offered rate changed, averaged across the files.")}


def _at(points: tuple, rate: float) -> float:
    rates = np.array([r for r, _ in points])
    values = np.array([a for _, a in points])
    return float(np.interp(rate, rates, values))


def _gain(points: tuple, typical: float | None, cut: float) -> float | None:
    """Acceptance points won by pricing ``cut`` below the typical rate."""
    if typical is None:
        return None
    return round((_at(points, typical - cut) - _at(points, typical)) * 100, 1)


def floor_components() -> dict[str, Any]:
    """The parts of the break-even floor that are the same for every file."""
    econ = policy().get("economics") or {}
    return {"cost_of_funds_pct": econ.get("funding_cost_pct"),
            "operating_cost_pct": econ.get("servicing_cost_pct"),
            "capital_charge_pct": econ.get("capital_charge_pct"),
            "source": "config/pricing_policy.yaml (economics)"}


@lru_cache(maxsize=2)
def _measured(trained_at: str | None, bps: float) -> tuple[tuple[str, float, float, float], ...]:
    from ..governance import model_registry

    model = model_registry.load_bundle().uplift
    out = []
    for segment, rows in _recent(portfolio.frame()).groupby("segment"):
        curve = model.curve(rows, [bps])
        out.append((segment, round(float(curve["uplift_pts"].mean()), 1),
                    round(float(curve["uplift_low_pts"].mean()), 1),
                    round(float(curve["uplift_high_pts"].mean()), 1)))
    return tuple(out)


def discount_effect(bps: float = 25.0) -> dict[str, Any] | None:
    """What a discount of ``bps`` buys, by segment: measured from the discounts the
    bank actually gave (the uplift model, which adjusts for which files got one),
    beside what the acceptance curve alone implies."""
    from ..governance import model_registry

    model = model_registry.load_bundle().uplift
    if model is None:
        return None
    card = getattr(model, "card", None)
    curves = {s["segment"]: s for s in (acceptance_curves() or {}).get("series") or []}
    rows = []
    for segment, pts, low, high in _measured(getattr(card, "trained_at", None), bps):
        curve = curves.get(segment) or {}
        typical = curve.get("typical_rate")
        rows.append({"segment": segment, "label": SEGMENT_LABELS.get(segment, segment),
                     "typical_rate": typical,
                     "acceptance_at_typical": round(_at(tuple((p["rate"], p["acceptance"])
                                                             for p in curve["points"]), typical), 4)
                     if curve and typical else None,
                     "measured_pts": pts, "measured_low_pts": low, "measured_high_pts": high,
                     "curve_pts": curve.get("discount_25bps_pts") if bps == 25 else None})
    order = list(curves)
    rows.sort(key=lambda r: order.index(r["segment"]) if r["segment"] in order else 99)
    return {"discount_bps": bps, "rows": rows,
            "method": ("Measured: the uplift model, learned from the discounts actually given and "
                       "adjusted for which files were given one, averaged over each segment's "
                       f"latest {FILES_PER_SEGMENT} files, with its 10-90% range. Curve: the "
                       "acceptance curve read 0.25% below the segment's typical rate.")}
