"""How customers behave around a competitor offer, read from the loan book.

Each table answers one question a relationship manager or area manager asks:
does a documented quote change the outcome, does another round of negotiation
buy the file or only a discount, do month-end and target pressure produce
discounts the customer never asked for, and do the files we win while dearer
than the competitor stay?

*Discount* is the system recommendation minus the rate sales offered, in basis
points: positive means sales went below the recommendation. Most files get
none, so the median is near zero everywhere; the tables report the average and
the share discounted by more than ``MEANINGFUL_BPS``. *Win* is
disbursement. *Gap* is our offered rate minus the competitor's quote: positive
means we were dearer. Cells built from fewer than ``min_n_rate`` files are
suppressed rather than shown, as in the competitor tables.

Context for reviewers only: nothing here is read by the pricing path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

RESPONSES = ["accepted", "negotiated", "deferred", "rejected"]
MEANINGFUL_BPS = 5

# (label, low, high] in bps of our rate over the competitor's quote.
GAP_BANDS = [("We were 25+ bps cheaper", -np.inf, -25), ("Within 25 bps", -25, 25),
             ("25–75 bps dearer", 25, 75), ("75+ bps dearer", 75, np.inf)]
CHANNELS = {"branch_walkin": "Branch walk-in", "dsa": "DSA", "connector": "Connector",
            "digital": "Digital"}
TARGET_BANDS = [("Below 70% of target", -np.inf, 70), ("70–90% of target", 70, 90),
                ("90%+ of target", 90, np.inf)]


def _r(value: Any, digits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


class Behaviour:
    def __init__(self, loans: pd.DataFrame, min_n_rate: int = 30):
        self.min_n = int(min_n_rate)
        df = loans.copy()
        df["win"] = df["disbursed"].astype(bool)
        df["discount_bps"] = (df["ai_recommended_rate"] - df["sales_offered_rate"]) * 100
        if "competitor_gap_bps" not in df:
            df["competitor_gap_bps"] = (df["offered_rate"] - df["competitor_rate"]) * 100
        self.df = df

    # ------------------------------------------------------------- one cell

    def cell(self, label: str, rows: pd.DataFrame) -> dict[str, Any]:
        n = int(len(rows))
        if n < self.min_n:
            return {"label": label, "n": n, "suppressed": True}
        won = rows[rows["win"]]
        return {
            "label": label, "n": n, "suppressed": False,
            "win_rate": _r(rows["win"].mean()),
            "avg_discount_bps": _r(rows["discount_bps"].mean(), 1),
            "discounted_share": _r((rows["discount_bps"] > MEANINGFUL_BPS).mean()),
            "bt_out_24m": _r(won["bt_out_24m"].mean()) if len(won) >= self.min_n else None,
            "responses": {k: _r((rows["customer_response"] == k).mean()) for k in RESPONSES},
        }

    def _bands(self, column: str, bands: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
        values = self.df[column]
        return [self.cell(label, self.df[(values > lo) & (values <= hi)]) for label, lo, hi in bands]

    # ------------------------------------------------------------- the tables

    def summary(self) -> dict[str, Any]:
        df = self.df
        rounds = df["prior_negotiation_rounds"].clip(upper=3)
        negotiating = df[df["prior_negotiation_rounds"] > 0]
        evidence = df["competitor_evidence_verified"].astype(bool)
        month_end = df["days_to_month_end"] <= 3
        relationship = df["existing_relationship"].astype(bool)

        documented, verbal = self.cell("Documented quote", df[evidence]), \
            self.cell("Verbal quote only", df[~evidence])
        early, late = self.cell("Rest of month", df[~month_end]), \
            self.cell("Last 3 days of month", df[month_end])
        negotiation = [self.cell(f"{int(k)}{'+' if k == 3 else ''} round{'s' if k != 1 else ''}",
                                 df[rounds == k]) for k in range(4)]
        return {
            "headline": {
                "files": int(len(df)),
                "win_rate": _r(df["win"].mean()),
                "negotiating_share": _r((df["prior_negotiation_rounds"] > 0).mean()),
                "avg_discount_when_negotiating_bps": _r(negotiating["discount_bps"].mean(), 1),
                "avg_discount_otherwise_bps": _r(df.loc[df["prior_negotiation_rounds"] == 0,
                                                        "discount_bps"].mean(), 1),
                "documented_quote_share": _r(evidence.mean()),
                "bt_out_24m": _r(df.loc[df["win"], "bt_out_24m"].mean()),
            },
            "by_gap": self._bands("competitor_gap_bps", GAP_BANDS),
            "by_evidence": [documented, verbal],
            "by_negotiation": negotiation,
            "by_month_end": [early, late],
            "by_target": self._bands("sales_target_pct", TARGET_BANDS),
            "by_channel": sorted((self.cell(CHANNELS.get(c, str(c).replace("_", " ").title()), g)
                                  for c, g in df.groupby("channel")),
                                 key=lambda c: -c["n"]),
            "by_relationship": [self.cell("New to us", df[~relationship]),
                                self.cell("Existing relationship", df[relationship])],
            "findings": self.findings(documented, verbal, early, late, negotiation),
            "definitions": {
                "win": "the file was disbursed",
                "discount": "System recommendation minus the rate sales offered, in bps (positive = below the recommendation)",
                "discounted_share": f"share of files discounted by more than {MEANINGFUL_BPS} bps",
                "gap": "our offered rate minus the competitor's quote, in bps (positive = we were dearer)",
                "bt_out_24m": "share of disbursed files that moved to another lender within 24 months",
            },
            "settings": {"min_n_rate": self.min_n},
        }

    def findings(self, documented, verbal, early, late, negotiation) -> list[str]:
        """Plain sentences, each computed from the tables above it."""
        out: list[str] = []
        if not (early["suppressed"] or late["suppressed"]):
            lift = (late["win_rate"] or 0) - (early["win_rate"] or 0)
            out.append(
                f"In the last three days of the month the average discount is "
                f"{late['avg_discount_bps']:.1f} bps against {early['avg_discount_bps']:.1f} the rest "
                f"of the month, while the win rate moves only {lift * 100:+.1f} points.")
        first, last = negotiation[0], negotiation[-1]
        if not (first["suppressed"] or last["suppressed"]):
            out.append(
                f"Customers who negotiate are not won by it: the win rate is {first['win_rate']:.0%} "
                f"with no negotiation and {last['win_rate']:.0%} after three or more rounds, while the "
                f"average discount goes from {first['avg_discount_bps']:.1f} to "
                f"{last['avg_discount_bps']:.1f} bps.")
        if not (documented["suppressed"] or verbal["suppressed"]):
            out.append(
                f"A documented competitor quote wins {documented['win_rate']:.0%} of files against "
                f"{verbal['win_rate']:.0%} for a verbal one (average discount "
                f"{documented['avg_discount_bps']:.1f} vs {verbal['avg_discount_bps']:.1f} bps).")
        bands = self._bands("competitor_gap_bps", GAP_BANDS)
        cheaper, dearest = bands[0], bands[-1]
        if cheaper.get("bt_out_24m") is not None and dearest.get("bt_out_24m") is not None:
            out.append(
                f"Files won while 75+ bps dearer than the competitor leave within 24 months "
                f"{dearest['bt_out_24m']:.0%} of the time, against {cheaper['bt_out_24m']:.0%} when "
                f"we were 25+ bps cheaper.")
        return out
