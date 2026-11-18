"""What the branch's own history says about upselling, competitors and the
conversations behind them - the sub-tabs of the area manager's review.

**Upselling.** Files the branch priced *above* the system's recommendation (a
premium override), against files left at it: how often, by how much, and whether
the customer still took the loan and what the loan was worth. For the branch,
across every branch, by the reason given, and for files like this one (same
segment and risk grade).

**Competitor strength.** For each lender named on the branch's files: how often it
came up, what it quoted against the rate we sanctioned, how often the quote was
documented, and how often the customer still took our loan - so the lenders that
actually win files stand out from the ones that are only mentioned.

**Notes.** The sales and conversation notes from the branch's past files, newest
first, with what then happened; the ones naming this file's lender first.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..ingestion import loader
from . import portfolio
from .approval_briefing import scope

MIN_N = 20
NOTES_SHOWN = 30


def _rate(series: pd.Series) -> float | None:
    return round(float(series.mean()), 4) if len(series) >= MIN_N else None


def _num(value: Any, digits: int = 1) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), digits)


def _profile(rows: pd.DataFrame) -> dict[str, Any]:
    upsold = rows[rows["override_direction"] == "premium"]
    held = rows[rows["override_direction"] == "none"]
    return {
        "files": int(len(rows)), "upsold": int(len(upsold)),
        "upsell_share": round(len(upsold) / len(rows), 4) if len(rows) else None,
        "median_premium_bps": _num(upsold["override_bps"].median()) if len(upsold) else None,
        "took_when_upsold": _rate(upsold["disbursed"].astype(float)),
        "took_at_recommendation": _rate(held["disbursed"].astype(float)),
        "value_upsold_inr": _num(upsold["realized_value_inr"].mean(), 0) if len(upsold) >= MIN_N else None,
        "value_at_recommendation_inr": _num(held["realized_value_inr"].mean(), 0) if len(held) >= MIN_N else None,
    }


def upselling(branch: pd.DataFrame, everyone: pd.DataFrame, segment: str | None,
              grade: str | None) -> dict[str, Any]:
    upsold = branch[branch["override_direction"] == "premium"]
    by_reason = []
    for reason, group in upsold.groupby("override_reason"):
        by_reason.append({"reason": str(reason), "files": int(len(group)),
                          "median_premium_bps": _num(group["override_bps"].median()),
                          "took": _rate(group["disbursed"].astype(float))})
    by_reason.sort(key=lambda r: -r["files"])
    like = branch[(branch["segment"] == segment) & (branch["risk_grade"] == grade)] if segment and grade else branch.iloc[0:0]
    return {"branch": _profile(branch), "all_branches": _profile(everyone), "by_reason": by_reason[:6],
            "similar": {**_profile(like), "segment": segment, "risk_grade": grade}}


def competitor_strength(branch: pd.DataFrame, lender: str | None) -> dict[str, Any]:
    named = branch[branch["competitor_name"].notna() & (branch["competitor_name"] != "")]
    rows = []
    for name, group in named.groupby("competitor_name"):
        took = group["disbursed"].astype(float)
        gap = (group["competitor_rate"] - group["sanctioned_rate"]) * 100
        rows.append({"lender": str(name), "files": int(len(group)),
                     "share": round(len(group) / len(branch), 4) if len(branch) else None,
                     "median_quote": _num(group["competitor_rate"].median(), 2),
                     "median_gap_bps": _num(gap.median()),
                     "documented": _rate(group["competitor_evidence_verified"].astype(float)),
                     "we_won": _rate(took), "they_won": round(1 - float(took.mean()), 4) if len(group) >= MIN_N else None})
    rows.sort(key=lambda r: -(r["they_won"] or 0))
    return {"lenders": rows, "named_files": int(len(named)), "files": int(len(branch)),
            "this_lender": next((r for r in rows if lender and r["lender"].lower() == lender.lower()), None),
            "lender": lender}


def notes(branch_files: pd.DataFrame, city: str | None, lender: str | None) -> dict[str, Any]:
    all_notes = loader.load_notes()
    if not len(all_notes) or not city:
        return {"notes": [], "counts": {}, "total": 0}
    here = all_notes[all_notes["city"] == city].copy()
    outcome = branch_files.set_index("loan_id")[["disbursed", "sanctioned_rate", "override_direction"]]
    here = here.join(outcome, on="loan_id")
    here["names_lender"] = here["competitor_name"].fillna("").str.lower().eq((lender or "").lower()) & bool(lender)
    here = here.sort_values(["names_lender", "note_date"], ascending=[False, False])
    shown = here.head(NOTES_SHOWN)
    return {
        "total": int(len(here)), "counts": {str(k): int(v) for k, v in here["note_type"].value_counts().items()},
        "naming_lender": int(here["names_lender"].sum()),
        "notes": [{"date": str(r.note_date), "type": str(r.note_type), "text": str(r.note_text),
                   "lender": None if pd.isna(r.competitor_name) else str(r.competitor_name),
                   "quote": _num(r.competitor_rate, 2), "names_lender": bool(r.names_lender),
                   "took": None if pd.isna(r.disbursed) else bool(r.disbursed),
                   "rate": _num(r.sanctioned_rate, 2),
                   "priced": None if pd.isna(r.override_direction) else str(r.override_direction)}
                  for r in shown.itertuples()],
    }


def history(request: dict[str, Any], applicant: dict[str, Any], *, risk_grade: str | None) -> dict[str, Any]:
    everyone = portfolio.frame()
    city = applicant.get("city")
    branch, where = scope(everyone, city)
    sales = {**{k: applicant.get(k) for k in ("competitor_name", "competitor_rate")},
             **(applicant.get("sales_context") or {})}
    lender = sales.get("competitor_name")
    return {
        "request_id": request.get("request_id"), "where": where,
        "upselling": upselling(branch, everyone, applicant.get("segment"), risk_grade),
        "competitors": competitor_strength(branch, lender),
        "notes": notes(branch, city if where["level"] == "branch" else None, lender),
        "file_notes": {"conversation": (applicant.get("conversation_notes") or "").strip() or None,
                       "rm_request": (request.get("note") or "").strip() or None,
                       "rm_reason": request.get("reason"),
                       "competitor": {"name": lender, "rate": sales.get("competitor_rate"),
                                      "evidence": bool(sales.get("competitor_evidence_verified"))}},
        "note": ("From the loan book the pricing models learn from, and the sales notes written on "
                 f"those files. Rates on fewer than {MIN_N} files are withheld."),
    }
