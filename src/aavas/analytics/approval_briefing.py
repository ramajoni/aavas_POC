"""What an area manager reads before approving a lower rate.

Three questions, each answered from the loan book for the branch the file is in
with the all-branch figure beside each one for reference:

* **Overrides** - how often this branch prices below the recommendation, how
  deep, for which reasons, and whether those discounts paid for themselves.
  "Paid for themselves" is the propensity-adjusted read from
  ``learning/override_learning.py``: a discounted file is compared with files
  that were as likely to be discounted and were not, because a raw comparison
  makes every discount look as if it lost customers.
* **Competitors** - who quotes against us in this city, how far under our rate,
  how much of it is documented, and how often we still won; for the lender on
  this file, its usual quotes here and our record at this gap.
* **Performance** - how the branch's book has done (conversion, yield, balance
  transfers out, delinquency, default, value per file), how past discounts like
  this one turned out, and the requesting RM's record since go-live.

All of it is history for the person deciding. Nothing here is read by the price.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..learning import override_learning

MIN_RATE_N = 30          # a rate on fewer files than this is not shown
SIMILAR_DEPTH_BPS = 10   # "a discount like this one": within this many bps
MONTHS = 12
DISCOUNT = "discount"


def _num(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, digits)


def _share(flags: pd.Series) -> dict[str, Any]:
    """A rate with its n; withheld when the sample is too small to mean anything."""
    flags = flags.dropna().astype(bool)
    n = int(len(flags))
    return {"n": n, "rate": _num(flags.mean()) if n >= MIN_RATE_N else None}


def _discounts(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["override_direction"] == DISCOUNT]


# ------------------------------------------------------------------ the branch

def scope(history: pd.DataFrame, city: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The branch's files, or every branch when the city has no history."""
    if city and city in set(history["city"]):
        branch = history[history["city"] == city]
        return branch, {"level": "branch", "city": city,
                        "branch_id": str(branch["branch_id"].mode().iloc[0]),
                        "state": str(branch["state"].mode().iloc[0]), "files": int(len(branch))}
    return history, {"level": "all", "city": city, "branch_id": None, "state": None,
                     "files": int(len(history)),
                     "note": f"No loan history for {city or 'this city'}; every branch is shown instead."}


# ---------------------------------------------------------------- overrides

def override_profile(df: pd.DataFrame) -> dict[str, Any]:
    disc = _discounts(df)
    competitor = disc["override_reason"] == "competitor_lower_rate"
    return {
        "files": int(len(df)),
        "discount_rate": _share(df["override_direction"] == DISCOUNT),
        "median_discount_bps": _num(-disc["override_bps"].median(), 1) if len(disc) else None,
        "competitor_share_of_discounts": _share(competitor) if len(disc) else {"n": 0, "rate": None},
        "conversion_discounted": _share(disc["disbursed"]),
        "conversion_untouched": _share(df.loc[~df["was_overridden"].astype(bool), "disbursed"]),
    }


def by_reason(df: pd.DataFrame, top: int = 6) -> list[dict[str, Any]]:
    disc = _discounts(df)
    rows = []
    for reason, group in disc.groupby("override_reason"):
        rows.append({"reason": str(reason), "files": int(len(group)),
                     "share": _num(len(group) / len(disc)) if len(disc) else None,
                     "median_discount_bps": _num(-group["override_bps"].median(), 1),
                     "conversion": _share(group["disbursed"])["rate"]})
    return sorted(rows, key=lambda r: -r["files"])[:top]


def adjusted_effect(df: pd.DataFrame, propensity: np.ndarray | None, reason: str | None,
                    label: str) -> dict[str, Any] | None:
    """Did discounts here (for this reason) pay for themselves, propensity-adjusted."""
    if propensity is None or len(df) == 0:
        return None
    mask = (df["override_direction"] == DISCOUNT).to_numpy()
    if reason:
        mask &= (df["override_reason"] == reason).to_numpy()
    effect = override_learning.propensity_adjusted(df, propensity, group_mask=mask, label=label)
    out = effect.as_dict()
    for key in ("s2d_lift_pts", "value_lift_inr", "value_lift_se_inr", "mean_override_bps"):
        out[key] = _num(out[key], 2)
    return out


def similar_decisions(df: pd.DataFrame, reason: str | None, depth_bps: float) -> dict[str, Any]:
    """Past discounts for the same reason, about as deep as this one, and what followed."""
    disc = _discounts(df)
    if reason:
        disc = disc[disc["override_reason"] == reason]
    depth = -disc["override_bps"]
    like = disc[(depth - depth_bps).abs() <= SIMILAR_DEPTH_BPS]
    funded = like[like["disbursed"].astype(bool)]
    return {"reason": reason, "depth_bps": _num(depth_bps, 1),
            "window_bps": [_num(max(depth_bps - SIMILAR_DEPTH_BPS, 0), 1), _num(depth_bps + SIMILAR_DEPTH_BPS, 1)],
            "files": int(len(like)),
            "deepest_bps": _num(depth.max(), 1) if len(depth) else None,
            "conversion": _share(like["disbursed"])["rate"],
            "bt_out_24m": _share(funded["bt_out_24m"])["rate"],
            "default_rate": _share(funded["defaulted"])["rate"],
            "median_yield_pct": _num(funded["realized_yield_pct"].median(), 2) if len(funded) >= MIN_RATE_N else None,
            "value_per_file_inr": _num(like["realized_value_inr"].mean(), 0) if len(like) >= MIN_RATE_N else None}


# -------------------------------------------------------------- performance

def performance(df: pd.DataFrame) -> dict[str, Any]:
    funded = df[df["disbursed"].astype(bool)]
    return {
        "files": int(len(df)),
        "conversion": _share(df["disbursed"])["rate"],
        "avg_sanctioned_rate": _num(funded["sanctioned_rate"].mean(), 2) if len(funded) else None,
        "avg_discount_to_card_bps": (_num(((funded["policy_card_rate"] - funded["sanctioned_rate"]) * 100).mean(), 1)
                                     if len(funded) else None),
        "median_yield_pct": _num(funded["realized_yield_pct"].median(), 2) if len(funded) >= MIN_RATE_N else None,
        "bt_out_24m": _share(funded["bt_out_24m"])["rate"],
        "delinquent_12m": _share(funded["delinquent_12m"])["rate"],
        "default_rate": _share(funded["defaulted"])["rate"],
        "value_per_file_inr": _num(df["realized_value_inr"].mean(), 0),
    }


def monthly(df: pd.DataFrame, months: int = MONTHS) -> list[dict[str, Any]]:
    keyed = df.assign(month=pd.to_datetime(df["application_date"]).dt.to_period("M").astype(str))
    rows = []
    for month, group in sorted(keyed.groupby("month"), key=lambda kv: kv[0])[-months:]:
        rows.append({"month": month, "files": int(len(group)),
                     "discount_rate": _num((group["override_direction"] == DISCOUNT).mean()),
                     "conversion": _num(group["disbursed"].astype(bool).mean())})
    return rows


def rm_record(requests: list[dict[str, Any]], offers: list[dict[str, Any]]) -> dict[str, Any]:
    """The requesting RM's own record since go-live, from the live store."""
    decided = [r for r in requests if r.get("status") in ("approved", "declined")]
    depths = [(r["recommended_rate"] - r["requested_rate"]) * 100 for r in requests
              if r.get("recommended_rate") is not None and r.get("requested_rate") is not None]
    answered = [o for o in offers if o.get("status") in ("accepted", "not_accepted")]
    return {
        "requests": len(requests),
        "approved": sum(r.get("status") == "approved" for r in requests),
        "declined": sum(r.get("status") == "declined" for r in requests),
        "pending": sum(r.get("status") == "pending" for r in requests),
        "approval_rate": _num(sum(r["status"] == "approved" for r in decided) / len(decided)) if decided else None,
        "avg_requested_discount_bps": _num(sum(depths) / len(depths), 1) if depths else None,
        "offers_sent": len(offers),
        "offers_accepted": sum(o.get("status") == "accepted" for o in offers),
        "acceptance_rate": _num(sum(o["status"] == "accepted" for o in answered) / len(answered)) if answered else None,
    }


# --------------------------------------------------------------- competitors

def competitor_landscape(analysis, city: str | None, branch_level: bool, top: int = 6) -> list[dict[str, Any]]:
    """Who quotes against us in this city, from the sales notes joined to outcomes."""
    lc = analysis.lc
    if branch_level and city:
        lc = lc[lc["city"] == city]
    files_here = int(analysis.loans.loc[analysis.loans["city"] == city, "loan_id"].nunique()) \
        if branch_level and city else analysis.total_loans
    rows = []
    for name, group in lc.groupby("competitor"):
        quoted = group[group["quote"].notna()]
        n_q = len(quoted)
        rows.append({
            "competitor": str(name), "lender_type": analysis.registry.lender_type(name),
            "files": int(group["loan_id"].nunique()),
            "share_of_files": _num(group["loan_id"].nunique() / files_here) if files_here else None,
            "median_quote": _num(quoted["quote"].median(), 2) if n_q >= 10 else None,
            "median_gap_bps": _num(quoted["gap_ours_bps"].median(), 1) if n_q >= 10 else None,
            "documented_share": (_num((quoted["evidence"] == "documented").mean()) if n_q >= MIN_RATE_N else None),
            "win_rate": _share(group.drop_duplicates("loan_id")["disbursed"])["rate"],
        })
    return sorted(rows, key=lambda r: -r["files"])[:top]


def this_competitor(analysis, name: str | None, quote: float | None, requested_rate: float,
                    city: str | None, state: str | None, evidence_seen: bool) -> dict[str, Any] | None:
    canonical = analysis.registry.resolve(name) if name else None
    if not canonical or quote is None:
        return None
    bench = analysis.benchmark(canonical, city=city, state=state)
    gap = (float(quote) - float(requested_rate)) * 100
    win = analysis.win_rate_at_gap(canonical, gap, city=city)
    p5 = bench.get("p5")
    return {
        "competitor": canonical, "quote": _num(quote, 2), "evidence_seen": evidence_seen,
        "benchmark": {k: bench.get(k) for k in ("level", "n", "median", "p5", "p10", "p90")},
        "unusually_low": bool(p5 is not None and float(quote) < p5),
        "gap_to_requested_bps": _num(gap, 1),
        "win_rate_at_gap": {"bucket": win.get("bucket"),
                            "city": (win.get("city") or {}).get("rate"),
                            "city_n": (win.get("city") or {}).get("n"),
                            "all_cities": (win.get("all_cities") or {}).get("rate"),
                            "all_n": (win.get("all_cities") or {}).get("n"),
                            "beyond_history": win.get("beyond_history", False)},
    }


# -------------------------------------------------------------------- briefing

def briefing(request: dict[str, Any], applicant: dict[str, Any], *, history: pd.DataFrame,
             propensity: np.ndarray | None = None, analysis=None,
             rm_requests: list[dict[str, Any]] | None = None,
             rm_offers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    city = applicant.get("city")
    # A stored application nests the sales context; a priced snapshot carries it flat.
    sales = {**{k: applicant.get(k) for k in ("competitor_name", "competitor_rate",
                                              "competitor_evidence_verified")},
             **(applicant.get("sales_context") or {})}
    reason = request.get("reason")
    depth = (float(request["recommended_rate"]) - float(request["requested_rate"])) * 100
    branch, where = scope(history, city)
    at_branch = where["level"] == "branch"
    branch_propensity = None
    if propensity is not None:
        branch_propensity = propensity[history.index.get_indexer(branch.index)]

    overrides = {
        "branch": override_profile(branch), "all_branches": override_profile(history),
        "by_reason": by_reason(branch),
        "effect_all_discounts": adjusted_effect(branch, branch_propensity, None, "all discounts"),
        "effect_this_reason": adjusted_effect(branch, branch_propensity, reason, str(reason)) if reason else None,
        "similar_here": similar_decisions(branch, reason, depth),
    }
    competitors = None
    if analysis is not None:
        competitors = {
            "landscape": competitor_landscape(analysis, city, at_branch),
            "this_file": this_competitor(analysis, sales.get("competitor_name"), sales.get("competitor_rate"),
                                         request["requested_rate"], city if at_branch else None,
                                         where.get("state"), bool(sales.get("competitor_evidence_verified"))),
        }
    result = {
        "request_id": request.get("request_id"), "loan_id": request.get("loan_id"),
        "requested_discount_bps": _num(depth, 1), "reason": reason, "where": where,
        "window": {"from": str(pd.to_datetime(history["application_date"]).min().date()),
                   "to": str(pd.to_datetime(history["application_date"]).max().date())},
        "overrides": overrides, "competitors": competitors,
        "performance": {"branch": performance(branch), "all_branches": performance(history),
                        "monthly": monthly(branch),
                        "rm": rm_record(rm_requests or [], rm_offers or [])},
        "caveats": [
            "History is the loan book used to train the pricing models; rates on fewer than "
            f"{MIN_RATE_N} files are withheld.",
            "Raw conversion of discounted files is confounded (people discount the hardest files); "
            "the 'paid for itself' figures compare like with like by override propensity.",
            "Competitor figures come from the sales notes joined to what then happened; they "
            "describe the book, they do not predict this customer.",
        ],
    }
    result["signals"] = signals(result)
    return result


def _pct(rate: float | None) -> str:
    return "—" if rate is None else f"{rate * 100:.0f}%"


VERDICTS = {"value creating": "paid for themselves", "value destroying": "lost value",
            "no measurable effect": "made no measurable difference"}


def _rupees(value: float) -> str:
    return f"{'−' if value < 0 else '+'}₹{abs(value):,.0f}"


def signals(b: dict[str, Any]) -> list[str]:
    """The few lines worth reading first."""
    out = []
    where = b["where"]
    name = where["city"] if where["level"] == "branch" else "All branches"
    ob, oa = b["overrides"]["branch"], b["overrides"]["all_branches"]
    if where["level"] == "branch":
        out.append(f"{name} discounts {_pct(ob['discount_rate']['rate'])} of files below the recommendation "
                   f"(all branches {_pct(oa['discount_rate']['rate'])}), median {ob['median_discount_bps'] or 0:.0f} bps.")
    effect = b["overrides"]["effect_this_reason"] or b["overrides"]["effect_all_discounts"]
    if effect and effect.get("verdict") != "insufficient evidence":
        label = (f"{str(b['reason']).replace('_', ' ')} discounts" if b["overrides"]["effect_this_reason"]
                 else "Discounts")
        out.append(f"{label[:1].upper()}{label[1:]} here {VERDICTS.get(effect['verdict'], effect['verdict'])}: "
                   f"{effect['s2d_lift_pts']:+.1f} pts conversion and {_rupees(effect['value_lift_inr'])} a file "
                   f"against comparable files that were not discounted ({effect['n_override']} discounted files).")
    similar = b["overrides"]["similar_here"]
    if similar["files"] and similar["conversion"] is not None:
        out.append(f"{similar['files']} past {str(similar['reason'] or '').replace('_', ' ')} discounts of "
                   f"{similar['window_bps'][0]:.0f}–{similar['window_bps'][1]:.0f} bps here: "
                   f"{_pct(similar['conversion'])} converted"
                   + (f", {_pct(similar['bt_out_24m'])} moved the loan away within 2 years." if similar["bt_out_24m"] is not None else "."))
    elif similar["deepest_bps"] is not None and b["requested_discount_bps"] > similar["deepest_bps"]:
        out.append(f"No past {str(similar['reason'] or '').replace('_', ' ')} discount here went this deep: "
                   f"{b['requested_discount_bps']:.0f} bps asked, the deepest was {similar['deepest_bps']:.0f} bps.")
    comp = (b.get("competitors") or {}).get("this_file")
    if comp and comp["benchmark"].get("median") is not None:
        out.append(f"{comp['competitor']} usually quotes {comp['benchmark']['median']:.2f}% "
                   f"({comp['benchmark']['level']}, {comp['benchmark']['n']} quotes); this quote is {comp['quote']:.2f}%"
                   + (", under its usual 5th percentile." if comp["unusually_low"] else "."))
    return out
