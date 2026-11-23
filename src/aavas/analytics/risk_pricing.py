"""Risk-aware pricing for one lower-rate request: what the file's risk is, where it sits in
its grade, what moves it, and what each candidate rate does to affordability, risk cost,
acceptance and value.

This is decision support for the area manager. The recommended price, the floor and the
band are set by the pricing engine and the policy, never by this analysis.

The per-rate table is the shared offer engine (``offer_evaluator.evaluate_offers``) with
the amount, tenure and fee held at the file's own values, so it shows the same numbers as
any other view built on that engine. Acceptance carries this branch's shift exactly as the
scenario table does.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ..config import model_config, policy
from ..optimization import constraints, yield_calculator
from . import offer_evaluator, portfolio

GRADES = ["A+", "A", "B", "C"]
BORDERLINE_SHARE = 0.10      # PD within 10% of the band's width of a boundary
MIN_PEERS = 30               # fewest book files to quote a percentile
STEP = 0.25

PD_NOTE = ("The risk model reads affordability at a fixed assessment rate, so the default risk does not change with the "
           "rate; payment shock is covered by the stress test.")
NOTE = ("Decision support for the area manager. The recommended price, the floor and the band are set by "
        "the pricing engine and the policy, never by this analysis.")

LABELS = {
    "verified_monthly_income": "Verified monthly income", "income_verification_haircut": "Income cut at verification",
    "sanctioned_loan_amount": "Loan amount", "appraised_property_value": "Property value", "ltv": "Loan to property value",
    "bureau_score": "Credit bureau score", "bureau_vintage_months": "Length of credit history",
    "delinquency_count_24m": "Missed payments in 2 years", "banking_behaviour_score": "Banking behaviour score",
    "foir": "Share of income going to EMIs", "tenure_years": "Loan tenure", "applicant_age": "Applicant age",
    "deviation_count": "Policy deviations", "legal_flag": "Legal issue on the property",
    "technical_flag": "Technical issue on the property", "segment": "Customer segment", "city": "City",
    "channel": "How the loan came in", "property_type": "Property type", "new_to_credit": "New to credit",
}


# ------------------------------------------------------------------ small helpers

def _grade_bounds() -> dict[str, tuple[float, float | None]]:
    """Each grade's annual-PD band: above the previous cut-off, up to its own. D has no top."""
    cuts = sorted(model_config()["risk"]["grade_cutoffs"].items(), key=lambda kv: kv[1])
    out, low = {}, 0.0
    for i, (grade, cut) in enumerate(cuts):
        out[grade] = (low, None if i == len(cuts) - 1 else float(cut))
        low = float(cut)
    return out


def _card_for(snap: dict[str, Any], grade: str, tenure_months: float, amount: float | None = None) -> float | None:
    row = dict(snap)
    row.update({"risk_grade": grade, "tenure_years": tenure_months / 12.0})
    if amount and row.get("appraised_property_value"):
        row["ltv"] = amount / float(row["appraised_property_value"])
    value = float(constraints.card_rate(pd.DataFrame([row]))[0])
    return None if np.isnan(value) else round(value, 2)


def _floor_of(card: float | None) -> float | None:
    if card is None:
        return None
    pol = policy()
    return round(max(card - pol["constraints"]["absolute_discount_floor_bps"] / 100.0,
                     pol["reference"]["card_rate_floor"]), 2)


def band(snap: dict[str, Any], risk: dict[str, Any], tenure_months: float) -> dict[str, Any]:
    """Where this file's PD sits in its grade's band, and what a one-grade move does to the card."""
    bounds = _grade_bounds()
    order = list(bounds)
    grade = str(risk.get("risk_grade") or "C")
    pd_value = risk.get("pd")
    low, high = bounds.get(grade, (0.0, None))
    i = order.index(grade) if grade in order else None
    better = order[i - 1] if i else None
    worse = order[i + 1] if i is not None and i + 1 < len(order) else None
    position = to_better = to_worse = None
    borderline = False
    if pd_value is not None:
        pd_value = float(pd_value)
        if high is not None and high > low:
            position = round(min(max((pd_value - low) / (high - low), 0.0), 1.0), 4)
            width = high - low
            near_lower = better is not None and (pd_value - low) <= BORDERLINE_SHARE * width
            near_upper = worse is not None and (high - pd_value) <= BORDERLINE_SHARE * width
            borderline = bool(near_lower or near_upper)
        to_better = round(pd_value - low, 5) if better else None
        to_worse = round(high - pd_value, 5) if high is not None and worse else None
    card_now = _card_for(snap, grade, tenure_months)
    card_better = _card_for(snap, better, tenure_months) if better else None
    card_worse = _card_for(snap, worse, tenure_months) if worse else None
    return {
        "grade": grade, "pd_low": round(low, 5), "pd_high": None if high is None else round(high, 5),
        "position_in_band": position, "borderline": borderline,
        "better_grade": better, "worse_grade": worse,
        "pd_distance_to_better_grade": to_better, "pd_distance_to_worse_grade": to_worse,
        "card_rate_this_grade": card_now, "card_rate_better_grade": card_better, "card_rate_worse_grade": card_worse,
        "card_change_if_better_pts": None if card_better is None or card_now is None else round(card_better - card_now, 2),
        "card_change_if_worse_pts": None if card_worse is None or card_now is None else round(card_worse - card_now, 2),
        "tenure_months": int(round(tenure_months)),
    }


# ------------------------------------------------------------------ the book, scored once

def _trained_at(model: Any) -> str:
    return str(getattr(getattr(model, "card", None), "trained_at", None))


@lru_cache(maxsize=2)
def _book(trained_at: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The disbursed book with the risk model's PD, grade and expected loss, and the typical
    value of each model feature. Scored once per process (per trained artefact)."""
    from ..governance import model_registry

    model = model_registry.load_bundle().risk
    book = portfolio.frame()
    book = book[book["disbursed"].astype(bool)].copy()
    margin_base = yield_calculator.breakeven_rate(0.0)
    if model is not None:
        scored = model.assess(book)
        book["_grade"] = scored["risk_grade"].astype(str).to_numpy()
        book["_pd"] = scored["pd"].to_numpy(float)
        book["_el"] = scored["expected_loss_pct"].to_numpy(float)
        book["_margin"] = book["sanctioned_rate"].to_numpy(float) - (margin_base + book["_el"].to_numpy(float))
    else:
        book["_grade"] = book["risk_grade"].astype(str)
        book["_pd"] = np.nan
        book["_el"] = np.nan
        book["_margin"] = np.nan
    typical: dict[str, Any] = {}
    for feature in getattr(getattr(model, "card", None), "features", None) or []:
        if feature not in book.columns:
            continue
        col = book[feature]
        if col.dtype == bool or not pd.api.types.is_numeric_dtype(col) or col.nunique() <= 2:
            mode = col.mode()
            typical[feature] = mode.iloc[0] if len(mode) else None
        else:
            typical[feature] = float(col.median())
    return book, typical


def _tenure_band(months: float) -> tuple[str, float, float]:
    bands = sorted(policy().get("tenure_loading_bps") or [], key=lambda b: b["max_months"])
    low = 0.0
    for b in bands:
        if months <= b["max_months"]:
            first = low == 0.0
            label = f"up to {int(b['max_months'])} months" if first else f"{int(low) + 1} to {int(b['max_months'])} months"
            return label, low, float(b["max_months"])
        low = float(b["max_months"])
    return f"over {int(low)} months", low, 1e9


def percentile(rates: np.ndarray, rate: float | None) -> float | None:
    """Share of book rates at or below this rate, in per cent."""
    rates = np.asarray(rates, dtype=float)
    rates = rates[~np.isnan(rates)]
    if rate is None or len(rates) == 0:
        return None
    return round(float((rates <= rate + 1e-9).mean() * 100), 1)


def peers(snap: dict[str, Any], risk: dict[str, Any], tenure_months: float, ask: float, rec: float,
          book: pd.DataFrame | None) -> dict[str, Any]:
    if book is None or book.empty:
        return {"grades": [], "this_file": None}
    horizon = float(model_config()["risk"]["pd_horizon_years"])
    bounds = _grade_bounds()
    grade = str(risk.get("risk_grade") or "C")
    rows = []
    for g in GRADES:
        part = book[book["_grade"] == g]
        rate = part["sanctioned_rate"].to_numpy(float)
        card = _card_for(snap, g, tenure_months)
        low, high = bounds[g]
        default = float(part["defaulted"].astype(float).mean()) if len(part) else None
        rows.append({
            "grade": g, "files": int(len(part)), "pd_low": round(low, 5), "pd_high": high,
            "median_pd": None if part["_pd"].isna().all() else round(float(part["_pd"].median()), 5),
            "card_rate": card, "floor_rate": _floor_of(card),
            "median_rate": round(float(np.median(rate)), 2) if len(rate) else None,
            "rate_p25": round(float(np.percentile(rate, 25)), 2) if len(rate) else None,
            "rate_p75": round(float(np.percentile(rate, 75)), 2) if len(rate) else None,
            "median_margin_over_risk_cost": None if part["_margin"].isna().all() else round(float(part["_margin"].median()), 2),
            "realised_default_rate": None if default is None else round(default, 4),
            "realised_default_rate_per_year": None if default is None else round(default / horizon, 4),
            "this_file": g == grade,
        })
    same = book[book["_grade"] == grade]
    label, lo, hi = _tenure_band(tenure_months)
    months = same["tenure_years"].to_numpy(float) * 12
    in_band = same[(months > lo) & (months <= hi)]
    enough = len(in_band) >= MIN_PEERS
    here = {
        "grade": grade, "files_same_grade": int(len(same)),
        "ask_percentile_same_grade": percentile(same["sanctioned_rate"].to_numpy(float), ask),
        "recommended_percentile_same_grade": percentile(same["sanctioned_rate"].to_numpy(float), rec),
        "tenure_band": label, "files_same_grade_and_tenure": int(len(in_band)),
        "ask_percentile_same_grade_and_tenure": percentile(in_band["sanctioned_rate"].to_numpy(float), ask) if enough else None,
        "recommended_percentile_same_grade_and_tenure": percentile(in_band["sanctioned_rate"].to_numpy(float), rec) if enough else None,
        "median_margin_over_risk_cost": next((r["median_margin_over_risk_cost"] for r in rows if r["grade"] == grade), None),
    }
    return {"grades": rows, "this_file": here, "book_files": int(len(book)),
            "basis": "Disbursed loans in the book, graded by the same risk model; realised default rate is "
                     "defaults over the book's whole window (about four years), also shown per year."}


def clear_caches() -> None:
    """Drop the scored book: it is built from the models and the policy."""
    _book.cache_clear()


# ------------------------------------------------------------------ drivers

def _fmt(feature: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if feature in ("ltv", "foir", "income_verification_haircut"):
        return f"{float(value) * 100:.0f}%"
    if feature in ("verified_monthly_income", "sanctioned_loan_amount", "appraised_property_value"):
        return f"₹{float(value):,.0f}"
    if feature == "tenure_years":
        return f"{float(value):g} years"
    if feature == "bureau_vintage_months":
        return f"{float(value):.0f} months"
    if feature in ("legal_flag", "technical_flag", "new_to_credit"):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):g}" if float(value) == int(float(value)) else f"{float(value):.1f}"
    return str(value).replace("_", " ")


def drivers(model: Any, base: pd.DataFrame, typical: dict[str, Any], top: int = 5) -> list[dict[str, Any]]:
    """What moves this file's PD against a typical borrower: each feature set to the book's
    typical value (median, or mode for categories) one at a time, and the file rescored."""
    features = [f for f in (getattr(getattr(model, "card", None), "features", None) or [])
                if f in base.columns and f in typical and typical[f] is not None]
    if not features:
        return []
    frames = [base.iloc[[0]]]
    for f in features:
        row = base.iloc[[0]].copy()
        row[f] = typical[f]
        frames.append(row)
    scores = model.pd(pd.concat(frames, ignore_index=True))
    this_pd = float(scores[0])
    out = []
    for f, at_typical in zip(features, scores[1:]):
        change = (this_pd - float(at_typical)) * 100       # points of annual PD this value adds
        out.append({"feature": f, "label": LABELS.get(f, f.replace("_", " ")),
                    "this_value": _fmt(f, base.iloc[0][f]), "typical_value": _fmt(f, typical[f]),
                    "pd_change_pts": round(change, 3),
                    "direction": "raises risk" if change > 0.0005 else "lowers risk" if change < -0.0005 else "no effect"})
    out.sort(key=lambda d: abs(d["pd_change_pts"]), reverse=True)
    return [d for d in out if d["direction"] != "no effect"][:top]


# ------------------------------------------------------------------ the rates

def rate_grid(env: dict[str, Any], asked: float, rec: float, card: float | None) -> list[tuple[float, list[str]]]:
    """The rates shown: the floor, the ask, the recommendation and the card exactly as they are (an ask
    below 13% is scored at its own rate and flagged outside the band), plus steps of 0.25 between the
    lowest and highest of those, which stay inside the policy band and the envelope."""
    ref = policy()["reference"]
    lo_band, hi_band = float(ref["card_rate_floor"]), float(ref["card_rate_ceiling"])
    env_lo = max(lo_band, float(env.get("min_rate") or lo_band))
    env_hi = min(hi_band, float(env.get("max_rate") or hi_band))
    marks: dict[float, list[str]] = {}
    for name, rate in (("floor", env.get("min_rate")), ("ask", asked), ("recommended", rec), ("card", card)):
        if rate is None or (isinstance(rate, float) and np.isnan(rate)):
            continue
        marks.setdefault(round(float(rate), 2), []).append(name)      # a marked rate is shown and scored as it is
    if not marks:
        return []
    lo, hi = min(marks), max(marks)
    steps = np.round(np.arange(np.ceil(lo / STEP) * STEP, hi + 1e-9, STEP), 2)
    out = [(r, m) for r, m in marks.items()]
    for r in steps:
        r = float(r)
        if env_lo - 1e-9 <= r <= env_hi + 1e-9 and all(abs(r - k) >= 0.05 for k in marks):
            out.append((r, []))
    return sorted(out)


def _shift(city: str | None) -> tuple[float, str | None]:
    from . import branch_scenarios

    try:
        check = branch_scenarios.calibration(city)
        used = check["branch"] or check["all_branches"]
        return used["shift"] - check["all_branches"]["shift"], None
    except Exception as exc:                                              # noqa: BLE001
        return 0.0, f"The branch's own acceptance history could not be read ({exc}); the general model is shown."


def rates_table(payload: dict[str, Any], grid: list[tuple[float, list[str]]], city: str | None,
                loan: float, months: float) -> tuple[list[dict[str, Any]], list[str]]:
    from . import branch_scenarios

    if not grid:
        return [], []
    offers = pd.DataFrame({"rate": [r for r, _ in grid], "loan_amount": loan, "tenure_months": months})
    scored = offer_evaluator.evaluate_offers(payload, offers)
    limits = [t for t in scored.attrs.get("limitations", []) if t != offer_evaluator.FEE_ASSUMPTION]
    shift, problem = _shift(city)
    if problem:
        limits.append(problem)
    p = scored["acceptance_model"].to_numpy(float)
    p_branch = branch_scenarios._sigmoid(branch_scenarios._logit(p) + shift) if not np.isnan(p).all() else p
    rows = []
    for i, (rate, marks) in enumerate(grid):
        r = scored.iloc[i]
        value = r["value_inr"]
        value_branch = value * p_branch[i] / p[i] if p[i] and not np.isnan(value) else np.nan
        cap = float(r["foir_cap"])
        rows.append({
            "rate": round(float(rate), 2), "marks": marks, "emi": None if np.isnan(r["emi"]) else int(r["emi"]),
            "foir": float(r["foir"]), "stressed_foir": float(r["stressed_foir"]), "foir_cap": cap,
            "stressed_foir_ok": bool(r["stressed_foir_ok"]),
            "pd_annual": float(r["pd_annual"]), "risk_grade_at_rate": str(r["risk_grade"]),
            "expected_loss_pct": float(r["expected_loss_pct"]), "breakeven_rate": float(r["breakeven_rate"]),
            "margin_over_risk_cost": float(r["margin_over_risk_cost"]),
            "acceptance_model": None if np.isnan(p[i]) else round(float(p[i]), 4),
            "acceptance_branch": None if np.isnan(p_branch[i]) else round(float(p_branch[i]), 4),
            "value_model_inr": None if np.isnan(value) else int(round(value)),
            "value_branch_inr": None if np.isnan(value_branch) else int(round(value_branch)),
            "apr_pct": float(r["apr_pct"]), "in_band": bool(r["in_band"]), "failures": list(r["failures"]),
        })
    return rows, limits


# ------------------------------------------------------------------ the verdict

def verdict(*, ask: float, floor: float | None, grade: str, band_info: dict[str, Any], rows: list[dict[str, Any]],
            ask_row: dict[str, Any] | None, rec_row: dict[str, Any] | None,
            book_margin: float | None, recommended: float) -> dict[str, Any]:
    """One or two plain sentences from fixed rules, in this order: ask under the risk-based
    floor; stressed FOIR failing at a shown rate; PD near a grade boundary; the ask's margin
    over risk cost against the same-grade book; """
    points: list[tuple[str, str]] = []                      # (tone, sentence)
    if floor is not None and ask < floor - 1e-9:
        points.append(("bad", f"The ask of {ask:.2f}% is {floor - ask:.2f} points below the risk-based floor of "
                              f"{floor:.2f}% for a grade {grade} file at this tenure, so policy does not allow it."))
    failing = [r for r in rows if not r["stressed_foir_ok"]]
    if failing:
        first = min(failing, key=lambda r: r["rate"])
        points.append(("bad", (f"At {first['rate']:.2f}% and above, EMIs would take {first['stressed_foir'] * 100:.0f}% of income "
                              f"if rates rose {constraints.stress_add_pct():g} points, over the {first['foir_cap'] * 100:.0f}% limit for grade {grade}."
                              if first["stressed_foir"] == first["stressed_foir"] else
                              f"At {first['rate']:.2f}% the stress test cannot be run (no income on the file), so the rate is not allowed.")))
    if band_info.get("borderline"):
        near_better = (band_info.get("pd_distance_to_better_grade") is not None and
                       (band_info.get("pd_distance_to_worse_grade") is None or
                        band_info["pd_distance_to_better_grade"] <= band_info["pd_distance_to_worse_grade"]))
        other = band_info["better_grade"] if near_better else band_info["worse_grade"]
        move = band_info["card_change_if_better_pts" if near_better else "card_change_if_worse_pts"]
        tail = f", which would move the card rate by {move:+.2f} points" if move is not None else ""
        points.append(("warn", f"This file's risk sits close to the grade {other} boundary{tail}."))
    margin_sentence = None
    if ask_row is not None and book_margin is not None:
        m = ask_row["margin_over_risk_cost"]
        gap = m - book_margin
        how = ("in line with" if abs(gap) < 0.15 else "above" if gap > 0 else "below")
        margin_sentence = (f"At the ask the margin over risk cost is {m:.2f} points, {how} the {book_margin:.2f} points "
                           f"the book earned on a typical grade {grade} loan.")
        points.append(("warn" if gap <= -0.15 else "good" if gap >= 0.15 else "info", margin_sentence))
    order = {"bad": 0, "warn": 1, "good": 2, "info": 3}
    chosen = sorted(points, key=lambda p: order[p[0]])[:2]
    chosen = chosen if chosen else [("info", "No risk concern stands out at the ask.")]
    return {"tone": "bad" if any(t == "bad" for t, _ in chosen) else "warn" if any(t == "warn" for t, _ in chosen) else "good"
            if any(t == "good" for t, _ in chosen) else "info",
            "text": " ".join(s for _, s in chosen), "points": [s for _, s in points]}


# ------------------------------------------------------------------ the analysis

def risk_pricing(request: dict[str, Any], payload: dict[str, Any], room: dict[str, Any] | None = None) -> dict[str, Any]:
    from ..governance import model_registry

    snap = dict(payload.get("application_snapshot") or {})
    risk = dict(payload.get("risk") or {})
    env = dict(payload.get("envelope") or {})
    rec = float((payload.get("recommendation") or {}).get("recommended_rate") or request["recommended_rate"])
    asked = float(request["requested_rate"])
    defaults = offer_evaluator.file_defaults(payload)
    loan = float(request.get("loan_amount") or defaults["loan_amount"])
    months = float(defaults["tenure_months"])
    years = months / 12.0
    limitations: list[str] = []

    try:
        bundle = model_registry.load_bundle()
        model = getattr(bundle, "risk", None)
    except Exception:                                                     # noqa: BLE001
        bundle, model = None, None
    if model is None:
        limitations.append("The risk model is not available, so the risk shown is the one fixed on the decision: "
                           "it is not rescored at each rate, and the drivers are not shown.")

    economics = (payload.get("recommendation") or {}).get("economics") or {}
    life = float(economics.get("expected_life_years") or 0.0)
    el_pct = float(risk.get("expected_loss_pct") or 0.0)
    el_rupees = None
    if life and loan:
        avg = float(yield_calculator.average_outstanding_factor(life, years))
        el_rupees = int(round(el_pct / 100.0 * loan * avg * life))
    score = {
        "risk_grade": risk.get("risk_grade"), "pd_annual": risk.get("pd"), "pd_lifetime": risk.get("lifetime_pd"),
        "lgd": risk.get("lgd"), "expected_loss_pct_per_year": risk.get("expected_loss_pct"),
        "expected_loss_inr_over_life": el_rupees, "expected_life_years": life or None,
        "breakeven_rate": risk.get("breakeven_rate"), "source": risk.get("source"),
        "lgd_source": risk.get("lgd_source"),
    }
    band_info = band(snap, risk, months)

    book = typical = None
    try:
        book, typical = _book(_trained_at(model))
    except Exception as exc:                                              # noqa: BLE001
        limitations.append(f"The loan book could not be scored ({exc}); peer comparison and drivers are not shown.")

    base = None
    driver_rows: list[dict[str, Any]] = []
    if model is not None and typical:
        try:
            base = offer_evaluator.model_frame(snap, pd.DataFrame({"rate": [rec], "loan_amount": [loan], "tenure_months": [months]}))
            driver_rows = drivers(model, base, typical)
        except Exception as exc:                                          # noqa: BLE001
            limitations.append(f"The drivers could not be worked out ({exc}).")

    card = env.get("card_rate")
    grid = rate_grid(env, asked, rec, None if card is None else float(card))
    rows, row_limits = rates_table(payload, grid, snap.get("city"), loan, months)
    limitations.extend(row_limits)
    at = lambda name: next((r for r in rows if name in r["marks"]), None)              # noqa: E731
    rec_row, ask_row = at("recommended"), at("ask")
    if rec_row and risk.get("pd") is not None and model is not None and abs(rec_row["pd_annual"] - float(risk["pd"])) > 0.0005:
        limitations.append(
            f"The risk rescored here ({rec_row['pd_annual'] * 100:.2f}% a year) differs from the figure on the decision "
            f"({float(risk['pd']) * 100:.2f}%): the decision was scored on a different version of the file.")

    peer = peers(snap, risk, months, asked, rec, book)
    here = peer.get("this_file") or {}
    floor = env.get("min_rate")
    say = verdict(ask=asked, floor=None if floor is None else float(floor), grade=str(risk.get("risk_grade")),
                  band_info=band_info, rows=rows, ask_row=ask_row, rec_row=rec_row,
                  book_margin=here.get("median_margin_over_risk_cost"), recommended=rec)
    return {
        "request_id": request.get("request_id"), "loan_amount": loan, "tenure_years": years,
        "city": snap.get("city"), "requested_rate": round(asked, 2), "recommended_rate": round(rec, 2),
        "floor_rate": None if floor is None else round(float(floor), 2),
        "card_rate": None if card is None else round(float(card), 2),
        "stress_add_pct": constraints.stress_add_pct(), "verdict": say, "pd_note": PD_NOTE, "score": score, "band": band_info, "drivers": driver_rows,
        "rates": rows, "peers": peer, "limitations": limitations, "note": NOTE,
    }
