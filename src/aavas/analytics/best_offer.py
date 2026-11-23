"""Best offer: the offers a file could be given by moving four levers (rate, loan amount,
tenure, processing fee), scored on the shared engine, and the best of them by each of the
objectives in ``config/best_offer.yaml``.

Decision support for the area manager. The engine and the pricing policy set the card rate,
the floor and the band for each grade and tenure; every offer here is checked against the
policy as it stands, and only eligible offers can be best. The area manager's decision today
changes only the rate: acting on a tenure, amount or fee change needs the relationship manager
to update the file and re-price.

Search, in two passes. Pass 1 scores every (amount, tenure, fee) at the recommended rate to
learn its grade; pass 2 expands each combination into every rate from its floor to its card.
Each offer keeps its own eligibility, since the grade can move with the rate.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from ..config import best_offer_config, policy
from ..optimization import constraints
from . import branch_scenarios, offer_evaluator, risk_pricing

NOTE = ("Decision support only. The area manager's decision today changes only the rate; acting on a tenure, "
        "amount or fee change needs the relationship manager to update the file and re-price. The recommended "
        "price, floor and band are set by the engine and the policy, never by this analysis.")
REDUCED_AMOUNT = ("Offers for a smaller loan than the customer asked for are scored with the acceptance model as "
                  "it is, but it has no customers who were offered less than they asked, so their acceptance is "
                  "an extrapolation.")
TENURE_CHANGED = ("Offers with a different tenure from the one the customer asked for are scored with the acceptance model "
                  "as it is, but it has not seen customers offered a different tenure from the one they chose, so "
                  "their acceptance is an extrapolation.")
POINTS_MAX = 400
CACHE_SIZE = 16
_cache: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()
_lock = threading.Lock()      # the server runs requests on a thread pool

REASON_WORDS = {
    "grade_declined": "grade has no standard offer", "bureau_below_minimum": "bureau score below the policy minimum",
    "rate_below_floor": "rate below the lowest allowed for the grade and tenure",
    "rate_above_card": "rate above the card rate", "rate_outside_13_17": "rate outside 13% to 17%",
    "foir_over_cap": "EMIs take too much of income",
    "stressed_foir_unavailable": "no income on the file to test the EMI against",
    "ltv_over_cap": "loan too large for the property", "age_at_maturity_over_limit": "borrower too old at maturity",
    "amount_over_max_loan": "amount above the maximum loan", "tenure_outside_policy": "tenure outside policy"}
COMFORT = {"emi_over_comfort"}          # a comfort limit set in best_offer.yaml, not a policy rule


def reason_words() -> dict[str, str]:
    """Reason codes in plain words; the stress add-on is the policy's, not a number written here."""
    stress = constraints.stress_add_pct()
    return {**REASON_WORDS, "stressed_foir_over_cap": f"EMIs too heavy if rates rose {stress:g} points"}


def clear_caches() -> None:
    """Drop everything built from the config, the policy or the models."""
    with _lock:
        _cache.clear()


def comfort_words(pct: float) -> str:
    return f"EMI more than {pct:g}% above what the customer asked for"


def rupees(x: float) -> str:
    n = int(round(abs(x)))
    s = str(n)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return f"Rs {s}"


# ------------------------------------------------------------------ the config

def validate_config(cfg: dict[str, Any]) -> None:
    """Reject a best_offer.yaml the search cannot run on. Raises ValueError saying what is wrong."""
    def need(ok: bool, message: str) -> None:
        if not ok:
            raise ValueError(f"best_offer.yaml: {message}")

    def number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    need(isinstance(cfg, dict), "the file is empty")
    lv = cfg.get("levers") or {}
    for key in ("rate", "tenure", "loan_amount", "processing_fee"):
        need(isinstance(lv.get(key), dict), f"levers.{key} is missing")
    need(number(lv["rate"].get("step_pct")) and lv["rate"]["step_pct"] > 0, "levers.rate.step_pct must be above 0")
    t = lv["tenure"]
    need(number(t.get("step_months")) and t["step_months"] > 0, "levers.tenure.step_months must be above 0")
    need(number(t.get("min_months")) and number(t.get("max_months")) and 0 < t["min_months"] <= t["max_months"],
         "levers.tenure needs 0 < min_months <= max_months")
    reductions = lv["loan_amount"].get("reductions_pct") or []
    need(isinstance(reductions, list) and all(number(r) and 0 < r < 100 for r in reductions),
         "levers.loan_amount.reductions_pct must be percentages between 0 and 100")
    published = float(policy()["economics"]["processing_fee_pct"])
    levels = lv["processing_fee"].get("levels_pct")
    need(isinstance(levels, list) and levels and all(number(f) and 0 <= f <= published for f in levels),
         f"levers.processing_fee.levels_pct must each be between 0 and the published fee ({published:g}%)")
    emi = lv.get("emi")
    if emi is not None:
        need(isinstance(emi, dict) and number(emi.get("max_increase_pct")) and emi["max_increase_pct"] >= 0,
             "levers.emi.max_increase_pct must be 0 or more")
    objectives = cfg.get("objectives")
    need(isinstance(objectives, list) and objectives, "objectives is missing")
    measures, references = {"value", "acceptance"}, {"recommendation", "ask"}
    for o in objectives:
        need(isinstance(o, dict) and o.get("key") and o.get("title"), "every objective needs a key and a title")
        need(o.get("maximise") in measures, f"objective {o.get('key')}: maximise must be one of {sorted(measures)}")
        need(isinstance(o.get("at_least"), dict) and o["at_least"].get("measure") in measures,
             f"objective {o.get('key')}: at_least.measure must be one of {sorted(measures)}")
        need(o["at_least"].get("reference") in references,
             f"objective {o.get('key')}: at_least.reference must be one of {sorted(references)}")
    need(len({o["key"] for o in objectives}) == len(objectives), "objective keys must be different")
    tie = cfg.get("tie_break")
    need(isinstance(tie, list) and tie and set(tie) <= {"fewer_changes", "value", "acceptance"},
         "tie_break must list fewer_changes, value and acceptance")
    need(number(cfg.get("tie_tolerance_pct", 0.5)) and cfg.get("tie_tolerance_pct", 0.5) >= 0, "tie_tolerance_pct must be 0 or more")
    rows = (cfg.get("display") or {}).get("table_rows")
    need(isinstance(rows, int) and not isinstance(rows, bool) and rows > 0, "display.table_rows must be a whole number above 0")


# ------------------------------------------------------------------ the grid

def eligible_amount(payload: dict[str, Any], request_amount: float, rate: float) -> float | None:
    """The most the file qualifies for, when that is below the request: the pipeline's stressed
    capacity (``constraints.resize_terms``), the grade's LTV cap and the maximum loan, to the
    nearest Rs 1,000 below. None when the request itself qualifies."""
    snap = dict(payload.get("application_snapshot") or {})
    pol = policy()
    grade = str((payload.get("risk") or {}).get("risk_grade") or snap.get("risk_grade") or "C")
    snap["risk_grade"] = grade
    limits = [float(pol["constraints"]["max_loan_amount"])]
    terms = constraints.resize_terms(pd.DataFrame([snap]), rate)
    if terms.get("possible"):
        limits.append(float(terms["max_affordable_amount"]))
    elif terms.get("possible") is False:
        return 0.0
    value = pd.to_numeric(snap.get("appraised_property_value"), errors="coerce")
    caps = (pol.get("eligibility_limits") or {}).get(grade)
    if pd.notna(value) and caps:
        limits.append(float(np.floor(min(float(caps["hl_ltv_max"]), float(pol["constraints"]["max_ltv"])) * value / 1000.0) * 1000.0))
    best = min(limits)
    return float(best) if best < request_amount - 1e-9 else None


def grid(payload: dict[str, Any], request_amount: float, own_months: float, rec: float, cfg: dict[str, Any]) -> pd.DataFrame:
    """Every (amount, tenure, fee) combination, with what each lever is relative to the file's own."""
    pol = policy()
    published = float(pol["economics"]["processing_fee_pct"])
    lv = cfg["levers"]
    t = lv["tenure"]
    lo = max(int(t["min_months"]), int(pol["constraints"]["min_tenure_months"]))
    hi = min(int(t["max_months"]), int(pol["constraints"]["max_tenure_months"]))
    tenures = set(range(lo, hi + 1, int(t["step_months"])))
    if t.get("include_requested", True):
        tenures.add(int(round(own_months)))
    amounts: list[tuple[float, str]] = [(request_amount, "requested")]
    cap = eligible_amount(payload, request_amount, rec)
    if cap is not None and cap > 0:
        amounts.append((cap, "eligible_maximum"))
    for pct in lv["loan_amount"].get("reductions_pct") or []:
        amounts.append((float(np.floor(request_amount * (1 - float(pct) / 100.0) / 1000.0) * 1000.0), f"reduced_{int(pct)}"))
    seen, unique = set(), []
    for amount, basis in amounts:
        amount = min(amount, request_amount)               # never above the request
        if amount > 0 and round(amount) not in seen:
            seen.add(round(amount))
            unique.append((amount, basis))
    fees = sorted({min(float(f), published) for f in lv["processing_fee"]["levels_pct"]}, reverse=True)
    rows = [{"loan_amount": a, "amount_basis": b, "tenure_months": m, "fee_pct": f}
            for a, b in unique for m in sorted(tenures) for f in fees]
    return pd.DataFrame(rows)


def rate_ladder(floor: float, card: float, step: float) -> np.ndarray:
    ref = policy()["reference"]
    lo, hi = max(floor, float(ref["card_rate_floor"])), min(card, float(ref["card_rate_ceiling"]))
    if np.isnan(lo) or np.isnan(hi) or hi < lo:
        return np.array([])
    return np.unique(np.round(np.arange(lo, hi + 1e-9, step), 2))


# ------------------------------------------------------------------ selection

def changes(row: pd.Series, own: dict[str, float], rec: float) -> int:
    return int(abs(row["rate"] - rec) > 0.004) + int(round(row["tenure_months"]) != round(own["tenure_months"])) \
        + int(abs(row["fee_pct"] - own["fee_pct"]) > 1e-9) + int(abs(row["loan_amount"] - own["loan_amount"]) > 0.5)


def pick(offers: pd.DataFrame, objective: dict[str, Any], refs: dict[str, pd.Series], tie: list[str],
         tolerance_pct: float) -> tuple[pd.Series | None, str | None]:
    """The eligible offer that maximises the objective's measure subject to its at_least constraint;
    among offers within the tie tolerance of the best, the tie-break order decides."""
    measure = {"acceptance": "acceptance", "value": "value"}
    top = measure[objective["maximise"]]
    need = objective["at_least"]
    ref_row = refs[need["reference"]]
    floor_value = float(ref_row[measure[need["measure"]]])
    pool = offers[offers["eligible"] & offers[top].notna() & (offers[measure[need["measure"]]] >= floor_value - 1e-9)]
    if pool.empty:
        return None, (f"No eligible offer has {need['measure'].replace('value', 'risk-adjusted value')} at least as good as "
                      f"the {need['reference']} ({'%.0f%%' % (floor_value * 100) if need['measure'] == 'acceptance' else rupees(floor_value)}).")
    best = float(pool[top].max())
    near = pool[pool[top] >= best - abs(best) * tolerance_pct / 100.0]
    keys = {"fewer_changes": near["changes"], "value": -near["value"], "acceptance": -near["acceptance"]}
    order = near.assign(**{f"_{k}": keys[k] for k in tie}).sort_values([f"_{k}" for k in tie]) if tie else near
    return order.iloc[0], None


def pareto(offers: pd.DataFrame) -> pd.DataFrame:
    ordered = offers.sort_values(["acceptance", "value"], ascending=[False, False])
    keep, best = [], -np.inf
    for idx, v in ordered["value"].items():
        if v > best + 1e-9:
            keep.append(idx)
            best = v
    return ordered.loc[keep].sort_values("acceptance")


def _lever_view(row: pd.Series) -> dict[str, Any]:
    return {"rate": float(row["rate"]), "loan_amount": float(row["loan_amount"]), "tenure_months": int(row["tenure_months"]),
            "fee_pct": float(row["fee_pct"]), "emi": None if pd.isna(row["emi"]) else int(row["emi"]),
            "apr_pct": float(row["apr_pct"]), "risk_grade": str(row["risk_grade"]), "pd_annual": float(row["pd_annual"]),
            "acceptance": None if pd.isna(row["acceptance"]) else round(float(row["acceptance"]), 4),
            "value_inr": None if pd.isna(row["value"]) else int(round(row["value"])),
            "eligible": bool(row["eligible"]), "failures": list(row["failures"]),
            "amount_reduced": bool(row.get("amount_reduced", False)), "tenure_changed": bool(row.get("tenure_changed", False))}


def sentence(row: pd.Series, own: dict[str, float], rec: float, ask_row: pd.Series, ask: float) -> str:
    acts = []
    if abs(row["rate"] - rec) <= 0.004:
        acts.append(f"Keep {rec:.2f}%")
    else:
        acts.append(f"Price at {row['rate']:.2f}% instead of {rec:.2f}%")
    if round(row["tenure_months"]) != round(own["tenure_months"]):
        acts.append(f"set the tenure to {int(row['tenure_months'])} months (from {int(own['tenure_months'])})")
    if abs(row["fee_pct"] - own["fee_pct"]) > 1e-9:
        acts.append("waive the processing fee" if row["fee_pct"] <= 1e-9 else f"charge {row['fee_pct']:.2f}% processing fee instead of {own['fee_pct']:.2f}%")
    if abs(row["loan_amount"] - own["loan_amount"]) > 0.5:
        acts.append(f"sanction {rupees(row['loan_amount'])} instead of {rupees(own['loan_amount'])}")
    head = acts[0] + ("" if len(acts) == 1 else (" and " + ", ".join(acts[1:]) if len(acts) == 2 else ", " + ", ".join(acts[1:-1]) + " and " + acts[-1]))
    if len(acts) > 1 and abs(row["rate"] - rec) <= 0.004:
        head += f" instead of cutting to {ask:.2f}%"
    delta = row["value"] - ask_row["value"]
    more = f"{rupees(delta)} more value than the ask" if delta > 0.5 else f"{rupees(-delta)} less value than the ask" if delta < -0.5 else "about the same value as the ask"
    if len(acts) == 1 and abs(row["rate"] - rec) <= 0.004:
        head = f"Keep {rec:.2f}% as recommended"
    return f"{head}: {row['acceptance'] * 100:.0f}% likely to accept ({ask_row['acceptance'] * 100:.0f}% at the ask), {more}."


# ------------------------------------------------------------------ the analysis

def _bundle_identity(bundle: Any) -> list[Any]:
    """Which models priced this: whether each is present and, when it has a card, its version and training time."""
    out = []
    for name in ("risk", "conversion", "prepayment"):
        model = getattr(bundle, name, None)
        card = getattr(model, "card", None)
        out.append([name, model is not None, getattr(card, "version", None), getattr(card, "trained_at", None)])
    return out


def _fingerprint(request: dict[str, Any], payload: dict[str, Any], cfg: dict[str, Any], bundle: Any) -> str:
    keep = {k: payload.get(k) for k in ("application_snapshot", "risk", "envelope", "recommendation")}
    blob = json.dumps([keep, request.get("requested_rate"), request.get("recommended_rate"), request.get("loan_amount"), cfg,
                       policy(), _bundle_identity(bundle)], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def best_offers(request: dict[str, Any], payload: dict[str, Any], room: dict[str, Any] | None = None) -> dict[str, Any]:
    from ..governance import model_registry

    cfg = best_offer_config()
    bundle = model_registry.load_bundle()
    key = (str(request.get("request_id")), _fingerprint(request, payload, cfg, bundle))
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    out = _compute(request, payload, cfg, bundle)
    with _lock:
        _cache[key] = out
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def _compute(request: dict[str, Any], payload: dict[str, Any], cfg: dict[str, Any], bundle: Any) -> dict[str, Any]:
    snap = dict(payload.get("application_snapshot") or {})
    rec = float((payload.get("recommendation") or {}).get("recommended_rate") or request["recommended_rate"])
    ask = float(request["requested_rate"])
    own = offer_evaluator.file_defaults(payload)
    loan = float(request.get("loan_amount") or own["loan_amount"])
    own = {**own, "loan_amount": loan}
    limitations: list[str] = []
    step = float(cfg["levers"]["rate"]["step_pct"])

    combos = grid(payload, loan, own["tenure_months"], rec, cfg)
    # pass 1: each combination at the recommended rate, for its grade, card and floor
    mode = offer_evaluator.foir_mode(snap)                      # found once, used by every batch below
    first = offer_evaluator.evaluate_offers(payload, combos.assign(rate=rec), bundle=bundle, mode=mode)
    limitations.extend(t for t in first.attrs.get("limitations", []) if t not in limitations)
    parts = []
    skipped = 0
    for i, c in combos.iterrows():
        ladder = rate_ladder(float(first.at[i, "floor_rate"]), float(first.at[i, "card_rate"]), step)
        if len(ladder) == 0:
            skipped += 1
            continue
        parts.append(pd.DataFrame({"rate": ladder, "loan_amount": c["loan_amount"], "tenure_months": c["tenure_months"],
                                   "fee_pct": c["fee_pct"], "amount_basis": c["amount_basis"]}))
    sweep = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["rate", "loan_amount", "tenure_months", "fee_pct", "amount_basis"])

    refs_in = pd.DataFrame({"rate": [rec, ask], "loan_amount": loan, "tenure_months": own["tenure_months"],
                            "fee_pct": own["fee_pct"], "amount_basis": "requested"})
    lever_rows = _lever_inputs(combos, first, rec, own, loan)
    everything = pd.concat([sweep, refs_in, lever_rows.drop(columns=["lever", "label"])], ignore_index=True)
    scored = offer_evaluator.evaluate_offers(payload, everything.drop(columns=["amount_basis"]), bundle=bundle, mode=mode)
    limitations.extend(t for t in scored.attrs.get("limitations", []) if t not in limitations)
    scored["amount_basis"] = everything["amount_basis"].to_numpy()

    shift, problem = risk_pricing._shift(snap.get("city"))
    if problem:
        limitations.append(problem)
    p = scored["acceptance_model"].to_numpy(float)
    p_branch = branch_scenarios._sigmoid(branch_scenarios._logit(p) + shift) if not np.isnan(p).all() else p
    scored["acceptance"] = p_branch
    with np.errstate(divide="ignore", invalid="ignore"):
        scored["value"] = np.where(p > 0, scored["value_inr"].to_numpy(float) * p_branch / p, np.nan)
    scored["amount_reduced"] = scored["loan_amount"] < loan - 0.5
    scored["tenure_changed"] = scored["tenure_months"].round() != round(own["tenure_months"])
    scored["changes"] = [changes(r, own, rec) for _, r in scored.iterrows()]
    comfort_pct = ((cfg["levers"].get("emi") or {}).get("max_increase_pct"))
    n_sweep = len(sweep)
    if comfort_pct is not None:
        own_emi = float(scored["emi"].iloc[n_sweep])              # the file's own EMI at the recommendation
        over = scored["emi"].to_numpy(float) > own_emi * (1 + float(comfort_pct) / 100.0) + 0.5
        scored["failures"] = [list(f) + (["emi_over_comfort"] if o else []) for f, o in zip(scored["failures"], over)]
        scored["eligible"] = scored["eligible"] & ~over

    n_refs = 2
    sweep_scored = scored.iloc[:n_sweep].copy()
    rec_row, ask_row = scored.iloc[n_sweep], scored.iloc[n_sweep + 1]
    lever_scored = scored.iloc[n_sweep + n_refs:].reset_index(drop=True)

    refs = {"recommendation": rec_row, "ask": ask_row}
    references = {"recommendation": _lever_view(rec_row), "ask": _lever_view(ask_row)}
    eligible = sweep_scored[sweep_scored["eligible"]]
    if sweep_scored["amount_reduced"].any():
        limitations.append(REDUCED_AMOUNT)
    if sweep_scored["tenure_changed"].any():
        limitations.append(TENURE_CHANGED)
    if skipped:
        limitations.append(f"{skipped} combination(s) of amount, tenure and fee have no standard offer at the file's grade "
                           "and were not searched.")
    from collections import Counter
    reasons = Counter(code for fails in sweep_scored["failures"] for code in fails)
    words = {**reason_words(), "emi_over_comfort": comfort_words(float(comfort_pct)) if comfort_pct is not None else "EMI above the comfort limit"}

    best: list[dict[str, Any]] = []
    tolerance = float(cfg.get("tie_tolerance_pct", 0.5))      # the yaml sets it; 0.5 only if it is missing
    no_model = scored["acceptance"].isna().all()
    for objective in cfg["objectives"]:
        entry = {"key": objective["key"], "title": objective["title"], "maximise": objective["maximise"],
                 "at_least": objective["at_least"], "offer": None, "reason": None}
        if no_model:
            entry["reason"] = "The acceptance model is not available, so offers cannot be compared."
        else:
            chosen, why = pick(eligible, objective, refs, list(cfg.get("tie_break") or []), tolerance) if len(eligible) else (None, "No eligible offer was found.")
            if chosen is None:
                entry["reason"] = why
            else:
                view = _lever_view(chosen)
                entry["offer"] = {**view, "amount_basis": str(chosen["amount_basis"]), "changes_count": int(chosen["changes"]),
                                  "changes": {"rate_bps": round((chosen["rate"] - rec) * 100), "tenure_months": int(chosen["tenure_months"] - own["tenure_months"]),
                                              "fee_pct": round(float(chosen["fee_pct"] - own["fee_pct"]), 4),
                                              "loan_amount_inr": int(round(chosen["loan_amount"] - loan))},
                                  "vs_ask": {"acceptance_pts": round((chosen["acceptance"] - ask_row["acceptance"]) * 100, 1),
                                             "value_inr": int(round(chosen["value"] - ask_row["value"]))},
                                  "vs_recommendation": {"acceptance_pts": round((chosen["acceptance"] - rec_row["acceptance"]) * 100, 1),
                                                        "value_inr": int(round(chosen["value"] - rec_row["value"]))},
                                  "sentence": sentence(chosen, own, rec, ask_row, ask)}
        best.append(entry)

    levers = []
    for i, meta in lever_rows.reset_index(drop=True).iterrows():
        row = lever_scored.iloc[i]
        levers.append({"lever": meta["lever"], "label": meta["label"], **_lever_view(row),
                       "acceptance_change_pts": None if pd.isna(row["acceptance"]) else round((row["acceptance"] - rec_row["acceptance"]) * 100, 1),
                       "value_change_inr": None if pd.isna(row["value"]) else int(round(row["value"] - rec_row["value"]))})

    frontier = pareto(eligible.dropna(subset=["acceptance", "value"])) if len(eligible) else eligible
    pts = eligible.dropna(subset=["acceptance", "value"]).sort_values("acceptance")
    if len(pts) > POINTS_MAX:
        pts = pts.iloc[np.unique(np.linspace(0, len(pts) - 1, POINTS_MAX).round().astype(int))]
    short = lambda df: [{"rate": float(r["rate"]), "loan_amount": float(r["loan_amount"]), "tenure_months": int(r["tenure_months"]),      # noqa: E731
                         "fee_pct": float(r["fee_pct"]), "acceptance": round(float(r["acceptance"]), 4),
                         "value_inr": int(round(r["value"])), "amount_reduced": bool(r["amount_reduced"]),
                         "tenure_changed": bool(r["tenure_changed"])} for _, r in df.iterrows()]
    table = eligible.dropna(subset=["value"]).sort_values("value", ascending=False).head(int(cfg["display"]["table_rows"]))
    return {
        "request_id": request.get("request_id"), "config_version": cfg.get("best_offer_config_version"),
        "own": {"loan_amount": loan, "tenure_months": int(own["tenure_months"]), "fee_pct": own["fee_pct"]},
        "references": references, "best": best, "levers": levers,
        "frontier": short(frontier) if cfg["display"].get("frontier", True) and len(frontier) else [],
        "points": short(pts), "table": [{**_lever_view(r), "amount_basis": str(r["amount_basis"])} for _, r in table.iterrows()],
        "ineligible": [{"code": c, "words": words.get(c, c), "offers": int(n),
                       "kind": "comfort" if c in COMFORT else "policy"} for c, n in reasons.most_common()],
        "counts": {"scored": int(len(sweep_scored)), "eligible": int(len(eligible))},
        "limitations": limitations, "note": NOTE,
    }


def _lever_inputs(combos: pd.DataFrame, first: pd.DataFrame, rec: float, own: dict[str, float], loan: float) -> pd.DataFrame:
    """What each lever does on its own: one change from the recommendation reference. The floor is the one
    the first pass already found for the file's own amount, tenure and fee."""
    mine = (abs(combos["loan_amount"] - loan) < 0.5) & (combos["tenure_months"].round() == round(own["tenure_months"])) \
        & ((combos["fee_pct"] - own["fee_pct"]).abs() < 1e-9)
    floor = float(first.loc[mine.to_numpy(), "floor_rate"].iloc[0]) if mine.any() else float("nan")
    rows = []
    if np.isfinite(floor):          # a grade with no card rate has no floor, so there is no such lever
        rows.append(("rate", f"Rate to the floor, {floor:.2f}%", floor, loan, own["tenure_months"], own["fee_pct"], "requested"))
    for m in sorted(combos["tenure_months"].unique()):
        if int(m) != int(round(own["tenure_months"])):
            rows.append(("tenure", f"{int(m)} months", rec, loan, float(m), own["fee_pct"], "requested"))
    for f in sorted(combos["fee_pct"].unique(), reverse=True):
        if abs(f - own["fee_pct"]) > 1e-9:
            rows.append(("fee", "Waive the fee" if f <= 1e-9 else f"{f:.2f}% fee", rec, loan, own["tenure_months"], float(f), "requested"))
    for a, basis in combos[["loan_amount", "amount_basis"]].drop_duplicates().itertuples(index=False):
        if abs(a - loan) > 0.5:
            rows.append(("amount", f"{rupees(a)}", rec, float(a), own["tenure_months"], own["fee_pct"], basis))
    return pd.DataFrame(rows, columns=["lever", "label", "rate", "loan_amount", "tenure_months", "fee_pct", "amount_basis"])
