"""One engine that scores an offer on all four levers the lender controls: the rate,
the loan amount, the tenure and the processing fee.

``evaluate_offers(payload, offers)`` takes a table of candidate offers for one file and
returns one row per offer, scored in a single batch (one pass through each model, not
one per offer). Everything is recomputed for the offer, not read off the file:

* EMI, FOIR and stressed FOIR (rate plus the policy stress add-on) against the grade's cap,
  all at the offer's own rate, used only for the eligibility checks.
* Model FOIR, the input the risk and conversion models are given: computed by the pipeline's
  own feature code (``stage2_features.build``) at the rate the pipeline used for this file,
  so only the offer's amount and tenure move it and the numbers agree with the decision and
  with the scenario table. (The pipeline's convention is to be replaced by a policy
  assessment rate; until then this matches it exactly.)
* LTV against its cap, age at maturity against the policy limit.
* PD and grade, rescored with the risk model on the offer's model FOIR, LTV, amount and tenure.
* The policy envelope for that grade and tenure (card rate from the grade x tenure grid,
  floor = card less the absolute discount floor, never below the product floor) and
  whether the rate is inside it.
* Loss given default, expected loss and the break-even rate; expected life from the
  prepayment model; acceptance from the conversion model.
* APR including the fee (the KFS calculator) and the risk-adjusted expected value in rupees.
* Every eligibility failure, as reason codes.

**The fee assumption.** No model was fitted with the processing fee in it. So the effect of
the fee on acceptance is an explicit assumption: the conversion model is scored at the
rate plus however much the offer's APR differs from the APR at the published fee, i.e. a
customer reacts to the all-in cost. At the published fee the extra is zero, so the numbers
equal the rate-only analysis. The fee is never raised above ``economics.processing_fee_pct``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import policy
from ..features import stage2_features
from ..features.feature_pipeline import first_available, numeric_column
from ..optimization import constraints, yield_calculator

FEE_ASSUMPTION = ("The processing fee is in no model. Its effect on acceptance is an assumption: the "
                  "conversion model is scored at the rate plus the difference between this offer's APR and "
                  "the APR at the published fee (customers react to the all-in cost). At the published fee "
                  "this changes nothing; the fee is never raised above the published fee.")

OFFER_COLUMNS = ["rate", "loan_amount", "tenure_months", "fee_pct"]


def apr_vector(amount: np.ndarray, rate: np.ndarray, months: np.ndarray, upfront: np.ndarray) -> np.ndarray:
    """APR on the amount actually received, for many offers at once: the monthly IRR that makes
    the level EMI (to the whole rupee, as the KFS rounds it) worth the amount less the upfront fee,
    times 12. Same definition as ``intake.kfs.apr``, which stays the source of truth (a test holds
    this to it); the KFS's last instalment, which clears the balance, differs by a few rupees."""
    amount, rate, n, upfront = (np.asarray(a, dtype=float) for a in (amount, rate, months, upfront))
    i = rate / 1200.0
    with np.errstate(divide="ignore", invalid="ignore"):
        emi = np.where(i == 0, amount / n, amount * i * (1 + i) ** n / ((1 + i) ** n - 1))
    emi = np.floor(emi + 0.5)
    net = amount - upfront
    lo, hi = np.zeros_like(net), np.full_like(net, 0.1)
    for _ in range(60):
        mid = (lo + hi) / 2
        pv = emi * (1 - (1 + mid) ** -n) / mid
        above = pv > net
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return (lo + hi) / 2 * 1200.0     # unrounded: callers round for display, the KFS rounds to 2 dp


def _upfront(amount: np.ndarray, fee_pct: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(amount, dtype=float) * np.asarray(fee_pct, dtype=float) / 100.0 + 0.5)


def _snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload.get("application_snapshot") or {})


def _num(snap: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = pd.to_numeric(snap.get(key), errors="coerce")
        if pd.notna(value):
            return float(value)
    return default


def file_defaults(payload: dict[str, Any]) -> dict[str, float]:
    """The file's own amount, tenure and the published fee: what an offer holds fixed."""
    snap = _snapshot(payload)
    return {"loan_amount": _num(snap, "sanctioned_loan_amount", "requested_loan_amount"),
            "tenure_months": round(_num(snap, "tenure_years", "requested_tenure_years", default=20.0) * 12),
            "fee_pct": float(policy()["economics"]["processing_fee_pct"])}


def _frame(snap: dict[str, Any], offers: pd.DataFrame, *, rates: np.ndarray | None = None,
           mode: str = "recommended") -> pd.DataFrame:
    """The file repeated once per offer with the offer's amount and tenure; FOIR and LTV come from
    the pipeline's own feature code (``stage2_features.build``).

    ``rates`` given: FOIR is worked out at each offer's own rate (the affordability view). Not
    given: at whatever rate the pipeline used for this file (``mode``, see ``foir_mode``), so only
    the amount and tenure move it (the view the models are fed)."""
    n = len(offers)
    base = pd.DataFrame([snap])
    frame = base.iloc[np.zeros(n, dtype=int)].reset_index(drop=True)
    frame["sanctioned_loan_amount"] = offers["loan_amount"].to_numpy(float)
    frame["tenure_years"] = offers["tenure_months"].to_numpy(float) / 12.0
    frame["requested_tenure_years"] = frame["tenure_years"]
    columns = ("ai_recommended_rate", "policy_card_rate")
    original = {c: frame[c].copy() for c in columns if c in frame.columns}
    if rates is not None:
        frame["ai_recommended_rate"] = np.asarray(rates, dtype=float)
    elif mode == "fallback":
        frame = frame.drop(columns=[c for c in columns if c in frame.columns])
    elif mode == "card":
        frame = frame.drop(columns=[c for c in ("ai_recommended_rate",) if c in frame.columns])
    frame = frame.drop(columns=[c for c in ("foir", "ltv") if c in frame.columns])
    built = stage2_features.build(frame)
    for column in columns:
        if column in original:
            built[column] = original[column].to_numpy()
        elif column in built.columns:
            built = built.drop(columns=[column])
    return built


def foir_mode(snap: dict[str, Any]) -> str | None:
    """Which rate the pipeline's FOIR used for this file, found by reproducing the file's own FOIR
    with the pipeline's code: the recommended rate, the card rate, or the fallback it uses when
    neither is known yet (scoring time). None when none of them reproduces it."""
    stored = pd.to_numeric(snap.get("foir"), errors="coerce")
    if pd.isna(stored):
        return "recommended"
    own = pd.DataFrame({"loan_amount": [_num(snap, "sanctioned_loan_amount", "requested_loan_amount")],
                        "tenure_months": [_num(snap, "tenure_years", "requested_tenure_years", default=20.0) * 12]})
    for mode in ("fallback", "card", "recommended"):
        if mode == "card" and "policy_card_rate" not in snap:
            continue
        if mode == "recommended" and "ai_recommended_rate" not in snap:
            continue
        if abs(float(_frame(snap, own, mode=mode)["foir"].iloc[0]) - float(stored)) < 1e-6:
            return mode
    return None


def model_frame(snap: dict[str, Any], offers: pd.DataFrame) -> pd.DataFrame:
    """The frame the models are scored on: FOIR as the pipeline computes it for this file."""
    return _frame(snap, offers, mode=foir_mode(snap) or "fallback")


def _gap(frame: pd.DataFrame, rates: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["offered_rate"] = rates
    if "competitor_rate" in out.columns:
        competitor = pd.to_numeric(out["competitor_rate"], errors="coerce")
        out["competitor_gap_bps"] = ((out["offered_rate"] - competitor) * 100).fillna(0.0)
    return out


def _regulatory_ltv(amount: np.ndarray) -> np.ndarray:
    out = np.full(len(amount), np.nan)
    slabs = policy().get("housing_regulatory_ltv") or []
    for slab in reversed([s for s in slabs if "max_amount" in s]):
        out = np.where(amount <= float(slab["max_amount"]), float(slab["max_ltv"]), out)
    rest = [s for s in slabs if s.get("else")]
    if rest:
        out = np.where(np.isnan(out), float(rest[0]["max_ltv"]), out)
    return out


_AUTO = object()


def evaluate_offers(payload: dict[str, Any], offers: pd.DataFrame, *, bundle: Any = None, mode: Any = _AUTO) -> pd.DataFrame:
    """Score every offer (columns ``rate``, ``loan_amount``, ``tenure_months``, ``fee_pct``;
    the last three default to the file's own) in one batch. ``attrs['limitations']`` lists
    anything that could not be scored with a model and the fee assumption."""
    pol = policy()
    econ = pol["economics"]
    published_fee = float(econ["processing_fee_pct"])
    defaults = file_defaults(payload)
    snap = _snapshot(payload)
    risk_block = payload.get("risk") or {}

    table = pd.DataFrame(offers).copy().reset_index(drop=True)
    for column in OFFER_COLUMNS[1:]:
        if column not in table.columns:
            table[column] = defaults[column]
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(defaults[column])
    table["rate"] = pd.to_numeric(table["rate"], errors="coerce").astype(float)
    fee_asked = table["fee_pct"].to_numpy(float)
    fee = np.clip(fee_asked, 0.0, published_fee)
    table["fee_pct"] = fee
    n = len(table)
    limitations: list[str] = [FEE_ASSUMPTION]
    if n == 0:
        out = table.copy()
        out.attrs["limitations"] = limitations
        return out

    if bundle is None:
        from ..governance import model_registry
        bundle = model_registry.load_bundle()

    mode = foir_mode(snap) if mode is _AUTO else mode      # a caller scoring many batches finds it once
    if mode is None:
        limitations.append("The file's own FOIR could not be reproduced with the pipeline's rate convention, so the models "
                           "are fed FOIR at the pipeline's fallback rate.")
    # Models see the pipeline's FOIR (only amount and tenure move it); affordability is judged at the offer's own rate.
    frame = _frame(snap, table, mode=mode or "fallback")
    affordability = _frame(snap, table, rates=table["rate"].to_numpy(float))
    rate = table["rate"].to_numpy(float)
    amount = table["loan_amount"].to_numpy(float)
    months = table["tenure_months"].to_numpy(float)
    years = months / 12.0
    # Income as every other FOIR in the pipeline reads it: verified, else declared. No income means
    # the stress test cannot be run, and an offer that cannot be tested is not eligible.
    income = first_available(frame, ["verified_monthly_income", "declared_monthly_income"], 0.0).to_numpy(float)
    income = np.where(income > 0, income, np.nan)
    obligations = constraints.numeric_column(frame, "declared_obligations", 0.0).to_numpy(float) \
        if hasattr(constraints, "numeric_column") else np.zeros(n)

    # ---- risk: PD, LGD and grade at this offer's FOIR / LTV / amount / tenure
    model = getattr(bundle, "risk", None)
    scored = None
    if model is not None:
        try:
            scored = model.assess(frame)
        except Exception as exc:                                       # noqa: BLE001 - degrade, say so
            limitations.append(f"The risk model could not rescore the offers ({exc}); the file's own risk is used.")
    elif model is None:
        limitations.append("No risk model is available; the file's own PD, LGD and grade are used at every offer.")
    if scored is not None:
        pd_annual = scored["pd"].to_numpy(float)
        pd_life = scored["lifetime_pd"].to_numpy(float)
        lgd = scored["lgd"].to_numpy(float)
        el_pct = scored["expected_loss_pct"].to_numpy(float)
        grade = scored["risk_grade"].astype(str).to_numpy()
    else:
        pd_annual = np.full(n, float(risk_block.get("pd") or 0.0))
        pd_life = np.full(n, float(risk_block.get("lifetime_pd") or 0.0))
        lgd = np.full(n, float(risk_block.get("lgd") or 0.0))
        el_pct = np.full(n, float(risk_block.get("expected_loss_pct") or 0.0))
        grade = np.full(n, str(risk_block.get("risk_grade") or snap.get("risk_grade") or "C"))
    frame["risk_grade"] = grade

    # ---- affordability, collateral and age
    foir = affordability["foir"].to_numpy(float)
    model_foir = frame["foir"].to_numpy(float)
    emi = constraints._emi(amount, rate, years)
    stressed_emi = constraints._emi(amount, rate + constraints.stress_add_pct(), years)
    stressed = (stressed_emi + obligations) / income
    foir_cap = constraints.grade_foir_cap(frame)
    limits = {g: v for g, v in (pol.get("eligibility_limits") or {}).items() if v}
    grade_ltv = pd.Series(grade).map({g: float(v["hl_ltv_max"]) for g, v in limits.items()}).to_numpy(float)
    ltv_cap = np.fmin(np.fmin(grade_ltv, _regulatory_ltv(amount)), float(pol["constraints"]["max_ltv"]))
    ltv = frame["ltv"].to_numpy(float)
    age = _num(snap, "applicant_age", default=np.nan)
    ages = pol["constraints"].get("max_age_at_maturity") or {}
    salaried = str(snap.get("segment", "")).startswith("salaried")
    age_limit = float(ages.get("salaried" if salaried else "self_employed", 99)) if ages else 99.0
    age_at_maturity = age + years

    # ---- the policy envelope for THIS grade and tenure
    card = constraints.card_rate(frame)
    floor_ref = float(pol["reference"]["card_rate_floor"])
    ceiling_ref = float(pol["reference"]["card_rate_ceiling"])
    discount = float(pol["constraints"]["absolute_discount_floor_bps"]) / 100.0
    floor = np.maximum(card - discount, floor_ref)
    in_band = (~np.isnan(card)) & (rate >= floor - 1e-9) & (rate <= np.minimum(card, ceiling_ref) + 1e-9)

    # ---- acceptance (conversion), life (prepayment), APR and the fee assumption
    published_apr = apr_vector(amount, rate, months, _upfront(amount, np.full(n, published_fee)))
    offer_apr = np.where(np.isclose(fee, published_fee), published_apr, apr_vector(amount, rate, months, _upfront(amount, fee)))
    effective = np.round(rate + (offer_apr - published_apr), 4)
    p_conv = np.full(n, np.nan)
    conversion = getattr(bundle, "conversion", None)
    if conversion is not None:
        sc = _gap(frame, effective)
        p_conv = conversion.model.predict_proba(conversion.transformer.transform(sc))[:, 1]
    else:
        limitations.append("No conversion model is available; acceptance and value are not scored.")
    life = np.full(n, np.nan)
    p_bt = np.full(n, 0.0)
    prepayment = getattr(bundle, "prepayment", None)
    if prepayment is not None:
        sp = _gap(frame, rate)
        p_bt = prepayment.model.predict_proba(prepayment.transformer.transform(sp))[:, 1]
        life = prepayment.life_from_probability(p_bt)
    else:
        limitations.append("No prepayment model is available; expected life and value are not scored.")

    breakeven = np.array([yield_calculator.breakeven_rate(float(e)) for e in el_pct])
    value = np.full(n, np.nan)
    if conversion is not None and prepayment is not None:
        results = yield_calculator.compute(
            rate=rate, conversion_probability=p_conv, expected_loss_pct=el_pct, expected_life_years=life,
            loan_amount=amount, tenure_years=years, bt_out_probability=p_bt)
        # The calculator books the published fee; the offer's own fee replaces it.
        value = np.array([r.conversion_probability * (r.lifetime_value_pct + (f - published_fee)) / 100.0 * a
                          for r, a, f in zip(results, amount, fee)])

    # ---- every eligibility failure
    declined = np.isnan(card)
    bureau = bool(constraints.bureau_below_minimum(frame.iloc[[0]])[0])
    checks = [
        ("grade_declined", declined),
        ("bureau_below_minimum", np.full(n, bureau)),
        ("rate_below_floor", (~declined) & (rate < floor - 1e-9)),
        ("rate_above_card", (~declined) & (rate > np.minimum(card, ceiling_ref) + 1e-9)),
        ("rate_outside_13_17", (rate < floor_ref - 1e-9) | (rate > ceiling_ref + 1e-9)),
        ("foir_over_cap", foir > foir_cap + 1e-9),
        ("stressed_foir_over_cap", stressed > foir_cap + 1e-9),
        ("stressed_foir_unavailable", np.isnan(stressed)),
        ("ltv_over_cap", ltv > ltv_cap + 1e-9),
        ("age_at_maturity_over_limit", age_at_maturity > age_limit + 1e-9),
        ("amount_over_max_loan", amount > float(pol["constraints"]["max_loan_amount"]) + 1e-9),
        ("tenure_outside_policy", (months < float(pol["constraints"]["min_tenure_months"]) - 1e-9)
         | (months > float(pol["constraints"]["max_tenure_months"]) + 1e-9)),
    ]
    reasons = [[code for code, hit in checks if bool(hit[i])] for i in range(n)]

    out = pd.DataFrame({
        "rate": np.round(rate, 2), "loan_amount": amount, "tenure_months": months.astype(int),
        "fee_pct": np.round(fee, 4), "fee_clipped": fee_asked > published_fee + 1e-9,
        "emi": np.round(emi, 0), "foir": np.round(foir, 4), "model_foir": np.round(model_foir, 6), "stressed_foir": np.round(stressed, 4),
        "foir_cap": foir_cap, "stressed_foir_ok": ~(np.isnan(stressed) | (stressed > foir_cap + 1e-9)), "ltv": np.round(ltv, 4), "ltv_cap": np.round(ltv_cap, 4),
        "age_at_maturity": np.round(age_at_maturity, 1), "age_limit": age_limit,
        "pd_annual": np.round(pd_annual, 5), "pd_lifetime": np.round(pd_life, 5), "risk_grade": grade,
        "card_rate": np.round(card, 2), "floor_rate": np.round(floor, 2), "in_band": in_band,
        "lgd": np.round(lgd, 4), "expected_loss_pct": np.round(el_pct, 4), "breakeven_rate": np.round(breakeven, 2),
        "margin_over_risk_cost": np.round(rate - breakeven, 2),
        "expected_life_years": np.round(life, 2), "bt_out_probability": np.round(p_bt, 4),
        "acceptance_model": np.round(p_conv, 4), "acceptance_rate_used": np.round(effective, 2),
        "apr_pct": np.round(offer_apr, 2), "value_inr": np.round(value, 0),
        "failures": reasons, "eligible": [not r for r in reasons],
    })
    out.attrs["limitations"] = limitations
    return out
