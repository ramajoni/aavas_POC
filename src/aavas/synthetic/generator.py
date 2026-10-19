"""Synthetic home-loan lifecycle generator.

The POC has no access to Aavas production data, so this module fabricates a
lifecycle with an explicit causal structure:

    customer + property + sales context
        -> latent credit risk (PD/LGD)
        -> policy card rate (from pricing_policy.yaml, not from a fitted model)
        -> legacy AI/rule recommendation
        -> human override (driven by competitor gap, targets, negotiation)
        -> customer response and sanction
        -> disbursal, balance-transfer-out, delinquency
        -> realized yield

Two properties matter for the POC to be meaningful:

1. Price elasticity is heterogeneous and *known*. Each application carries a
   latent ``price_elasticity``; the conversion model has to recover it from
   observable features. That gives us a ground truth to evaluate against, which
   real data would never hand us.
2. Overrides are informative, not noise. They are driven by competitor gap,
   month-end target pressure and negotiation, and some of them genuinely create
   value while others destroy it. The override-learning module has to tell the
   two apart.

Latent columns (``true_*``, ``price_elasticity``) are written to the raw file for
evaluation only and are excluded from every feature contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd

from ..config import model_config, policy, seed as config_seed
from ..logging_utils import get_logger
from ..optimization import constraints
from ..paths import ensure_dirs, resolve

log = get_logger(__name__)

# Aavas-like affordable-housing footprint: tier 2/3 towns, not metros.
GEOGRAPHY = {
    "Rajasthan": ["Jaipur", "Jodhpur", "Kota", "Alwar"],
    "Gujarat": ["Ahmedabad", "Surat", "Rajkot"],
    "Maharashtra": ["Pune", "Nagpur", "Nashik"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Jabalpur"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra"],
}
SEGMENTS = ["salaried_formal", "salaried_informal", "self_employed_professional",
            "self_employed_non_professional"]
SEGMENT_MIX = [0.30, 0.22, 0.13, 0.35]
CHANNELS = ["branch_walkin", "dsa", "connector", "digital"]
CHANNEL_MIX = [0.38, 0.34, 0.18, 0.10]
PROPERTY_TYPES = ["resale_flat", "builder_flat", "self_construction", "plot_plus_construction"]
PROPERTY_MIX = [0.30, 0.22, 0.34, 0.14]
COMPETITORS = ["HDFC", "LIC HF", "Bajaj HF", "Local Bank", "Shubham HF", "Aadhar HF"]

# Competitive intensity is not uniform across the footprint. Gujarat and
# Maharashtra are contested affordable-housing markets where competitors bid
# harder and branches discount more; Rajasthan is the home market with a
# stronger franchise. This regional structure is what regional personalization
# in the federated module exists to capture - without it, one global model is
# provably sufficient and federation buys nothing.
STATE_COMPETITIVE_INTENSITY = {
    "Rajasthan": -0.35,
    "Gujarat": 0.55,
    "Maharashtra": 0.40,
    "Madhya Pradesh": -0.10,
    "Uttar Pradesh": 0.20,
}

OVERRIDE_REASONS = [
    "competitor_lower_rate", "customer_threatening_bt", "strong_negotiation",
    "branch_competitive_pressure", "month_end_target", "salesperson_incentive",
    "improve_s2d_conversion", "relationship_value", "local_market_knowledge", "other",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _emi(principal: np.ndarray, annual_rate: np.ndarray, years: np.ndarray) -> np.ndarray:
    r = annual_rate / 100.0 / 12.0
    n = years * 12
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def generate(n: int | None = None, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (lifecycle dataframe, sales notes dataframe)."""
    cfg = model_config()
    pol = policy()
    n = int(n or cfg["data"]["n_applications"])
    rng = np.random.default_rng(config_seed() if seed is None else seed)

    d: dict[str, np.ndarray] = {}

    # Oversample: files that breach hard policy would have been declined at credit,
    # so they are filtered out at the end rather than written as impossible rows.
    n_target, n = n, int(n * 1.3)

    # ------------------------------------------------------------- identity / time
    day_offset = rng.integers(0, 730, n)  # 24 months of history
    application_date = pd.Timestamp("2024-09-01") + pd.to_timedelta(day_offset, unit="D")
    d["application_date"] = application_date

    states = rng.choice(list(GEOGRAPHY), n, p=[0.34, 0.22, 0.16, 0.16, 0.12])
    d["state"] = states
    d["city"] = np.array([rng.choice(GEOGRAPHY[s]) for s in states])
    # blake2b, not the builtin hash(): Python salts string hashing per process
    # (PYTHONHASHSEED), so hash(city) remapped every city to a different branch on
    # every regeneration. Metrics never moved - branch_id is an identity column,
    # not a feature - but branch-level analysis was not reproducible, and AutoML's
    # locality search uses branch_id as its unit. Found by the dataset content
    # hash reporting a different dataset for a run whose metrics were identical.
    d["branch_id"] = np.array([f"BR{int(hashlib.blake2b(c.encode(), digest_size=4).hexdigest(), 16) % 40:03d}"
                               for c in d["city"]])
    d["sales_officer_id"] = np.array([f"SO{i:04d}" for i in rng.integers(1, 260, n)])

    # ------------------------------------------------------------------- customer
    d["segment"] = rng.choice(SEGMENTS, n, p=SEGMENT_MIX)
    d["channel"] = rng.choice(CHANNELS, n, p=CHANNEL_MIX)
    d["property_type"] = rng.choice(PROPERTY_TYPES, n, p=PROPERTY_MIX)
    d["applicant_age"] = np.clip(rng.normal(37, 8, n), 21, 62).round().astype(int)

    seg = pd.Series(d["segment"])
    income_mu = seg.map({
        "salaried_formal": np.log(38000), "salaried_informal": np.log(29000),
        "self_employed_professional": np.log(62000), "self_employed_non_professional": np.log(44000),
    }).to_numpy()
    declared_income = np.exp(rng.normal(income_mu, 0.45)).round(-2)
    d["declared_monthly_income"] = declared_income

    # Informal and self-employed files lose more income at verification.
    haircut_mu = seg.map({
        "salaried_formal": 0.02, "salaried_informal": 0.16,
        "self_employed_professional": 0.10, "self_employed_non_professional": 0.21,
    }).to_numpy()
    haircut = np.clip(rng.normal(haircut_mu, 0.07), 0.0, 0.45)
    verified_income = (declared_income * (1 - haircut)).round(-2)
    d["verified_monthly_income"] = verified_income
    d["income_verification_haircut"] = haircut.round(4)

    # New-to-credit is common in this segment and cannot simply be declined.
    ntc = rng.random(n) < seg.map({
        "salaried_formal": 0.06, "salaried_informal": 0.22,
        "self_employed_professional": 0.08, "self_employed_non_professional": 0.19,
    }).to_numpy()
    d["new_to_credit"] = ntc
    score = np.clip(rng.normal(724, 58, n), 560, 880).round().astype(int)
    d["bureau_score"] = np.where(ntc, -1, score)              # -1 is the NTC sentinel
    d["bureau_vintage_months"] = np.where(ntc, 0, np.clip(rng.normal(74, 36, n), 6, 260).round().astype(int))
    d["delinquency_count_24m"] = np.where(ntc, 0, rng.poisson(0.28, n))
    d["banking_behaviour_score"] = np.clip(rng.normal(62, 16, n) - haircut * 40, 5, 100).round(1)
    d["declared_obligations"] = np.where(rng.random(n) < 0.42, (declared_income * rng.uniform(0.04, 0.22, n)).round(-2), 0.0)

    # --------------------------------------------------------- property and ticket
    ltv_target = np.clip(rng.normal(0.70, 0.09, n), 0.35, 0.90)
    requested_amount = np.clip(verified_income * rng.normal(34, 7, n), 300000, pol["constraints"]["max_loan_amount"]).round(-3)
    d["requested_loan_amount"] = requested_amount
    estimated_value = (requested_amount / ltv_target).round(-3)
    d["estimated_property_value"] = estimated_value
    d["estimated_ltv"] = (requested_amount / estimated_value).round(4)

    # Valuation typically comes in below the customer's estimate.
    appraised = (estimated_value * np.clip(rng.normal(0.96, 0.06, n), 0.75, 1.10)).round(-3)
    d["appraised_property_value"] = appraised
    d["requested_tenure_years"] = np.clip(rng.normal(19, 4, n), 5, pol["constraints"]["max_tenure_years"]).round().astype(int)
    d["tenure_years"] = np.clip(d["requested_tenure_years"] + rng.integers(-3, 2, n), 5, pol["constraints"]["max_tenure_years"])

    # Credit trims the ticket to respect LTV and FOIR. The FOIR cap is applied by
    # inverting the EMI formula at an indicative rate, which is what a credit
    # officer does with the eligibility calculator.
    max_by_ltv = appraised * pol["constraints"]["max_ltv"]
    indicative_rate = 11.5
    r_m = indicative_rate / 100.0 / 12.0
    n_m = d["tenure_years"] * 12
    growth = (1 + r_m) ** n_m
    max_emi = np.maximum(pol["constraints"]["max_foir"] * verified_income - d["declared_obligations"], 500.0)
    max_by_foir = max_emi * (growth - 1) / (r_m * growth)

    sanctioned_amount = np.minimum(requested_amount, max_by_ltv)
    sanctioned_amount = np.minimum(sanctioned_amount, max_by_foir)
    sanctioned_amount = np.minimum(sanctioned_amount, verified_income * 46).round(-3)
    sanctioned_amount = np.clip(sanctioned_amount, 250000, pol["constraints"]["max_loan_amount"])
    d["sanctioned_loan_amount"] = sanctioned_amount
    d["ltv"] = np.round(np.clip(sanctioned_amount / appraised, 0.2, pol["constraints"]["max_ltv"]), 4)

    d["deviation_count"] = rng.poisson(0.55, n) + (haircut > 0.25).astype(int)
    d["legal_flag"] = (rng.random(n) < 0.07).astype(int)
    d["technical_flag"] = (rng.random(n) < 0.09).astype(int)

    df = pd.DataFrame(d)

    # --------------------------------------------------------- latent credit risk
    score_z = np.where(ntc, -0.55, (df["bureau_score"].to_numpy() - 724) / 58.0)
    emi = _emi(sanctioned_amount, np.full(n, indicative_rate), df["tenure_years"].to_numpy())
    foir = np.clip((emi + df["declared_obligations"].to_numpy()) / verified_income, 0.08, 0.95)
    df["foir"] = foir.round(4)

    pd_logit = (
        -4.85
        - 0.72 * score_z
        + 2.35 * (foir - 0.40)
        + 2.05 * (df["ltv"].to_numpy() - 0.65)
        + 0.44 * df["delinquency_count_24m"].to_numpy()
        - 0.016 * (df["banking_behaviour_score"].to_numpy() - 60)
        + 0.30 * df["deviation_count"].to_numpy()
        + 0.42 * ntc.astype(float)
        + seg.map({"salaried_formal": -0.30, "salaried_informal": 0.22,
                   "self_employed_professional": -0.05, "self_employed_non_professional": 0.30}).to_numpy()
        + 0.35 * df["legal_flag"].to_numpy()
        + 0.25 * df["technical_flag"].to_numpy()
        + rng.normal(0, 0.28, n)
    )
    true_pd = _sigmoid(pd_logit)
    df["true_pd"] = true_pd.round(5)
    lgd_base = df["property_type"].map(cfg["risk"]["lgd_by_property_type"]).to_numpy()
    df["true_lgd"] = np.clip(lgd_base + (df["ltv"].to_numpy() - 0.70) * 0.35, 0.10, 0.65).round(4)

    cuts = cfg["risk"]["grade_cutoffs"]
    df["risk_grade"] = pd.cut(
        true_pd, bins=[-1, cuts["A"], cuts["B"], cuts["C"], cuts["D"], 1.01],
        labels=["A", "B", "C", "D", "E"],
    ).astype(str)

    # ----------------------------------------------------------- sales context
    # Single source of truth: the same function the policy agent enforces at runtime.
    df["policy_card_rate"] = constraints.card_rate(df).round(2)

    # Competitors bid harder for better files; that is what makes elasticity
    # correlate with quality rather than being random noise.
    intensity = df["state"].map(STATE_COMPETITIVE_INTENSITY).to_numpy(float)
    df["state_competitive_intensity"] = intensity
    # A contested market means competitors undercut harder.
    comp_edge = (rng.normal(28, 34, n) + 22 * score_z - 60 * (df["ltv"].to_numpy() - 0.70)
                 + 26 * intensity)
    df["competitor_rate"] = (df["policy_card_rate"] - comp_edge / 100.0).round(2)
    df["competitor_name"] = rng.choice(COMPETITORS, n)
    df["competitor_evidence_verified"] = (rng.random(n) < 0.34)

    dom = application_date.day
    days_in_month = application_date.days_in_month
    df["days_to_month_end"] = (days_in_month - dom).astype(int)
    df["month_end"] = df["days_to_month_end"] <= 5
    quarter_end = application_date.month.isin([3, 6, 9, 12])
    df["sales_target_pct"] = np.clip(rng.normal(78, 16, n) + np.where(quarter_end, -6, 0), 20, 135).round(1)
    df["branch_target_pct"] = np.clip(rng.normal(82, 13, n), 30, 130).round(1)
    df["existing_relationship"] = rng.random(n) < 0.12
    df["prior_negotiation_rounds"] = rng.poisson(0.9, n)
    df["customer_negotiating"] = (df["prior_negotiation_rounds"].to_numpy() > 0) & (rng.random(n) < 0.78)

    # ------------------------------------------------- latent price elasticity
    # Percentage-point drop in conversion logit per 100 bps of price. Thin-file,
    # low-income, high-LTV customers are captive; prime salaried files shop.
    elasticity = (
        1.55
        + 0.55 * np.clip(score_z, -2, 2)
        + seg.map({"salaried_formal": 0.45, "salaried_informal": -0.18,
                   "self_employed_professional": 0.22, "self_employed_non_professional": -0.12}).to_numpy()
        + 0.40 * (np.log(sanctioned_amount) - np.log(1000000))
        - 0.75 * (df["ltv"].to_numpy() - 0.70)
        + df["channel"].map({"branch_walkin": -0.12, "dsa": 0.30, "connector": 0.10, "digital": 0.35}).to_numpy()
        + 0.30 * df["customer_negotiating"].to_numpy().astype(float)
        - 0.25 * df["existing_relationship"].to_numpy().astype(float)
        + rng.normal(0, 0.18, n)
    )
    df["price_elasticity"] = np.clip(elasticity, 0.35, 4.0).round(4)

    # ------------------------------------------- legacy recommendation + override
    # Before this platform existed, pricing was a grid plus judgement. The legacy
    # recommendation is the card rate with a small statistical adjustment.
    legacy_adjust = (-14 * score_z + 30 * (df["ltv"].to_numpy() - 0.70) + rng.normal(0, 9, n)) / 100.0
    df["ai_recommended_rate"] = np.round(df["policy_card_rate"] + legacy_adjust, 2)

    comp_gap_bps = (df["ai_recommended_rate"] - df["competitor_rate"]) * 100
    override_pressure = (
        0.0115 * np.clip(comp_gap_bps, 0, None)
        + 0.020 * np.clip(100 - df["sales_target_pct"].to_numpy(), 0, None)
        + 0.013 * np.clip(100 - df["branch_target_pct"].to_numpy(), 0, None)
        + 0.85 * df["customer_negotiating"].to_numpy().astype(float)
        + 0.55 * df["month_end"].to_numpy().astype(float)
        + 0.45 * df["existing_relationship"].to_numpy().astype(float)
        + 0.60 * intensity
    )
    overridden = rng.random(n) < _sigmoid(override_pressure - 2.35)

    discount_bps = np.round(np.clip(rng.gamma(2.1, 8.0, n) * (0.55 + 0.45 * _sigmoid(override_pressure - 2.0)), 0, 85))
    premium_bps = np.round(np.clip(rng.gamma(1.4, 6.0, n), 0, 45))
    # Most overrides are discounts; a minority price up when the file is captive.
    is_discount = rng.random(n) < _sigmoid(0.9 * override_pressure - 0.6)
    override_bps = np.where(overridden, np.where(is_discount, -discount_bps, premium_bps), 0.0)

    # A human may not breach the absolute discount floor either.
    max_discount = pol["constraints"]["absolute_discount_floor_bps"]
    override_bps = np.maximum(override_bps, -max_discount)
    df["override_bps"] = override_bps

    reason = np.full(n, "", dtype=object)
    reason = np.where(overridden & (comp_gap_bps > 25), "competitor_lower_rate", reason)
    reason = np.where(overridden & (reason == "") & df["customer_negotiating"].to_numpy(), "strong_negotiation", reason)
    reason = np.where(overridden & (reason == "") & df["month_end"].to_numpy(), "month_end_target", reason)
    reason = np.where(overridden & (reason == "") & (df["branch_target_pct"].to_numpy() < 75), "branch_competitive_pressure", reason)
    reason = np.where(overridden & (reason == "") & df["existing_relationship"].to_numpy(), "relationship_value", reason)
    remaining = overridden & (reason == "")
    reason = np.where(remaining, rng.choice(["improve_s2d_conversion", "local_market_knowledge",
                                             "salesperson_incentive", "customer_threatening_bt", "other"],
                                            n, p=[0.30, 0.24, 0.18, 0.18, 0.10]), reason)
    df["override_reason"] = np.where(overridden, reason, "no_override")

    # Branches in contested markets also shade the rate down outside of a formal
    # override, which is the regional effect a central model cannot see.
    regional_shading = -6.0 * intensity + rng.normal(0, 2.5, n)
    df["sales_offered_rate"] = np.round(
        df["ai_recommended_rate"] + (override_bps + regional_shading) / 100.0, 2)

    # Credit may claw back part of an aggressive discount on weak files.
    clawback = ((df["risk_grade"].isin(["D", "E"])) & (override_bps < -30) & (rng.random(n) < 0.45))
    df["sanctioned_rate"] = np.round(np.where(clawback, df["sales_offered_rate"] + 0.15, df["sales_offered_rate"]), 2)
    df["credit_clawback"] = clawback

    # ------------------------------------------------------------ customer response
    gap_bps = (df["sanctioned_rate"] - df["competitor_rate"]) * 100
    conv_logit = (
        1.42
        - df["price_elasticity"].to_numpy() * gap_bps / 100.0
        - 0.40 * df["customer_negotiating"].to_numpy().astype(float)
        + 0.45 * df["existing_relationship"].to_numpy().astype(float)
        + df["channel"].map({"branch_walkin": 0.20, "dsa": -0.05, "connector": 0.02, "digital": -0.22}).to_numpy()
        - 0.55 * (df["ltv"].to_numpy() > 0.85).astype(float)
        - 0.30 * df["legal_flag"].to_numpy()
        + rng.normal(0, 0.30, n)
    )
    true_conversion = _sigmoid(conv_logit)
    df["true_conversion_probability"] = true_conversion.round(5)
    df["sanctioned"] = True                      # every row in this dataset reached sanction
    disbursed = rng.random(n) < true_conversion
    df["disbursed"] = disbursed
    df["customer_response"] = np.where(
        disbursed, "accepted",
        np.where(rng.random(n) < 0.45, "rejected", np.where(rng.random(n) < 0.6, "negotiated", "deferred")),
    )

    # ------------------------------------------------ post-disbursal behaviour
    # Market drifts after disbursal; being priced above it drives balance transfer.
    market_drift = rng.normal(-0.10, 0.22, n)
    future_gap_bps = gap_bps + market_drift * 100
    bt_logit = (
        -1.95
        + 0.0115 * np.clip(future_gap_bps, -150, 250)
        + 0.55 * np.clip(score_z, -2, 2)
        + 0.32 * (np.log(sanctioned_amount) - np.log(1000000))
        - 0.60 * (df["ltv"].to_numpy() - 0.70)
        + df["segment"].map({"salaried_formal": 0.40, "salaried_informal": -0.10,
                             "self_employed_professional": 0.18, "self_employed_non_professional": -0.15}).to_numpy()
        + rng.normal(0, 0.35, n)
    )
    bt_prob = _sigmoid(bt_logit)
    df["true_bt_probability"] = bt_prob.round(5)
    bt_out = disbursed & (rng.random(n) < bt_prob)
    df["bt_out_24m"] = bt_out

    # true_pd is an annualised hazard; scale to a lifetime view for the event draw.
    defaulted = disbursed & (rng.random(n) < np.clip(true_pd * 4.0, 0, 0.6))
    df["delinquent_12m"] = disbursed & (rng.random(n) < true_pd * 2.0)
    df["defaulted"] = defaulted

    months_to_exit = np.where(bt_out, rng.integers(7, 25, n), rng.integers(48, 121, n))
    df["months_to_exit"] = np.where(disbursed, months_to_exit, 0)
    realized_life = np.where(disbursed, months_to_exit / 12.0, 0.0)
    df["realized_life_years"] = realized_life.round(2)

    # ------------------------------------------------------------ realized yield
    econ = pol["economics"]
    # Annual carry on the outstanding book, in percentage points.
    carry = (
        df["sanctioned_rate"].to_numpy()
        - econ["funding_cost_pct"] - econ["servicing_cost_pct"] - econ["capital_charge_pct"]
    )
    life = np.where(disbursed, np.maximum(realized_life, 0.5), np.nan)
    # One-off fee income and acquisition cost amortise over the *realized* life.
    one_off_annual = (econ["processing_fee_pct"] - econ["acquisition_cost_pct"]) / life
    # Realized (not expected) credit cost: only defaulted files carry a loss.
    credit_cost_annual = np.where(defaulted, df["true_lgd"].to_numpy() * 100.0 / life, 0.0)
    realized_yield = np.where(disbursed, carry + one_off_annual - credit_cost_annual, np.nan)
    df["realized_yield_pct"] = np.round(realized_yield, 3)

    # Annualised yield alone hides the damage an early balance transfer does: the
    # spread is earned for fewer years. realized_value_pct is the lifetime economic
    # value of the decision as a % of the disbursed amount, which is what the
    # optimizer's expected value is actually trying to predict.
    disc = econ["discount_rate_pct"] / 100.0
    # Amortising balance -> average outstanding is roughly 62% of original over a
    # full-tenure HL, closer to 95% for a loan that exits in the first two years.
    avg_outstanding = np.where(realized_life <= 2.5, 0.95, np.clip(0.95 - 0.055 * realized_life, 0.55, 0.95))
    discount_factor = np.where(disbursed, (1 - np.exp(-disc * np.maximum(realized_life, 0.01))) / disc, 0.0)
    realized_value = np.where(
        disbursed,
        carry * avg_outstanding * discount_factor
        + econ["processing_fee_pct"] - econ["acquisition_cost_pct"]
        - np.where(defaulted, df["true_lgd"].to_numpy() * 100.0 * avg_outstanding, 0.0),
        0.0,
    )
    df["realized_value_pct"] = np.round(realized_value, 3)
    df["realized_value_inr"] = np.round(realized_value / 100.0 * sanctioned_amount, 0)

    # ------------------------------------------ historical Stage-1 indicative quote
    # What the branch verbally quoted at enquiry, before any assessment.
    quote_mid = df["policy_card_rate"].to_numpy() + rng.normal(-0.05, 0.22, n)
    half_band = np.clip(rng.normal(0.30, 0.10, n), 0.12, 0.65)
    df["stage1_min_rate"] = np.round(quote_mid - half_band, 2)
    df["stage1_max_rate"] = np.round(quote_mid + half_band, 2)

    # ------------------------------------------------- drop files credit would decline
    con = pol["constraints"]
    ref = pol["reference"]
    keep = (
        (df["foir"] <= con["max_foir"] + 1e-9)
        & (df["ltv"] <= con["max_ltv"] + 1e-9)
        & (df["sanctioned_rate"].between(ref["card_rate_floor"], ref["card_rate_ceiling"]))
        & (df["ai_recommended_rate"].between(ref["card_rate_floor"], ref["card_rate_ceiling"]))
        & (df["sales_offered_rate"].between(ref["card_rate_floor"], ref["card_rate_ceiling"]))
        & (df["stage1_min_rate"] >= ref["card_rate_floor"])
        & (df["stage1_max_rate"] <= ref["card_rate_ceiling"])
    )
    dropped = int((~keep).sum())
    df = df.loc[keep].copy()
    if len(df) < n_target:
        log.warning("only %s policy-compliant rows generated for a target of %s", len(df), n_target)
    df = df.sort_values("application_date").head(n_target).reset_index(drop=True)
    df.insert(0, "loan_id", [f"AAV{i:07d}" for i in range(1, len(df) + 1)])
    log.info("dropped %s files that breached hard policy (declined at credit)", dropped)

    notes = _generate_notes(df, rng)
    return df, notes


def _generate_notes(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Free-text sales/credit notes - the unstructured competitive intelligence
    that section 6 of the problem statement says never reaches the model."""
    templates = {
        "competitor_lower_rate": [
            "Customer showed {comp} sanction letter at {crate}%. Asked us to match or he will go with them.",
            "{comp} has offered {crate}% with lower processing fee. Customer comparing both offers.",
        ],
        "customer_threatening_bt": [
            "Customer says if we do not improve rate he will take {comp} offer and transfer later anyway.",
            "Existing {comp} customer, clearly indicated BT if our rate stays above {crate}%.",
        ],
        "strong_negotiation": [
            "Third round of negotiation. Customer firm on {crate}%, will not sign above that.",
            "Customer negotiated hard, referred to a neighbour who got a better rate from us last month.",
        ],
        "month_end_target": [
            "Month end case, need this file for target. Requesting {disc} bps relaxation.",
            "Disbursement pending for month close, requesting rate approval to convert today.",
        ],
        "branch_competitive_pressure": [
            "{city} market very competitive this quarter, {comp} aggressive on affordable segment.",
            "Losing files in {city} to {comp}. Branch requesting pricing relaxation on this profile.",
        ],
        "relationship_value": [
            "Existing customer with clean track record, second loan. Requesting relationship pricing.",
            "Family already has a loan with us, repayment record clean. Worth retaining at {crate}%.",
        ],
        "improve_s2d_conversion": [
            "Sanction done but customer not coming for disbursement, rate is the only open point.",
            "File may lapse, customer delaying disbursement citing rate. Requesting small relaxation.",
        ],
        "local_market_knowledge": [
            "This locality in {city} is seeing better collections, profile stronger than the score suggests.",
            "Known builder project in {city}, our past files here have performed well.",
        ],
        "salesperson_incentive": ["Requesting approval to close this file in the current cycle."],
        "other": ["Customer requested rate review before disbursement."],
        "no_override": [
            "Customer accepted the recommended rate without negotiation.",
            "Rate discussed and agreed, no relaxation requested.",
            "Profile explained to customer, comfortable with the offered pricing.",
        ],
    }
    rows = []
    for row in df.itertuples(index=False):
        pool = templates.get(row.override_reason, templates["other"])
        text = pool[rng.integers(0, len(pool))].format(
            comp=row.competitor_name, crate=f"{row.competitor_rate:.2f}", city=row.city,
            disc=int(abs(row.override_bps)) or 10,
        )
        rows.append({
            "loan_id": row.loan_id,
            "application_date": row.application_date,
            "city": row.city,
            "state": row.state,
            "branch_id": row.branch_id,
            "note_type": "sales_negotiation",
            "note_text": text,
            "override_reason": row.override_reason,
            "competitor_name": row.competitor_name,
            "competitor_rate": row.competitor_rate,
            "evidence_verified": bool(row.competitor_evidence_verified),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic Aavas lifecycle dataset")
    parser.add_argument("-n", "--rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = model_config()["data"]
    df, notes = generate(args.rows, args.seed)

    raw_path = Path(resolve(cfg["raw_path"]))
    notes_path = Path(resolve(cfg["notes_path"]))
    df.to_csv(raw_path, index=False)
    notes.to_csv(notes_path, index=False)

    log.info("wrote %s rows -> %s", len(df), raw_path)
    log.info("wrote %s notes -> %s", len(notes), notes_path)
    dis = df[df["disbursed"]]
    log.info(
        "disbursal %.1f%% | override %.1f%% | BT-out(24m) %.1f%% | mean sanctioned rate %.2f%%",
        100 * df["disbursed"].mean(),
        100 * (df["override_bps"] != 0).mean(),
        100 * df.loc[df["disbursed"], "bt_out_24m"].mean(),
        df["sanctioned_rate"].mean(),
    )
    log.info(
        "mean realized yield %.2f%% p.a. | mean lifetime value %.2f%% of disbursal | grade mix %s",
        dis["realized_yield_pct"].mean(),
        dis["realized_value_pct"].mean(),
        df["risk_grade"].value_counts(normalize=True).round(3).to_dict(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
