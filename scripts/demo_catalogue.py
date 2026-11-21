"""A catalogue of every area-manager case the live pricing engine can produce.

demo_scenarios.py hand-tunes a dozen files. This script finds the reachable cases by
probing the engine instead, then builds them through the real API.

    # 1. Find the cases (throwaway API only; --scratch is required, nothing else is touched)
    PYTHONPATH=src python scripts/demo_catalogue.py probe --base http://127.0.0.1:8011 --scratch

    # 2. Create the cases on a server (RM files with a pending rate request for the area manager)
    PYTHONPATH=src python scripts/demo_catalogue.py create --base http://127.0.0.1:8000
    PYTHONPATH=src python scripts/demo_catalogue.py create --base http://127.0.0.1:8000 --decide-history

    # 3. Write the coverage matrix to docs/demo/area_manager_cases.md
    python scripts/demo_catalogue.py report

    # 4. Cut the area manager's list to 10 pending requests that still show every scene
    #    (works on the store named by AAVAS_STORE_URL; dry run unless --apply, which backs the store up first)
    PYTHONPATH=src python scripts/demo_catalogue.py trim --keep 10
    PYTHONPATH=src python scripts/demo_catalogue.py trim --keep 10 --apply

probe prices a few hundred applicant profiles (pre-assessment and full pricing, the same calls
the RM makes), records grade, tenure band, card, floor, recommendation and flags for each, then
picks one profile per cell and writes scripts/demo_catalogue.json. The primary cells are
grade (A+, A, B, C) x tenure band (36-60, 61-84, 85-120 months) x ask position (within room,
at the floor, below the floor) x evidence (a verified letter below our floor, between the floor
and our recommendation, at or above it; a claim not verified; none). A cell with no case is
marked impossible (a policy rule forbids it) or not found (possible under the policy, but no
probed profile produced it), each with the reason. Secondary values (segment, channel, property type,
override reason, existing customer, negotiating) are rotated across the cells, and a set of
special situations gets one case each. D never reaches the area manager and LAP is not
accepted by the intake API, so both are listed as unreachable.

create posts the cases, checks each landed in its intended cell and, if not, says so and goes by
what was observed. Every rate (ask, approved, counter, uplift) must be inside the policy band
(reference.card_rate_floor to card_rate_ceiling, 13.00-17.00%); a rate outside it stops the
script before anything is posted. With --decide-history a small separate set is decided by the
area manager (approve as asked, counter, decline, uplift) and set back in time, so the
precedent and decision-memory panels have something to show.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import demo_scenarios as ds  # noqa: E402  (call, login, upload, DOCS, PRE_KEYS, backdate)

CATALOGUE = HERE / "demo_catalogue.json"
REPORT = ROOT / "docs" / "demo" / "area_manager_cases.md"

GRADES = ("A+", "A", "B", "C")
BANDS = ("36-60", "61-84", "85-120")
POSITIONS = ("within_room", "at_floor", "below_floor")
EVIDENCE = ("below_floor", "between", "at_or_above_rec", "claimed", "none")
SEGMENTS = ("salaried_formal", "salaried_informal", "self_employed_professional", "self_employed_non_professional")
CHANNELS = ("branch_walkin", "dsa", "connector", "digital")
PROPERTIES = ("resale_flat", "builder_flat", "self_construction", "plot_plus_construction")
EXISTING = (True, False)
NEGOTIATING = (True, False)
CITIES = ("Jodhpur", "Kota", "Alwar", "Jaipur", "Rajkot", "Ahmedabad", "Surat", "Indore", "Nagpur", "Jabalpur",
          "Bhopal", "Pune", "Nashik", "Agra", "Lucknow", "Kanpur")
# the matrix (card) rate by grade for 36-60 / 61-84 / 85-120 months, to reason about cells before probing
MATRIX = {"A+": (13.00, 13.50, 14.00), "A": (14.00, 14.50, 15.00), "B": (15.00, 15.50, 16.00), "C": (16.00, 16.50, 17.00)}
RM_BPS = 0.25          # the RM may send an ask within this of the recommendation; below it needs the area manager
DISCOUNT_FLOOR = 0.60  # constraints.absolute_discount_floor_bps / 100
HISTORY_OUTCOMES = ("approve_as_asked", "approve_at_floor", "counter", "decline", "uplift")
TOL = 0.004
MIN_DAYS_BEFORE_DECISION = 3  # hours between an RM request and the area manager's decision, in the back-dated history


# ---------------------------------------------------------------------------------------- pure helpers
def policy_band() -> tuple[float, float]:
    """The permitted rate band: reference.card_rate_floor / card_rate_ceiling, 13-17 if unavailable."""
    try:
        from aavas.config import policy
        ref = policy().get("reference") or {}
    except Exception:  # noqa: BLE001 - the package may not be importable (tests without PYTHONPATH)
        ref = {}
    return float(ref.get("card_rate_floor") or 13.0), float(ref.get("card_rate_ceiling") or 17.0)


def check_band(rate: float, label: str, lo: float, hi: float) -> float:
    """Refuse any rate outside the policy band, before it is posted."""
    if not lo - 1e-9 <= rate <= hi + 1e-9:
        raise SystemExit(f"{label}: rate {rate:.2f}% is outside the policy band {lo:.2f}-{hi:.2f}%; fix the case")
    return rate


def band_of(years: float) -> str:
    months = round(years * 12)
    return "36-60" if months <= 60 else "61-84" if months <= 84 else "85-120"


def floor_of(card: float, lo: float = 13.0) -> float:
    return round(max(card - DISCOUNT_FLOOR, lo), 2)


def position_of(rate: float, floor: float) -> str:
    return "below_floor" if rate < floor - TOL else "at_floor" if rate < floor + TOL else "within_room"


def evidence_class(comp, floor: float, rec: float) -> str:
    """comp is (rate, lender, letter_seen) or None."""
    if not comp:
        return "none"
    rate, _, seen = comp[0], comp[1], comp[2]
    if not seen:
        return "claimed"
    if rate < floor - TOL:
        return "below_floor"
    return "between" if rate < rec - TOL else "at_or_above_rec"


def ask_for(position: str, rec: float, floor: float, lo: float = 13.0, hi: float = 17.0) -> float | None:
    """The ask for a position, or None when no valid ask exists. A valid ask is more than 25 bps
    under the recommendation (the RM's own authority ends there) and inside the policy band."""
    if position == "within_room":
        rate = round(rec - 0.30, 2)
        ok = rate >= floor + TOL          # strictly above the floor, or it is the at-floor ask
    elif position == "at_floor":
        rate = round(floor, 2)
        ok = floor < round(rec - RM_BPS, 2) - TOL
    elif position == "below_floor":
        rate = round(min(floor - 0.25, rec - 0.30), 2)
        ok = rate < floor - TOL
    else:
        raise ValueError(position)
    return rate if ok and lo - 1e-9 <= rate <= hi + 1e-9 else None


def available_positions(rec: float, floor: float, lo: float = 13.0, hi: float = 17.0) -> list[str]:
    return [p for p in POSITIONS if ask_for(p, rec, floor, lo, hi) is not None]


def ceiling_of(obs: dict) -> float:
    """The highest rate the area manager may uplift to: the policy card, or the top of the quoted range."""
    ceil = float(obs.get("ceiling") or obs["card"])
    quoted = obs.get("quoted_max")
    if quoted is not None and float(quoted) < ceil and float(quoted) >= obs["rec"] - 1e-9:
        ceil = float(quoted)
    return ceil


def uplift_rate(obs: dict, hi: float = 17.0) -> float | None:
    gap = ceiling_of(obs) - obs["rec"]
    if gap < 0.05:
        return None
    rate = round(obs["rec"] + min(0.15, gap / 2), 2)
    return rate if rate > obs["rec"] + TOL and rate <= min(hi, ceiling_of(obs)) + 1e-9 else None


def cell_key(grade: str, band: str, position: str, evidence: str) -> str:
    return f"{grade}|{band}|{position}|{evidence}"


def all_cells() -> list[tuple[str, str, str, str]]:
    return [(g, b, p, e) for g in GRADES for b in BANDS for p in POSITIONS for e in EVIDENCE]


def static_unreachable(grade: str, band: str, position: str, evidence: str, lo: float = 13.0) -> str | None:
    """Reasons that follow from the matrix and the 13% floor alone, before any probing."""
    card = MATRIX[grade][BANDS.index(band)]
    floor = floor_of(card, lo)
    if card <= lo + 1e-9:
        return (f"the card rate for {grade} at {band} months is {card:.2f}%, the same as the policy floor: nothing can be "
                f"conceded, and an ask more than 25 bps under the recommendation would be under {lo:.2f}%")
    if position == "below_floor" and floor - 0.25 < lo - 1e-9:
        return (f"the floor for {grade} at {band} months is {floor:.2f}%, the lowest rate allowed, so an ask below "
                f"it would be under {lo:.2f}%")
    return None


def eff_ltv_cap(grade: str, loan: float, limits: dict) -> float:
    grade_cap = float((limits.get(grade) or {}).get("hl_ltv_max") or 0.9)
    return min(grade_cap, 0.90 if loan <= 3_000_000 else 0.80)


def stressed_foir(card: float, years: int, loan: float, income: float, obligations: float) -> float:
    rate = (card + 2.0) / 1200
    n = years * 12
    emi = loan * rate / (1 - (1 + rate) ** -n)
    return (emi + obligations) / income


# ---------------------------------------------------------------------------------------- the pricing flow
def new_loan_id() -> str:
    return "APP" + "".join(secrets.choice("0123456789") for _ in range(13))


def applicant_of(p: dict, loan_id: str) -> dict:
    comp = p.get("comp")
    sales = ({"competitor_rate": comp[0], "competitor_name": comp[1], "competitor_evidence_verified": bool(comp[2])}
             if comp else {})
    return {"loan_id": loan_id, "name": p.get("name") or "Probe Applicant", "mobile": p.get("mobile") or "9820013000",
            "email": "", "segment": p["segment"], "city": p["city"], "channel": p["channel"],
            "property_type": p["property_type"], "applicant_age": p["age"], "declared_monthly_income": p["income"],
            "declared_obligations": p["obligations"], "requested_loan_amount": p["loan"],
            "requested_tenure_years": p["years"], "estimated_property_value": p["value"], "bureau_score": p["score"],
            "new_to_credit": p["score"] < 0, "sales_context": sales}


def final_parts(p: dict, applicant: dict) -> tuple[dict, dict]:
    """(the fields the RM adds at verification, the final-assessment request body)."""
    v = p["verified"]
    comp = p.get("comp")
    sales = applicant["sales_context"]
    full_sales = {**sales, "competitor_quote": bool(comp), "customer_negotiating": p["negotiating"],
                  "prior_negotiation_rounds": p["rounds"] if p["negotiating"] else None,
                  "existing_relationship": p["existing"]}
    complete = {"verified_monthly_income": v["income"], "appraised_property_value": v["appraised"],
                "sanctioned_loan_amount": v["loan"], "tenure_years": v["years"], "bureau_vintage_months": v["history"],
                "delinquency_count_24m": v["missed"], "banking_behaviour_score": v["banking"],
                "deviation_count": v["deviations"], "legal_flag": 0, "technical_flag": 0}
    body = {**{k: applicant[k] for k in ds.PRE_KEYS if k != "sales_context"},
            "sales_context": {k: x for k, x in full_sales.items() if x is not None}, **complete}
    return {**complete, "sales_context": full_sales, "conversation_notes": p.get("notes") or ""}, body


def observe(priced: dict, full: dict, p: dict, limits: dict, lo: float) -> dict:
    pay = full["payload"]
    env, risk = pay.get("envelope") or {}, pay.get("risk") or {}
    grade, card = risk.get("risk_grade"), float(env.get("card_rate") or 0)
    rec = float(priced["recommendation"]["recommended_rate"])
    v = p["verified"]
    foir = stressed_foir(card, v["years"], v["loan"], v["income"], p["obligations"]) if card else None
    return {"grade": grade, "band": band_of(v["years"]), "card": round(card, 2), "floor": floor_of(card, lo) if card else None,
            "ceiling": float(env.get("max_rate") or card), "rec": round(rec, 2),
            "breakeven": risk.get("breakeven_rate"), "infeasible": bool(env.get("infeasible")),
            "resize_required": bool(env.get("resize_required")), "no_offer": bool(env.get("no_offer")),
            "required_approval": full.get("required_approval"),
            "verdict": (pay.get("policy") or {}).get("verdict"),
            "quoted_max": (pay.get("indicative_band") or {}).get("max_rate"),
            "sensitivity": (pay.get("price_sensitivity") or {}).get("sensitivity_label"),
            "foir_stressed": round(foir, 4) if foir is not None else None,
            "foir_cap": (limits.get(grade) or {}).get("foir_max") if grade else None,
            "ltv": round(v["loan"] / v["appraised"], 4),
            "versions": priced.get("versions") or {}, "model_versions": pay.get("model_versions") or {}}


def price(base: str, rm: str, am: str, p: dict, limits: dict, lo: float, *, full_flow: bool = False):
    """Run a profile through pricing. With full_flow, also the application, documents and verification."""
    loan_id = new_loan_id()
    applicant = applicant_of(p, loan_id)
    ds.call(base, "POST", "/pricing/pre-assessment", {k: applicant[k] for k in ds.PRE_KEYS}, rm)
    if full_flow:
        ds.call(base, "POST", "/applications", {"loan_id": loan_id, "applicant": applicant}, rm)
        for kind in ds.DOCS:
            ds.upload(base, rm, loan_id, kind)
    complete, body = final_parts(p, applicant)
    if full_flow:
        ds.call(base, "PUT", f"/applications/{loan_id}", {"applicant": complete}, rm)
    priced = ds.call(base, "POST", "/pricing/final-assessment", body, rm)
    full = ds.call(base, "GET", f"/pricing/decision/{priced['decision_id']}", None, am)
    return loan_id, priced, full, observe(priced, full, p, limits, lo)


# ---------------------------------------------------------------------------------------- the probe grid
def _emi_per_rupee(rate_pct: float, years: int) -> float:
    r = rate_pct / 1200
    return r / (1 - (1 + r) ** -(years * 12))


def make_profile(rng: random.Random, *, segment=None, years=None, loan=None, ltv=None, foir=None, score=None,
                 age=None, hist=None, bank=None, missed=None, deviations=None, comp=None, negotiating=None,
                 existing=None) -> dict:
    segment = segment or rng.choice(SEGMENTS)
    years = years or rng.choice([3, 4, 5, 6, 7, 8, 9, 10])
    limit = 60 if segment.startswith("salaried") else 70
    age = age if age is not None else rng.randint(24, max(25, min(limit - years, 58)))
    loan = loan or rng.choice([1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000,
                               7_000_000, 7_400_000])
    cap = 0.90 if loan <= 3_000_000 else 0.80
    ltv = ltv if ltv is not None else rng.uniform(0.35, cap)
    value = int(round(loan / ltv, -4))
    share = rng.choice([0.0, 0.0, 0.05, 0.10])
    foir = foir if foir is not None else rng.uniform(share + 0.15, 0.62)
    income = int(round(loan * _emi_per_rupee(16.0, years) / (foir - share), -3))
    obligations = int(round(income * share, -3))
    negotiating = rng.random() < 0.5 if negotiating is None else negotiating
    existing = rng.random() < 0.4 if existing is None else existing
    score = score if score is not None else rng.randint(675, 850)
    p = {"name": "", "mobile": "", "city": rng.choice(CITIES), "segment": segment, "channel": rng.choice(CHANNELS),
         "property_type": rng.choice(PROPERTIES), "age": age, "income": income, "obligations": obligations,
         "loan": loan, "years": years, "value": value, "score": score, "comp": comp,
         "negotiating": negotiating or bool(comp), "rounds": rng.choice([1, 2, 3]), "existing": existing,
         "verified": {"income": int(round(income * 0.98, -3)), "appraised": value, "loan": loan,
                      "years": years, "history": hist if hist is not None else rng.choice([12, 24, 48, 60, 84, 120, 180]),
                      "missed": missed if missed is not None else rng.choice([0, 0, 0, 0, 1, 2, 3]),
                      "banking": bank if bank is not None else rng.randint(35, 95),
                      "deviations": deviations if deviations is not None else rng.choice([0, 0, 0, 1, 2, 3])}}
    return p


def special_profile(kind: str, rng: random.Random) -> dict:
    if kind == "new_to_credit":
        return make_profile(rng, segment="salaried_formal", years=10, loan=1_500_000, ltv=rng.uniform(0.5, 0.65),
                            foir=rng.uniform(0.25, 0.40), score=-1, hist=0, age=rng.randint(23, 30), missed=0, deviations=0)
    if kind == "score_675_685":
        return make_profile(rng, score=rng.randint(675, 685))
    if kind == "ltv_near_cap":
        cap = rng.choice([0.90, 0.80, 0.75, 0.70])
        loan = rng.choice([2_000_000, 2_500_000, 2_800_000]) if cap == 0.90 else rng.choice([3_500_000, 4_500_000, 5_500_000])
        return make_profile(rng, loan=loan, ltv=cap - rng.uniform(0.0, 0.019), foir=rng.uniform(0.25, 0.45))
    if kind == "large_ticket":
        return make_profile(rng, loan=rng.choice([7_000_000, 7_200_000, 7_400_000, 7_500_000]), ltv=rng.uniform(0.40, 0.78),
                            foir=rng.uniform(0.28, 0.5), years=rng.choice([8, 9, 10]))
    if kind == "foir_near_cap":
        return make_profile(rng, foir=rng.choice([0.55, 0.50, 0.45, 0.40]) - rng.uniform(0.0, 0.03))
    if kind == "age_near_limit":
        segment = rng.choice(SEGMENTS)
        years = rng.choice([5, 7, 8, 10])
        limit = 60 if segment.startswith("salaried") else 70
        return make_profile(rng, segment=segment, years=years, age=limit - years - rng.randint(0, 2))
    if kind == "missed_and_deviations":
        return make_profile(rng, missed=rng.choice([1, 2, 3]), deviations=rng.choice([1, 2, 3]), score=rng.randint(675, 760))
    raise ValueError(kind)


SPECIAL_TEMPLATES = ("new_to_credit", "score_675_685", "ltv_near_cap", "large_ticket", "foir_near_cap",
                     "age_near_limit", "missed_and_deviations")


def competitor_variants(base: dict, obs: dict) -> list[tuple[str, tuple]]:
    """Competitor letters positioned relative to the base file's own floor and recommendation."""
    floor, rec = obs["floor"], obs["rec"]
    lenders = ["HDFC Bank", "SBI", "ICICI Bank", "Bajaj Housing Finance", "LIC Housing Finance", "PNB Housing Finance"]
    pick = lambda i: lenders[(base["loan"] // 100000 + i) % len(lenders)]  # noqa: E731
    return [("below_floor", (round(max(floor - 0.40, 11.0), 2), pick(0), True)),
            ("between", (round(floor + max(0.10, 0.5 * (rec - floor)), 2), pick(1), True)),
            ("at_or_above_rec", (round(min(rec + 0.25, 17.5), 2), pick(2), True)),
            ("claimed", (round(floor + 0.10, 2), pick(3), False))]


def run_probe(args) -> None:
    lo, hi = policy_band()
    from aavas.config import policy
    limits = policy().get("eligibility_limits") or {}
    rng = random.Random(args.seed)
    if args.from_records:  # reselect from saved probe results, no API needed
        saved = json.loads(Path(args.from_records).read_text(encoding="utf-8"))
        return finish_probe(args, saved["records"], rng, lo, hi, limits, saved["seconds"], saved["rejected"])
    rm, am = ds.login(args.base, *args.rm.split(":", 1)), ds.login(args.base, *args.am.split(":", 1))
    started = time.time()
    records: list[dict] = []
    failures = 0

    def work(item):
        tag, profile = item
        try:
            _, _, _, obs = price(args.base, rm, am, profile, limits, lo)
            return {"tag": tag, "profile": profile, "obs": obs}
        except BaseException as exc:  # noqa: BLE001 - a rejected profile is a result, not a crash
            return {"tag": tag, "error": str(exc)[:160]}

    def run(items, label):
        nonlocal failures
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for out in pool.map(work, items):
                done += 1
                if "obs" in out:
                    records.append(out)
                else:
                    failures += 1
                if done % 25 == 0 or done == len(items):
                    print(f"  {label}: {done}/{len(items)} priced, {len(records)} usable, {failures} rejected "
                          f"({time.time() - started:.0f}s)", flush=True)

    print(f"phase 1: {args.profiles} applicant profiles, no competitor", flush=True)
    run([("base", make_profile(rng, comp=None)) for _ in range(args.profiles)]
        + [("base", make_profile(rng, comp=None, foir=rng.uniform(0.38, 0.58),
                                 loan=rng.choice([1_000_000, 1_500_000, 2_000_000, 3_000_000])))
           for _ in range(args.stretched)], "phase 1")
    bases = [r for r in records if r["tag"] == "base" and not r["obs"]["infeasible"] and r["obs"]["grade"] in GRADES]

    print("phase 2: competitor letters placed against each base file's own floor and recommendation", flush=True)
    groups: dict[tuple, list] = {}
    for r in bases:
        groups.setdefault((r["obs"]["grade"], r["obs"]["band"]), []).append(r)
    items = []
    for (grade, band), rows in sorted(groups.items()):
        rows.sort(key=lambda r: (-(r["obs"]["rec"] - r["obs"]["floor"]), rng.random()))
        chosen = rows[:3] + rng.sample(rows[3:], min(len(rows[3:]), args.bases_per_group - 3))
        for r in chosen:
            for _, comp in competitor_variants(r["profile"], r["obs"]):
                items.append(("variant", {**r["profile"], "comp": list(comp), "negotiating": True,
                                          "channel": rng.choice(CHANNELS), "property_type": rng.choice(PROPERTIES),
                                          "city": rng.choice(CITIES)}))
    run(items, "phase 2")

    print("phase 3: special situations", flush=True)
    items = [(f"special:{k}", special_profile(k, rng)) for k in SPECIAL_TEMPLATES for _ in range(args.special_each)]
    run(items, "phase 3")

    run_seconds = round(time.time() - started, 1)
    if args.records:
        Path(args.records).write_text(json.dumps({"records": records, "rejected": failures, "seconds": run_seconds}),
                                      encoding="utf-8", newline="\n")
    finish_probe(args, records, rng, lo, hi, limits, run_seconds, failures)


def finish_probe(args, records, rng, lo, hi, limits, run_seconds, failures) -> None:
    versions = next((r["obs"]["versions"] for r in records), {})
    models = next((r["obs"]["model_versions"] for r in records), {})
    data = build_catalogue(records, rng, lo, hi, limits, run_seconds, failures, versions, models)
    data["probe"]["seed"] = args.seed
    out = Path(args.out) if args.out else CATALOGUE
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    s = data["summary"]
    print(f"\nwrote {out}: {s['cells_filled']} of {s['cells_total']} cells filled, {s['cells_impossible']} impossible "
          f"under the policy, {s['cells_not_found']} not found by the probe, {s['special_cases']} special, "
          f"{s['history_cases']} history ({run_seconds}s, {len(records)} profiles priced)")


# ---------------------------------------------------------------------------------------- selection
FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karan", "Nikhil", "Manish", "Suresh", "Rajendra", "Imran", "Harpreet",
         "Dinesh", "Anil", "Prakash", "Mahesh", "Ananya", "Priya", "Neha", "Kavita", "Sunita", "Meera", "Pooja", "Rekha",
         "Shalini", "Divya", "Farida", "Lakshmi", "Swati", "Anjali", "Vandana", "Rashmi", "Sanjay", "Vikas", "Yogesh", "Tarun"]
LAST = ["Sharma", "Verma", "Gupta", "Mehta", "Patel", "Shah", "Joshi", "Iyer", "Nair", "Reddy", "Kulkarni", "Deshmukh",
        "Agarwal", "Saxena", "Chauhan", "Rathore", "Sheikh", "Ansari", "Qureshi", "Bhatt", "Yadav", "Mishra", "Pandey",
        "Tiwari", "Dubey", "Jain", "Kapoor", "Malhotra", "Bansal", "Solanki", "Parmar", "Rawat", "Menon", "Pillai"]
LENDERS = ("HDFC Bank", "SBI", "ICICI Bank", "Bajaj Housing Finance", "LIC Housing Finance", "PNB Housing Finance")
COMPETITOR_REASONS = ["competitor_lower_rate", "customer_threatening_bt", "branch_competitive_pressure", "local_market_knowledge"]
OTHER_REASONS = ["strong_negotiation", "relationship_value", "month_end_target", "improve_s2d_conversion",
                 "salesperson_incentive", "other", "local_market_knowledge", "branch_competitive_pressure"]

SPECIALS = {
    "new_to_credit": "New to credit, no bureau score: the policy refers it for manual review",
    "score_675_685": "Bureau score just above the 675 minimum",
    "ltv_near_cap": "Loan to value within 2 points of the cap for its grade and loan size",
    "large_ticket": "Loan of Rs 70 lakh or more (the cap is Rs 75 lakh)",
    "foir_near_cap": "Stressed EMI burden within 4 points of the grade's FOIR cap",
    "age_near_limit": "Age at maturity close to the limit (salaried 60, self-employed 70)",
    "missed_and_deviations": "Missed payments and credit deviations on the file",
    "near_break_even": "An ask within 1 point of break-even",
    "balance_transfer": "Balance-transfer threat, competitor letter seen",
    "month_end_no_evidence": "Month-end target, nothing from the customer",
    "price_insensitive_existing": "Price-insensitive existing customer: little to gain from a cut",
}


def _eff(r: dict) -> list[str]:
    return available_positions(r["obs"]["rec"], r["obs"]["floor"])


def _ev(r: dict) -> str:
    p = r["profile"]
    return evidence_class(p.get("comp"), r["obs"]["floor"], r["obs"]["rec"])


def _verified(r: dict) -> bool:
    comp = r["profile"].get("comp")
    return bool(comp and comp[2])


def _usable(r: dict) -> bool:
    o = r["obs"]
    return (o["grade"] in GRADES and not o["infeasible"] and not o["resize_required"] and not o["no_offer"]
            and o["floor"] is not None)


def secondary_values(p: dict) -> set:
    return {("segment", p["segment"]), ("channel", p["channel"]), ("property", p["property_type"]),
            ("existing", p["existing"]), ("negotiating", p["negotiating"]), ("city", p["city"])}


def pick(cands: list[dict], covered: set, used: set, rng: random.Random):
    pool = [c for c in cands if c["_id"] not in used]
    if not pool:
        return None
    best = max(len(secondary_values(c["profile"]) - covered) for c in pool)
    return rng.choice([c for c in pool if len(secondary_values(c["profile"]) - covered) == best])


def special_match(kind: str, r: dict, limits: dict) -> bool:
    p, o = r["profile"], r["obs"]
    positions = _eff(r)
    comp = p.get("comp")
    if kind == "new_to_credit":
        return p["score"] < 0
    if kind == "score_675_685":
        return 675 <= p["score"] <= 685
    if kind == "ltv_near_cap":
        cap = eff_ltv_cap(o["grade"], p["loan"], limits)
        return cap - 0.02 - 1e-9 <= o["ltv"] <= cap + 1e-9
    if kind == "large_ticket":
        return p["loan"] >= 7_000_000
    if kind == "foir_near_cap":
        cap = o.get("foir_cap")
        return bool(cap) and cap - 0.04 <= o["foir_stressed"] <= cap - 0.002
    if kind == "age_near_limit":
        limit = 60 if p["segment"].startswith("salaried") else 70
        return limit - 2 <= p["age"] + p["years"] <= limit
    if kind == "missed_and_deviations":
        return p["verified"]["missed"] >= 1 and p["verified"]["deviations"] >= 1 and p["score"] >= 675
    if kind == "near_break_even":
        be = o.get("breakeven")
        asks = [ask_for(x, o["rec"], o["floor"]) for x in positions]
        return be is not None and any(a is not None and a - be <= 1.0 for a in asks)
    if kind == "balance_transfer":
        return bool(comp and comp[2] and p["negotiating"] and _ev(r) in ("between", "below_floor") and positions)
    if kind == "month_end_no_evidence":
        return not comp and not p["negotiating"] and "within_room" in positions
    if kind == "price_insensitive_existing":
        return not comp and not p["negotiating"] and p["existing"] and "within_room" in positions
    raise ValueError(kind)


def position_for(r: dict, prefer=POSITIONS) -> str | None:
    have = _eff(r)
    return next((x for x in prefer if x in have), None)


def probe_stats(records: list[dict]) -> dict:
    stats: dict[str, dict] = {}
    for r in records:
        if not _usable(r):
            continue
        key = f"{r['obs']['grade']}|{r['obs']['band']}"
        s = stats.setdefault(key, {"profiles": 0, "max_room_bps": 0, "min_gap_to_breakeven": None})
        s["profiles"] += 1
        s["max_room_bps"] = max(s["max_room_bps"], round((r["obs"]["rec"] - r["obs"]["floor"]) * 100))
        if r["obs"].get("breakeven") is not None:
            gap = round(r["obs"]["floor"] - float(r["obs"]["breakeven"]), 2)
            s["min_gap_to_breakeven"] = gap if s["min_gap_to_breakeven"] is None else min(s["min_gap_to_breakeven"], gap)
    return stats


# How far the recommendation must sit above the floor (strictly more than, in bps) for an ask
# position to exist: a within-room ask is at least 30 bps under the recommendation and above the
# floor; an at-floor ask must still be more than 25 bps under the recommendation.
NEEDED_ROOM_BPS = {"within_room": 30, "at_floor": 25}


def explain_unreachable(cell, stats: dict, lo: float) -> tuple[str, str]:
    """Why a cell has no case: ("impossible", why) when the policy rules it out, or
    ("not_found", why) when only the probe did - possible in principle, not produced."""
    grade, band, position, evidence = cell
    static = static_unreachable(grade, band, position, evidence, lo)
    if static:
        return "impossible", static[0].upper() + static[1:]
    s = stats.get(f"{grade}|{band}")
    if not s:
        return "not_found", (f"Not found: no probed profile was priced as grade {grade} at {band} months within policy "
                             f"(the grade follows the risk model)")
    room, n = s["max_room_bps"], s["profiles"]
    need = NEEDED_ROOM_BPS.get(position, -1)
    if evidence == "between":                     # a letter between our floor and our recommendation
        need = max(need, 0)
    if need >= 0 and room <= need:
        return "not_found", (f"Not found in {n} probed grade {grade} profiles at {band} months: the recommendation never "
                             f"rose more than {room} bps above the floor, and this cell needs more than {need} bps")
    return "not_found", (f"Not found in {n} probed grade {grade} profiles at {band} months: possible under the policy "
                         f"(the room reached {room} bps), but no probed file combined this evidence with this ask "
                         f"position; a wider probe may fill it")


def summarise_cells(cells: list[dict]) -> dict:
    """Filled, impossible under the policy, and not found by the probe."""
    count = lambda status: sum(1 for c in cells if c["status"] == status)   # noqa: E731
    return {"cells_total": len(cells), "cells_filled": count("filled"),
            "cells_impossible": count("impossible"), "cells_not_found": count("not_found")}


def build_catalogue(records, rng, lo, hi, limits, run_seconds, failures, versions, models) -> dict:
    for i, r in enumerate(records):
        r["_id"] = i
    usable = [r for r in records if _usable(r)]
    stats = probe_stats(records)
    used: set = set()
    covered: set = set()
    cases: list[dict] = []
    cells_out = []

    # primary matrix, rarest cells first
    cands = {}
    for cell in all_cells():
        g, b, pos, ev = cell
        if static_unreachable(*cell, lo=lo):
            cands[cell] = []
            continue
        cands[cell] = [r for r in usable if r["tag"] in ("base", "variant") and r["obs"]["grade"] == g
                       and r["obs"]["band"] == b and _ev(r) == ev and pos in _eff(r)]
    chosen: dict[tuple, dict] = {}
    for cell in sorted(cands, key=lambda c: (len(cands[c]), c)):
        r = pick(cands[cell], covered, used, rng)
        if r is None:
            continue
        used.add(r["_id"])
        covered |= secondary_values(r["profile"])
        chosen[cell] = r

    # special situations: dedicated probes first, then unused records
    special_cases = {}
    for kind in list(SPECIALS):
        pool = [r for r in usable if r["_id"] not in used and _eff(r) and special_match(kind, r, limits)
                and (r["tag"] == f"special:{kind}" or r["tag"] in ("base", "variant", "special:" + kind))]
        if kind == "price_insensitive_existing":
            pool.sort(key=lambda r: 0 if "insensitive" in str(r["obs"].get("sensitivity") or "") else 1)
            pool = [r for r in pool if "insensitive" in str(r["obs"].get("sensitivity") or "")] or pool
        if not pool:
            continue
        rows = [r for r in pool if r["tag"] == f"special:{kind}"] or pool
        r = pick(rows, covered, used, rng)
        used.add(r["_id"])
        covered |= secondary_values(r["profile"])
        special_cases[kind] = r

    # history: one profile per outcome, from records the other cases did not use
    history_cases = []
    wanted = [("approve_as_asked", "within_room"), ("approve_at_floor", "at_floor"), ("counter", "below_floor"),
              ("decline", "within_room"), ("decline", "below_floor"), ("uplift", None), ("uplift", None),
              ("approve_as_asked", "within_room"), ("counter", "below_floor")]
    for outcome, position in wanted:
        def fits(r, outcome=outcome, position=position):
            o, positions = r["obs"], _eff(r)
            if outcome == "uplift":
                return uplift_rate(o, hi) is not None and bool(positions)
            if outcome == "counter":
                return o["rec"] > o["floor"] + TOL and "below_floor" in positions and _verified(r)
            if outcome == "decline" and position == "within_room":
                return position in positions and _ev(r) in ("claimed", "none")
            if outcome == "decline":
                return position in positions
            return position in positions and _verified(r)
        pool = [r for r in usable if r["_id"] not in used and fits(r)]
        r = pick(pool, set(), used, rng)
        if r is None:
            continue
        used.add(r["_id"])
        history_cases.append((outcome, position, r))

    # names, mobiles, reasons, notes
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    counters = {"mobile": 0, "reason": {}}

    def next_mobile() -> str:
        counters["mobile"] += 1
        return f"9820013{counters['mobile']:03d}"

    def reason_for(ev: str) -> str:
        pool = COMPETITOR_REASONS if ev in ("below_floor", "between", "at_or_above_rec", "claimed") else OTHER_REASONS
        reason = min(pool, key=lambda x: (counters["reason"].get(x, 0), pool.index(x)))
        counters["reason"][reason] = counters["reason"].get(reason, 0) + 1
        return reason

    def finish(case_id, kind, r, position, *, situation=None, reason=None, outcome=None):
        p = dict(r["profile"])
        ev = _ev(r)
        p["name"], p["mobile"] = names.pop(), next_mobile()
        reason = reason or reason_for(ev)
        p["reason"] = reason
        p["notes"] = make_note(p, ev, situation)
        o = r["obs"]
        ask = ask_for(position, o["rec"], o["floor"], lo, hi) if position else None
        if ask is not None:
            check_band(ask, case_id, lo, hi)
        out = {"case_id": case_id, "kind": kind, "situation": situation, "outcome": outcome,
               "cell": {"grade": o["grade"], "band": o["band"], "position": position, "evidence": ev},
               "ask": ask, "reason": reason, "days": 1 + (len(cases) * 3) % 19, "profile": p,
               "observed": {k: v for k, v in o.items() if k not in ("versions", "model_versions")}}
        if outcome:
            out["decision"] = history_decision(outcome, o, ask, hi)
        return out

    matrix_n = 0
    for cell in sorted(chosen):
        matrix_n += 1
        g, b, pos, ev = cell
        cases.append(finish(f"M-{matrix_n:03d}", "matrix", chosen[cell], pos))
    cell_case = {cell_key(*c): cases[i]["case_id"] for i, c in enumerate(sorted(chosen))}
    for n, kind in enumerate(special_cases, 1):
        r = special_cases[kind]
        prefer = ("within_room", "at_floor", "below_floor")
        reason = {"balance_transfer": "customer_threatening_bt", "month_end_no_evidence": "month_end_target",
                  "price_insensitive_existing": "relationship_value"}.get(kind)
        position = position_for(r, prefer)
        cases.append(finish(f"S-{n:03d}", "special", r, position, situation=kind, reason=reason))
    for n, (outcome, position, r) in enumerate(history_cases, 1):
        o = r["obs"]
        if outcome == "uplift":
            position = position_for(r, ("within_room", "below_floor", "at_floor"))
        cases.append(finish(f"H-{n:03d}", "history", r, position, outcome=outcome,
                            reason="competitor_lower_rate" if _ev(r) != "none" else "strong_negotiation"))

    for cell in all_cells():
        key = cell_key(*cell)
        if key in cell_case:
            cells_out.append({"key": key, "grade": cell[0], "band": cell[1], "position": cell[2], "evidence": cell[3],
                              "status": "filled", "case_id": cell_case[key]})
        else:
            status, why = explain_unreachable(cell, stats, lo)
            cells_out.append({"key": key, "grade": cell[0], "band": cell[1], "position": cell[2], "evidence": cell[3],
                              "status": status, "reason": why})
    unreachable_other = [
        {"item": "Grade D", "reason": "Declined under standard policy before it reaches the area manager: no card rate, "
                                      "no offer, so no rate request can exist (risk_grade_loading_bps D: null)"},
        {"item": "LAP residential / LAP commercial", "reason": "The intake API accepts only the four housing property types "
                                                              "(resale_flat, builder_flat, self_construction, plot_plus_construction)"},
        {"item": "An ask within 25 bps of the recommendation", "reason": "The RM may send it alone, with a reason "
                                                                         "(offer_below_recommendation_bps 25): it never reaches the area manager"},
        {"item": "An ask beyond the area manager's 100 bps", "reason": "The band is [card - 60 bps, card], narrower than her "
                                                                       "100 bps, so the policy floor always binds first"},
        {"item": "An ask above the recommendation", "reason": "A rate request is a request for a lower rate; going above the "
                                                              "recommendation is the area manager's uplift, never above the matrix rate"},
    ]
    for kind, label in SPECIALS.items():
        if kind not in special_cases:
            gap = min((s["min_gap_to_breakeven"] for s in stats.values() if s["min_gap_to_breakeven"] is not None), default=None)
            why = (f"No probed profile matched; the smallest gap between a floor and break-even was {gap:.2f} points"
                   if kind == "near_break_even" and gap is not None else "No probed profile matched this situation")
            unreachable_other.append({"item": label, "reason": why})
    for outcome in ("approve_as_asked", "approve_at_floor", "counter", "decline", "uplift"):
        if not any(h[0] == outcome for h in history_cases):
            unreachable_other.append({"item": f"History outcome: {outcome}", "reason": "No probed profile allows it (see the probe statistics)"})
    secondary = secondary_coverage(cases)
    return {"generated": date.today().isoformat(), "versions": versions, "model_versions": models,
            "policy_band": [lo, hi],
            "probe": {"profiles_priced": len(records), "rejected": failures, "seconds": run_seconds,
                      "seed": None, "stats": stats},
            "summary": {**summarise_cells(cells_out), "matrix_cases": matrix_n,
                        "special_cases": len(special_cases), "history_cases": len(history_cases),
                        "pending_cases": len(cases) - len(history_cases)},
            "cells": cells_out, "other_unreachable": unreachable_other, "secondary": secondary, "cases": cases}


def secondary_coverage(cases: list[dict]) -> dict:
    out: dict[str, dict] = {k: {} for k in ("segment", "channel", "property_type", "reason", "existing", "negotiating")}
    for c in cases:
        p = c["profile"]
        for key, val in (("segment", p["segment"]), ("channel", p["channel"]), ("property_type", p["property_type"]),
                         ("reason", c["reason"]), ("existing", str(p["existing"])), ("negotiating", str(p["negotiating"]))):
            out[key].setdefault(val, []).append(c["case_id"])
    return out


# ---------------------------------------------------------------------------------------- notes
SEGMENT_NOTE = {"salaried_formal": "Salaried at a listed company.",
                "salaried_informal": "Paid by a small firm, part of it in cash.",
                "self_employed_professional": "Runs a professional practice.",
                "self_employed_non_professional": "Runs a small business."}
PROPERTY_NOTE = {"resale_flat": "Buying a resale flat.", "builder_flat": "Buying a flat from a builder.",
                 "self_construction": "Building on a plot the family owns.",
                 "plot_plus_construction": "Buying a plot and building on it."}
SITUATION_NOTE = {
    "new_to_credit": "First loan, no credit card before, so no bureau score.",
    "score_675_685": "Credit score is only just above our minimum.",
    "ltv_near_cap": "Putting down very little of the property's value.",
    "large_ticket": "One of the largest loans we make.",
    "foir_near_cap": "The EMI takes almost all of what the policy lets us count.",
    "age_near_limit": "Loan will run almost to the age limit.",
    "missed_and_deviations": "Has a few late EMIs and some credit deviations on the file.",
    "near_break_even": "The ask is close to what the loan costs us.",
    "balance_transfer": "Will move the existing loan to the other lender unless we match.",
    "month_end_no_evidence": "The branch is short of the month-end target and wants it closed this week.",
    "price_insensitive_existing": "Says the EMI is fine; only wants it sanctioned in time.",
}


def make_note(p: dict, ev: str, situation: str | None) -> str:
    comp = p.get("comp")
    parts = [SEGMENT_NOTE[p["segment"]], PROPERTY_NOTE[p["property_type"]]]
    if comp and comp[2]:
        lead = (f"Moving the existing loan to {comp[1]} at {comp[0]:.2f}% unless we match; the letter is attached."
                if situation == "balance_transfer"
                else f"{comp[1]} sanction letter at {comp[0]:.2f}% seen; will sign with them unless we come closer.")
        parts.append(lead)
    elif comp:
        parts.append(f"Says {comp[1]} offered {comp[0]:.2f}%; has not shown the letter yet.")
    elif p["negotiating"]:
        parts.append("Asked if the rate can be lower; no other offer.")
    else:
        parts.append("Not asking for a lower rate; happy with the service.")
    if p["existing"]:
        parts.append("Existing Aavas customer.")
    if situation and situation in SITUATION_NOTE and situation != "balance_transfer":
        parts.append(SITUATION_NOTE[situation])
    return " ".join(parts)


# ---------------------------------------------------------------------------------------- history decisions
def history_decision(outcome: str, obs: dict, ask: float | None, hi: float) -> dict:
    rec, floor = obs["rec"], obs["floor"]
    if outcome == "approve_as_asked":
        return {"approve": True, "rate": ask, "note": "Approved as asked: inside the room above the floor and the file supports it."}
    if outcome == "approve_at_floor":
        return {"approve": True, "rate": floor, "note": "Approved at the policy floor, the lowest the policy allows."}
    if outcome == "counter":
        return {"approve": True, "rate": floor, "note": "Countered at the floor: nothing lower is allowed, and the file does not need more."}
    if outcome == "decline":
        return {"approve": False, "rate": None, "note": "Declined: the recommended rate stands; no evidence for a cut."}
    if outcome == "uplift":
        return {"approve": True, "uplift": True, "rate": uplift_rate(obs, hi), "reason": "customer_accepts_higher_rate",
                "note": "Declined the discount and priced a little above the recommendation; the customer is not price sensitive."}
    raise ValueError(outcome)


# ---------------------------------------------------------------------------------------- create
def load_catalogue(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or CATALOGUE).read_text(encoding="utf-8"))


def stamp(days: float, hours: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days, hours=hours)).isoformat(timespec="seconds")


def backdate_decisions(decided: list[tuple[str, str, int]]) -> None:
    from aavas.governance import store
    with store.connect() as conn:
        for loan_id, request_id, days in decided:
            when = stamp(days, -MIN_DAYS_BEFORE_DECISION)
            conn.execute("UPDATE rate_requests SET decided_at = ? WHERE request_id = ?", (when, request_id))
            conn.execute("UPDATE offers SET sent_at = ? WHERE request_id = ?", (when, request_id))
    print("set back", len(decided), "decisions")


def run_create(args) -> None:
    data = load_catalogue(args.catalogue)
    lo, hi = policy_band()
    from aavas.config import policy
    limits = policy().get("eligibility_limits") or {}
    rm, am = ds.login(args.base, *args.rm.split(":", 1)), ds.login(args.base, *args.am.split(":", 1))
    cases = [c for c in data["cases"] if args.decide_history or c["kind"] != "history"]
    if args.only:
        cases = [c for c in cases if c["kind"] in args.only.split(",")]
    made, decided, mismatches = [], [], []
    started = time.time()
    for n, c in enumerate(cases, 1):
        p = c["profile"]
        p["comp"] = tuple(p["comp"]) if p.get("comp") else None
        loan_id, priced, full, obs = price(args.base, rm, am, p, limits, lo, full_flow=True)
        ev = evidence_class(p["comp"], obs["floor"], obs["rec"])
        intended = c["cell"]
        seen = {"grade": obs["grade"], "band": obs["band"], "evidence": ev}
        position = intended["position"]
        if position not in available_positions(obs["rec"], obs["floor"], lo, hi):
            alt = position_for({"obs": obs, "profile": p}, ("within_room", "at_floor", "below_floor"))
            if alt is None:
                mismatches.append((c["case_id"], f"no valid ask now (rec {obs['rec']:.2f}, floor {obs['floor']:.2f}); skipped"))
                print(f"{n:3}. {c['case_id']} {p['name']:24} SKIPPED: no valid ask")
                continue
            position = alt
        ask = check_band(ask_for(position, obs["rec"], obs["floor"], lo, hi), f"{c['case_id']} {p['name']}", lo, hi)
        seen["position"] = position
        diff = {k: (intended[k], seen[k]) for k in seen if intended.get(k) != seen[k]}
        if diff:
            mismatches.append((c["case_id"], diff))
        req = ds.call(args.base, "POST", "/pricing/rate-requests",
                      {"loan_id": loan_id, "rate": ask, "reason": c["reason"], "note": None}, rm)
        made.append((loan_id, req["request_id"], c["days"]))
        line = (f"{n:3}. {c['case_id']} {p['name']:24} {p['city']:10} grade {obs['grade']:2} {obs['band']:6} card {obs['card']:5} "
                f"floor {obs['floor']:5} rec {obs['rec']:5} ask {ask:5} {position:11} letter {ev}")
        if c["kind"] == "history":
            plan = history_decision(c["outcome"], obs, ask, hi)
            if plan.get("rate") is None and plan["approve"]:
                mismatches.append((c["case_id"], f"{c['outcome']} not possible now; left pending"))
            else:
                if plan.get("rate") is not None:
                    check_band(plan["rate"], f"{c['case_id']} {c['outcome']}", lo, hi)
                body = {k: v for k, v in plan.items() if v is not None or k == "approve"}
                ds.call(args.base, "POST", f"/pricing/rate-requests/{req['request_id']}/decide", body, am)
                decided.append(made[-1])
                line += f"  -> {c['outcome']} {plan.get('rate') or ''}"
        print(line, flush=True)
    if not args.no_backdate:
        ds.backdate(made)
        if decided:
            backdate_decisions(decided)
    print(f"\ncreated {len(made)} of {len(cases)} cases in {time.time() - started:.0f}s "
          f"({len(decided)} decided as the area manager)")
    if mismatches:
        print(f"{len(mismatches)} case(s) landed differently from the catalogue; each is labelled by what was observed:")
        for case_id, what in mismatches:
            print("  ", case_id, what)
    else:
        print("every case landed in its intended cell")


# ---------------------------------------------------------------------------------------- report
POS_LABEL = {"within_room": "within room", "at_floor": "at floor", "below_floor": "below floor"}
EV_LABEL = {"below_floor": "letter < floor", "between": "letter floor..rec", "at_or_above_rec": "letter >= rec",
            "claimed": "claim, no letter", "none": "no competitor"}
OUTCOME_LABEL = {"approve_as_asked": "Approve as asked", "approve_at_floor": "Approve at the floor", "counter": "Counter at the floor",
                 "decline": "Decline", "uplift": "Uplift above the recommendation"}


def render_report(data: dict) -> str:
    cases = {c["case_id"]: c for c in data["cases"]}
    s = data["summary"]
    lo, hi = data["policy_band"]
    v = data.get("versions") or {}
    secs = data["probe"].get("seconds")
    probe_time = (f"in {secs} s" if secs is not None
                  else f"(run time not recorded: {data['probe'].get('seconds_note') or 'unknown'})")
    lines = ["# Area manager cases: the coverage matrix", "",
             f"Probed {data['generated']} against app {v.get('app_version', '?')}, policy {v.get('policy_version', '?')}, "
             f"model config {v.get('model_config_version', '?')}. Every rate is inside the policy band "
             f"({lo:.2f}-{hi:.2f}%). Generated from `scripts/demo_catalogue.json` by "
             f"`python scripts/demo_catalogue.py report`; do not edit by hand.", "",
             "## What is covered", "",
             f"- Primary cells (grade x tenure band x ask position x evidence): {s['cells_total']} in all, "
             f"**{s['cells_filled']} filled, {s['cells_impossible']} impossible under the policy, "
             f"{s['cells_not_found']} not found by the probe** (possible under the policy, but no probed file "
             f"produced them).",
             f"- Pending cases for the area manager: {s['pending_cases']} ({s['matrix_cases']} matrix, "
             f"{s['special_cases']} special situations). History cases decided by the area manager: {s['history_cases']}.",
             f"- Probe: {data['probe']['profiles_priced']} profiles priced {probe_time} "
             f"({data['probe']['rejected']} rejected by the intake rules).", "",
             "Ask position: **within room** is at least 30 bps under the recommendation and above the floor; "
             "**at floor** is exactly the policy floor; **below floor** is under it (the area manager can approve only "
             "down to the floor). Evidence is read against the file's own floor and recommendation: a verified letter "
             "below our floor, between the floor and the recommendation, or at or above the recommendation; a claim "
             "the RM has not seen the letter for; or no competitor.", ""]
    cell_by = {c["key"]: c for c in data["cells"]}
    lines += ["## Matrix", "", "Each entry is a case id (see the case list); `x` with a note number for a cell the policy "
              "makes impossible; `?` with a note number for a cell that is possible under the policy but that no "
              "probed file produced.", ""]
    notes: dict[str, int] = {}

    def note_no(reason: str) -> int:
        return notes.setdefault(reason, len(notes) + 1)

    for g in GRADES:
        lines += [f"### Grade {g}", "", "| Tenure | Ask position | " + " | ".join(EV_LABEL[e] for e in EVIDENCE) + " |",
                  "|---|---|" + "---|" * len(EVIDENCE)]
        for b in BANDS:
            for pos in POSITIONS:
                row = []
                for e in EVIDENCE:
                    c = cell_by[cell_key(g, b, pos, e)]
                    mark = "x" if c["status"] == "impossible" else "?"
                    row.append(c["case_id"] if c["status"] == "filled" else f"{mark} ({note_no(c['reason'])})")
                lines.append(f"| {b} | {POS_LABEL[pos]} | " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## Cells without a case, and why", ""]
    lines += [f"{i}. {r}" for r, i in sorted(notes.items(), key=lambda kv: kv[1])]
    lines += ["", "Outside the matrix:", ""]
    lines += [f"- **{o['item']}**: {o['reason']}" for o in data["other_unreachable"]]
    lines += ["", "## Secondary values", "", "| Dimension | Value | Cases |", "|---|---|---|"]
    expected = {"segment": SEGMENTS, "channel": CHANNELS, "property_type": PROPERTIES,
                "existing": ("True", "False"), "negotiating": ("True", "False")}
    for dim, vals in data["secondary"].items():
        want = expected.get(dim) or tuple(sorted(vals))
        for val in want:
            ids = vals.get(val, [])
            lines.append(f"| {dim} | {val} | {len(ids)} ({', '.join(ids[:4])}{', ...' if len(ids) > 4 else ''}) |"
                         if ids else f"| {dim} | {val} | none |")
    lines += ["", "## Special situations", "", "| Situation | Case | Detail |", "|---|---|---|"]
    done = {c["situation"]: c for c in data["cases"] if c["kind"] == "special"}
    for kind, label in SPECIALS.items():
        c = done.get(kind)
        if c:
            o = c["observed"]
            lines.append(f"| {label} | {c['case_id']} | {c['profile']['name']}, {c['profile']['city']}: grade {o['grade']}, "
                         f"card {o['card']:.2f}, floor {o['floor']:.2f}, rec {o['rec']:.2f}, ask {c['ask']:.2f} |")
        else:
            why = next((x["reason"] for x in data["other_unreachable"] if x["item"] == label), "not reached")
            lines.append(f"| {label} | none | {why} |")
    lines += ["", "## All cases", "",
              "| Case | Customer | City | Segment | Channel | Property | Tenure | Loan | Score | Letter | Existing | Negotiating "
              "| Grade | Card | Floor | Rec | Ask | Position | Reason |", "|" + "---|" * 19]
    for c in data["cases"]:
        if c["kind"] == "history":
            continue
        p, o = c["profile"], c["observed"]
        ev = c["cell"]["evidence"]
        lines.append(f"| {c['case_id']} | {p['name']} | {p['city']} | {p['segment']} | {p['channel']} | {p['property_type']} "
                     f"| {p['years']}y | {p['loan'] / 100000:.1f}L | {p['score'] if p['score'] >= 0 else 'none'} | {EV_LABEL[ev]} "
                     f"| {'yes' if p['existing'] else 'no'} | {'yes' if p['negotiating'] else 'no'} | {o['grade']} "
                     f"| {o['card']:.2f} | {o['floor']:.2f} | {o['rec']:.2f} | {c['ask']:.2f} "
                     f"| {POS_LABEL.get(c['cell']['position'], '-')} | {c['reason']} |")
    lines += ["", "## Decision history", "",
              "With `--decide-history` these are decided by the area manager (meera) and set back in time, for the precedent "
              "and decision-memory panels.", "", "| Case | Customer | Grade | Tenure | Card | Floor | Rec | Ask | Decision | Rate |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in data["cases"]:
        if c["kind"] != "history":
            continue
        p, o, d = c["profile"], c["observed"], c["decision"]
        lines.append(f"| {c['case_id']} | {p['name']}, {p['city']} | {o['grade']} | {c['cell']['band']} | {o['card']:.2f} "
                     f"| {o['floor']:.2f} | {o['rec']:.2f} | {c['ask']:.2f} | {OUTCOME_LABEL[c['outcome']]} "
                     f"| {d['rate']:.2f} |" if d.get("rate") is not None else
                     f"| {c['case_id']} | {p['name']}, {p['city']} | {o['grade']} | {c['cell']['band']} | {o['card']:.2f} "
                     f"| {o['floor']:.2f} | {o['rec']:.2f} | {c['ask']:.2f} | {OUTCOME_LABEL[c['outcome']]} | rec stands |")
    lines += ["", "## How to run", "",
              "```", "# 1. Probe the engine (throwaway API only, never the live server)",
              "PYTHONPATH=src python scripts/demo_catalogue.py probe --base http://127.0.0.1:8011 --scratch", "",
              "# 2. Create the pending cases on a server", "PYTHONPATH=src python scripts/demo_catalogue.py create --base http://127.0.0.1:8000", "",
              "# 2b. Also create and decide the history cases", "PYTHONPATH=src python scripts/demo_catalogue.py create --base http://127.0.0.1:8000 --decide-history", "",
              "# 3. Rewrite this page from the catalogue", "python scripts/demo_catalogue.py report", "",
              "# 4. Cut the area manager's list to 10 pending requests that still show every scene (store in AAVAS_STORE_URL)",
              "PYTHONPATH=src python scripts/demo_catalogue.py trim --keep 10            # dry run",
              "PYTHONPATH=src python scripts/demo_catalogue.py trim --keep 10 --apply    # backs the store up, then removes", "```", "",
              "### Probing on another machine (the server's models differ from a laptop's)", "",
              "Recommendations move a little between model sets, so probe where the cases will live, writing a separate "
              "catalogue, then create from it:", "",
              "```", "# a throwaway API on a copy of the store (works on Linux and Windows; see scripts/scratch_api.py)",
              "PYTHONPATH=src python scripts/scratch_api.py start --source /path/to/aavas_governance.db --port 8011 --dir /tmp/scratch",
              "PYTHONPATH=src python scripts/demo_catalogue.py probe --base http://127.0.0.1:8011 --scratch --out /tmp/catalogue.json",
              "PYTHONPATH=src python scripts/scratch_api.py stop --dir /tmp/scratch     # stops the API, deletes the copy",
              "PYTHONPATH=src python scripts/demo_catalogue.py create --base http://127.0.0.1:8000 --catalogue /tmp/catalogue.json",
              "python scripts/demo_catalogue.py report --catalogue /tmp/catalogue.json --out /tmp/cases.md", "```", "",
              "`create` checks that every case landed in its intended cell, prints any that did not (and labels them by what "
              "was observed), and refuses to post a rate outside the policy band. It back-dates the files so the worklists "
              "show aging; pass `--no-backdate` to skip that. Run it once per server: a second run adds a second set of cases.", ""]
    return "\n".join(lines)


def run_report(args=None) -> None:
    out = Path(getattr(args, "out", None) or REPORT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(load_catalogue(getattr(args, "catalogue", None))), encoding="utf-8", newline="\n")
    print("wrote", out)


# ---------------------------------------------------------------------------------------- trim
# Cut the area manager's list to N pending requests that between them still show every scene.
HAND_WRITTEN_PREFIXES = ("9820011", "9820012")   # demo_scenarios.py core and extended; the catalogue is 9820013xxx
LOAN_TABLES_NEVER_TRIMMED = ("requests", "loan_book", "audit_log")   # the API request log, the history book, the audit log
WEIGHT = {"grade": 4, "band": 4, "position": 4, "evidence": 4, "reason": 3, "segment": 3, "special": 3, "pair": 1}
SCENE_SPECIALS = {
    "new_to_credit": "new to credit, manual review", "balance_transfer": "balance-transfer threat",
    "month_end_no_evidence": "month-end target, nothing from the customer", "price_insensitive_existing": "price-insensitive existing customer",
    "large_ticket": "ticket of Rs 70 lakh or more", "ltv_near_cap": "LTV near its cap", "foir_near_cap": "EMI burden near its cap",
    "age_near_limit": "age at maturity near the limit", "score_675_685": "bureau score just above 675",
    "late_or_deviations": "late EMIs or deviations",
}
EVIDENCE_PHRASE = {"below_floor": "verified letter under our floor", "between": "verified letter between floor and recommendation",
                   "at_or_above_rec": "verified letter at or above our rate", "claimed": "claim, letter not seen",
                   "none": "no competitor"}
POSITION_PHRASE = {"within_room": "asks within the room", "at_floor": "asks the floor", "below_floor": "asks below the floor"}


def _json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    return value or {}


def request_features(req: dict, decision: dict, applicant: dict, limits: dict, lo: float = 13.0) -> dict | None:
    """What a rate request shows: its grade, band, ask position, evidence, reason, segment and special
    situations, read from the stored request, its priced decision and the applicant record."""
    payload = _json(decision.get("payload"))
    grade = (payload.get("risk") or {}).get("risk_grade")
    card = decision.get("card_rate")
    if grade not in GRADES or card is None:
        return None
    a = applicant or {}
    sc = a.get("sales_context") or {}
    floor, rec, ask = floor_of(float(card), lo), float(req["recommended_rate"]), float(req["requested_rate"])
    years = int(a.get("tenure_years") or a.get("requested_tenure_years") or 10)
    comp = ((float(sc["competitor_rate"]), sc.get("competitor_name"), bool(sc.get("competitor_evidence_verified")))
            if sc.get("competitor_rate") is not None else None)
    evidence, position = evidence_class(comp, floor, rec), position_of(ask, floor)
    loan = float(a.get("sanctioned_loan_amount") or a.get("requested_loan_amount") or 0)
    value = float(a.get("appraised_property_value") or a.get("estimated_property_value") or 0)
    income = float(a.get("verified_monthly_income") or a.get("declared_monthly_income") or 0)
    score = a.get("bureau_score")
    segment = a.get("segment")
    limit = 60 if str(segment).startswith("salaried") else 70
    foir_cap = (limits.get(grade) or {}).get("foir_max")
    foir = stressed_foir(float(card), years, loan, income, float(a.get("declared_obligations") or 0)) if loan and income else None
    special = set()
    if a.get("new_to_credit") or (score is not None and float(score) < 0):
        special.add("new_to_credit")
    if req.get("reason") == "customer_threatening_bt":
        special.add("balance_transfer")
    if req.get("reason") == "month_end_target" and comp is None:
        special.add("month_end_no_evidence")
    if (comp is None and sc.get("customer_negotiating") is False and sc.get("existing_relationship")
            and "insensitive" in str((payload.get("price_sensitivity") or {}).get("sensitivity_label") or "")):
        special.add("price_insensitive_existing")
    if loan >= 7_000_000:
        special.add("large_ticket")
    if loan and value:
        cap = eff_ltv_cap(grade, loan, limits)
        if cap - 0.02 - 1e-9 <= loan / value <= cap + 1e-9:
            special.add("ltv_near_cap")
    if foir is not None and foir_cap and float(foir_cap) - 0.04 <= foir <= float(foir_cap) - 0.002:
        special.add("foir_near_cap")
    if a.get("applicant_age") and limit - 2 <= int(a["applicant_age"]) + years <= limit:
        special.add("age_near_limit")
    if score is not None and 675 <= float(score) <= 685:
        special.add("score_675_685")
    if int(a.get("delinquency_count_24m") or 0) >= 1 or int(a.get("deviation_count") or 0) >= 1:
        special.add("late_or_deviations")
    elements = {("grade", grade), ("band", band_of(years)), ("position", position), ("evidence", evidence),
                ("reason", req.get("reason")), ("segment", segment)}
    elements |= {("special", s) for s in special}
    elements |= {("pair", f"grade-band:{grade}|{band_of(years)}"), ("pair", f"grade-position:{grade}|{position}"),
                 ("pair", f"grade-evidence:{grade}|{evidence}")}
    return {"request_id": req["request_id"], "loan_id": req["loan_id"], "status": req["status"],
            "requested_at": req.get("requested_at") or "", "mobile": str(a.get("mobile") or ""),
            "name": a.get("name") or req.get("applicant_name"), "grade": grade, "years": years, "band": band_of(years),
            "evidence": evidence, "position": position, "reason": req.get("reason"), "segment": segment,
            "special": sorted(special), "elements": elements, "ask": ask, "rec": rec, "floor": floor}


def scene_label(f: dict) -> str:
    """A short human label of the scene a request shows."""
    parts = [f["grade"], f"{f['years']}y", EVIDENCE_PHRASE[f["evidence"]], POSITION_PHRASE[f["position"]]]
    if f["special"]:
        scene = "; ".join(SCENE_SPECIALS[s] for s in f["special"][:2])
    elif f["position"] == "below_floor":
        scene = "counter at the floor" if f["rec"] > f["floor"] + TOL else "nothing to give, hold the recommendation"
    elif f["evidence"] in ("none",):
        scene = "no evidence for a cut, hold" if f["position"] == "within_room" else "approve only at the floor"
    elif f["evidence"] == "claimed":
        scene = "verify the letter first"
    else:
        scene = "approve if the letter holds" if f["position"] == "within_room" else "approve at the floor"
    return f"{', '.join(parts)}: {scene}"


def is_hand_written(f: dict) -> bool:
    return f["mobile"].startswith(HAND_WRITTEN_PREFIXES)


def upload_folder(loan_id) -> Path | None:
    """The upload folder of a loan, or None when the id is not a plain loan id or would resolve
    anywhere but directly under the upload root: a stored id is data, never a path to trust."""
    from aavas.intake import applications
    if not applications.LOAN_ID.match(str(loan_id or "")):
        return None
    root = applications.upload_dir().resolve()
    folder = (root / str(loan_id)).resolve()
    return folder if folder.parent == root else None


def select_scenes(features: list[dict], keep: int) -> list[dict]:
    """Greedy weighted set cover over the pending requests; deterministic. Ties prefer the
    hand-written scenarios, then the older request, then the request id."""
    pool = sorted((f for f in features if f["status"] == "pending"), key=lambda f: (f["requested_at"], f["request_id"]))
    covered: set = set()
    chosen: list[dict] = []
    while pool and len(chosen) < keep:
        def gain(f):
            return sum(WEIGHT[dim] for dim, _ in f["elements"] - covered)
        best = min(pool, key=lambda f: (-gain(f), 0 if is_hand_written(f) else 1, f["requested_at"], f["request_id"]))
        chosen.append(best)
        covered |= best["elements"]
        pool.remove(best)
    return improve_by_swaps(chosen, pool)


def _coverage_score(chosen: list[dict]) -> tuple[int, int]:
    """Single values covered first (every grade, band, position, evidence, reason, segment,
    special situation), then the weighted total, pairs included."""
    covered = set().union(*(f["elements"] for f in chosen)) if chosen else set()
    return sum(1 for dim, _ in covered if dim != "pair"), sum(WEIGHT[dim] for dim, _ in covered)


def improve_by_swaps(chosen: list[dict], pool: list[dict]) -> list[dict]:
    """Greedy can spend a slot on a pair and leave a single value out. Swap a kept request for
    one left in the pool while the score strictly rises; deterministic, with the same
    preferences as the greedy step for the incoming request."""
    chosen, pool = list(chosen), list(pool)
    while True:
        base = _coverage_score(chosen)
        best = None
        for i in range(len(chosen)):
            for c in pool:
                trial = chosen[:i] + [c] + chosen[i + 1:]
                score = _coverage_score(trial)
                if score <= base:
                    continue
                key = (tuple(-s for s in score), 0 if is_hand_written(c) else 1, c["requested_at"], c["request_id"], i)
                if best is None or key < best[0]:
                    best = (key, i, c)
        if best is None:
            return chosen
        _, i, c = best
        pool.append(chosen[i])
        pool.remove(c)
        chosen[i] = c


def coverage_table(features: list[dict], kept: list[dict], lo_allowed: list[str]) -> list[tuple[str, list, list]]:
    """(dimension, covered values, missed values) for values present among the pending requests."""
    pending = [f for f in features if f["status"] == "pending"]
    universe = {"grade": GRADES, "band": BANDS, "position": POSITIONS, "evidence": EVIDENCE,
                "reason": lo_allowed, "segment": SEGMENTS, "special": tuple(SCENE_SPECIALS)}
    rows = []
    for dim, values in universe.items():
        present = {v for f in pending for d, v in f["elements"] if d == dim}
        have = {v for f in kept for d, v in f["elements"] if d == dim}
        rows.append((dim, [v for v in values if v in have],
                     [v for v in values if v in present and v not in have]))
    return rows


def load_request_features(conn, limits: dict, lo: float) -> list[dict]:
    out = []
    for req in conn.execute("SELECT * FROM rate_requests").fetchall():
        req = dict(req)
        dec = conn.execute("SELECT * FROM decisions WHERE decision_id = ?", (req["decision_id"],)).fetchone()
        app = conn.execute("SELECT applicant FROM applications WHERE loan_id = ?", (req["loan_id"],)).fetchone()
        if dec is None or app is None:
            continue
        f = request_features(req, dict(dec), _json(app["applicant"]), limits, lo)
        if f:
            out.append(f)
    return out


def _tables(conn) -> dict[str, list[str]]:
    return {t: [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
            for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")}


def removal_plan(conn, loan_ids: list[str]) -> dict[str, tuple[str, list]]:
    """For each table with rows to remove: the WHERE clause and its parameters. A table is trimmed when it
    has a loan_id (or request_id / decision_id) column; the API request log, the history book and the audit log never are."""
    if not loan_ids:
        return {}
    ph = ",".join("?" for _ in loan_ids)
    req_ids = [r[0] for r in conn.execute(f"SELECT request_id FROM rate_requests WHERE loan_id IN ({ph})", loan_ids)]
    dec_ids = [r[0] for r in conn.execute(f"SELECT decision_id FROM decisions WHERE loan_id IN ({ph})", loan_ids)]
    plan: dict[str, tuple[str, list]] = {}
    for table, cols in _tables(conn).items():
        if table in LOAN_TABLES_NEVER_TRIMMED:
            continue
        if "loan_id" in cols:
            plan[table] = (f"loan_id IN ({ph})", list(loan_ids))
        elif "request_id" in cols and req_ids:
            plan[table] = (f"request_id IN ({','.join('?' for _ in req_ids)})", req_ids)
        elif "decision_id" in cols and dec_ids:
            plan[table] = (f"decision_id IN ({','.join('?' for _ in dec_ids)})", dec_ids)
    # a borrower account goes with its files, unless a file that stays still belongs to it
    gone = {r[0] for r in conn.execute(f"SELECT owner FROM applications WHERE loan_id IN ({ph})", loan_ids)}
    gone |= {r[0] for r in conn.execute(f"SELECT username FROM customer_logins WHERE loan_id IN ({ph})", loan_ids)}
    stay = {r[0] for r in conn.execute(f"SELECT owner FROM applications WHERE loan_id NOT IN ({ph})", loan_ids)}
    stay |= {r[0] for r in conn.execute(f"SELECT username FROM customer_logins WHERE loan_id NOT IN ({ph})", loan_ids)}
    accounts = sorted(u for u in gone - stay if u)
    if accounts:
        plan["borrower_accounts"] = (f"username IN ({','.join('?' for _ in accounts)})", accounts)
    return plan


def run_trim(args) -> None:
    import shutil
    import sqlite3
    from aavas.config import policy
    from aavas.governance import audit, store
    from aavas.intake import applications
    lo, _ = policy_band()
    limits = policy().get("eligibility_limits") or {}
    allowed = list((policy().get("override") or {}).get("allowed_reasons") or [])
    with store.connect() as conn:
        features = load_request_features(conn, limits, lo)
        total_requests = conn.execute("SELECT COUNT(*) FROM rate_requests").fetchone()[0]
        kept = select_scenes(features, args.keep)
        kept_loans = {f["loan_id"] for f in kept}
        every_loan = [r[0] for r in conn.execute("SELECT DISTINCT loan_id FROM rate_requests")]
        remove = sorted(set(every_loan) - kept_loans)
        plan = removal_plan(conn, remove)
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t} WHERE {w}", p).fetchone()[0] for t, (w, p) in plan.items()}
        audit_before = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        requests_before = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    pending = sum(f["status"] == "pending" for f in features)
    print(f"{total_requests} rate requests ({pending} pending, {total_requests - pending} decided); keeping {len(kept)} of the pending")
    print("\nKept:")
    for n, f in enumerate(kept, 1):
        origin = "hand-written" if is_hand_written(f) else "catalogue" if f["mobile"].startswith("9820013") else "other"
        print(f"{n:3}. {f['request_id']} {f['loan_id']} {str(f['name'])[:22]:22} [{origin}] {scene_label(f)}")
    print("\nCoverage (values present among the pending requests):")
    for dim, have, miss in coverage_table(features, kept, allowed):
        print(f"  {dim:9} covered {len(have)}: {', '.join(map(str, have)) or '-'}")
        print(f"  {'':9} missed  {len(miss)}: {', '.join(map(str, miss)) or '-'}")
    print(f"\nWould remove {len(remove)} loan files (every other file with a rate request):")
    for t, n in sorted(counts.items()):
        print(f"  {t:24} {n}")
    print(f"  {'upload folders':24} {sum(1 for l in remove if (upload_folder(l) or Path('/nonexistent')).exists())}")
    unsafe = [l for l in remove if upload_folder(l) is None]
    if unsafe:
        print(f"  upload folders skipped, loan id not a plain id: {unsafe}")
    print(f"  not touched: requests ({requests_before} rows), audit_log ({audit_before} rows), loan_book, files with no rate request")
    if not args.apply:
        print("\nDry run: nothing written. Add --apply to back up the store and remove.")
        return
    path = store.database_path()
    backup = path.with_name(f"{path.stem}.before-trim-{time.strftime('%Y%m%d%H%M%S')}{path.suffix}")
    source, target = sqlite3.connect(path), sqlite3.connect(backup)
    source.backup(target)
    target.close()
    source.close()
    if not backup.exists():
        raise SystemExit("the backup was not written; nothing removed")
    print(f"\nbacked up to {backup}")
    removed = {}
    with store.connect() as conn:
        plan = removal_plan(conn, remove)          # re-read inside the transaction we delete in
        for table, (where, params) in plan.items():
            removed[table] = conn.execute(f"DELETE FROM {table} WHERE {where}", params).rowcount
    folders = 0
    for loan_id in remove:
        folder = upload_folder(loan_id)          # None for anything but a plain id directly under the upload root
        if folder is not None and folder.exists():
            shutil.rmtree(folder)
            folders += 1
    audit.record("demo_trim", "trimmed_rate_requests", "store", "demo",
                 {"kept": [f["request_id"] for f in kept], "removed_files": len(remove), "removed": removed})
    with store.connect() as conn:
        audit_after = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        requests_after = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        left = conn.execute("SELECT COUNT(*) FROM rate_requests").fetchone()[0]
    print("removed:")
    for t, n in sorted(removed.items()):
        print(f"  {t:24} {n}")
    print(f"  {'upload folders':24} {folders}")
    print(f"rate requests left: {left}. requests log: {requests_before} -> {requests_after} rows (untouched). "
          f"audit_log: {audit_before} -> {audit_after} rows (nothing removed; the trim itself is recorded)")
    print("decision memory is read from the sqlite table on every request, so no restart or index rebuild is needed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "create"):
        sp = sub.add_parser(name)
        sp.add_argument("--base", default=None if name == "create" else "http://127.0.0.1:8011", required=name == "create")
        sp.add_argument("--rm", default="rohit:Rohit@2026")
        sp.add_argument("--am", default="meera:Meera@2026")
    pr = sub.choices["probe"]
    pr.add_argument("--out", help="write the catalogue here instead of scripts/demo_catalogue.json")
    pr.add_argument("--records", help="also save every probe result here, so --from-records can reselect without the API")
    pr.add_argument("--from-records", help="rebuild the catalogue from saved probe results (no API calls)")
    pr.add_argument("--scratch", action="store_true", help="required: confirms --base is a throwaway copy")
    pr.add_argument("--profiles", type=int, default=260)
    pr.add_argument("--stretched", type=int, default=140, help="extra profiles with a high EMI burden and a small loan")
    pr.add_argument("--bases-per-group", type=int, default=8)
    pr.add_argument("--special-each", type=int, default=12)
    pr.add_argument("--workers", type=int, default=4)
    pr.add_argument("--seed", type=int, default=2026)
    cr = sub.choices["create"]
    cr.add_argument("--catalogue", help="catalogue JSON to post (default scripts/demo_catalogue.json)")
    cr.add_argument("--decide-history", action="store_true")
    cr.add_argument("--no-backdate", action="store_true")
    cr.add_argument("--only", help="comma list of kinds: matrix, special, history")
    tr = sub.add_parser("trim", help="cut the area manager's list to N pending requests that still show every scene")
    tr.add_argument("--keep", type=int, default=10)
    tr.add_argument("--apply", action="store_true", help="back up the store, then remove (default is a dry run)")
    rp = sub.add_parser("report")
    rp.add_argument("--catalogue", help="catalogue JSON to read (default scripts/demo_catalogue.json)")
    rp.add_argument("--out", help="write the page here instead of docs/demo/area_manager_cases.md")
    args = ap.parse_args()
    if args.cmd == "probe":
        if not args.scratch and not args.from_records:
            raise SystemExit("probe prices hundreds of files and writes them to the server's store: run it only against a "
                             "throwaway copy, and say so with --scratch")
        if args.base.rstrip("/").endswith(":8000"):
            raise SystemExit("refusing to probe port 8000, the live local server")
        run_probe(args)
    elif args.cmd == "create":
        run_create(args)
    elif args.cmd == "trim":
        run_trim(args)
    else:
        run_report(args)


if __name__ == "__main__":
    main()
