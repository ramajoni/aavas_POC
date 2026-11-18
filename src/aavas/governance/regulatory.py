"""RBI artefacts for home-loan pricing, each with the platform's evidence.

``config/rbi_artefacts.yaml`` lists the directions, circulars and drafts that bear
on how a housing finance company prices a home loan and governs its pricing
models, and what each asks for. This module puts a live number beside every ask:
how many priced decisions carry a rate build-up, which models have been tested
for bias, whether the LTV cap in the policy follows the RBI tiers.

Some evidence decides its own coverage. Whether the policy's LTV cap holds for a
₹50 lakh loan is a fact about the policy, not a judgement written in the YAML, so
the row says "Gap" for as long as the policy allows more than RBI does.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Callable

import yaml

from ..config import load, policy
from ..logging_utils import get_logger
from ..paths import CONFIG_DIR
from . import approvals, model_registry, monitors, policy_versions, store

log = get_logger(__name__)

ARTEFACTS = CONFIG_DIR / "rbi_artefacts.yaml"
LAKH = 100_000
# RBI's LTV ceilings on housing loans to individuals, by loan size (HFC Directions).
LTV_TIERS = ((0, 30 * LAKH, 0.90), (30 * LAKH, 75 * LAKH, 0.80), (75 * LAKH, None, 0.75))
MONITOR_FRESH_DAYS = 31


def _lakh(amount: float) -> str:
    return f"₹{amount / LAKH:,.0f} lakh"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _date(stamp: str | None) -> str:
    if not stamp:
        return "never"
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).strftime("%d %b %Y")
    except ValueError:
        return str(stamp)[:10]


# ------------------------------------------------------------------ the facts

def _decisions() -> list[dict[str, Any]]:
    rows = store.query("SELECT payload, policy_version FROM decisions WHERE stage = 'post_assessment' "
                       "ORDER BY created_at")
    out = []
    for row in rows:
        payload = store.decode(row.get("payload"))
        if isinstance(payload, dict):
            payload["_policy_version"] = row.get("policy_version")
            out.append(payload)
    return out


def _book() -> list[tuple[float, float]]:
    """(sanctioned amount, LTV) for every booked loan in the portfolio."""
    from ..analytics import portfolio
    try:
        frame = portfolio.frame()
    except Exception:                                              # noqa: BLE001 - no book loaded
        log.warning("portfolio not available for the LTV check", exc_info=True)
        return []
    booked = frame[frame["disbursed"].astype(bool)]
    return list(zip(booked["sanctioned_loan_amount"].astype(float), booked["ltv"].astype(float)))


def facts() -> dict[str, Any]:
    decisions = _decisions()
    explanation = [d.get("explanation") or {} for d in decisions]
    decomposed = [d for d in decisions if (d.get("recommendation") or {}).get("value_decomposition")]
    lives = [float(e) for d in decisions
             if (e := ((d.get("recommendation") or {}).get("economics") or {}).get("expected_life_years")) is not None]
    priced = []
    for d in decisions:
        snap = d.get("application_snapshot") or {}
        amount = snap.get("sanctioned_loan_amount") or snap.get("requested_loan_amount")
        if amount and snap.get("ltv") is not None:
            priced.append((float(amount), float(snap["ltv"])))

    cards = model_registry.load_bundle().cards()
    registered = sorted({r["name"] for r in model_registry.registered() if r.get("role") == "champion"})
    overrides = store.query("SELECT reason, evidence_verified, within_authority, approved_by FROM overrides")
    signatures = store.query("SELECT COUNT(*) AS n FROM approval_signatures")
    return {
        "decisions": decisions, "explanation": explanation, "decomposed": decomposed,
        "latest_decomposition": (decomposed[-1]["recommendation"]["value_decomposition"] if decomposed else {}),
        "lives": lives, "priced": priced, "book": _book(), "cards": cards, "registered": registered,
        "overrides": overrides, "signatures": int(signatures[0]["n"]) if signatures else 0,
        "audit_entries": store.count("audit_log"), "versions": policy_versions.versions(),
        "approvals": approvals.summary(), "monitors": monitors.status()["monitors"],
        "policy": policy(),
    }


# ------------------------------------------------------- one sentence per key
# Each returns {"text": ..., "coverage": ...?, "figures": {...}}. A returned
# coverage replaces the one written in the YAML.

def _value_decomposition(f):
    n, total = len(f["decomposed"]), len(f["decisions"])
    return {"text": f"{n} of {total} priced decisions store the rate build-up: cost of funds, servicing, "
                    "capital charge, expected credit loss and margin.",
            "figures": {"with_build_up": n, "decisions": total}}


def _reason_codes(f):
    total = len(f["decisions"])
    coded = sum(bool(e.get("reason_codes")) for e in f["explanation"])
    flagged = sum(bool((e.get("fair_pricing_review") or {}).get("flagged")) for e in f["explanation"])
    return {"text": f"{coded} of {total} priced decisions carry reason codes; {flagged} "
                    f"{'was' if flagged == 1 else 'were'} flagged for fair-pricing review because the "
                    "commercial uplift passed the policy threshold.",
            "figures": {"with_reason_codes": coded, "decisions": total, "fair_pricing_flags": flagged}}


def _policy_versions(f):
    versions = f["versions"]
    live = next((v["version"] for v in versions if v["current"]), None)
    stamped = sum(bool(d.get("_policy_version")) for d in f["decisions"])
    return {"text": f"{len(versions)} policy versions on record, {live} live; {stamped} of "
                    f"{len(f['decisions'])} priced decisions record the version they were priced under.",
            "figures": {"versions": len(versions), "live": live, "stamped": stamped}}


def _ltv(low: float, high: float | None, cap: float):
    def check(f):
        pol = f["policy"].get("constraints") or {}
        allowed, largest = float(pol.get("max_ltv") or 1), float(pol.get("max_loan_amount") or 0)
        band = f"above {_lakh(low)}" if low else f"up to {_lakh(high)}"
        if low and high:
            band = f"above {_lakh(low)} up to {_lakh(high)}"
        if largest and largest <= low:
            return {"text": f"The product's largest loan is {_lakh(largest)}, so no loan reaches this tier.",
                    "coverage": "in_place", "figures": {"max_loan_amount": largest}}

        def in_band(rows):
            return [(a, ltv) for a, ltv in rows if a > low and (high is None or a <= high)]
        booked, priced = ([r for r in in_band(rows) if r[1] > cap + 1e-9] for rows in (f["book"], f["priced"]))
        priced_in_band = len(in_band(f["priced"]))
        if allowed <= cap + 1e-9:
            text = (f"The policy caps LTV at {_pct(allowed)}, within RBI's {_pct(cap)} for loans {band}.")
            coverage = "in_place"
        else:
            text = (f"The policy allows {_pct(allowed)} on every loan up to {_lakh(largest)}, above RBI's "
                    f"{_pct(cap)} for loans {band}.")
            coverage = "gap"
        if booked:
            text += (f" {len(booked)} booked loans in this band run above {_pct(cap)} "
                     f"(highest {_pct(max(l for _, l in booked))}).")
        if priced_in_band:
            text += f" {len(priced)} of the {priced_in_band} priced files in this band are over it."
        # The control is the policy cap and what it lets through on new files; loans
        # already on the book above the cap are a finding to act on, not the control.
        return {"text": text, "coverage": "gap" if priced else coverage, "attention": bool(booked),
                "figures": {"policy_max_ltv": allowed, "rbi_cap": cap, "booked_over": len(booked),
                            "priced_over": len(priced), "priced_in_band": priced_in_band}}
    return check


def _all_in_cost(f):
    fee = float((f["policy"].get("economics") or {}).get("processing_fee_pct") or 0)
    return {"text": f"Offers show the rate, EMI, a {fee:.2f}% processing fee and the interest over the full "
                    "term. An APR and a repayment schedule are not produced.",
            "figures": {"processing_fee_pct": fee}}


def _rate_components(f):
    parts = [k for k in f["latest_decomposition"] if str(k).startswith("less_")]
    named = ", ".join(k.removeprefix("less_").replace("_", " ") for k in parts) or "none recorded"
    penal = any("penal" in str(k) for k in f["latest_decomposition"])
    return {"text": f"The latest build-up deducts {named} from the customer rate"
                    + ("; it includes a penal element." if penal else "; there is no penal element."),
            "coverage": "gap" if penal else None, "figures": {"components": parts}}


def _prepayment(f):
    econ = f["policy"].get("economics") or {}
    fee_keys = [k for k in econ if "prepay" in k or "foreclos" in k]
    life = f"{median(f['lives']):.1f} years" if f["lives"] else "not recorded"
    return {"text": f"The economics carry {'a pre-payment fee' if fee_keys else 'no pre-payment or foreclosure fee'}; "
                    f"the median expected life of a priced loan is {life}, from the prepayment model.",
            "coverage": "gap" if fee_keys else None, "figures": {"median_expected_life_years": median(f["lives"]) if f["lives"] else None}}


def _change_control(f):
    s = f["approvals"]
    model = (load("governance").get("artefacts") or {}).get("model") or {}
    roles = ", ".join(r.replace("_", " ") for r in model.get("requires") or [])
    gates = ", ".join(g.replace("_", " ") for g in model.get("gates") or [])
    return {"text": f"{s['total']} change requests on record. A model promotion needs {roles} to sign and "
                    f"must clear: {gates}.",
            "figures": {"change_requests": s["total"], "by_state": s["by_state"]}}


def _model_inventory(f):
    cards, registered = f["cards"], f["registered"]
    missing = [n for n in cards if n not in registered]
    text = f"{len(cards)} models in use, each with a model card: {', '.join(cards)}."
    if missing:
        listed = len(cards) - len(missing)
        text += (f" The model registry lists {listed or 'none'} of them as champion; not listed: "
                 f"{', '.join(missing)}.")
    return {"text": text, "attention": bool(missing),
            "figures": {"models": len(cards), "registered": registered, "not_registered": missing}}


def _model_validation(f):
    cards = f["cards"].values()
    tested = [c for c in cards if (c.get("n_test") or 0) > 0 and c.get("metrics")]
    sizes = [int(c["n_test"]) for c in tested]
    text = (f"{len(tested)} of {len(f['cards'])} models carry metrics on a held-out test set"
            + (f" ({min(sizes):,}–{max(sizes):,} files)" if sizes else "")
            + ". Promotion needs model-risk sign-off, and whoever asks for a change may not approve it.")
    return {"text": text, "coverage": "in_place" if tested and len(tested) == len(f["cards"]) else "partial",
            "figures": {"tested": len(tested), "models": len(f["cards"])}}


def _model_fairness(f):
    passed, failed, unmeasured = [], [], []
    for name, card in f["cards"].items():
        verdict = (card.get("fairness") or {}).get("passed")
        (unmeasured if verdict is None else passed if verdict else failed).append(name)
    parts = [f"Measured on {len(passed) + len(failed)} of {len(f['cards'])} models"]
    if passed:
        parts.append(f"pass: {', '.join(passed)}")
    if failed:
        parts.append(f"fail: {', '.join(failed)}")
    if unmeasured:
        parts.append(f"not measured: {', '.join(unmeasured)}")
    coverage = "in_place" if f["cards"] and not failed and not unmeasured else "partial" if passed else "gap"
    return {"text": "; ".join(parts) + ".", "coverage": coverage, "attention": bool(failed),
            "figures": {"passed": passed, "failed": failed, "unmeasured": unmeasured}}


def _model_limitations(f):
    counts = {n: len(c.get("limitations") or []) for n, c in f["cards"].items()}
    documented = sum(1 for v in counts.values() if v)
    return {"text": f"{sum(counts.values())} known limitations written on {documented} of {len(counts)} "
                    "model cards, with training window, features and learner.",
            "coverage": "in_place" if counts and documented == len(counts) else "partial",
            "figures": {"limitations": sum(counts.values()), "documented": documented}}


def _monitors(f):
    now = datetime.now(timezone.utc)
    parts, stale = [], []
    for name, m in f["monitors"].items():
        parts.append(f"{name.replace('_', ' ')} {m['last_status'] or 'never run'}"
                     + (f" on {_date(m['last_run'])}" if m["last_run"] else ""))
        try:
            last = datetime.fromisoformat(str(m["last_run"]).replace("Z", "+00:00")) if m["last_run"] else None
        except ValueError:
            last = None
        if last is None or now - last > timedelta(days=MONITOR_FRESH_DAYS):
            stale.append(name)
    text = "Last runs: " + "; ".join(parts) + "."
    if stale:
        text += (f" Not run in the last {MONITOR_FRESH_DAYS} days: "
                 f"{', '.join(n.replace('_', ' ') for n in stale)}.")
    # No run on record at all is no monitoring, not a stale one.
    never = all(not m["last_run"] for m in f["monitors"].values())
    return {"text": text, "coverage": "gap" if never else "partial" if stale else "in_place",
            "attention": bool(stale),
            "figures": {"stale": stale}}


def _overrides(f):
    rows = f["overrides"]
    reasoned = sum(bool(r.get("reason")) for r in rows)
    named = sum(bool(r.get("approved_by")) for r in rows)
    verified = sum(bool(r.get("evidence_verified")) for r in rows)
    return {"text": f"{len(rows)} rates changed by a person on record: {reasoned} with a reason, {named} with "
                    f"a named approver, {verified} with verified evidence.",
            "figures": {"overrides": len(rows), "with_reason": reasoned, "named": named, "verified": verified}}


def _audit(f):
    return {"text": f"{f['audit_entries']:,} audit-log entries and {f['signatures']} signatures on change "
                    "requests; decisions are append-only.",
            "figures": {"audit_entries": f["audit_entries"], "signatures": f["signatures"]}}


EVIDENCE: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "value_decomposition": _value_decomposition, "reason_codes": _reason_codes,
    "policy_versions": _policy_versions,
    "ltv_up_to_30l": _ltv(*LTV_TIERS[0]), "ltv_30l_to_75l": _ltv(*LTV_TIERS[1]),
    "ltv_above_75l": _ltv(*LTV_TIERS[2]),
    "all_in_cost": _all_in_cost, "rate_components": _rate_components, "prepayment": _prepayment,
    "change_control": _change_control, "model_inventory": _model_inventory,
    "model_validation": _model_validation, "model_fairness": _model_fairness,
    "model_limitations": _model_limitations, "monitors": _monitors, "overrides": _overrides,
    "audit": _audit,
}


def register() -> dict[str, Any]:
    spec = yaml.safe_load(ARTEFACTS.read_text(encoding="utf-8")) or {}
    f = facts()
    cache: dict[str, dict[str, Any]] = {}
    artefacts, tally = [], {k: 0 for k in spec.get("coverage") or {}}
    for art in spec.get("artefacts") or []:
        rows = []
        for req in art.get("requirements") or []:
            key = req.get("evidence")
            found = None
            if key:
                if key not in EVIDENCE:
                    raise KeyError(f"rbi_artefacts.yaml names unknown evidence {key!r}")
                if key not in cache:
                    cache[key] = EVIDENCE[key](f)
                found = cache[key]
            coverage = (found or {}).get("coverage") or req.get("coverage") or "partial"
            tally[coverage] = tally.get(coverage, 0) + 1
            rows.append({**{k: v for k, v in req.items() if k != "coverage"}, "coverage": coverage,
                         "attention": coverage == "gap" or bool((found or {}).get("attention")),
                         "evidence_text": (found or {}).get("text"), "figures": (found or {}).get("figures")})
        artefacts.append({**{k: v for k, v in art.items() if k != "requirements"}, "requirements": rows})
    # What a validator should act on: every gap, and rows whose evidence shows a
    # failure even where the control exists (a model failing its fairness test).
    # One entry per evidence finding, so a failing model shows once, not under
    # every artefact that cites it.
    attention, seen = [], set()
    for a in artefacts:
        for r in a["requirements"]:
            mark = r.get("evidence") or (a["id"], r["ask"])
            if r["attention"] and mark not in seen:
                seen.add(mark)
                attention.append({"artefact": a["title"], "id": a["id"], "ask": r["ask"],
                                  "coverage": r["coverage"], "evidence": r["evidence_text"], "page": r.get("page")})
    return {"version": spec.get("version"), "checked_on": spec.get("checked_on"), "entity": spec.get("entity"),
            "coverage_labels": spec.get("coverage"), "status_labels": spec.get("status"),
            "policy_version": f["policy"].get("policy_version"),
            "summary": {"artefacts": len(artefacts), "requirements": sum(tally.values()), "by_coverage": tally},
            "attention": attention, "artefacts": artefacts,
            "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
