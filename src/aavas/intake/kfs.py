"""Key Facts Statement for a home-loan offer (RBI NBFC Responsible Business Conduct
Directions, 2025, para 29, applied to HFCs by HFC Directions para 175).

A KFS is given before the loan contract is signed. It carries a unique proposal
number, a validity of at least three working days during which its terms bind the
company (29(2)), the annual percentage rate on the net amount disbursed with every
charge in it (29(3)-(4)), the repayment schedule, the contingent charges (penal
charges 30(6), pre-payment 34(8), switching 31(4)), the effect of a benchmark change
on the EMI and the tenure (31(1)), the document-release and grievance commitments,
and a disclosure that models helped price the loan (FREE-AI rec. 18).

Rates come from the offer; charges and commitments from config/fair_practices.yaml.
The APR method: IRR of the cash flows on the net amount disbursed, reducing balance,
monthly rate x 12, as fair_practices.yaml states. Check it against Annex I of the RBC
Directions before using a KFS outside this platform.
"""

from __future__ import annotations

import secrets
from datetime import date, datetime, timedelta, timezone

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ..config import load, policy
from ..governance import audit, decision_logger, store


# Dates a borrower reads (issue date, validity) are Indian dates; IST has no daylight saving.
IST = timezone(timedelta(hours=5, minutes=30))


class KFSError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def rupees(x: float) -> int:
    """Nearest rupee, half a rupee and above rounded up (RBC para 103)."""
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def emi(principal: float, rate_pct: float, months: int) -> float:
    r = rate_pct / 1200
    return principal / months if r == 0 else principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def schedule(principal: float, rate_pct: float, months: int) -> list[dict[str, int]]:
    """Monthly amortisation in whole rupees; the last instalment clears what is left."""
    r, instalment = rate_pct / 1200, rupees(emi(principal, rate_pct, months))
    rows, balance = [], float(principal)
    for m in range(1, months + 1):
        interest = rupees(balance * r)
        pay = instalment if m < months else rupees(balance) + interest
        repaid = pay - interest
        rows.append({"instalment": m, "opening": rupees(balance), "emi": pay, "interest": interest,
                     "principal": repaid, "closing": max(0, rupees(balance - repaid))})
        balance -= repaid
    return rows


def _irr_monthly(net: float, payments: list[float]) -> float:
    """The monthly rate at which the instalments are worth exactly the net amount disbursed."""
    def pv(r: float) -> float:
        return sum(p / (1 + r) ** (i + 1) for i, p in enumerate(payments))
    lo, hi = 0.0, 0.1
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pv(mid) > net else (lo, mid)
    return (lo + hi) / 2


def apr(principal: float, rate_pct: float, months: int, upfront: float) -> dict[str, float]:
    """Annual percentage rate on the amount actually received (principal less every upfront charge)."""
    rows = schedule(principal, rate_pct, months)
    net = principal - upfront
    monthly = _irr_monthly(net, [r["emi"] for r in rows])
    return {"apr_pct": round(monthly * 1200, 2), "effective_annual_pct": round(((1 + monthly) ** 12 - 1) * 100, 2),
            "net_disbursed": rupees(net), "total_interest": sum(r["interest"] for r in rows),
            "total_repaid": sum(r["emi"] for r in rows)}


def months_to_repay(principal: float, rate_pct: float, instalment: float) -> int | None:
    """Months to clear the loan at a fixed instalment; None when the instalment does not
    cover the interest, which RBC 31(5) forbids (no negative amortisation)."""
    r, balance, months = rate_pct / 1200, float(principal), 0
    if instalment <= balance * r:
        return None
    while balance > 0.5 and months < 1200:
        balance -= instalment - balance * r
        months += 1
    return months


def benchmark_impact(principal: float, rate_pct: float, months: int, shocks_bps: list[int]) -> list[dict]:
    base = rupees(emi(principal, rate_pct, months))
    out = []
    for bps in shocks_bps:
        new_rate = rate_pct + bps / 100
        stretched = months_to_repay(principal, new_rate, base)
        out.append({"benchmark_change_bps": bps, "rate_pct": round(new_rate, 2),
                    "emi_if_tenure_unchanged": rupees(emi(principal, new_rate, months)),
                    "months_if_emi_unchanged": stretched,
                    "note": None if stretched else "the EMI would not cover the interest: the EMI must rise "
                                                   "(no negative amortisation, RBC 31(5))"})
    return out


def working_days_after(start: date, n: int) -> date:
    """n working days after ``start``, skipping Saturdays and Sundays. Holidays are not known
    here; counting them as working days only ever shortens nothing, so the HFC should extend
    the date by its holiday calendar."""
    d, left = start, n
    while left:
        d += timedelta(days=1)
        if d.weekday() < 5:
            left -= 1
    return d


def prepayment_pct(fp: dict[str, Any], *, rate_type: str, borrower: str, purpose: str) -> float:
    """RBC 33-34 and HFC 163: nothing on a floating-rate loan to an individual for purposes
    other than business, whatever the config says."""
    if rate_type == "floating" and borrower == "individual" and purpose != "business":
        return 0.0
    return float((fp["charges"].get("prepayment") or {}).get("other_pct") or 0)


def build(offer: dict[str, Any], decision: dict[str, Any], *, language: str = "en",
          today: date | None = None) -> dict[str, Any]:
    fp, pol = load("fair_practices"), policy()
    snap = decision.get("application_snapshot") or {}
    principal = snap.get("sanctioned_loan_amount") or snap.get("requested_loan_amount")
    years = snap.get("tenure_years") or snap.get("requested_tenure_years")
    if not principal or not years:
        raise KFSError(409, "the priced file has no loan amount or tenure: a KFS cannot be built")
    if language not in fp["kfs"]["languages"]:
        raise KFSError(422, f"language must be one of {', '.join(fp['kfs']['languages'])}")
    principal, months, rate = float(principal), int(round(float(years) * 12)), float(offer["rate"])
    prod, bench = fp["product"], fp["benchmark"]

    fee_pct = float((pol.get("economics") or {}).get("processing_fee_pct") or 0)
    to_hfc = [{"name": "Processing fee", "amount_inr": rupees(principal * fee_pct / 100),
               "basis": f"{fee_pct:.2f}% of the loan amount, one time", "payable_to": "the company"}]
    third = [{**c, "amount_inr": rupees(float(c["amount_inr"]))} for c in fp["charges"].get("third_party") or []]
    upfront = sum(c["amount_inr"] for c in to_hfc + third)
    cost = apr(principal, rate, months, upfront)
    rows = schedule(principal, rate, months)
    issued = today or datetime.now(IST).date()
    pre_pct = prepayment_pct(fp, rate_type=prod["rate_type"], borrower=prod["borrower"], purpose=prod["purpose"])
    approved = bool(fp["approval"].get("approved"))

    return {
        "illustrative": not approved,
        "banner": None if approved else ("ILLUSTRATIVE: the charges schedule in this statement is not yet "
                                         "board-approved. Do not give it to a borrower."),
        "language": language,
        "translation_required": language != "en",
        "versions": {"policy": pol.get("policy_version"), "fair_practices": fp.get("fair_practices_config_version")},
        "part_1": {
            "loan_id": offer["loan_id"], "offer_id": offer["offer_id"],
            "loan_type": prod["default_loan_type"],
            "sanctioned_amount_inr": rupees(principal),
            "disbursal": "in stages linked to construction where the property is under construction (HFC 183)",
            "loan_term_months": months,
            "instalments": {"type": "Equated monthly instalment", "number": months,
                            "emi_inr": rows[0]["emi"], "final_emi_inr": rows[-1]["emi"],
                            "starts": "as set out in the sanction letter and loan agreement"},
            "interest": {"rate_type": prod["rate_type"], "annualised_rate_pct": round(rate, 2),
                         "benchmark": bench["name"], "benchmark_rate_pct": bench["rate_pct"],
                         "spread_pct": round(rate - float(bench["rate_pct"]), 2),
                         "reset_every_months": bench["reset_every_months"],
                         "if_the_benchmark_rises": benchmark_impact(principal, rate, months,
                                                                    list(bench["impact_shocks_bps"]))},
            "fees_and_charges": {"payable_to_company": to_hfc, "payable_to_third_parties": third,
                                 "third_party_basis": "recovered at actuals, with receipts (RBC 29(4))"},
            "apr": {"apr_pct": cost["apr_pct"], "effective_annual_pct": cost["effective_annual_pct"],
                    "net_disbursed_inr": cost["net_disbursed"], "method": fp["kfs"]["apr_method"]},
            "contingent_charges": {
                "penal": [{**c, "nature": "charge, not interest; on the amount in default only; never capitalised"}
                          for c in fp["charges"]["penal"]],
                "prepayment_pct": pre_pct,
                "prepayment_basis": ("none: floating-rate loan to an individual for purposes other than business, "
                                     "from any source, with no lock-in (RBC 34)") if pre_pct == 0 else
                                    "on the amount prepaid, per the board-approved policy (RBC 34(5))",
                "switching": fp["charges"]["switching"],
            },
        },
        "part_2": {
            "change_of_terms": (f"Notice {fp['change_of_terms']['notice']}; interest and charges change only "
                                f"prospectively. If a change is to your disadvantage you may close or switch the "
                                f"loan within {fp['change_of_terms']['exit_window_days']} days without extra "
                                f"charges or interest."),
            "document_release": (f"Original property documents are released within "
                                 f"{fp['document_release']['days_after_closure']} days of full repayment, at "
                                 f"{fp['document_release']['where']}. For each day of delay attributable to us "
                                 f"we pay ₹{fp['document_release']['compensation_inr_per_day']:,}."),
            "grievance": (f"{fp['grievance']['officer']}. We acknowledge a complaint within "
                          f"{fp['grievance']['acknowledge_within_days']} days and reply in full within "
                          f"{fp['grievance']['final_response_within_days']} days. With no reply within "
                          f"{fp['grievance']['escalate_after_days']} days, or if dissatisfied, you may approach the "
                          f"{fp['grievance']['escalation_body']}."),
            "non_discrimination": " ".join(fp["non_discrimination"]["statement"].split()),
            "use_of_models": " ".join(fp["ai_disclosure"]["text"].split()),
        },
        "validity": {"issued_on": issued.isoformat(),
                     "valid_until": working_days_after(issued, int(fp["kfs"]["validity_working_days"])).isoformat(),
                     "binding": "The terms in this statement bind the company until the date above (RBC 29(2))."},
        "not_in_kfs": "No fee or charge that is not in this statement may be levied without your explicit consent "
                      "(RBC 29(5)).",
        "totals": {"total_interest_inr": cost["total_interest"], "total_repaid_inr": cost["total_repaid"],
                   "upfront_charges_inr": upfront},
        "repayment_schedule": rows,
    }


def _offer(offer_id: str) -> dict[str, Any]:
    row = store.fetch_one("SELECT * FROM offers WHERE offer_id = ?", (offer_id,))
    if row is None:
        raise KFSError(404, f"no offer {offer_id}")
    return row


def issue(offer_id: str, *, issued_by: str, language: str = "en") -> dict[str, Any]:
    """Give the borrower a KFS for an offer. A KFS still valid for the same rate is returned
    again rather than replaced: its terms already bind the company."""
    offer = _offer(offer_id)
    if offer["status"] != "sent":
        raise KFSError(409, f"offer {offer_id} is {offer['status']}: a KFS is only given for a live offer")
    today = datetime.now(IST).date()
    held = store.fetch_one("SELECT * FROM kfs_statements WHERE offer_id = ? AND language = ? "
                           "ORDER BY issued_at DESC LIMIT 1", (offer_id, language))
    if held and held["valid_until"] >= today.isoformat() and abs(float(held["rate_pct"]) - float(offer["rate"])) < 1e-9:
        return _shape(held)
    decision = decision_logger.get_decision(offer["decision_id"])
    if decision is None:
        raise KFSError(409, "the decision behind this offer is not on record")
    kfs = build(offer, decision.get("payload") or decision, language=language, today=today)
    row = {"kfs_id": f"KFS-{secrets.token_hex(5).upper()}", "offer_id": offer_id, "loan_id": offer["loan_id"],
           "issued_by": issued_by, "issued_at": store.now(), "valid_until": kfs["validity"]["valid_until"],
           "rate_pct": float(offer["rate"]), "apr_pct": kfs["part_1"]["apr"]["apr_pct"], "language": language,
           "illustrative": kfs["illustrative"], "payload": kfs, "acknowledged_at": None}
    store.insert("kfs_statements", row)
    audit.record(issued_by, "issued_kfs", "application", offer["loan_id"],
                 {"kfs_id": row["kfs_id"], "offer_id": offer_id, "apr_pct": row["apr_pct"],
                  "valid_until": row["valid_until"], "illustrative": kfs["illustrative"]})
    return _shape(row)


def acknowledge(kfs_id: str, *, recorded_by: str) -> dict[str, Any]:
    """The borrower's acknowledgement that the KFS was explained (RBC 29(1))."""
    row = store.fetch_one("SELECT * FROM kfs_statements WHERE kfs_id = ?", (kfs_id,))
    if row is None:
        raise KFSError(404, f"no KFS {kfs_id}")
    if row["acknowledged_at"]:
        return _shape(row)
    with store.connect() as conn:
        conn.execute("UPDATE kfs_statements SET acknowledged_at = ? WHERE kfs_id = ?", (store.now(), kfs_id))
    audit.record(recorded_by, "kfs_acknowledged", "application", row["loan_id"], {"kfs_id": kfs_id})
    return _shape(store.fetch_one("SELECT * FROM kfs_statements WHERE kfs_id = ?", (kfs_id,)))


def statements(offer_id: str | None = None, loan_id: str | None = None) -> list[dict[str, Any]]:
    clauses, params = [], []
    if offer_id:
        clauses.append("offer_id = ?"); params.append(offer_id)
    if loan_id:
        clauses.append("loan_id = ?"); params.append(loan_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return [_shape(r) for r in store.query(f"SELECT * FROM kfs_statements{where} ORDER BY issued_at DESC",
                                           tuple(params))]


def _shape(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["payload"] = store.decode(out.get("payload"))
    out["illustrative"] = bool(out.get("illustrative"))
    return out
