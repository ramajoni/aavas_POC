"""The area manager's checklist for a lower-rate request: the checks a manager runs
before signing off, with what the file and the branch say for each.

Three things the other tabs do not show are worked out here:

**Alternatives to a rate cut.** What the customer's EMI and the bank's interest look
like if the manager approves the ask, approves a smaller cut, holds the rate and
waives the processing fee, or holds the rate over a longer tenure.

**Precedent.** Where the asked rate would sit among the rates the branch has
sanctioned for loans like this one (same segment and risk grade).

**Why the customer is pushing.** Words in the conversation notes that point to the
EMI, the fee or speed rather than the rate itself.
"""

from __future__ import annotations

from typing import Any

from ..config import policy
from . import portfolio

HINTS = {
    "emi": ("emi", "monthly", "instalment", "installment", "budget"),
    "fee": ("processing fee", "fee", "charges", "insurance"),
    "speed": ("faster", "quick", "tat", "turnaround", "disburse", "urgent", "by next week", "registration"),
    "rate": ("rate", "roi", "interest", "cheaper", "lower"),
}
HINT_TEXT = {"emi": "the EMI", "fee": "fees or charges", "speed": "speed of disbursement", "rate": "the rate itself"}


def _emi(amount: float, rate: float, years: float) -> float:
    r, n = rate / 1200, years * 12
    return amount * r * (1 + r) ** n / ((1 + r) ** n - 1) if r else amount / n


def alternatives(*, loan: float, years: float, age: float | None, ask: float, rec: float,
                 counter: float | None) -> dict[str, Any]:
    """The ways to close the file, from the bank's side and the customer's."""
    econ = policy().get("economics") or {}
    fee_pct = float(econ.get("processing_fee_pct") or 0)
    max_years = int((policy().get("constraints") or {}).get("max_tenure_years") or 30)
    n = years * 12
    emi_rec, emi_ask = _emi(loan, rec, years), _emi(loan, ask, years)
    interest_rec = emi_rec * n - loan

    def option(key, label, rate, tenure, bank_change, note):
        emi = _emi(loan, rate, tenure)
        return {"key": key, "label": label, "rate": round(rate, 2), "tenure_years": tenure, "emi": round(emi),
                "emi_vs_ask": round(emi - emi_ask), "bank_change_inr": round(bank_change), "note": note,
                "age_at_end": round(age + tenure) if age else None}

    out = [option("ask", f"Approve the ask ({ask:.2f}%)", ask, years, -(emi_rec - emi_ask) * n,
                  "The customer's EMI as asked; the bank gives up the interest over the full loan.")]
    if counter is not None and ask + 0.005 < counter < rec - 0.005:
        out.append(option("counter", f"Approve {counter:.2f}% instead", counter, years, -(emi_rec - _emi(loan, counter, years)) * n,
                          "The suggested rate: a smaller cut, worth more to the bank."))
    fee = loan * fee_pct / 100
    if fee:
        out.append(option("fee", f"Hold {rec:.2f}%, waive the processing fee", rec, years, -fee,
                          f"A one-time saving of about ₹{fee:,.0f} for the customer; the rate and EMI stay."))
    longer = next((t for t in range(int(years) + 1, max_years + 1) if _emi(loan, rec, t) <= emi_ask + 1), None)
    tenure = longer or max_years
    if tenure > years:
        out.append(option("tenure", f"Hold {rec:.2f}%, extend to {tenure} years", rec, tenure,
                          (_emi(loan, rec, tenure) * tenure * 12 - loan) - interest_rec,
                          ("The EMI matches the ask without cutting the rate." if longer else
                           f"Even at the {max_years}-year maximum the EMI stays above the ask.")
                          + " Check the customer's age at the end of the loan."))
    return {"options": out, "processing_fee_pct": fee_pct, "max_tenure_years": max_years,
            "emi_at_model_rate": round(emi_rec), "emi_at_ask": round(emi_ask)}


def precedent(city: str | None, segment: str | None, grade: str | None, ask: float) -> dict[str, Any]:
    """Where the asked rate sits among the rates the branch sanctioned for loans like this."""
    book = portfolio.frame()
    booked = book[book["disbursed"].astype(bool)]
    like = booked[(booked["city"] == city) & (booked["segment"] == segment) & (booked["risk_grade"] == grade)]
    scope = "loans like this in the branch"
    if len(like) < 20:
        like = booked[(booked["segment"] == segment) & (booked["risk_grade"] == grade)]
        scope = "loans like this across every branch"
    if not len(like):
        return {"files": 0, "scope": scope}
    rates = like["sanctioned_rate"].astype(float)
    return {"files": int(len(like)), "scope": scope,
            "share_at_or_below": round(float((rates <= ask + 1e-9).mean()), 4),
            "median_rate": round(float(rates.median()), 2), "lowest_rate": round(float(rates.min()), 2)}


def reason_hints(*texts: str | None) -> dict[str, Any]:
    """What the notes say the customer cares about, beyond the rate."""
    text = " ".join(t for t in texts if t).lower()
    if not text.strip():
        return {"found": [], "text": None}
    found = [k for k, words in HINTS.items() if any(w in text for w in words)]
    return {"found": found, "labels": [HINT_TEXT[k] for k in found], "text": text[:400]}


def month_to_date(requests: list[dict[str, Any]], city: str | None, rm: str, now_month: str,
                  where) -> dict[str, Any]:
    """Lower rates asked and approved this month, in this branch and by this RM."""
    month = [r for r in requests if str(r.get("requested_at") or "")[:7] == now_month]
    branch = [r for r in month if city and where(r).get("city") == city]

    def tally(rows: list[dict[str, Any]]) -> dict[str, Any]:
        approved = [r for r in rows if r.get("status") == "approved"]
        depth = [(r["recommended_rate"] - (r.get("approved_rate") or r["requested_rate"])) * 100 for r in approved
                 if r.get("recommended_rate") is not None]
        return {"asked": len(rows), "approved": len(approved),
                "declined": sum(r.get("status") == "declined" for r in rows),
                "pending": sum(r.get("status") == "pending" for r in rows),
                "avg_approved_cut_bps": round(sum(depth) / len(depth), 1) if depth else None}
    return {"month": now_month, "branch": tally(branch), "rm": tally([r for r in month if r.get("requested_by") == rm])}


def relationship(applicant: dict[str, Any]) -> list[str]:
    sc = applicant.get("sales_context") or {}
    flags = []
    if sc.get("existing_relationship") or applicant.get("existing_relationship"):
        flags.append("Existing customer")
    channel = applicant.get("channel")
    flags.append({"branch_walkin": "Walked into the branch", "dsa": "Sourced by a DSA", "connector": "Sourced by a connector",
                  "digital": "Applied online"}.get(channel, f"Channel: {channel}") if channel else "Channel not recorded")
    if applicant.get("property_type") == "builder_flat":
        flags.append("Builder project (possible tie-up)")
    if sc.get("customer_negotiating"):
        flags.append(f"Negotiating ({sc.get('prior_negotiation_rounds') or 0} rounds so far)")
    return flags
