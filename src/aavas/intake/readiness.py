"""What a file still needs before it is priced and sent, step by step.

The relationship manager works a file in three steps, each unlocking the next
(``readiness`` in config/intake.yaml):

1. **Indicative pricing** - the customer's own figures, enough for the range.
2. **Compute loan risk** - every verified figure and both checks, which give the
   risk-adjusted rate. Until all
   are in, the file is not given a risk-adjusted rate: pricing on declared
   figures in their place would be a guess presented as an assessment.
3. **Customer interaction and negotiation** - whether the customer is negotiating,
   is an existing customer, and what they said. A competitor's quote is recorded
   when there is one, never required.
   The acceptance score is read only once these are answered.

Nothing goes to the customer or the area manager until all three are complete.

A field is missing when it is empty; a yes/no question is missing until it has
been answered either way. On top of that, figures are sense-checked against the
pricing policy (a loan of 1250% of the property's value is a typing error, not a
risk), and a flagged legal or technical check stops the file at step 2.
"""

from __future__ import annotations

from typing import Any

from ..config import policy
from . import applications

SALES_KEYS = {"customer_negotiating", "prior_negotiation_rounds", "existing_relationship"}


def _inr(x: float) -> str:
    """Indian digit grouping: 5000000 -> ₹50,00,000."""
    whole = str(int(round(abs(x))))
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ("−" if x < 0 else "") + "₹" + ",".join(groups + [tail])


def _given(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() != ""
    return value is not None


def _num(value: Any) -> float | None:
    try:
        return float(value) if _given(value) else None
    except (TypeError, ValueError):
        return None


def _value(applicant: dict[str, Any], key: str) -> Any:
    if key in SALES_KEYS:
        return (applicant.get("sales_context") or {}).get(key)
    return applicant.get(key)


def _missing(applicant: dict[str, Any], key: str) -> bool:
    if key == "bureau_score":
        return not (applicant.get("new_to_credit") or _num(applicant.get("bureau_score")) is not None)
    if key == "prior_negotiation_rounds":
        return _value(applicant, "customer_negotiating") is True and (_num(_value(applicant, key)) or 0) < 1
    if key in ("customer_negotiating", "existing_relationship"):
        return not isinstance(_value(applicant, key), bool)
    return not _given(_value(applicant, key))


def _limits() -> dict[str, float]:
    con = policy().get("constraints") or {}
    return {"max_ltv": float(con.get("max_ltv") or 1), "max_loan": float(con.get("max_loan_amount") or 0),
            "max_tenure": float(con.get("max_tenure_years") or 30)}


def _loan_checks(loan: float | None, value: float | None, tenure: float | None, *, loan_label: str,
                 value_label: str, lim: dict[str, float]) -> list[str]:
    out = []
    if loan and value:
        ltv = loan / value
        if ltv > lim["max_ltv"] + 1e-9:
            out.append(f"The {loan_label} ({_inr(loan)}) is {ltv:.0%} of the {value_label} ({_inr(value)}); "
                       f"the policy lends up to {lim['max_ltv']:.0%}. Check both figures.")
    if loan and lim["max_loan"] and loan > lim["max_loan"]:
        out.append(f"The {loan_label} ({_inr(loan)}) is above the largest loan the policy allows "
                   f"({_inr(lim['max_loan'])}).")
    if tenure and tenure > lim["max_tenure"]:
        out.append(f"A tenure of {tenure:g} years is longer than the policy allows ({lim['max_tenure']:g} years).")
    return out


def _indicative(a: dict[str, Any], lim) -> tuple[list[str], list[str]]:
    income, obligations = _num(a.get("declared_monthly_income")), _num(a.get("declared_obligations"))
    loan = _num(a.get("requested_loan_amount"))
    problems = _loan_checks(loan, _num(a.get("estimated_property_value")), _num(a.get("requested_tenure_years")),
                            loan_label="loan asked", value_label="estimated property value", lim=lim)
    warnings = []
    if income and obligations is not None and obligations >= income:
        problems.append(f"Monthly obligations ({_inr(obligations)}) are not below the declared monthly income "
                        f"({_inr(income)}).")
    if income and loan and income >= loan:
        warnings.append(f"Declared monthly income ({_inr(income)}) is as much as the whole loan asked "
                        f"({_inr(loan)}). Check it is the monthly figure, not a yearly one.")
    return problems, warnings


def _risk(a: dict[str, Any], lim) -> tuple[list[str], list[str]]:
    problems = _loan_checks(_num(a.get("sanctioned_loan_amount")), _num(a.get("appraised_property_value")),
                            _num(a.get("tenure_years")), loan_label="assessed loan",
                            value_label="appraised property value", lim=lim)
    for key, name in (("legal_flag", "legal"), ("technical_flag", "technical")):
        if _num(a.get(key)) == 1:
            problems.append(f"The {name} check flagged an issue. Clear it before the file is priced and sent.")
    banking = _num(a.get("banking_behaviour_score"))
    if banking is not None and not 0 <= banking <= 100:
        problems.append("Banking behaviour is a score from 0 to 100.")
    for key, label in (("bureau_vintage_months", "Credit history"), ("delinquency_count_24m", "Delinquencies"),
                       ("deviation_count", "Credit deviations")):
        if (_num(a.get(key)) or 0) < 0:
            problems.append(f"{label} cannot be negative.")
    warnings = []
    declared, verified = _num(a.get("declared_monthly_income")), _num(a.get("verified_monthly_income"))
    if declared and verified and verified < declared * 0.5:
        warnings.append(f"Verified income ({_inr(verified)}) is less than half the declared figure "
                        f"({_inr(declared)}).")
    return problems, warnings


CHECKS = {"indicative": _indicative, "risk": _risk}


def steps_config() -> list[dict[str, Any]]:
    return list(applications.settings().get("readiness") or [])


def check(applicant: dict[str, Any] | None) -> dict[str, Any]:
    a = applicant or {}
    lim = _limits()
    steps, open_before = [], False
    for step in steps_config():
        fields = step.get("fields") or {}
        missing = [{"field": k, "label": label} for k, label in fields.items() if _missing(a, k)]
        problems, warnings = CHECKS[step["key"]](a, lim) if step["key"] in CHECKS else ([], [])
        done = not open_before and not missing and not problems
        steps.append({"key": step["key"], "label": step["label"], "unlocks": step.get("unlocks"),
                      "action": step.get("action") or step["label"],
                      "done": done, "locked": open_before, "missing": missing,
                      "problems": problems, "warnings": warnings})
        open_before = open_before or not done
    following = next((s for s in steps if not s["done"]), None)
    return {"steps": steps, "complete": following is None,
            "next": following["key"] if following else None,
            "next_label": following["label"] if following else None,
            "due": (len(following["missing"]) + len(following["problems"])) if following else 0,
            "done": {s["key"]: s["done"] for s in steps}}


def summary(applicant: dict[str, Any] | None) -> dict[str, Any]:
    """What a worklist row needs: whether it is complete, and the next step."""
    full = check(applicant)
    return {k: full[k] for k in ("complete", "next", "next_label", "due", "done")}


def refusal(applicant: dict[str, Any] | None, action: str) -> str | None:
    """Why ``action`` (sending an offer, asking the area manager) is refused, or None."""
    full = check(applicant)
    if full["complete"]:
        return None
    step = next(s for s in full["steps"] if not s["done"])
    due = [m["label"] for m in step["missing"]] + step["problems"]
    return (f"Complete the file before {action}. {step['label']} still needs: "
            + "; ".join(due[:6]) + ("…" if len(due) > 6 else "") + ".")
