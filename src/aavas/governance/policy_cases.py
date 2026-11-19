"""Policy sample data: applicants with the outcome the policy requires.

Each row of ``tests/data/policy_cases.csv`` is an applicant, the rate proposed for
them, and what must happen: the card rate (or ``none`` for no standard offer), the
verdict (pass / refer / block), the approval level, and words the reason must
contain. Expected values are worked out from the policy, never read back from the
engine, so a row fails when the engine and the policy disagree.

``tests/test_policy_cases.py`` runs every row as a test; the agentic workflow's
policy step runs them all before training.

    rate column: ``card`` | ``card-35`` / ``card+5`` (bps from the card rate) | a rate in %
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..optimization import constraints
from ..paths import ROOT

CASES = ROOT / "tests" / "data" / "policy_cases.csv"


def load(path: Path | None = None) -> list[dict[str, str]]:
    with Path(path or CASES).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def application(case: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{
        "segment": "salaried_formal", "property_type": "resale_flat", "deviation_count": 0,
        "risk_grade": case["risk_grade"],
        "tenure_years": float(case["tenure_months"]) / 12,
        "bureau_score": float(case["bureau_score"]),
        "new_to_credit": case["new_to_credit"].strip().lower() == "true",
        "sanctioned_loan_amount": float(case["loan_amount"]),
        "verified_monthly_income": float(case["monthly_income"]),
        "declared_obligations": float(case["other_emis"]),
        "ltv": float(case["ltv"]),
    }])


def proposed_rate(spec: str, card: float) -> float:
    spec = spec.strip().lower()
    if spec.startswith("card"):
        offset = spec[4:]
        return round(card + (float(offset) / 100 if offset else 0.0), 2)
    return float(spec)


def evaluate(case: dict[str, str]) -> dict[str, Any]:
    """Run one case; ``problems`` lists every way the engine disagreed with the policy."""
    app = application(case)
    card = float(constraints.card_rate(app)[0])
    problems: list[str] = []
    expected_card = case["expected_card_rate"].strip().lower()
    if expected_card == "none":
        if not np.isnan(card):
            problems.append(f"expected no standard offer, got a card rate of {card:.2f}%")
    elif np.isnan(card) or abs(card - float(expected_card)) > 1e-6:
        problems.append(f"card rate {card:.2f}%, expected {float(expected_card):.2f}%")

    check = constraints.check(app, proposed_rate(case["rate"], card),
                              override_reason=case["override_reason"] or None,
                              evidence_verified=case["evidence_verified"].strip().lower() == "true")
    said = check.violations + check.conditions + check.warnings
    if check.verdict != case["expected_verdict"]:
        problems.append(f"verdict {check.verdict}, expected {case['expected_verdict']}")
    if case["expected_approval"] and check.required_approval != case["expected_approval"]:
        problems.append(f"approval {check.required_approval}, expected {case['expected_approval']}")
    if case["expected_message"] and not any(case["expected_message"] in s for s in said):
        problems.append(f"no reason mentions '{case['expected_message']}'")
    return {"case_id": case["case_id"], "rule": case["rule"], "description": case["description"],
            "card_rate": None if np.isnan(card) else round(card, 2), "verdict": check.verdict,
            "approval": check.required_approval, "reasons": said, "ok": not problems, "problems": problems}


def run(path: Path | None = None) -> list[dict[str, Any]]:
    return [evaluate(case) for case in load(path)]
