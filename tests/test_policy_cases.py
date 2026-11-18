"""Policy sample data: every row of tests/data/policy_cases.csv is one test.

Each row is an applicant, the rate proposed for them, and what the pricing policy
says must happen: the card rate (or ``none`` for no standard offer), the verdict
(pass / refer / block), the approval level, and words the reason must contain.

The expected values are worked out from config/pricing_policy.yaml - the matrix,
the limits, EMI arithmetic at the stressed rate - not read back from the engine,
so a row fails when the engine and the policy disagree. Credit can add a case by
adding a row; no code change is needed.

    rate column: ``card`` | ``card-35`` / ``card+5`` (bps from the card rate) | a rate in %
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aavas.optimization import constraints

CASES = Path(__file__).parent / "data" / "policy_cases.csv"


def _cases() -> list[dict]:
    with CASES.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _application(case: dict) -> pd.DataFrame:
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


def _rate(spec: str, card: float) -> float:
    spec = spec.strip().lower()
    if spec.startswith("card"):
        offset = spec[4:]
        return round(card + (float(offset) / 100 if offset else 0.0), 2)
    return float(spec)


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c['case_id']}-{c['rule']}")
def test_policy_case(case: dict) -> None:
    app = _application(case)
    card = float(constraints.card_rate(app)[0])

    if case["expected_card_rate"].strip().lower() == "none":
        assert np.isnan(card), f"{case['case_id']}: expected no standard offer, got a card rate of {card:.2f}%"
    else:
        assert card == pytest.approx(float(case["expected_card_rate"])), f"{case['case_id']}: card rate"

    check = constraints.check(app, _rate(case["rate"], card),
                              override_reason=case["override_reason"] or None,
                              evidence_verified=case["evidence_verified"].strip().lower() == "true")
    said = check.violations + check.conditions + check.warnings
    assert check.verdict == case["expected_verdict"], f"{case['case_id']}: verdict {check.verdict}; {said}"
    if case["expected_approval"]:
        assert check.required_approval == case["expected_approval"], f"{case['case_id']}: approval"
    if case["expected_message"]:
        assert any(case["expected_message"] in s for s in said), \
            f"{case['case_id']}: expected '{case['expected_message']}' in {said}"
