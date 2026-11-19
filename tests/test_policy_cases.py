"""Policy sample data: every row of tests/data/policy_cases.csv is one test.

The evaluation lives in ``aavas.governance.policy_cases`` so the agentic workflow
runs exactly what this test runs. Credit can add a case by adding a row.
"""

from __future__ import annotations

import pytest

from aavas.governance import policy_cases


@pytest.mark.parametrize("case", policy_cases.load(), ids=lambda c: f"{c['case_id']}-{c['rule']}")
def test_policy_case(case: dict) -> None:
    result = policy_cases.evaluate(case)
    assert result["ok"], f"{case['case_id']} ({case['description']}): {result['problems']}; said {result['reasons']}"
