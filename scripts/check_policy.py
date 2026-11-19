#!/usr/bin/env python3
"""Policy file consistency checks (invariants INV-01..INV-11 in the policy's Part C).

    python scripts/check_policy.py [config/pricing_policy.yaml] [config/model_config.yaml]

Prints PASS / FAIL / SKIP per check and exits 1 if any check fails. The checks live
in ``aavas.governance.policy_checks``; the agentic workflow runs the same ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aavas.governance import policy_checks  # noqa: E402


def main(argv: list[str]) -> int:
    policy_path = Path(argv[0]) if argv else ROOT / "config" / "pricing_policy.yaml"
    model_path = Path(argv[1]) if len(argv) > 1 else ROOT / "config" / "model_config.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    cutoffs = yaml.safe_load(model_path.read_text(encoding="utf-8"))["risk"]["grade_cutoffs"]
    results = policy_checks.run(policy, cutoffs)
    for r in results:
        word = "SKIP" if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")
        print(f"{word} {r['check']}" + (f"  {r['detail']}" if r["detail"] else ""))
    s = policy_checks.summary(results)
    print(f"\n{s['passed']} passed, {len(s['failed'])} failed, {s['skipped']} skipped of {s['checks']}")
    return 0 if s["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
