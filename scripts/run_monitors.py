#!/usr/bin/env python3
"""Run the monitoring passes and record that they ran.

    python scripts/run_monitors.py [--monitor drift|fairness|retraining]

This is the thing a scheduler calls. Drift computed when someone opens a page is
a report; a monitor records having run, what it found, and - by its absence from
the history - when it did not run at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aavas.governance import monitors  # noqa: E402
from aavas.logging_utils import get_logger  # noqa: E402

log = get_logger("run_monitors")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the recorded monitoring passes")
    parser.add_argument("--monitor", choices=["drift", "fairness", "retraining"],
                        help="run one pass instead of all of them")
    parser.add_argument("--actor", default="scheduler")
    parser.add_argument("--json", action="store_true", help="print the findings as JSON")
    args = parser.parse_args(argv)

    runners = {"drift": monitors.run_drift, "fairness": monitors.run_fairness,
               "retraining": monitors.run_retraining}
    results = ({args.monitor: runners[args.monitor](actor=args.actor)} if args.monitor
               else monitors.run_all(actor=args.actor))

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"\n{'MONITOR':14s} {'STATUS':8s} {'TRIGGERED':10s} FINDING")
        print("-" * 96)
        for name, result in results.items():
            findings = result.get("findings") or {}
            headline = (", ".join(f"{m['model']} gap {m['gaps']}" for m in findings["failing"])
                        if findings.get("failing") else
                        "; ".join(findings.get("triggers") or []) if findings.get("triggers") else
                        str(findings.get("overall") or findings.get("error") or "nothing to report"))
            print(f"{name:14s} {result.get('status',''):8s} "
                  f"{'yes' if result.get('triggered') else 'no':10s} {headline[:60]}")

    # A triggered monitor is not a failure of this script - it is the script
    # doing its job - so the exit code reports only whether a pass errored.
    failed = [n for n, r in results.items() if r.get("status") == "failed"]
    if failed:
        log.error("monitor(s) failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
