#!/usr/bin/env python3
"""Detect data, model and concept drift and ship the findings to OpenObserve.

    python scripts/run_drift.py                         # the live book, as it is
    python scripts/run_drift.py --scenario price_war    # simulate drift to prove the alerts fire
    python scripts/run_drift.py --provision             # (re)create the OpenObserve alerts and dashboard
    python scripts/run_drift.py --no-ship               # report only

Scenarios: none, data_shift, price_war, default_surge (src/aavas/monitoring/drift.py).
Settings: config/monitoring.yaml; OpenObserve connection from .env.openobserve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aavas.monitoring import drift, openobserve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data, model and concept drift -> OpenObserve")
    parser.add_argument("--scenario", default="none", choices=drift.SCENARIOS)
    parser.add_argument("--provision", action="store_true", help="create or update the OpenObserve alerts and dashboard")
    parser.add_argument("--no-ship", action="store_true", help="do not send the findings to OpenObserve")
    args = parser.parse_args(argv)

    report = drift.report(args.scenario)
    out = drift.save(report)

    s = report["status"]
    print(f"drift run {report['run_id']} (scenario {args.scenario}): overall {s['overall']} | data {s['data']} | "
          f"model {s['model']} | concept {s['concept']}")
    print(f"  reference {report['windows']['reference']}  current {report['windows']['current']}")
    for kind in ("data", "model", "concept"):
        for f in [f for f in report[kind] if f["status"] != "ok"][:6]:
            print(f"  {kind:<7} {f['status']:<5} {f['name']:<36} psi={f.get('psi')} ks={f.get('ks_stat')} "
                  f"ref={f.get('reference_value')} cur={f.get('current_value')}"
                  + (f" changed {f['change_detected_at']}" if f.get("change_detected_at") else ""))

    if not args.no_ship:
        client = openobserve.Client()
        if not client.health():
            print(f"OpenObserve at {client.url} is not reachable; findings kept in {out}")
            return 2
        print("shipped:", json.dumps(openobserve.ship(report, client)))
        if args.provision:                       # after shipping: an alert needs its stream to exist
            print("provisioned:", json.dumps(client.provision(), default=str))
    return 1 if s["overall"] == "alert" else 0


if __name__ == "__main__":
    raise SystemExit(main())
