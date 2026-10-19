"""Run the Swagger examples against the running API, recording a synthetic loan.

Usage: python3 scripts/run_api_flow.py [http://localhost:8000]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")

    def call(path, payload=None):
        request = Request(
            base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=120) as response:
            body = json.load(response)
            print(f"{response.status} {path}", flush=True)
            return body

    schemas = call("/openapi.json")["components"]["schemas"]
    loan_id = "DEMO-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")

    def example(name):
        return {**schemas[name]["examples"][0], "loan_id": loan_id}

    assert call("/health")["status"] == "ok", "Models must be ready"
    call("/pricing/pre-assessment", example("Stage1Request"))
    decision = call("/pricing/final-assessment?include_scenarios=true", example("Stage2Request"))
    call("/pricing/override/reasons")
    override = example("OverrideRequest")
    override["human_rate"] = round(decision["policy"]["card_rate"] - 0.05, 2)
    override["reason_detail"] = "Synthetic demo: negotiated five basis points below card rate."
    record = call("/pricing/override", override)
    assert record["policy_check"]["allowed"], record["policy_check"]
    assert record["within_authority"], record
    outcome = example("OutcomeRequest")
    outcome["sanctioned_rate"] = override["human_rate"]
    call("/pricing/outcome", outcome)
    call("/pricing/decision/" + decision["decision_id"])
    history = call("/pricing/audit/" + loan_id)
    actions = {entry["action"] for entry in history["audit_entries"]}
    assert {"post_assessment_decision", "capture_override", "capture_outcome"} <= actions
    for path in ["/model-cards", "/pricing/policy/explain?question=branch%20manager%20discount%20limit",
                 "/pricing/learning/override-review", "/pricing/learning/questions"]:
        call(path)
    print(f"Complete flow recorded for {loan_id}; decision {decision['decision_id']}")


if __name__ == "__main__":
    main()
