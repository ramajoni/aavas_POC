#!/usr/bin/env python3
"""Run the walkthrough deck's three test cases through the pricing service.

    python scripts/run_test_cases.py                      # in-process, all three
    python scripts/run_test_cases.py --cases TC2          # one
    python scripts/run_test_cases.py --base-url http://localhost:8000
    python scripts/run_test_cases.py --narrative          # also ask Claude to explain

The deck's appendix shows each case's *dataset* values - what the synthetic
generator priced and what the customer did. Those are references, not outputs of
this prototype (open item 8). This script replaces them: every number it writes
comes back from the same endpoints the LOS and the UI call, so the walkthrough can
show the prototype's own rate, band, approval level and policy verdict.

Each case is a loan from the synthetic book, replayed as an application:

* TC1 ``AAV0010358`` - prime borrower, verified competitor quote below card. The RM
  asks to match it; the approval matrix decides who may say yes.
* TC2 ``AAV0009223`` - grade D, the RM cuts the rate to 12.82%. The cut is recorded
  at sales-officer authority first, to show the refer/block, then escalated.
* TC3 ``AAV0000582`` - new to credit. The band narrows from Stage 1 to Stage 2, the
  RM's notes are read, and no discount is needed.

Outcomes are the dataset's own recorded customer response, replayed so the audit
trail shows a closed loop. They are history, not a prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aavas.ingestion import loader  # noqa: E402
from aavas.logging_utils import get_logger  # noqa: E402
from aavas.paths import REPORTS_DIR  # noqa: E402

log = get_logger("run_test_cases")
OUT_DIR = REPORTS_DIR / "test_cases"

# What the deck says each case is for, and the human step it scripts. The rates
# are the deck's own: the RM's proposed rate is an input, not a result.
CASES: dict[str, dict[str, Any]] = {
    "TC1": {"loan_id": "AAV0010358", "title": "Prime borrower, competitor below card",
            "flow": "Borrower -> RM pricing -> Manager",
            "override": {"rate": "competitor", "reason": "competitor_lower_rate",
                         "detail": "RM requests a match to the verified competitor sanction letter.",
                         "first_approver": "rm.demo:sales_officer", "escalate": True}},
    "TC2": {"loan_id": "AAV0009223", "title": "Higher-risk borrower, override",
            "flow": "RM pricing -> Manager",
            "override": {"rate": 12.82, "reason": "competitor_lower_rate",
                         "detail": "RM cuts the rate to hold the file against a verified competitor quote.",
                         "first_approver": "rm.demo:sales_officer", "escalate": True}},
    "TC3": {"loan_id": "AAV0000582", "title": "New to credit, self-employed",
            "flow": "RM application -> RM pricing",
            "override": None},
}

SALES_CONTEXT = ["competitor_rate", "competitor_name", "competitor_evidence_verified",
                 "sales_target_pct", "branch_target_pct", "customer_negotiating",
                 "prior_negotiation_rounds", "existing_relationship", "month_end",
                 "days_to_month_end"]
STAGE1 = ["segment", "city", "channel", "property_type", "declared_monthly_income",
          "requested_loan_amount", "requested_tenure_years", "applicant_age",
          "declared_obligations", "estimated_property_value", "bureau_score", "new_to_credit"]
STAGE2 = ["verified_monthly_income", "appraised_property_value", "sanctioned_loan_amount",
          "tenure_years", "bureau_vintage_months", "delinquency_count_24m",
          "banking_behaviour_score", "deviation_count", "legal_flag", "technical_flag"]


# ------------------------------------------------------------------ transport
class Client:
    """The same calls whether the service is in this process or across the network."""

    def __init__(self, base_url: str | None):
        self.base_url = base_url.rstrip("/") if base_url else None
        if self.base_url:
            import httpx

            self._http = httpx.Client(base_url=self.base_url, timeout=180)
        else:
            from fastapi.testclient import TestClient

            from aavas.api.main import app

            self._http = TestClient(app)

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, Any]:
        response = self._http.request(method, path, json=body)
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        return response.status_code, payload


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and value != value:     # NaN
        return None
    return value


def application(row: pd.Series, loan_id: str, final: bool) -> dict[str, Any]:
    body = {"loan_id": loan_id, **{k: _plain(row[k]) for k in STAGE1}}
    body["sales_context"] = {k: _plain(row[k]) for k in SALES_CONTEXT}
    if final:
        body.update({k: _plain(row[k]) for k in STAGE2})
    return body


# ------------------------------------------------------------------ one case
def run_case(key: str, spec: dict[str, Any], table: pd.DataFrame, notes: pd.DataFrame,
             client: Client, run_tag: str, narrative: bool) -> dict[str, Any]:
    row = table.loc[table["loan_id"] == spec["loan_id"]]
    if row.empty:
        raise ValueError(f"{spec['loan_id']} is not in the dataset")
    row = row.iloc[0]
    # A fresh application id per run: the decision store is append-only, and a
    # replay must not be read as a second decision on the original file.
    app_id = f"{key}-{spec['loan_id']}-{run_tag}"
    steps: list[dict[str, Any]] = []
    out: dict[str, Any] = {"case": key, "title": spec["title"], "flow": spec["flow"],
                           "source_loan_id": spec["loan_id"], "application_id": app_id,
                           "steps": steps}

    def step(name: str, method: str, path: str, body: dict | None = None) -> Any:
        status, payload = client.call(method, path, body)
        steps.append({"step": name, "call": f"{method} {path}", "status": status})
        if status >= 400:
            steps[-1]["error"] = payload
        return payload if status < 400 else None

    pre = step("GetPriceRange", "POST", "/pricing/pre-assessment", application(row, app_id, final=False))
    final = step("GetRiskBasedPriceRecommendation", "POST",
                 "/pricing/final-assessment?include_scenarios=true",
                 application(row, app_id, final=True))
    if final is None:
        out["error"] = "final assessment failed"
        return out

    rec, pol = final["recommendation"], final["policy"]
    out["pre_assessment"] = _pre_summary(pre)
    out["recommendation"] = {
        "decision_id": final.get("decision_id"),
        "recommended_rate": rec.get("recommended_rate"),
        "card_rate": pol.get("card_rate"),
        "permissible_band": pol.get("permissible_band") or {k: pol.get(k) for k in ("min_rate", "max_rate")},
        "required_approval": pol.get("required_approval"),
        "risk": final.get("risk"),
        "conversion_probability": rec.get("conversion_probability") or rec.get("expected_conversion"),
        "alternatives": rec.get("comparisons"),
        "policy_verdict": pol.get("verdict"),
        "conditions": pol.get("conditions"),
        "resize": pol.get("resize"),
        "notes": final.get("notes"),
        "fair_pricing": (final.get("explanation") or {}).get("fair_pricing_review"),
        "model_versions": final.get("model_versions"),
    }

    # The RM's own notes on the source file, read the way the RM screen reads them.
    file_notes = notes.loc[notes["loan_id"] == spec["loan_id"], "note_text"].astype(str).tolist()
    if file_notes:
        extracted = step("NoteExtraction", "POST", "/notes/extract",
                         {"notes": file_notes, "loan_id": app_id,
                          "sales_context": application(row, app_id, final=False)["sales_context"]})
        out["notes"] = {"text": file_notes, "extraction": extracted}
        context = step("CompetitorContext", "POST", "/competitors/context",
                       {"application": application(row, app_id, final=True), "notes": file_notes})
        if context is not None:
            out["competitor_context"] = context

    explained = step("LoanExplanation", "GET", f"/pricing/explain/{final['decision_id']}")
    if explained is not None:
        out["explanation"] = {
            "rate": explained.get("rate"), "bridge": explained.get("bridge"),
            "methods": {k: m.get("method") for k, m in (explained.get("models") or {}).items()},
            "headline": (explained.get("context") or {}).get("headline"),
            "warnings": explained.get("warnings"),
        }
        if narrative and (explained.get("narrative") or {}).get("available"):
            text = step("AIExplanation", "POST", f"/pricing/explain/{final['decision_id']}/narrative")
            if text is not None:
                out["explanation"]["narrative"] = text.get("sections")

    if spec["override"]:
        out["override"] = _override(spec["override"], row, final, app_id, step)

    outcome = {"loan_id": app_id, "decision_id": final["decision_id"],
               "disbursed": bool(row["disbursed"]),
               "customer_response": str(row["customer_response"]),
               "sanctioned_rate": (out.get("override") or {}).get("final_rate") or rec.get("recommended_rate")}
    if not bool(row["disbursed"]):
        outcome.update({"loss_reason": "competitor_lower_rate", "lost_to": str(row["competitor_name"]),
                        "lost_to_rate": float(row["competitor_rate"])})
    recorded = step("RecordPricingOutcome", "POST", "/pricing/outcome", outcome)
    if recorded is None and outcome.get("loss_reason"):
        # An unknown loss-reason code is the caller's error; record without it rather
        # than lose the outcome.
        outcome.pop("loss_reason")
        recorded = step("RecordPricingOutcome (retry)", "POST", "/pricing/outcome", outcome)
    out["outcome"] = {"replayed_from_dataset": True, **{k: outcome[k] for k in
                      ("disbursed", "customer_response", "sanctioned_rate")}, "recorded": recorded is not None}

    audit = step("AuditTrail", "GET", f"/pricing/audit/{app_id}")
    if audit is not None:
        out["audit_events"] = [e.get("action") for e in audit.get("audit_entries", [])]
    out["dataset_reference"] = {
        "stage1_range": [float(row["stage1_min_rate"]), float(row["stage1_max_rate"])],
        "recommended_rate": float(row["ai_recommended_rate"]),
        "override_bps": float(row["override_bps"]), "sanctioned_rate": float(row["sanctioned_rate"]),
        "customer_response": str(row["customer_response"]),
        "true_pd": float(row["true_pd"]), "true_lgd": float(row["true_lgd"]),
        "risk_grade": str(row["risk_grade"]),
    }
    return out


def _pre_summary(pre: dict | None) -> dict | None:
    if not pre:
        return None
    band = pre.get("indicative_band") or pre.get("band") or {}
    return {"decision_id": pre.get("decision_id"), "band": band,
            "most_likely_rate": pre.get("most_likely_rate") or band.get("mid_rate"),
            "confidence": pre.get("confidence"), "eligibility": pre.get("eligibility")}


def _override(spec: dict, row: pd.Series, final: dict, app_id: str, step) -> dict[str, Any]:
    rate = float(row["competitor_rate"]) if spec["rate"] == "competitor" else float(spec["rate"])
    evidence = bool(row["competitor_evidence_verified"])
    result: dict[str, Any] = {"proposed_rate": rate, "reason": spec["reason"],
                              "evidence_verified": evidence}

    # The uplift model is fitted on discounts; a proposal at or above the
    # recommendation is a premium it was never shown, so it is not asked.
    if rate < float(final["recommendation"]["recommended_rate"]):
        uplift = step("PricingUplift", "POST", "/pricing/uplift",
                      {"decision_id": final["decision_id"], "proposed_rate": rate})
        if uplift is not None:
            result["uplift"] = uplift
    else:
        result["uplift"] = {"skipped": "proposal is at or above the recommendation - a premium, "
                                       "which the uplift model does not cover"}

    first = step("Override (RM authority)", "POST", "/pricing/override",
                 {"loan_id": app_id, "human_rate": rate, "reason": spec["reason"],
                  "reason_detail": spec["detail"], "approved_by": spec["first_approver"],
                  "evidence_verified": evidence})
    if first is None:
        result["error"] = "override capture failed"
        return result
    result["required_approval"] = first.get("required_approval")
    result["within_rm_authority"] = first.get("within_authority")
    result["policy_check"] = first.get("policy_check")
    result["warning"] = first.get("warning")

    allowed = (first.get("policy_check") or {}).get("allowed", True)
    if not allowed:
        # The deck's manager can approve, reject or counter-offer. A blocked match
        # becomes a counter at the lowest rate policy permits - never below it.
        band = (first.get("policy_check") or {}).get("permissible_band") or {}
        counter = band.get("min_rate")
        if counter is not None and counter > rate:
            result["counter_offer"] = counter
            rate, allowed = float(counter), True
            first = step("Override (counter-offer at the policy floor)", "POST", "/pricing/override",
                         {"loan_id": app_id, "human_rate": rate, "reason": spec["reason"],
                          "reason_detail": spec["detail"] + " Counter-offer at the lowest permissible rate.",
                          "approved_by": spec["first_approver"], "evidence_verified": evidence}) or first
            result["counter_policy_check"] = first.get("policy_check")
            result["required_approval"] = first.get("required_approval")
    if spec.get("escalate") and first.get("within_authority") is False and allowed:
        approver = f"manager.demo:{first['required_approval']}"
        second = step("Override (escalated)", "POST", "/pricing/override",
                      {"loan_id": app_id, "human_rate": rate, "reason": spec["reason"],
                       "reason_detail": spec["detail"] + " Escalated to the required authority.",
                       "approved_by": approver, "evidence_verified": evidence})
        if second is not None:
            result["escalated_to"] = approver
            result["within_authority_after_escalation"] = second.get("within_authority")
    result["final_rate"] = rate if allowed else final["recommendation"]["recommended_rate"]
    result["released"] = bool(allowed and (result.get("within_authority_after_escalation")
                                           or first.get("within_authority")))
    return result


# ------------------------------------------------------------------- report
def summary_table(results: list[dict[str, Any]]) -> str:
    def pct(v):
        return "—" if v is None else f"{float(v):.2f}%"

    rows = [("Attribute", *[f"{r['case']} · {r['source_loan_id']}" for r in results])]
    pick = [
        ("Stage 1 range", lambda r: _band(r.get("pre_assessment"))),
        ("Card rate", lambda r: pct(r["recommendation"]["card_rate"])),
        ("Recommended rate", lambda r: pct(r["recommendation"]["recommended_rate"])),
        ("Permitted band", lambda r: _band_policy(r["recommendation"]["permissible_band"])),
        ("Risk (model)", lambda r: _risk(r["recommendation"]["risk"])),
        ("Approval at recommendation", lambda r: str(r["recommendation"]["required_approval"])),
        ("Guardrail at recommendation", lambda r: str(r["recommendation"]["policy_verdict"])
         + ("; " + "; ".join(r["recommendation"]["conditions"]) if r["recommendation"].get("conditions") else "")),
        ("RM proposal", lambda r: _proposal(r.get("override"))),
        ("Policy verdict", lambda r: _verdict(r.get("override"))),
        ("Outcome replayed", lambda r: f"{r['outcome']['customer_response']} at {pct(r['outcome']['sanctioned_rate'])}"),
        ("Dataset reference", lambda r: (f"range {r['dataset_reference']['stage1_range'][0]:.2f}–"
                                         f"{r['dataset_reference']['stage1_range'][1]:.2f}%, rec. "
                                         f"{r['dataset_reference']['recommended_rate']:.2f}%")),
    ]
    for label, fn in pick:
        rows.append((label, *[_safe(fn, r) for r in results]))
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _safe(fn, r) -> str:
    try:
        return str(fn(r))
    except Exception:                                    # noqa: BLE001 - a missing field is a dash
        return "—"


def _band(pre) -> str:
    band = (pre or {}).get("band") or {}
    lo = band.get("min_rate") or band.get("low")
    hi = band.get("max_rate") or band.get("high")
    return f"{lo:.2f}–{hi:.2f}%" if lo is not None and hi is not None else "—"


def _band_policy(band) -> str:
    band = band or {}
    return f"{band['min_rate']:.2f}–{band['max_rate']:.2f}%"


def _risk(risk) -> str:
    risk = risk or {}
    return f"grade {risk.get('risk_grade')}, PD {100 * float(risk.get('pd', 0)):.2f}%, LGD {100 * float(risk.get('lgd', 0)):.0f}%"


def _proposal(ovr) -> str:
    if not ovr:
        return "none"
    return f"{ovr['proposed_rate']:.2f}% ({ovr['reason']}, evidence {'verified' if ovr['evidence_verified'] else 'not verified'})"


def _verdict(ovr) -> str:
    if not ovr:
        return "no override"
    check = ovr.get("policy_check") or {}
    parts = [str(check.get("verdict") or ("pass" if check.get("allowed", True) else "block"))]
    if ovr.get("counter_offer") is not None:
        counter = ovr.get("counter_policy_check") or {}
        parts.append(f"counter-offer {ovr['counter_offer']:.2f}% ({counter.get('verdict', '—')})")
    parts.append(f"needs {ovr.get('required_approval')}")
    if ovr.get("escalated_to"):
        parts.append(f"escalated to {ovr['escalated_to'].split(':')[-1]}")
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    parser.add_argument("--base-url", default=None, help="a running service; default runs in-process")
    parser.add_argument("--narrative", action="store_true", help="also request the AI explanation")
    args = parser.parse_args(argv)

    from datetime import datetime, timezone

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    table = loader.load_training_table()
    notes = loader.load_notes()
    client = Client(args.base_url)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for key in args.cases:
        log.info("running %s (%s)", key, CASES[key]["loan_id"])
        result = run_case(key, CASES[key], table, notes, client, run_tag, args.narrative)
        results.append(result)
        with open(OUT_DIR / f"{key.lower()}.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        failed = [s for s in result["steps"] if s["status"] >= 400]
        log.info("%s: %s steps, %s failed%s", key, len(result["steps"]), len(failed),
                 "" if not failed else " (" + ", ".join(f"{s['step']} {s['status']}" for s in failed) + ")")

    table_md = summary_table([r for r in results if "recommendation" in r])
    with open(OUT_DIR / "summary.md", "w", encoding="utf-8") as fh:
        fh.write("# Walkthrough test cases - prototype outputs\n\n")
        fh.write(f"Run {run_tag} UTC via {'`' + args.base_url + '`' if args.base_url else 'the in-process API'}. "
                 "Every value except the dataset reference row came back from the pricing service.\n\n")
        fh.write(table_md + "\n")
    print(table_md)
    print(f"\nReports: {OUT_DIR}")
    return 0 if all("recommendation" in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
