"""Ten lower-rate requests for the area manager, each a different kind of decision.

Every file goes through the real flow, as an RM would take it: the indicative quote
on the customer's own figures, the application, the three documents, the verified
figures and the customer interaction, the risk-adjusted price, then the RM's request
for a lower rate. Nothing is written around the service except the dates: each file
is set back a few days so the worklists show aging.

    python scripts/demo_scenarios.py --base http://127.0.0.1:8000

The scenarios, and the decision each is built to show:
 1. Real competitor offer, letter seen          - a fair case to approve or counter
 2. No competitor, not bargaining                - little reason to cut: hold
 3. Competitor claimed, letter not seen          - verify before approving
 4. Customer barely responds to price            - room to uplift instead
 5. Ask below the lowest rate policy allows      - approve only down to the floor
 6. Missed payments and credit deviations        - risky credit: concern
 7. Long-standing Aavas customer                 - relationship worth keeping
 8. Self-employed, notes about the EMI           - a longer tenure may do instead
 9. Deep cut close to break-even                 - thin margin: concern
10. Large loan, PSU bank offer, letter seen      - big money either way
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

PDF = b"%PDF-1.4\n% demo document\n1 0 obj <<>> endobj\ntrailer <<>>\n%%EOF\n"
DOCS = ("income_proof", "bank_statement", "identity_address")
PRE_KEYS = ("loan_id", "segment", "city", "channel", "property_type", "declared_monthly_income",
            "requested_loan_amount", "requested_tenure_years", "applicant_age", "declared_obligations",
            "estimated_property_value", "bureau_score", "sales_context")

# name, mobile, city, segment, channel, property, age, income, other EMIs, loan, years, property value,
# score, competitor (rate, lender, letter seen), verified (income, appraised, loan, years, history months,
# missed payments, banking, deviations), interaction (negotiating, rounds, existing), notes,
# the ask (bps below the system recommendation, or "floor" to go below the policy floor), reason, days waiting
SCENARIOS = [
    ("Ananya Iyer", "9820011001", "Pune", "salaried_formal", "branch_walkin", "resale_flat", 33, 140000, 12000,
     3500000, 20, 5200000, 792, (9.10, "HDFC", True), (138000, 5100000, 3500000, 20, 96, 0, 82, 0), (True, 2, False),
     "Customer has an HDFC sanction letter at 9.10% and will sign this week if we come closer.", 45, "competitor_lower_rate", 1),
    ("Vikram Rathore", "9820011002", "Jaipur", "salaried_formal", "branch_walkin", "builder_flat", 41, 120000, 18000,
     2500000, 15, 4200000, 801, None, (118000, 4100000, 2500000, 15, 140, 0, 78, 0), (False, 0, False),
     "Asked if the rate can be lower; no other offer, happy with the service.", 35, "strong_negotiation", 0),
    ("Meenakshi Sundaram", "9820011003", "Indore", "salaried_informal", "dsa", "resale_flat", 37, 85000, 9000,
     3000000, 20, 4300000, 748, (9.40, "Bajaj HF", False), (80000, 4200000, 3000000, 20, 60, 0, 70, 0), (True, 1, False),
     "Says Bajaj offered 9.40%, will share the letter later.", 40, "competitor_lower_rate", 2),
    ("Arjun Mehta", "9820011004", "Ahmedabad", "self_employed_professional", "connector", "resale_flat", 39, 260000, 30000,
     4500000, 20, 7000000, 815, None, (255000, 6900000, 4500000, 20, 150, 0, 88, 0), (True, 1, True),
     "Doctor, wants a quick sanction; rate is not the main issue, speed of disbursement is.", 30, "relationship_value", 0),
    ("Farhan Qureshi", "9820011005", "Lucknow", "salaried_formal", "branch_walkin", "resale_flat", 35, 95000, 8000,
     2000000, 20, 3200000, 770, (8.40, "SBI", True), (93000, 3100000, 2000000, 20, 84, 0, 74, 0), (True, 3, False),
     "SBI sanction letter at 8.40%; customer wants us to match it exactly.", "floor", "competitor_lower_rate", 3),
    ("Sunita Yadav", "9820011006", "Kota", "self_employed_non_professional", "dsa", "self_construction", 44, 70000, 15000,
     1800000, 15, 2600000, 662, None, (61000, 2500000, 1800000, 15, 48, 3, 41, 2), (True, 2, False),
     "Wants a lower rate; two late EMIs on a two-wheeler loan last year.", 40, "month_end_target", 1),
    ("Rajesh Gupta", "9820011007", "Udaipur", "salaried_formal", "branch_walkin", "resale_flat", 48, 180000, 20000,
     5000000, 15, 7500000, 806, None, (178000, 7400000, 5000000, 15, 200, 0, 90, 0), (True, 1, True),
     "Aavas customer for 8 years (earlier loan closed early); expects a loyalty rate.", 30, "relationship_value", 0),
    ("Pooja Nair", "9820011008", "Pune", "self_employed_professional", "connector", "builder_flat", 36, 150000, 25000,
     4000000, 20, 5600000, 774, (9.60, "ICICI HFC", False), (146000, 5500000, 4000000, 20, 72, 0, 76, 0), (True, 2, False),
     "The EMI is above her monthly budget; she asked whether a longer tenure would bring it down.", 40, "strong_negotiation", 2),
    ("Deepak Chauhan", "9820011009", "Jaipur", "salaried_formal", "branch_walkin", "resale_flat", 31, 110000, 10000,
     3000000, 25, 4000000, 742, (9.00, "LIC HF", True), (108000, 3900000, 3000000, 25, 54, 0, 72, 0), (True, 3, False),
     "LIC HF letter at 9.00%; customer is ready to walk away.", "deep", "competitor_lower_rate", 4),
    ("Kavitha Reddy", "9820011010", "Bhopal", "salaried_formal", "branch_walkin", "resale_flat", 42, 300000, 40000,
     7000000, 20, 9500000, 820, (9.05, "Bank of Baroda", True), (295000, 9400000, 7000000, 20, 180, 0, 92, 0), (True, 2, True),
     "Bank of Baroda offer at 9.05% with the letter; large ticket, long-standing customer.", 50, "competitor_lower_rate", 1),
]


def call(base, method, path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path}: {e.code} {e.read()[:300]!r}") from e


def upload(base, token, loan_id, kind):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"document_type\"\r\n\r\n{kind}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{kind}.pdf\"\r\n"
            "Content-Type: application/pdf\r\n\r\n").encode() + PDF + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{base}/applications/{loan_id}/documents", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                          "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def login(base, user, password):
    return call(base, "POST", "/auth/login", {"username": user, "password": password})["token"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--rm", default="rohit:Rohit@2026")
    ap.add_argument("--am", default="meera:Meera@2026")
    ap.add_argument("--no-backdate", action="store_true")
    args = ap.parse_args()
    rm = login(args.base, *args.rm.split(":", 1))
    am = login(args.base, *args.am.split(":", 1))
    made = []
    for n, (name, mobile, city, segment, channel, prop, age, income, emis, loan, years, value, score, comp,
            verified, interaction, notes, ask, reason, days) in enumerate(SCENARIOS, 1):
        loan_id = f"APP{int(time.time() * 1000) % 10**13:013d}"
        time.sleep(0.01)
        sales = {"competitor_rate": comp[0], "competitor_name": comp[1], "competitor_evidence_verified": comp[2]} if comp else {}
        applicant = {"loan_id": loan_id, "name": name, "mobile": mobile, "email": "", "segment": segment, "city": city,
                     "channel": channel, "property_type": prop, "applicant_age": age, "declared_monthly_income": income,
                     "declared_obligations": emis, "requested_loan_amount": loan, "requested_tenure_years": years,
                     "estimated_property_value": value, "bureau_score": score, "new_to_credit": False, "sales_context": sales}
        call(args.base, "POST", "/pricing/pre-assessment", {k: applicant[k] for k in PRE_KEYS}, rm)
        call(args.base, "POST", "/applications", {"loan_id": loan_id, "applicant": applicant}, rm)
        for kind in DOCS:
            upload(args.base, rm, loan_id, kind)
        v_income, appraised, v_loan, v_years, history, missed, banking, deviations = verified
        negotiating, rounds, existing = interaction
        full_sales = {**sales, "competitor_quote": bool(comp), "customer_negotiating": negotiating,
                      "prior_negotiation_rounds": rounds if negotiating else None, "existing_relationship": existing}
        complete = {"verified_monthly_income": v_income, "appraised_property_value": appraised, "sanctioned_loan_amount": v_loan,
                    "tenure_years": v_years, "bureau_vintage_months": history, "delinquency_count_24m": missed,
                    "banking_behaviour_score": banking, "deviation_count": deviations, "legal_flag": 0, "technical_flag": 0,
                    "sales_context": full_sales, "conversation_notes": notes}
        call(args.base, "PUT", f"/applications/{loan_id}", {"applicant": complete}, rm)
        priced = call(args.base, "POST", "/pricing/final-assessment",
                      {**{k: applicant[k] for k in PRE_KEYS if k != "sales_context"}, "sales_context": {k: v for k, v in full_sales.items() if v is not None},
                       **{k: v for k, v in complete.items() if k not in ("sales_context", "conversation_notes")}}, rm)
        rec = priced["recommendation"]["recommended_rate"]
        full = call(args.base, "GET", f"/pricing/decision/{priced['decision_id']}", None, am)["payload"]
        envelope, risk = full.get("envelope") or {}, full.get("risk") or {}
        if ask == "floor":
            rate = round(float(envelope["min_rate"]) - 0.25, 2)
        elif ask == "deep":
            rate = round(min(max(float(risk.get("breakeven_rate") or rec - 1) + 0.15, float(envelope["min_rate"])), rec - 0.30), 2)
        else:
            rate = round(rec - ask / 100, 2)
        req = call(args.base, "POST", "/pricing/rate-requests", {"loan_id": loan_id, "rate": rate, "reason": reason}, rm)
        made.append((loan_id, req["request_id"], days))
        print(f"{n:2}. {name:20} {city:10} rec {rec:.2f}% ask {rate:.2f}% grade {risk.get('risk_grade')} "
              f"floor {envelope.get('min_rate')} break-even {risk.get('breakeven_rate')}")
    if not args.no_backdate:
        backdate(made)


def backdate(made) -> None:
    """Set each file back a few days, so the worklists show realistic aging."""
    from aavas.governance import store
    now = datetime.now(timezone.utc)
    stamp = lambda d, extra=0: (now - timedelta(days=d, hours=extra)).isoformat(timespec="seconds")  # noqa: E731
    with store.connect() as conn:
        for loan_id, request_id, days in made:
            if not days:
                continue
            opened = stamp(days + 2, 3)
            conn.execute("UPDATE applications SET created_at = ?, updated_at = ? WHERE loan_id = ?", (opened, stamp(days, 2), loan_id))
            conn.execute("UPDATE decisions SET created_at = ? WHERE loan_id = ? AND stage = 'pre_assessment'", (opened, loan_id))
            conn.execute("UPDATE application_documents SET uploaded_at = ? WHERE loan_id = ?", (stamp(days + 1), loan_id))
            conn.execute("UPDATE decisions SET created_at = ? WHERE loan_id = ? AND stage = 'post_assessment'", (stamp(days, 1), loan_id))
            conn.execute("UPDATE rate_requests SET requested_at = ? WHERE request_id = ?", (stamp(days), request_id))
    print("set back", sum(1 for *_, d in made if d), "files to show aging")


if __name__ == "__main__":
    main()
