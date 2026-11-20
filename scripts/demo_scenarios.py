"""Ten lower-rate requests for the area manager, each a different kind of decision.

Every file goes through the real flow, as an RM would take it: the indicative quote
on the customer's own figures, the application, the three documents, the verified
figures and the customer interaction, the risk-adjusted price, then the RM's request
for a lower rate. Nothing is written around the service except the dates: each file
is set back a few days so the worklists show aging.

    python scripts/demo_scenarios.py --base http://127.0.0.1:8000                  # the ten core files
    python scripts/demo_scenarios.py --base http://127.0.0.1:8000 --set extended   # twelve more: other grades, tenures, situations
    python scripts/demo_scenarios.py --base http://127.0.0.1:8000 --set all        # both sets

--set core is the default and is what the script always did. The core set is skewed to grade A,
ten-year housing loans. The extended set spreads the queue over the kinds of decision the live
policy (HL-LAP-POLICY-2026.14) allows. Under it no rate may sit above the matrix (card) rate and
the floor is card - 60 bps (never below 13.00%), so the policy floor always binds before the area
manager's own 100 bps limit; no case is built "beyond her authority".

The core scenarios, and the decision each is built to show:
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

The extended scenarios (--set extended). Grades come from the risk model, so each file was sized
to land in the grade named; "below floor" means the ask is under the policy floor, because the
recommendation already sits at (or within 25 bps of) the floor and the RM may only send an ask
more than 25 bps under the recommendation:
 E1.  B, 4 years, letter under our floor          - best possible is the floor: counter or decline
 E2.  A, 7 years, CA, letter equals our rate       - already matched: nothing to give
 E3.  A+, 10 years, informal pay, claim only       - small cut inside the room, verify first
 E4.  B, 7 years, shop owner, Shriram letter seen  - letter supports only the floor: counter
 E5.  C, 10 years, score 681, late EMIs, no offer  - credit head approval, no ground to cut
 E6.  A+, 10 years, balance transfer, letter seen  - ask exactly at the floor; rate already below the letter
 E7.  New to credit, no bureau score               - policy sends it to manual review (credit head)
 E8.  A, month-end target, nothing from customer   - decline: no evidence for a cut
 E9.  A+, price-insensitive existing customer      - hold: uplift cannot go above the matrix
 E10. A, Rs 74 lakh loan, near the Rs 75 lakh cap  - big money, SBI letter seen
 E11. A, Rs 28 lakh at 89% LTV, near the 90% cap   - thin collateral cushion, little reason to cut
 E12. B, deep cut close to break-even              - thin margin: concern
Not covered: LAP residential / commercial (the intake API accepts only the four housing property
types), a D grade (declined before it reaches the area manager) and an ask beyond the area
manager's 100 bps (the policy floor always binds first).
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
# the ask (bps below the system recommendation, "floor" to go 25 bps below the policy floor, "at_floor" to ask
# exactly the policy floor, or "deep" for a cut close to break-even), reason, days waiting.
# A bureau score below zero means new to credit.
SCENARIOS = [
    ('Ananya Iyer', '9820011001', 'Pune', 'salaried_formal', 'branch_walkin', 'resale_flat', 33, 165000, 12000,
     3500000, 10, 5200000, 792, (13.4, 'HDFC', True), (163000, 5100000, 3500000, 10, 96, 0, 82, 0), (True, 2, False),
     'Customer has an HDFC sanction letter at 13.40% and will sign this week if we come closer.', 45, 'competitor_lower_rate', 1),
    ('Vikram Rathore', '9820011002', 'Jaipur', 'salaried_formal', 'branch_walkin', 'builder_flat', 41, 140000, 18000,
     2500000, 10, 4200000, 801, None, (138000, 4100000, 2500000, 10, 140, 0, 78, 0), (False, 0, False),
     'Asked if the rate can be lower; no other offer, happy with the service.', 35, 'strong_negotiation', 0),
    ('Meenakshi Sundaram', '9820011003', 'Indore', 'salaried_informal', 'dsa', 'resale_flat', 37, 140000, 9000,
     3000000, 10, 4300000, 748, (13.75, 'Bajaj HF', False), (132000, 4200000, 3000000, 10, 60, 0, 70, 0), (True, 1, False),
     'Says Bajaj offered 13.75%, will share the letter later.', 40, 'competitor_lower_rate', 2),
    ('Arjun Mehta', '9820011004', 'Ahmedabad', 'self_employed_professional', 'connector', 'resale_flat', 39, 260000, 30000,
     4500000, 10, 7000000, 815, None, (255000, 6900000, 4500000, 10, 150, 0, 88, 0), (True, 1, True),
     'Doctor, wants a quick sanction; rate is not the main issue, speed of disbursement is.', 30, 'relationship_value', 0),
    ('Farhan Qureshi', '9820011005', 'Lucknow', 'salaried_formal', 'branch_walkin', 'resale_flat', 35, 100000, 8000,
     2000000, 10, 3200000, 770, (12.9, 'SBI', True), (98000, 3100000, 2000000, 10, 84, 0, 74, 0), (True, 3, False),
     'SBI sanction letter at 12.90%; customer wants us to match it exactly.', 'floor', 'competitor_lower_rate', 3),
    ('Sunita Yadav', '9820011006', 'Kota', 'self_employed_non_professional', 'dsa', 'self_construction', 44, 130000, 15000,
     1800000, 10, 2600000, 690, None, (113000, 2500000, 1800000, 10, 48, 3, 41, 2), (True, 2, False),
     'Wants a lower rate; two late EMIs on a two-wheeler loan last year.', 40, 'month_end_target', 1),
    ('Rajesh Gupta', '9820011007', 'Udaipur', 'salaried_formal', 'branch_walkin', 'resale_flat', 48, 240000, 20000,
     5000000, 10, 7500000, 806, None, (237000, 7400000, 5000000, 10, 200, 0, 90, 0), (True, 1, True),
     'Aavas customer for 8 years (earlier loan closed early); expects a loyalty rate.', 30, 'relationship_value', 0),
    ('Pooja Nair', '9820011008', 'Pune', 'self_employed_professional', 'connector', 'builder_flat', 36, 240000, 25000,
     4000000, 7, 5600000, 774, (13.9, 'ICICI HFC', False), (234000, 5500000, 4000000, 7, 72, 0, 76, 0), (True, 2, False),
     'The EMI is above her monthly budget; she asked whether a longer tenure would bring it down.', 40, 'strong_negotiation', 2),
    ('Deepak Chauhan', '9820011009', 'Jaipur', 'salaried_formal', 'branch_walkin', 'resale_flat', 31, 140000, 10000,
     3000000, 10, 4000000, 742, (13.25, 'LIC HF', True), (137000, 3900000, 3000000, 10, 54, 0, 72, 0), (True, 3, False),
     'LIC HF letter at 13.25%; customer is ready to walk away.', 'deep', 'competitor_lower_rate', 4),
    ('Kavitha Reddy', '9820011010', 'Bhopal', 'salaried_formal', 'branch_walkin', 'resale_flat', 42, 360000, 40000,
     7000000, 10, 9500000, 820, (13.3, 'Bank of Baroda', True), (354000, 9400000, 7000000, 10, 180, 0, 92, 0), (True, 2, True),
     'Bank of Baroda offer at 13.30% with the letter; large ticket, long-standing customer.', 50, 'competitor_lower_rate', 1),
]

EXTENDED_SCENARIOS = [
    ('Neha Kulkarni', '9820012001', 'Nashik', 'salaried_formal', 'branch_walkin', 'resale_flat', 29, 200000, 5000,
     1800000, 4, 2800000, 740, (12.8, 'Axis Bank', True), (196000, 2750000, 1800000, 4, 60, 0, 70, 0), (True, 2, False),
     'Axis Bank letter at 12.80%, under our lowest rate; she wants us to match it. The most we can do on a 4-year loan is the floor.',
     'floor', 'competitor_lower_rate', 2),
    ('Sandeep Joshi', '9820012002', 'Rajkot', 'self_employed_professional', 'connector', 'builder_flat', 45, 160000, 0,
     2800000, 7, 6000000, 800, (13.9, 'ICICI Bank', True), (158000, 5900000, 2800000, 7, 100, 0, 92, 0), (True, 1, True),
     'Chartered accountant. ICICI letter at 13.90% for 7 years, the same as our own rate; he asks for a little less.',
     30, 'competitor_lower_rate', 3),
    ('Rukhsar Sheikh', '9820012003', 'Rajkot', 'salaried_informal', 'dsa', 'self_construction', 45, 105000, 10000,
     2500000, 10, 6250000, 800, (14.4, 'Bajaj Housing Finance', False), (103000, 6200000, 2500000, 10, 60, 0, 92, 0), (True, 1, True),
     'Paid partly in cash by a small firm. Says Bajaj quoted 14.40%; the letter is yet to come. Has a savings account with us.',
     30, 'competitor_lower_rate', 4),
    ('Mukesh Agarwal', '9820012004', 'Agra', 'self_employed_non_professional', 'connector', 'resale_flat', 45, 250000, 15000,
     3500000, 7, 5600000, 770, (14.9, 'Shriram Housing Finance', True), (245000, 5500000, 3500000, 7, 96, 0, 80, 0), (True, 2, False),
     'Runs a hardware wholesale shop. Has a Shriram Housing letter at 14.90%; asking us to come down a little.',
     30, 'competitor_lower_rate', 5),
    ('Hitesh Patel', '9820012005', 'Surat', 'self_employed_non_professional', 'dsa', 'self_construction', 50, 260000, 25000,
     2600000, 10, 4200000, 681, None, (255000, 4100000, 2600000, 10, 48, 2, 52, 2), (True, 2, False),
     'Textile trader building a house on his own plot; two late EMIs last year. Wants a lower rate, no other offer.',
     30, 'strong_negotiation', 6),
    ('Priyanka Deshmukh', '9820012006', 'Alwar', 'self_employed_professional', 'dsa', 'resale_flat', 45, 135000, 10000,
     3000000, 10, 5450000, 820, (14.3, 'Kotak Mahindra Bank', True), (133000, 5400000, 3000000, 10, 160, 0, 92, 0), (True, 3, True),
     'Architect moving her loan to Kotak at 14.30% unless we match; the Kotak letter is attached. Our rate is already below it.',
     'at_floor', 'customer_threatening_bt', 2),
    ('Aditya Verma', '9820012007', 'Lucknow', 'salaried_formal', 'branch_walkin', 'resale_flat', 26, 70000, 0,
     1500000, 10, 2400000, -1, None, (68000, 2350000, 1500000, 10, 0, 0, 55, 0), (True, 1, False),
     'First job, no loan or credit card before, so no bureau score. Asks for a lower rate.',
     30, 'strong_negotiation', 1),
    ('Gaurav Saxena', '9820012008', 'Pune', 'salaried_formal', 'branch_walkin', 'resale_flat', 32, 120000, 0,
     3000000, 10, 5000000, 700, None, (118000, 4900000, 3000000, 10, 100, 0, 92, 0), (False, 0, True),
     'No other offer and no complaint about the rate. The branch is short of the month-end target and wants it closed this week.',
     30, 'month_end_target', 1),
    ('Rohan Bhatt', '9820012009', 'Surat', 'salaried_formal', 'branch_walkin', 'builder_flat', 38, 80000, 0,
     2500000, 10, 5000000, 760, None, (79000, 4900000, 2500000, 10, 100, 0, 75, 0), (False, 0, True),
     'Existing Aavas customer. Says the EMI is fine; he only wants it sanctioned before the possession date.',
     30, 'relationship_value', 0),
    ('Surendra Rathi', '9820012010', 'Surat', 'salaried_formal', 'branch_walkin', 'resale_flat', 45, 260000, 0,
     7400000, 10, 14800000, 760, (13.8, 'State Bank of India', True), (257000, 14600000, 7400000, 10, 100, 0, 92, 0), (True, 2, True),
     'Near the largest loan we make. SBI letter at 13.80%; long-standing Aavas customer.',
     40, 'competitor_lower_rate', 3),
    ('Pallavi Chauhan', '9820012011', 'Alwar', 'salaried_formal', 'dsa', 'builder_flat', 28, 120000, 4000,
     2800000, 10, 3150000, 800, None, (118000, 3150000, 2800000, 10, 72, 0, 78, 0), (True, 1, False),
     'Young buyer putting down only 11%; asks if the rate can be lower to ease the EMI.',
     30, 'improve_s2d_conversion', 2),
    ('Jyoti Rawat', '9820012012', 'Jaipur', 'salaried_formal', 'branch_walkin', 'resale_flat', 39, 190000, 10000,
     3000000, 10, 4500000, 735, (14.8, 'PNB Housing Finance', True), (187000, 4400000, 3000000, 10, 80, 0, 70, 0), (True, 3, False),
     'PNB Housing letter at 14.80%; she is ready to walk away if we do not come close.',
     'deep', 'competitor_lower_rate', 4),
]


def policy_reference() -> dict:
    """The policy's reference block (13% floor, 17% ceiling); empty if the package is not importable."""
    try:
        from aavas.config import policy
        return policy().get("reference") or {}
    except Exception:  # noqa: BLE001 - fall back to the 13-17% band
        return {}


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
    ap.add_argument("--set", choices=("core", "extended", "all"), default="core",
                    help="which scenarios to create (default core: the original ten)")
    args = ap.parse_args()
    scenarios = {"core": SCENARIOS, "extended": EXTENDED_SCENARIOS, "all": SCENARIOS + EXTENDED_SCENARIOS}[args.set]
    rm = login(args.base, *args.rm.split(":", 1))
    am = login(args.base, *args.am.split(":", 1))
    made = []
    for n, (name, mobile, city, segment, channel, prop, age, income, emis, loan, years, value, score, comp,
            verified, interaction, notes, ask, reason, days) in enumerate(scenarios, 1):
        loan_id = f"APP{int(time.time() * 1000) % 10**13:013d}"
        time.sleep(0.01)
        sales = {"competitor_rate": comp[0], "competitor_name": comp[1], "competitor_evidence_verified": comp[2]} if comp else {}
        applicant = {"loan_id": loan_id, "name": name, "mobile": mobile, "email": "", "segment": segment, "city": city,
                     "channel": channel, "property_type": prop, "applicant_age": age, "declared_monthly_income": income,
                     "declared_obligations": emis, "requested_loan_amount": loan, "requested_tenure_years": years,
                     "estimated_property_value": value, "bureau_score": score, "new_to_credit": score < 0, "sales_context": sales}
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
        elif ask == "at_floor":
            rate = round(float(envelope["min_rate"]), 2)
        elif ask == "deep":
            rate = round(min(max(float(risk.get("breakeven_rate") or rec - 1) + 0.15, float(envelope["min_rate"])), rec - 0.30), 2)
        else:
            rate = round(rec - ask / 100, 2)
        ref = policy_reference()
        lo, hi = float(ref.get("card_rate_floor") or 13.0), float(ref.get("card_rate_ceiling") or 17.0)
        if not lo <= rate <= hi:
            raise SystemExit(f"scenario {name!r}: ask {rate:.2f}% is outside the policy band {lo:.2f}-{hi:.2f}%; fix the scenario")
        req = call(args.base, "POST", "/pricing/rate-requests", {"loan_id": loan_id, "rate": rate, "reason": reason}, rm)
        made.append((loan_id, req["request_id"], days))
        floor = float(envelope.get("min_rate") or 0)
        position = "below floor" if rate < floor - 0.004 else "at floor" if rate < floor + 0.004 else "within room"
        print(f"{n:2}. {name:20} {city:10} grade {risk.get('risk_grade')} card {envelope.get('max_rate')} floor {envelope.get('min_rate')} "
              f"rec {rec:.2f}% ask {rate:.2f}% ({position}) break-even {risk.get('breakeven_rate')}")
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
