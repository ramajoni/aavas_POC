"""Five demo cases for the relationship manager and the area manager.

Everything goes through the running API as Rohit, the same calls the lending workspace
makes, so every access rule, authority limit and audit entry applies:

  APP-PW-2601, APP-PW-2602   documents in, waiting for Rohit to price and send an offer
  APP-PW-2603 .. 2605        priced by Rohit and sent to Meera for a rate below his authority

Each assisted application issues the customer a login, as the workspace does; the
temporary passwords are printed once, at the end.

  python scripts/seed_demo_cases.py                       # API at http://127.0.0.1:8000
  python scripts/seed_demo_cases.py --api http://127.0.0.1:8001

Re-running skips cases that already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
DOCUMENTS = [("income_proof", "salary_or_itr.pdf", PDF), ("bank_statement", "bank_statement_6m.pdf", PDF),
             ("identity_address", "pan_aadhaar.png", PNG), ("property_papers", "sale_agreement.pdf", PDF)]

# Declared at application, then the verified assessment the RM's file carries.
BASE = {"city": "Pune", "channel": "branch_walkin", "property_type": "resale_flat", "requested_tenure_years": 20,
        "tenure_years": 20, "delinquency_count_24m": 0, "deviation_count": 0, "legal_flag": 0, "technical_flag": 0}

CASES = [
    {"loan_id": "APP-PW-2601", "name": "Ananya Kulkarni", "mobile": "9822000101", "email": "ananya.k@example.com",
     "stage": "rm",
     "story": "Strong salaried file, no competitor in play. Price it and send the recommended rate directly.",
     "file": {"segment": "salaried_formal", "property_type": "builder_flat", "applicant_age": 34,
              "declared_monthly_income": 120000, "verified_monthly_income": 118000, "declared_obligations": 8000,
              "requested_loan_amount": 6500000, "sanctioned_loan_amount": 6500000,
              "estimated_property_value": 9500000, "appraised_property_value": 9400000,
              "bureau_score": 812, "bureau_vintage_months": 120, "banking_behaviour_score": 88,
              "sales_context": {}}},
    {"loan_id": "APP-PW-2602", "name": "Sameer Joshi", "mobile": "9822000102", "email": "sameer.j@example.com",
     "stage": "rm",
     "story": "Salary-account customer asking for a little off; the 10.95% HDFC rate he mentions is unverified. "
              "Up to 0.15% below the recommendation, with a reason, is within the RM's own authority.",
     "file": {"segment": "salaried_formal", "applicant_age": 41,
              "declared_monthly_income": 85000, "verified_monthly_income": 82000, "declared_obligations": 12000,
              "requested_loan_amount": 3800000, "sanctioned_loan_amount": 3800000,
              "estimated_property_value": 5600000, "appraised_property_value": 5500000,
              "bureau_score": 768, "bureau_vintage_months": 84, "banking_behaviour_score": 80,
              "sales_context": {"existing_relationship": True, "customer_negotiating": True,
                                "competitor_name": "HDFC Bank", "competitor_rate": 10.95}}},
    {"loan_id": "APP-PW-2603", "name": "Farida Shaikh", "mobile": "9822000103", "email": "farida.s@example.com",
     "stage": "am", "ask_bps": 25, "reason": "competitor_lower_rate",
     "note": "Doctor with her own clinic. ICICI Home Finance sanction letter at 10.20% seen and uploaded; "
             "she will move this week if we do not beat it.",
     "story": "Verified competitor offer below our rate; asks 0.25% below the recommendation, within Meera's "
              "0.40% limit. Approve, or match the competitor for less.",
     "file": {"segment": "self_employed_professional", "applicant_age": 39,
              "declared_monthly_income": 160000, "verified_monthly_income": 150000, "declared_obligations": 20000,
              "requested_loan_amount": 7200000, "sanctioned_loan_amount": 7000000,
              "estimated_property_value": 10500000, "appraised_property_value": 10200000,
              "bureau_score": 786, "bureau_vintage_months": 96, "banking_behaviour_score": 82,
              "sales_context": {"customer_negotiating": True, "prior_negotiation_rounds": 2,
                                "competitor_name": "ICICI Home Finance", "competitor_rate": 10.20,
                                "competitor_evidence_verified": True}}},
    {"loan_id": "APP-PW-2604", "name": "Vikram Patil", "mobile": "9822000104", "email": "vikram.p@example.com",
     "stage": "am", "ask_bps": 30, "reason": "strong_negotiation",
     "note": "Hardware trader. Says Bajaj Housing quoted lower but has shown nothing in writing. Pushing hard.",
     "story": "No evidence of a lower offer; asks 0.30% below, down to the policy floor. Within Meera's limit, "
              "but is the discount buying anything? Decline, or approve a smaller cut.",
     "file": {"segment": "self_employed_non_professional", "channel": "connector", "applicant_age": 46,
              "declared_monthly_income": 95000, "verified_monthly_income": 72000, "declared_obligations": 15000,
              "requested_loan_amount": 3500000, "sanctioned_loan_amount": 3300000,
              "estimated_property_value": 4800000, "appraised_property_value": 4600000,
              "bureau_score": 724, "bureau_vintage_months": 60, "banking_behaviour_score": 62, "deviation_count": 1,
              "sales_context": {"customer_negotiating": True, "prior_negotiation_rounds": 3}}},
    {"loan_id": "APP-PW-2605", "name": "Neha Deshmukh", "mobile": "9822000105", "email": "neha.d@example.com",
     "stage": "am", "ask_bps": 55, "reason": "competitor_lower_rate",
     "note": "Threatening a balance transfer; SBI offer letter at 9.75% seen. Wants a clear gap below it to stay.",
     "story": "Balance-transfer threat backed by a verified SBI letter, 0.55% below: beyond Meera's 0.40% "
              "limit. She can decline, approve no lower than her limit, or it goes to the credit head.",
     "file": {"segment": "salaried_formal", "property_type": "self_construction", "applicant_age": 37,
              "declared_monthly_income": 105000, "verified_monthly_income": 104000, "declared_obligations": 6000,
              "requested_loan_amount": 5200000, "sanctioned_loan_amount": 5200000,
              "estimated_property_value": 8000000, "appraised_property_value": 7900000,
              "bureau_score": 801, "bureau_vintage_months": 132, "banking_behaviour_score": 90,
              "sales_context": {"customer_negotiating": True, "existing_relationship": True,
                                "competitor_name": "SBI Home Loans", "competitor_rate": 9.75,
                                "competitor_evidence_verified": True}}},
]
DECLARED = ["loan_id", "segment", "city", "channel", "property_type", "declared_monthly_income",
            "requested_loan_amount", "requested_tenure_years", "applicant_age", "declared_obligations",
            "estimated_property_value", "bureau_score", "sales_context"]


class Api:
    def __init__(self, base: str):
        self.base, self.token = base.rstrip("/"), None

    def call(self, method: str, path: str, body=None, raw: bytes | None = None, ctype: str | None = None):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        data = raw
        if body is not None:
            data, ctype = json.dumps(body).encode(), "application/json"
        if ctype:
            headers["Content-Type"] = ctype
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            try:
                return e.code, json.loads(text)
            except ValueError:
                return e.code, text

    def ok(self, method, path, body=None, **kw):
        status, data = self.call(method, path, body, **kw)
        if status != 200:
            raise SystemExit(f"{method} {path} -> {status}: {data}")
        return data

    def login(self, username: str, password: str) -> None:
        self.token = self.ok("POST", "/auth/login", {"username": username, "password": password})["token"]

    def upload(self, loan_id: str, kind: str, filename: str, content: bytes):
        boundary = uuid.uuid4().hex
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="document_type"\r\n\r\n{kind}\r\n'.encode(),
                 (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                  f"Content-Type: application/octet-stream\r\n\r\n").encode() + content + b"\r\n",
                 f"--{boundary}--\r\n".encode()]
        return self.ok("POST", f"/applications/{loan_id}/documents", raw=b"".join(parts),
                       ctype=f"multipart/form-data; boundary={boundary}")


def seed(api: Api, case: dict) -> dict:
    loan_id = case["loan_id"]
    if api.call("GET", f"/applications/{loan_id}")[0] == 200:
        return {"loan_id": loan_id, "skipped": "already exists"}
    file = {**BASE, **case["file"], "loan_id": loan_id}
    file["sales_context"] = {"competitor_evidence_verified": False, **file["sales_context"]}
    api.ok("POST", "/pricing/pre-assessment", {k: file[k] for k in DECLARED if k in file})
    applicant = {**file, "name": case["name"], "mobile": case["mobile"], "email": case["email"]}
    made = api.ok("POST", "/applications", {"loan_id": loan_id, "applicant": applicant})
    for kind, filename, content in DOCUMENTS:
        api.upload(loan_id, kind, filename, content)
    out = {"loan_id": loan_id, "name": case["name"], "login": made.get("customer_login")}
    if case["stage"] == "am":
        priced = api.ok("POST", "/pricing/final-assessment", file)
        rec = priced["recommendation"]["recommended_rate"]
        ask = round(rec - case["ask_bps"] / 100, 2)
        request = api.ok("POST", "/pricing/rate-requests",
                         {"loan_id": loan_id, "rate": ask, "reason": case["reason"], "note": case["note"]})
        out.update(recommended=rec, asked=ask, request_id=request.get("request_id"), status=request.get("status"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--rm", default="rohit")
    args = parser.parse_args()
    api = Api(args.api)
    # The README's demo password unless overridden.
    api.login(args.rm, os.environ.get("AAVAS_DEMO_RM_PASSWORD", "Rohit@2026"))
    results = [seed(api, c) for c in CASES]
    for case, r in zip(CASES, results):
        if r.get("skipped"):
            print(f"{r['loan_id']}  skipped: {r['skipped']}")
            continue
        where = (f"with area manager: recommended {r['recommended']:.2f}%, asked {r['asked']:.2f}% "
                 f"({case['ask_bps']} bps below), {case['reason']}" if case["stage"] == "am"
                 else "with RM: documents in, ready to price")
        print(f"{r['loan_id']}  {r['name']:<16} {where}")
        print(f"             {case['story']}")
    logins = [r for r in results if r.get("login")]
    if logins:
        print("\nCustomer logins issued (temporary passwords, shown once):")
        for r in logins:
            print(f"  {r['loan_id']}  {r['login']['username']}  {r['login'].get('temporary_password') or '(existing login)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
