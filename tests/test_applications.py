"""Online applications and borrower documents.

What must hold: a borrower reaches only the applications they submitted; only
the borrower, or the RM who entered an assisted application, uploads; a file is judged by its contents, not its name; and the
indicative range staff see is the one we recorded, not one the browser sent.
"""

from __future__ import annotations

import shutil

import pytest
import yaml
from fastapi.testclient import TestClient

from aavas.api import access
from aavas.api.main import app

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PASSWORDS = {"rahul": "Rahul@2026", "rohit": "Rohit@2026", "meera": "Meera@2026",
             "arjun": "Arjun@2026", "priya": "Priya@2026"}


@pytest.fixture
def client(monkeypatch, tmp_path, bundle) -> TestClient:
    """Session mode, with a second borrower to test ownership against."""
    copy = tmp_path / "access.yaml"
    shutil.copy(access.access_file(), copy)
    data = yaml.safe_load(copy.read_text())
    data["users"]["priya"] = {"name": "Priya Menon", "role": "borrower", "title": "Borrower",
                              "password": access.hash_password(PASSWORDS["priya"])}
    copy.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("AAVAS_ACCESS_FILE", str(copy))
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def sign_in(client: TestClient, username: str) -> dict[str, str]:
    token = client.post("/auth/login", json={"username": username,
                                             "password": PASSWORDS[username]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def applicant(loan_id: str, name: str = "Rahul Verma") -> dict:
    return {
        "loan_id": loan_id, "name": name, "mobile": "9811100000", "segment": "salaried_formal", "city": "Pune",
        "channel": "digital", "property_type": "resale_flat",
        "declared_monthly_income": 55000, "requested_loan_amount": 2000000,
        "requested_tenure_years": 20, "applicant_age": 35, "declared_obligations": 5000,
        "estimated_property_value": 2900000, "bureau_score": 790,
        "sales_context": {"competitor_rate": 11.5, "competitor_evidence_verified": False},
    }


def quote_and_apply(client, headers, loan_id, name="Rahul Verma", **extra) -> dict:
    a = {**applicant(loan_id, name), **extra}
    keys = ["loan_id", "segment", "city", "channel", "property_type", "declared_monthly_income",
            "requested_loan_amount", "requested_tenure_years", "applicant_age",
            "declared_obligations", "estimated_property_value", "bureau_score", "sales_context"]
    quoted = client.post("/pricing/pre-assessment", json={k: a[k] for k in keys}, headers=headers)
    assert quoted.status_code == 200, quoted.text
    response = client.post("/applications", json={"loan_id": loan_id, "applicant": a},
                           headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def upload(client, headers, loan_id, content=PDF, name="slips.pdf", kind="income_proof"):
    return client.post(f"/applications/{loan_id}/documents", headers=headers,
                       data={"document_type": kind}, files={"file": (name, content)})


# ------------------------------------------------------------------ applying

def test_a_borrower_cannot_apply_before_an_indicative_offer(client) -> None:
    response = client.post("/applications", headers=sign_in(client, "rahul"),
                           json={"loan_id": "APP-NO-QUOTE", "applicant": applicant("APP-NO-QUOTE")})
    assert response.status_code == 409


def test_staff_see_the_range_we_recorded_not_one_the_browser_sent(client) -> None:
    rahul = sign_in(client, "rahul")
    summary = quote_and_apply(client, rahul, "APP-RANGE",
                              indicative={"min_rate": 1.0, "max_rate": 1.5})
    assert summary["indicative"]["min_rate"] > 5
    assert summary["status"] == "awaiting_documents"

    listed = client.get("/applications", headers=sign_in(client, "rohit")).json()["applications"]
    row = next(r for r in listed if r["loan_id"] == "APP-RANGE")
    assert row["applicant_name"] == "Rahul Verma"
    assert row["indicative"] == summary["indicative"]


def test_a_borrower_lists_only_their_own_applications(client) -> None:
    quote_and_apply(client, sign_in(client, "rahul"), "APP-RAHUL-1")
    quote_and_apply(client, sign_in(client, "priya"), "APP-PRIYA-1", name="Priya Menon")
    mine = client.get("/applications", headers=sign_in(client, "priya")).json()["applications"]
    assert {r["loan_id"] for r in mine} == {"APP-PRIYA-1"}
    assert client.get("/applications/APP-RAHUL-1",
                      headers=sign_in(client, "priya")).status_code == 404


def test_a_borrower_cannot_take_over_another_borrowers_application(client) -> None:
    quote_and_apply(client, sign_in(client, "rahul"), "APP-TAKEOVER")
    response = client.post("/applications", headers=sign_in(client, "priya"),
                           json={"loan_id": "APP-TAKEOVER", "applicant": applicant("APP-TAKEOVER")})
    assert response.status_code == 403


# ----------------------------------------------------------------- documents

def test_the_borrower_uploads_and_the_rm_downloads_the_same_bytes(client) -> None:
    rahul = sign_in(client, "rahul")
    quote_and_apply(client, rahul, "APP-DOCS")
    stored = upload(client, rahul, "APP-DOCS", name="../../etc/passwd.pdf")
    assert stored.status_code == 200, stored.text
    row = stored.json()
    assert row["content_type"] == "application/pdf"
    assert "/" not in row["filename"] and ".." not in row["filename"].split(".")[0]

    rohit = sign_in(client, "rohit")
    detail = client.get("/applications/APP-DOCS", headers=rohit).json()
    assert detail["status"] == "documents_received"
    assert [d["document_type"] for d in detail["document_list"]] == ["income_proof"]

    got = client.get(f"/applications/APP-DOCS/documents/{row['document_id']}", headers=rohit)
    assert got.status_code == 200
    assert got.content == PDF
    assert got.headers["content-disposition"].startswith("attachment")
    assert got.headers["x-content-type-options"] == "nosniff"


def test_only_the_borrower_uploads(client) -> None:
    quote_and_apply(client, sign_in(client, "rahul"), "APP-WHO")
    assert upload(client, sign_in(client, "rohit"), "APP-WHO").status_code == 403
    assert upload(client, sign_in(client, "priya"), "APP-WHO").status_code == 404


def test_a_file_is_judged_by_its_contents_not_its_name(client) -> None:
    rahul = sign_in(client, "rahul")
    quote_and_apply(client, rahul, "APP-TYPES")
    assert upload(client, rahul, "APP-TYPES", content=b"MZ\x90\x00 not a pdf",
                  name="payslip.pdf").status_code == 415
    png = upload(client, rahul, "APP-TYPES", content=PNG, name="scan.pdf")
    assert png.status_code == 200 and png.json()["content_type"] == "image/png"


def test_unknown_types_empty_files_and_oversized_files_are_refused(client, monkeypatch) -> None:
    from aavas.intake import applications as intake

    rahul = sign_in(client, "rahul")
    quote_and_apply(client, rahul, "APP-LIMITS")
    assert upload(client, rahul, "APP-LIMITS", kind="selfie").status_code == 422
    assert upload(client, rahul, "APP-LIMITS", content=b"").status_code == 422
    real = intake.settings()
    monkeypatch.setattr(intake, "settings", lambda: {**real, "max_file_mb": 0.0001})
    assert upload(client, rahul, "APP-LIMITS", content=PDF + b"0" * 200).status_code == 413


def test_other_staff_roles_and_strangers_are_kept_out(client) -> None:
    quote_and_apply(client, sign_in(client, "rahul"), "APP-KEEP-OUT")
    assert client.get("/applications", headers=sign_in(client, "arjun")).status_code == 403
    assert client.get("/applications").status_code == 401


def test_an_rm_enters_an_assisted_application(client) -> None:
    rohit = sign_in(client, "rohit")
    summary = quote_and_apply(client, rohit, "APP-ASSISTED-1", name="Walk In Customer")
    assert summary["source"] == "assisted" and summary["created_by"] == "rohit"
    assert upload(client, rohit, "APP-ASSISTED-1").status_code == 200, "the RM uploads to the file they entered"
    listed = client.get("/applications", headers=sign_in(client, "rohit")).json()["applications"]
    assert any(r["loan_id"] == "APP-ASSISTED-1" and r["source"] == "assisted" for r in listed)
    mine = client.get("/applications", headers=sign_in(client, "rahul")).json()["applications"]
    assert all(r["loan_id"] != "APP-ASSISTED-1" for r in mine), "a borrower never sees another's file"


def test_the_source_cannot_be_claimed_by_the_request(client) -> None:
    rahul = sign_in(client, "rahul")
    summary = quote_and_apply(client, rahul, "APP-SOURCE-1", source="assisted", created_by="rohit")
    assert summary["source"] == "online" and summary["created_by"] == "rahul"


# ------------------------------------------------------------- staff edits

VERIFIED = {"verified_monthly_income": 52000, "appraised_property_value": 2900000,
            "sanctioned_loan_amount": 2000000, "tenure_years": 20}


def test_staff_edits_are_saved_and_who_entered_it_never_changes(client) -> None:
    rohit = sign_in(client, "rohit")
    quote_and_apply(client, rohit, "APP-EDIT-1", name="Walk In Customer")
    changed = client.put("/applications/APP-EDIT-1", headers=rohit, json={"applicant": {
        **applicant("APP-EDIT-1", "Walk-In Customer"), **VERIFIED,
        "source": "online", "created_by": "rahul"}})
    assert changed.status_code == 200, changed.text
    assert "verified_monthly_income" in changed.json()["changed"] and "name" in changed.json()["changed"]
    stored = client.get("/applications/APP-EDIT-1", headers=rohit).json()
    assert stored["applicant_name"] == "Walk-In Customer"
    assert stored["applicant"]["verified_monthly_income"] == 52000
    assert stored["source"] == "assisted" and stored["created_by"] == "rohit", "an edit cannot rewrite provenance"
    actions = [e["action"] for e in client.get("/pricing/audit/APP-EDIT-1",
                                                headers=sign_in(client, "meera")).json()["audit_entries"]]
    assert "updated_application" in actions


def test_a_borrower_cannot_edit_a_submitted_file_and_unknown_files_are_404(client) -> None:
    rahul = sign_in(client, "rahul")
    quote_and_apply(client, rahul, "APP-EDIT-2")
    assert client.put("/applications/APP-EDIT-2", headers=rahul,
                      json={"applicant": applicant("APP-EDIT-2")}).status_code == 403
    assert client.put("/applications/APP-NOPE", headers=sign_in(client, "rohit"),
                      json={"applicant": applicant("APP-NOPE")}).status_code == 404


def test_the_borrower_sees_the_offer_sent_to_them(client) -> None:
    rahul, rohit = sign_in(client, "rahul"), sign_in(client, "rohit")
    quote_and_apply(client, rahul, "APP-OFFER-1")
    assert client.get("/applications/APP-OFFER-1", headers=rahul).json()["offer"] is None
    priced = client.post("/pricing/final-assessment", headers=rohit,
                         json={k: v for k, v in {**applicant("APP-OFFER-1"), **VERIFIED}.items() if k != "name"})
    assert priced.status_code == 200, priced.text
    rate = priced.json()["recommendation"]["recommended_rate"]
    assert client.post("/pricing/offers", headers=rohit, json={"loan_id": "APP-OFFER-1", "rate": rate}).status_code == 200
    offer = client.get("/applications/APP-OFFER-1", headers=rahul).json()["offer"]
    assert offer["rate"] == rate and offer["status"] == "sent"
    assert "route" not in offer and "reason" not in offer, "how the rate was reached stays with staff"
    assert "offer" not in client.get("/applications/APP-OFFER-1", headers=rohit).json()
