"""Submitted applications and their documents.

A borrower submits an application once they have an indicative offer; the
indicative range shown to staff is read from the recorded pre-assessment, never
from what the browser sends. Each application belongs to the borrower who
submitted it, and only that borrower may add documents to it.

Files are stored under ``AAVAS_UPLOAD_DIR`` (default ``data/uploads``) by a
generated id, never by the name they arrived with, and their type is decided
from their first bytes rather than the name or the browser's claim.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from pathlib import Path
from typing import Any

from ..config import env, load
from ..governance import decision_logger, store
from ..ingestion.schemas import Stage
from ..paths import DATA_DIR, resolve

# The first bytes of each format we accept, and what we serve it back as.
_SIGNATURES = {
    "pdf": (b"%PDF-", "application/pdf"),
    "jpeg": (b"\xff\xd8\xff", "image/jpeg"),
    "png": (b"\x89PNG\r\n\x1a\n", "image/png"),
}
LOAN_ID = re.compile(r"^[A-Za-z0-9-]{3,40}$")


class IntakeError(Exception):
    """A request the caller can fix. ``status`` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def settings() -> dict[str, Any]:
    return load("intake")


def document_types() -> dict[str, str]:
    return dict(settings().get("document_types") or {})


def upload_dir() -> Path:
    configured = env("AAVAS_UPLOAD_DIR")
    return resolve(configured) if configured else DATA_DIR / "uploads"


# --------------------------------------------------------------- applications

def get(loan_id: str) -> dict[str, Any] | None:
    row = store.fetch_one("SELECT * FROM applications WHERE loan_id = ?", (loan_id,))
    if row:
        row["applicant"] = store.decode(row["applicant"])
    return row


def submit(loan_id: str, applicant: dict[str, Any], owner: str) -> dict[str, Any]:
    if not LOAN_ID.match(loan_id or ""):
        raise IntakeError(422, "the application id must be 3-40 letters, digits or dashes")
    existing = get(loan_id)
    if existing and existing["owner"] != owner:
        raise IntakeError(403, "this application belongs to someone else")
    if decision_logger.latest_for_loan(loan_id, stage=Stage.PRE_ASSESSMENT.value) is None:
        raise IntakeError(409, "get an indicative offer for this application before applying")
    name = str(applicant.get("name") or "").strip()[:120]
    if not name:
        raise IntakeError(422, "the applicant's name is required")
    now = store.now()
    store.insert("applications", {
        "loan_id": loan_id, "owner": owner, "applicant_name": name,
        "applicant": {**applicant, "loan_id": loan_id},
        "created_at": existing["created_at"] if existing else now, "updated_at": now})
    return summary(get(loan_id))


# What staff may change on a stored file: everything about the applicant and the
# loan, never who entered it or where it came from.
FIXED_FIELDS = ("loan_id", "source", "created_by")


def update(loan_id: str, changes: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Staff edit a stored application. Returns the new summary and the fields changed."""
    existing = get(loan_id)
    if existing is None:
        raise IntakeError(404, "no such application")
    before = existing["applicant"] or {}
    after = {**before, **{k: v for k, v in changes.items() if k not in FIXED_FIELDS}}
    name = str(after.get("name") or "").strip()[:120]
    if not name:
        raise IntakeError(422, "the applicant's name is required")
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    store.insert("applications", {
        "loan_id": loan_id, "owner": existing["owner"], "applicant_name": name,
        "applicant": {**after, "loan_id": loan_id},
        "created_at": existing["created_at"], "updated_at": store.now()})
    return summary(get(loan_id)), changed


def set_owner(loan_id: str, owner: str) -> None:
    """Hand a file to the customer's login (an RM issued it after entering the file)."""
    with store.connect() as conn:
        conn.execute("UPDATE applications SET owner = ? WHERE loan_id = ?", (owner, loan_id))


def stored(owner: str | None = None) -> list[dict[str, Any]]:
    """Every stored application, or only ``owner``'s, newest change first."""
    sql = "SELECT * FROM applications"
    params: tuple = ()
    if owner is not None:
        sql += " WHERE owner = ?"
        params = (owner,)
    rows = store.query(sql + " ORDER BY updated_at DESC", params)
    for row in rows:
        row["applicant"] = store.decode(row["applicant"])
    return rows


def visible_to(owner: str | None) -> list[dict[str, Any]]:
    """Every application, or only ``owner``'s when given."""
    return [summary(row) for row in stored(owner)]


def summary(row: dict[str, Any]) -> dict[str, Any]:
    """What a list needs: the applicant's own figures, and the range we quoted."""
    a = row["applicant"] or {}
    sales = a.get("sales_context") or {}
    pre = decision_logger.latest_for_loan(row["loan_id"], stage=Stage.PRE_ASSESSMENT.value)
    docs = documents(row["loan_id"])
    return {
        "loan_id": row["loan_id"], "owner": row["owner"], "applicant_name": row["applicant_name"],
        "city": a.get("city"), "segment": a.get("segment"),
        "loan_amount": a.get("requested_loan_amount"), "tenure_years": a.get("requested_tenure_years"),
        "property_value": a.get("estimated_property_value"),
        "competitor_rate": sales.get("competitor_rate"),
        "competitor_verified": bool(sales.get("competitor_evidence_verified")),
        "indicative": ({"min_rate": pre["min_rate"], "max_rate": pre["max_rate"],
                        "decision_id": pre["decision_id"]} if pre else None),
        "documents": len(docs), "document_types": sorted({d["document_type"] for d in docs}),
        "status": "documents_received" if docs else "awaiting_documents",
        "source": a.get("source", "online"), "created_by": a.get("created_by", row["owner"]),
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


# ------------------------------------------------------------------ documents

def documents(loan_id: str) -> list[dict[str, Any]]:
    return store.query("SELECT * FROM application_documents WHERE loan_id = ? "
                       "ORDER BY uploaded_at DESC", (loan_id,))


def document(loan_id: str, document_id: str) -> dict[str, Any] | None:
    return store.fetch_one("SELECT * FROM application_documents WHERE loan_id = ? "
                           "AND document_id = ?", (loan_id, document_id))


def detect_format(head: bytes) -> tuple[str, str] | None:
    allowed = settings().get("allowed_formats") or list(_SIGNATURES)
    for name in allowed:
        magic, content_type = _SIGNATURES.get(name, (None, None))
        if magic and head.startswith(magic):
            return name, content_type
    return None


def clean_filename(name: str | None, fmt: str) -> str:
    base = Path(name or "").name
    # ASCII only: the name goes back out in a Content-Disposition header.
    base = re.sub(r"[^\w.\- ()]+", "_", base, flags=re.ASCII).strip(" ._")[:100]
    return base or f"document.{'jpg' if fmt == 'jpeg' else fmt}"


def add_document(loan_id: str, document_type: str, filename: str | None, content: bytes,
                 uploaded_by: str) -> dict[str, Any]:
    application = get(loan_id)
    entered_by = ((application or {}).get("applicant") or {}).get("created_by")
    # The customer's own file, or one this RM entered for them.
    if application is None or uploaded_by not in (application["owner"], entered_by):
        # The same answer whether it does not exist or is someone else's.
        raise IntakeError(404, "no such application of yours")
    if document_type not in document_types():
        raise IntakeError(422, f"document type must be one of {sorted(document_types())}")
    limit = float(settings().get("max_file_mb", 10)) * 1024 * 1024
    if not content:
        raise IntakeError(422, "the file is empty")
    if len(content) > limit:
        raise IntakeError(413, f"files may be at most {settings().get('max_file_mb', 10)} MB")
    if len(documents(loan_id)) >= int(settings().get("max_documents_per_application", 20)):
        raise IntakeError(409, "this application already has the maximum number of documents")
    found = detect_format(content[:16])
    if found is None:
        raise IntakeError(415, f"upload a {', '.join(settings().get('allowed_formats') or [])} "
                               f"file; this one is not")
    fmt, content_type = found

    document_id = f"DOC-{secrets.token_hex(8).upper()}"
    folder = upload_dir() / loan_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / document_id).write_bytes(content)
    row = {
        "document_id": document_id, "loan_id": loan_id, "document_type": document_type,
        "filename": clean_filename(filename, fmt), "content_type": content_type,
        "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
        "uploaded_by": uploaded_by, "uploaded_at": store.now(),
    }
    store.insert("application_documents", row)
    store.insert("applications", {**application, "updated_at": row["uploaded_at"]})
    return row


def read_document(row: dict[str, Any]) -> bytes:
    return (upload_dir() / row["loan_id"] / row["document_id"]).read_bytes()
