"""Online applications and their documents.

    POST /applications                              a borrower submits their application
    GET  /applications                              staff: every application; borrower: their own
    GET  /applications/document-types               what a borrower may upload
    GET  /applications/{loan_id}                    one application and its documents
    POST /applications/{loan_id}/documents          a borrower uploads a document
    GET  /applications/{loan_id}/documents/{id}     download a document

Which roles reach these is set in config/access.yaml; which *application* a
borrower reaches is decided here: only the ones they submitted.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..governance import audit
from ..intake import applications as intake
from .security import Principal, current_principal

router = APIRouter(prefix="/applications", tags=["applications"])

BORROWER = "borrower"


class ApplicationSubmit(BaseModel):
    loan_id: str = Field(..., min_length=3, max_length=40)
    applicant: dict[str, Any]


def _is_borrower(principal: Principal) -> bool:
    return principal.has(BORROWER)


def _application_for(loan_id: str, principal: Principal) -> dict[str, Any]:
    row = intake.get(loan_id)
    if row is None or (_is_borrower(principal) and row["owner"] != principal.name):
        raise HTTPException(404, "no such application")
    return row


def _fail(exc: intake.IntakeError) -> HTTPException:
    return HTTPException(exc.status, str(exc))


@router.post("", summary="Submit an application after an indicative offer")
def submit(request: ApplicationSubmit,
           principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    try:
        summary = intake.submit(request.loan_id, request.applicant, owner=principal.name)
    except intake.IntakeError as exc:
        raise _fail(exc) from exc
    audit.record(principal.name, "submitted_application", "application", request.loan_id)
    return summary


@router.get("", summary="Submitted applications")
def index(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    owner = principal.name if _is_borrower(principal) else None
    return {"applications": intake.visible_to(owner)}


@router.get("/document-types", summary="Document types and upload limits")
def document_types() -> dict[str, Any]:
    settings = intake.settings()
    return {"document_types": intake.document_types(),
            "allowed_formats": settings.get("allowed_formats"),
            "max_file_mb": settings.get("max_file_mb")}


@router.get("/{loan_id}", summary="One application, with its documents")
def detail(loan_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    row = _application_for(loan_id, principal)
    return {**intake.summary(row), "applicant": row["applicant"],
            "document_list": intake.documents(loan_id)}


@router.post("/{loan_id}/documents", summary="Upload a document to your application")
async def upload(loan_id: str, document_type: str = Form(...), file: UploadFile = File(...),
                 principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    if not _is_borrower(principal):
        raise HTTPException(403, "only the borrower uploads documents to their application")
    # Read one byte past the limit: enough to know it is too big, never the whole file.
    limit = int(float(intake.settings().get("max_file_mb", 10)) * 1024 * 1024)
    content = await file.read(limit + 1)
    try:
        row = intake.add_document(loan_id, document_type, file.filename, content,
                                  uploaded_by=principal.name)
    except intake.IntakeError as exc:
        raise _fail(exc) from exc
    audit.record(principal.name, "uploaded_document", "application", loan_id,
                 {"document_id": row["document_id"], "document_type": document_type,
                  "sha256": row["sha256"], "size_bytes": row["size_bytes"]})
    return row


@router.get("/{loan_id}/documents/{document_id}", summary="Download a document")
def download(loan_id: str, document_id: str,
             principal: Principal = Depends(current_principal)) -> Response:
    _application_for(loan_id, principal)
    row = intake.document(loan_id, document_id)
    if row is None:
        raise HTTPException(404, "no such document")
    if not _is_borrower(principal):
        audit.record(principal.name, "viewed_document", "application", loan_id,
                     {"document_id": document_id})
    return Response(
        content=intake.read_document(row), media_type=row["content_type"],
        headers={"Content-Disposition": f'attachment; filename="{row["filename"]}"',
                 "X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"})
