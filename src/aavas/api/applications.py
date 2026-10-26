"""Online applications and their documents.

    POST /applications                              a borrower applies, or an RM enters an assisted one
    GET  /applications                              staff: every application; borrower: their own
    GET  /applications/document-types               what a borrower may upload
    GET  /applications/{loan_id}                    one application and its documents (a
                                                    borrower also sees the offer sent to them)
    PUT  /applications/{loan_id}                    staff change the details on a file
    POST /applications/{loan_id}/customer-login     RM issues the customer a login, or resets it
    POST /applications/{loan_id}/documents          the borrower, or the RM who entered it, uploads
    GET  /applications/{loan_id}/documents/{id}     download a document
    GET  /applications/in-progress                  staff: files short of verification, and sample leads
    POST /applications/nudges                       RM: remind a stalled customer (recorded, not delivered)
    POST /applications/{loan_id}/offer-decision     the borrower accepts or declines the offer sent

Which roles reach these is set in config/access.yaml; which *application* a
borrower reaches is decided here: only the ones they submitted. An RM
uploads only to an assisted application they entered.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..governance import audit
from ..intake import applications as intake
from ..intake import offers, worklist
from . import access
from .contracts import Stage2Request
from .deps import get_orchestrator
from .security import Principal, current_principal

router = APIRouter(prefix="/applications", tags=["applications"])

BORROWER = "borrower"


class ApplicationUpdate(BaseModel):
    applicant: dict[str, Any]


class ApplicationSubmit(BaseModel):
    loan_id: str = Field(..., min_length=3, max_length=40)
    applicant: dict[str, Any]


class NudgeIn(BaseModel):
    ref: str = Field(..., min_length=3, max_length=40)
    channel: str = Field("whatsapp", max_length=20)


class OfferDecisionIn(BaseModel):
    accepted: bool
    loss_reason: str | None = Field(None, max_length=64)


def _is_borrower(principal: Principal) -> bool:
    return principal.has(BORROWER)


def _application_for(loan_id: str, principal: Principal) -> dict[str, Any]:
    row = intake.get(loan_id)
    if row is None or (_is_borrower(principal) and row["owner"] != principal.name):
        raise HTTPException(404, "no such application")
    return row


def _fail(exc: intake.IntakeError) -> HTTPException:
    return HTTPException(exc.status, str(exc))


RM = "relationship_manager"


@router.post("", summary="Submit an application after an indicative offer")
def submit(request: ApplicationSubmit,
           principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    # Who entered it is recorded by the server, never taken from the request.
    applicant = {**request.applicant,
                 "source": "online" if _is_borrower(principal) else "assisted",
                 "created_by": principal.name}
    login = None
    if principal.has(RM) and not _is_borrower(principal):
        # The customer gets their own login, and the file is theirs to follow.
        login = _issue(applicant, request.loan_id, principal)
    try:
        summary = intake.submit(request.loan_id, applicant,
                                owner=login["username"] if login else principal.name)
    except intake.IntakeError as exc:
        raise _fail(exc) from exc
    audit.record(principal.name, "submitted_application", "application", request.loan_id)
    return {**summary, "customer_login": login} if login else summary


def _issue(applicant: dict[str, Any], loan_id: str, principal: Principal) -> dict[str, Any]:
    try:
        user, password, created = access.issue_customer_login(
            applicant.get("name"), applicant.get("mobile"), applicant.get("email"),
            issued_by=principal.name, loan_id=loan_id)
    except access.RegistrationRejected as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    if created:
        # The password itself is never recorded, only that one was issued.
        audit.record(principal.name, "issued_customer_login", "user", user.username, {"loan_id": loan_id})
    return {"username": user.username, "temporary_password": password, "created": created}


@router.post("/{loan_id}/customer-login", summary="Issue the customer a login, or reset its password")
def customer_login(loan_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    if not principal.has(RM) or _is_borrower(principal):
        raise HTTPException(403, "the relationship manager issues customer logins")
    row = intake.get(loan_id)
    if row is None:
        raise HTTPException(404, "no such application")
    owner = row["owner"]
    if access.is_borrower_account(owner):
        try:
            password = access.reset_customer_password(owner, issued_by=principal.name, loan_id=loan_id)
        except access.RegistrationRejected as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        audit.record(principal.name, "reset_customer_password", "user", owner, {"loan_id": loan_id})
        return {"username": owner, "temporary_password": password, "created": False, "reset": True}
    # A file entered before logins were issued: create one now and hand the file over.
    login = _issue(row["applicant"] or {}, loan_id, principal)
    intake.set_owner(loan_id, login["username"])
    if login["temporary_password"] is None:
        # The customer already had an account (registered online): the file is
        # linked to it and their password is left alone.
        return {**login, "linked": True}
    return login


def _with_login(summary: dict[str, Any]) -> dict[str, Any]:
    owner = summary.get("owner")
    return {**summary, "customer_login": owner if access.is_borrower_account(owner) else None}


@router.get("", summary="Submitted applications")
def index(principal: Principal = Depends(current_principal),
          orchestrator=Depends(get_orchestrator)) -> dict[str, Any]:
    if _is_borrower(principal):
        return {"applications": intake.visible_to(principal.name)}
    # Staff see where each file stands; a verified file carries the suggested offer.
    stored = intake.stored()
    rows = worklist.annotate([intake.summary(r) for r in stored], {r["loan_id"]: r for r in stored},
                             price=_pricer(orchestrator))
    return {"applications": [_with_login(r) for r in rows]}


def _pricer(orchestrator):
    return lambda row: orchestrator.process(Stage2Request(**row).to_row())


@router.get("/in-progress", summary="Files short of verification, and leads that stalled")
def in_progress(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    if _is_borrower(principal):
        raise HTTPException(403, "staff only")
    stored = intake.stored()
    files = [_with_login(r) for r in worklist.annotate([intake.summary(r) for r in stored], {})
             if r["stage"] == "in_progress"]
    settings = intake.settings()
    me = access.user(principal.name)
    first = (me.name if me else principal.name).split()[0]
    items = [{**i, "preview": worklist.nudge_message(i, first)} for i in worklist.in_progress(files)]
    return {"items": items,
            "required_documents": worklist.required_documents(),
            "required_text": worklist.missing_text(worklist.required_documents()),
            "document_types": intake.document_types(),
            "channels": settings.get("nudge_channels") or ["sms"],
            "note": "Nudges are recorded against the file. Delivering them needs an SMS or "
                    "WhatsApp gateway, which this service does not have."}


@router.post("/nudges", summary="Remind a customer whose file has stalled")
def nudge(body: NudgeIn, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    if not principal.has(RM) or _is_borrower(principal):
        raise HTTPException(403, "the relationship manager nudges customers")
    stored = intake.stored()
    files = [_with_login(r) for r in worklist.annotate([intake.summary(r) for r in stored], {})
             if r["stage"] == "in_progress"]
    row = next((r for r in worklist.in_progress(files) if r["ref"] == body.ref), None)
    if row is None:
        raise HTTPException(404, "nothing in progress with that reference")
    me = access.user(principal.name)
    try:
        sent = worklist.record_nudge(row, body.channel, principal.name,
                                     rm_name=(me.name if me else principal.name).split()[0])
    except intake.IntakeError as exc:
        raise _fail(exc) from exc
    audit.record(principal.name, "nudged_customer", "application", body.ref,
                 {"channel": body.channel, "nudge_id": sent["nudge_id"]})
    return sent


@router.get("/document-types", summary="Document types and upload limits")
def document_types() -> dict[str, Any]:
    settings = intake.settings()
    return {"document_types": intake.document_types(),
            "allowed_formats": settings.get("allowed_formats"),
            "max_file_mb": settings.get("max_file_mb")}


@router.get("/{loan_id}", summary="One application, with its documents")
def detail(loan_id: str, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    row = _application_for(loan_id, principal)
    out = {**intake.summary(row), "applicant": row["applicant"],
           "document_list": intake.documents(loan_id)}
    if _is_borrower(principal):
        out["offer"] = _offer_for_borrower(loan_id)
    else:
        out = _with_login(worklist.annotate([out], {})[0])
    return out


@router.post("/{loan_id}/offer-decision", summary="Accept or decline the offer you were sent")
def offer_decision(loan_id: str, body: OfferDecisionIn, principal: Principal = Depends(current_principal),
                   orchestrator=Depends(get_orchestrator)) -> dict[str, Any]:
    """The customer answers on their side of the portal; the RM's list updates on its own."""
    if not _is_borrower(principal):
        raise HTTPException(403, "the customer answers their own offer; staff record it on the file")
    _application_for(loan_id, principal)
    waiting = [o for o in offers.offers(loan_id=loan_id) if o.get("status") == "sent"]
    if not waiting:
        raise HTTPException(409, "there is no offer waiting for your answer")
    try:
        offers.customer_decision(waiting[0]["offer_id"], accepted=body.accepted, final_rate=None,
                                 loss_reason=None if body.accepted else (body.loss_reason or "no_reason_recorded"),
                                 lost_to=None, lost_to_rate=None, actor=principal.name,
                                 capture_outcome=orchestrator.capture_outcome)
    except offers.OfferError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return {"offer": _offer_for_borrower(loan_id)}


def _offer_for_borrower(loan_id: str) -> dict[str, Any] | None:
    """The latest offer sent on this file, in the terms a customer needs - not how it
    was reached (route, reason, approvals stay with staff)."""
    from ..intake import offers

    sent = [o for o in offers.offers(loan_id=loan_id) if o.get("status") in ("sent", "accepted", "not_accepted")]
    if not sent:
        return None
    latest = sent[0]
    return {"rate": latest["rate"], "sent_at": latest["sent_at"], "status": latest["status"],
            "decided_at": latest.get("customer_decided_at")}


@router.put("/{loan_id}", summary="Change the details on a file")
def update(loan_id: str, request: ApplicationUpdate,
           principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    if _is_borrower(principal):
        raise HTTPException(403, "a submitted application is changed by the bank's staff")
    try:
        summary, changed = intake.update(loan_id, request.applicant)
    except intake.IntakeError as exc:
        raise _fail(exc) from exc
    if changed:
        audit.record(principal.name, "updated_application", "application", loan_id, {"fields": changed})
    return {**summary, "changed": changed}


@router.post("/{loan_id}/documents", summary="Upload a document to your application")
async def upload(loan_id: str, document_type: str = Form(...), file: UploadFile = File(...),
                 principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    if not (_is_borrower(principal) or principal.has(RM)):
        raise HTTPException(403, "only the borrower uploads documents to their application")
    existing = intake.get(loan_id)
    if principal.has(RM) and existing is not None \
            and (existing["applicant"] or {}).get("created_by") != principal.name:
        # An RM uploads only to an application they entered (the customer's login
        # owns it now); a borrower's own online application is theirs to upload to.
        raise HTTPException(403, "only the borrower uploads documents to their online application")
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
