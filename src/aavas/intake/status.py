"""Where an application stands, step by step, for the RM and the area manager.

One timeline for a file, read from the store and never from a browser: the
application, the indicative range quoted on the customer's own figures, the
documents, the risk-adjusted rate on the verified file, the customer interaction
and acceptance, the area manager (when a lower rate was asked), the offer and the
customer's answer. Each step is ``done``, ``current`` (the one the file is waiting
on), ``todo``, ``skipped`` (not needed for this file) or ``warn`` (done, but not
the way it was asked: a manager's decline).

``label`` turns the same facts into the one-line status a list shows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..governance import decision_logger
from ..ingestion.schemas import Stage
from . import applications, offers, readiness, worklist


def _pct(rate: Any) -> str:
    return f"{float(rate):.2f}%" if rate is not None else "—"


def _acceptance(payload: dict[str, Any]) -> float | None:
    sens = payload.get("price_sensitivity") or {}
    value = sens.get("conversion_at_recommended")
    if value is None:
        value = ((payload.get("recommendation") or {}).get("economics") or {}).get("conversion_probability")
    return float(value) if value is not None else None


def facts(loan_id: str) -> dict[str, Any]:
    """Everything the timeline and the label are read from, for one file."""
    stored = applications.get(loan_id)
    docs = worklist.verification(applications.documents(loan_id)) if stored else None
    request_rows = offers.requests(loan_id=loan_id)
    offer_rows = [o for o in offers.offers(loan_id=loan_id) if o.get("status") != "superseded"]
    pre = decision_logger.latest_for_loan(loan_id, stage=Stage.PRE_ASSESSMENT.value)
    post = decision_logger.latest_for_loan(loan_id, stage=Stage.POST_ASSESSMENT.value)
    steps = readiness.check((stored or {}).get("applicant")) if stored else None
    return {"stored": stored, "docs": docs, "requests": request_rows, "offers": offer_rows,
            "pre": pre, "post": post, "steps": steps,
            "stage": worklist.stage(loan_id, bool(docs and docs["verified"]), offer_rows, request_rows)}


def label(f: dict[str, Any]) -> dict[str, str]:
    """The one-line status, and a tone for its badge."""
    stage, steps, docs = f["stage"], f["steps"], f["docs"]
    done = (steps or {}).get("done") or {}
    if stage == "accepted":
        return {"label": "Accepted by the customer", "tone": "green"}
    if stage == "not_accepted":
        return {"label": "Not taken by the customer", "tone": "red"}
    if stage == "offer_sent":
        return {"label": "Offer sent · awaiting the customer", "tone": "blue"}
    if stage == "pending_rate_approval":
        return {"label": "With the area manager", "tone": "amber"}
    if steps is not None and not done.get("indicative"):
        return {"label": "Customer details to fill", "tone": "amber"}
    if docs is not None and not docs["verified"]:
        return {"label": f"Documents due ({len(docs['missing'])})", "tone": "amber"}
    if steps is not None and not done.get("risk"):
        return {"label": "Loan risk to compute", "tone": "amber"}
    if steps is not None and not done.get("behaviour"):
        return {"label": "Customer interaction to record", "tone": "amber"}
    if any(r.get("status") == "declined" for r in f["requests"][:1]):
        return {"label": "Manager declined · send at the recommendation", "tone": "red"}
    return {"label": "Ready for the final offer", "tone": "green"}


def _ago(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        days = (datetime.now(timezone.utc) - datetime.fromisoformat(str(iso).replace("Z", "+00:00"))).days
    except ValueError:
        return ""
    return "today" if days <= 0 else "yesterday" if days == 1 else f"{days} days ago"


def _due(steps: dict[str, Any] | None, key: str) -> tuple[list[str], list[str]]:
    step = next((s for s in (steps or {}).get("steps", []) if s["key"] == key), None)
    return ([m["label"] for m in step["missing"]], step["problems"]) if step else ([], [])


def _listed(items: list[str], most: int = 3) -> str:
    shown = ", ".join(items[:most]).lower()
    return shown + (f" and {len(items) - most} more" if len(items) > most else "")


def next_action(f: dict[str, Any]) -> dict[str, Any]:
    """What happens next on the file, and whose move it is: the RM, the area manager
    or the customer. ``kind`` lets a screen offer the matching button."""
    stage, steps, docs = f["stage"], f["steps"], f["docs"]
    done = (steps or {}).get("done") or {}
    request = f["requests"][0] if f["requests"] else None
    offer = f["offers"][0] if f["offers"] else None
    rec = (((f["post"] or {}).get("payload") or {}).get("recommendation") or {}).get("recommended_rate")

    def act(kind, owner, text, **extra):
        return {"kind": kind, "owner": owner, "text": text, **extra}

    if stage == "accepted":
        return act("disburse", "RM", f"Move the file to sanction and disbursement: the customer accepted "
                                     f"{_pct(offer and offer.get('rate'))}.")
    if stage == "not_accepted":
        return act("closed", "RM", "Closed: the customer did not take the offer. Nothing more to do unless they come back.")
    if stage == "offer_sent":
        return act("await_customer", "Customer", f"Accept or decline the offer at {_pct(offer['rate'])}, sent "
                                                 f"{_ago(offer.get('sent_at'))}. The RM follows up and records the answer.")
    if stage == "pending_rate_approval" and request:
        return act("manager_review", "Area manager",
                   f"Approve or decline {_pct(request['requested_rate'])} against the {_pct(request['recommended_rate'])} "
                   f"recommendation, asked {_ago(request.get('requested_at'))}.", request_id=request["request_id"])
    if steps is not None and not done.get("indicative"):
        missing, problems = _due(steps, "indicative")
        return act("fill_details", "RM", f"Correct the figures: {problems[0]}" if problems
                   else f"Fill in the customer's details: {_listed(missing)}.")
    if docs is not None and not docs["verified"]:
        return act("documents", "Customer", f"Upload the {worklist.missing_text(docs['missing'])}. "
                                            "The RM can nudge the customer from My Applications.")
    if steps is not None and not done.get("risk"):
        missing, problems = _due(steps, "risk")
        return act("verify", "RM", f"Clear this first: {problems[0]}" if problems
                   else f"Enter the verified figures to compute the loan risk: {_listed(missing)}.")
    if steps is not None and not done.get("behaviour"):
        missing, _ = _due(steps, "behaviour")
        return act("behaviour", "RM", f"Record the customer interaction and negotiation to see the acceptance rate: "
                                      f"{_listed(missing)}.")
    if request and request.get("status") == "declined":
        return act("resend", "RM", f"The area manager declined {_pct(request['requested_rate'])}: send the offer at "
                                   f"{_pct(rec or request['recommended_rate'])} or above.")
    return act("send_offer", "RM", f"Send the final offer at {_pct(rec)}, or ask the area manager for a lower rate."
               if rec is not None else "Open the file to price it and send the final offer.")


def _days(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        return max((datetime.now(timezone.utc) - datetime.fromisoformat(str(iso).replace("Z", "+00:00"))).days, 0)
    except ValueError:
        return None


def aging(f: dict[str, Any]) -> dict[str, Any]:
    """How long the file has been open, and how long it has waited at its current step."""
    stored, docs, steps, post = f["stored"], f["docs"], f["steps"], f["post"]
    request = f["requests"][0] if f["requests"] else None
    offer = f["offers"][0] if f["offers"] else None
    done = (steps or {}).get("done") or {}
    opened = (stored or {}).get("created_at") or (f["pre"] or {}).get("created_at")
    stage = f["stage"]
    if stage in ("accepted", "not_accepted"):
        since, step = offer and offer.get("customer_decided_at"), "since the customer answered"
    elif stage == "offer_sent":
        since, step = offer and offer.get("sent_at"), "with the customer"
    elif stage == "pending_rate_approval":
        since, step = request and request.get("requested_at"), "with the area manager"
    elif docs is not None and not docs["verified"]:
        since, step = opened, "waiting for documents"
    elif request and request.get("status") == "declined":
        since, step = request.get("decided_at"), "since the manager declined"
    elif steps is not None and not done.get("behaviour"):
        since, step = (docs or {}).get("verified_at") or opened, "waiting for the RM's figures"
    else:
        since, step = (post or {}).get("created_at") or (docs or {}).get("verified_at"), "ready to send"
    return {"days_open": _days(opened), "days_in_step": _days(since), "step": step, "opened_at": opened}


def upsell(f: dict[str, Any]) -> dict[str, Any] | None:
    """The file from the upsell side: what holding the rate is worth over the RM's ask,
    the room above the rate to hold, or - before pricing - the top of the quoted range."""
    request = f["requests"][0] if f["requests"] else None
    offer = f["offers"][0] if f["offers"] else None
    payload = (f["post"] or {}).get("payload") or {}
    rec = (payload.get("recommendation") or {}).get("recommended_rate")
    band = ((f["pre"] or {}).get("payload") or {}).get("band") or {}
    if offer and rec is not None:
        bps = round((float(offer["rate"]) - float(rec)) * 100)
        return {"kind": "sent", "bps": bps, "rate": offer["rate"],
                "text": f"Sent at {_pct(offer['rate'])}: {'+' if bps > 0 else ''}{bps} bps against the rate to hold"
                        if bps else f"Sent at the rate to hold, {_pct(rec)}"}
    if request and request.get("status") == "pending":
        bps = round((float(request["recommended_rate"]) - float(request["requested_rate"])) * 100)
        return {"kind": "hold", "bps": bps, "rate": request["recommended_rate"],
                "text": f"Holding {_pct(request['recommended_rate'])} over the ask of {_pct(request['requested_rate'])}: +{bps} bps"}
    if rec is not None:
        room = offers.headroom(payload, f.get("loan_id"))
        if room["rate"] is not None and room["rate"] > float(rec) + 0.005:
            bps = round((room["rate"] - float(rec)) * 100)
            return {"kind": "room", "bps": bps, "rate": room["rate"],
                    "text": f"Room up to {_pct(room['rate'])}: +{bps} bps over {_pct(rec)}, customers like this still accept"}
        why = {"above_quote": "it is already above the quoted range", "quoted_range": "it is the top of the quoted range",
               "policy": "it is the policy ceiling"}.get(room["capped_by"], "fewer than 70% of customers like this accept even at it")
        return {"kind": "none", "bps": 0, "rate": rec, "text": f"No room above the rate to hold: {why}"}
    if band.get("max_rate") is not None:
        return {"kind": "quote", "bps": None, "rate": band["max_rate"],
                "text": f"Quoted {_pct(band.get('min_rate'))}–{_pct(band['max_rate'])}: aim for the top, {_pct(band['max_rate'])}"}
    return None


def timeline(loan_id: str, name=lambda username: username) -> dict[str, Any]:
    """``name`` turns a username into the person's name for the timeline."""
    f = facts(loan_id)
    f["loan_id"] = loan_id
    stored, docs, steps, pre, post = f["stored"], f["docs"], f["steps"], f["pre"], f["post"]
    done = (steps or {}).get("done") or {}
    # Files that reached the manager or the customer before every step was required.
    past = bool(f["requests"] or f["offers"])
    out: list[dict[str, Any]] = []

    def step(key, title, state, detail=None, at=None):
        out.append({"key": key, "label": title, "state": state, "detail": detail, "at": at})

    a = (stored or {}).get("applicant") or {}
    if stored:
        source = "Entered by " + name(a["created_by"]) if a.get("source") == "assisted" and a.get("created_by") else "Applied online"
        step("application", "Application", "done", source, stored.get("created_at"))
    else:
        step("application", "Application", "skipped", "Priced without a submitted application")

    band = ((pre or {}).get("payload") or {}).get("band") or {}
    step("indicative", "Indicative pricing", "done" if pre else "current" if stored else "skipped",
         f"{_pct(band.get('min_rate'))}–{_pct(band.get('max_rate'))} on the customer's own figures" if pre
         else "Not quoted yet", (pre or {}).get("created_at"))

    if docs is None:
        step("documents", "Documents", "skipped", "No uploaded documents")
    else:
        need = worklist.required_documents()
        step("documents", "Documents", "done" if docs["verified"] else "warn" if past else "current",
             f"All {len(need)} in" if docs["verified"] else
             f"{worklist.missing_text(docs['missing'])} still due", docs.get("verified_at"))

    risk_ready = (done.get("risk") or past) if steps is not None else bool(post)
    early = steps is not None and past and not done.get("risk")
    p = (post or {}).get("payload") or {}
    rec = (p.get("recommendation") or {}).get("recommended_rate")
    grade = (p.get("risk") or {}).get("risk_grade")
    if risk_ready and post:
        moved = ""
        if band.get("max_rate") is not None and rec is not None and float(rec) > float(band["max_rate"]) + 1e-9:
            moved = f", {float(rec) - float(band['max_rate']):.2f}% above the indicative range"
        step("risk", "Risk-adjusted pricing", "done", f"{_pct(rec)} · risk grade {grade or '—'}{moved}"
             + (" · priced before every verified figure was in" if early else ""), post.get("created_at"))
    else:
        risk_step = next((s for s in (steps or {}).get("steps", []) if s["key"] == "risk"), None)
        due = len(risk_step["missing"]) + len(risk_step["problems"]) if risk_step else 0
        step("risk", "Risk-adjusted pricing", "current" if out[-1]["state"] in ("done", "skipped") else "todo",
             f"{due} verified figure{'s' if due != 1 else ''} to fill" if due else "Not priced yet")

    behaviour_ready = (done.get("behaviour") or past) if steps is not None else bool(post)
    acceptance = _acceptance(p) if post else None
    step("behaviour", "Customer acceptance",
         "done" if behaviour_ready and post else "current" if risk_ready else "todo",
         (f"{round(acceptance * 100)}% accept at {_pct(rec)}" if acceptance is not None else "Scored")
         + (" · interaction not recorded" if steps is not None and past and not done.get("behaviour") else "")
         if behaviour_ready and post else "Customer interaction to record")

    latest_request = f["requests"][0] if f["requests"] else None
    live_offer = f["offers"][0] if f["offers"] else None
    if latest_request is None:
        step("manager", "Area manager", "skipped",
             "Not needed: sent within the RM's authority" if live_offer else "Only if a lower rate is asked")
    elif latest_request["status"] == "pending":
        step("manager", "Area manager", "current", f"Asked {_pct(latest_request['requested_rate'])}, waiting",
             latest_request.get("requested_at"))
    elif latest_request["status"] == "uplifted":
        step("manager", "Area manager", "done",
             f"Declined {_pct(latest_request['requested_rate'])}; uplifted to {_pct(latest_request.get('approved_rate'))}",
             latest_request.get("decided_at"))
    elif latest_request["status"] == "approved":
        step("manager", "Area manager", "done", f"Approved {_pct(latest_request.get('approved_rate'))}",
             latest_request.get("decided_at"))
    else:
        step("manager", "Area manager", "warn",
             f"Declined {_pct(latest_request['requested_rate'])}, kept {_pct(latest_request['recommended_rate'])}",
             latest_request.get("decided_at"))

    if live_offer:
        step("offer", "Final offer", "done", f"Sent at {_pct(live_offer['rate'])}", live_offer.get("sent_at"))
        status = live_offer.get("status")
        step("decision", "Customer decision",
             "done" if status == "accepted" else "warn" if status == "not_accepted" else "current",
             "Accepted" if status == "accepted" else "Not taken" if status == "not_accepted" else "Waiting for the customer",
             live_offer.get("customer_decided_at"))
    else:
        ready_to_send = behaviour_ready and not (latest_request and latest_request["status"] == "pending")
        step("offer", "Final offer", "current" if ready_to_send and post else "todo", "Not sent yet")
        step("decision", "Customer decision", "todo", None)
    return {"loan_id": loan_id, "status": label(f), "next": next_action(f), "aging": aging(f), "upsell": upsell(f), "steps": out,
            "indicative": {"min_rate": band.get("min_rate"), "max_rate": band.get("max_rate")} if pre else None,
            "risk_adjusted": {"rate": rec, "risk_grade": grade} if post and risk_ready else None,
            "acceptance": acceptance if behaviour_ready else None}
