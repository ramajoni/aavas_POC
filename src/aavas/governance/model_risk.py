"""Model risk management: the inventory, risk tiers and independent validation.

Implements, for this platform, the RBI Draft Guidance on Regulatory Principles for
Model Risk Management, 2026 ("MRM <para>") and the inventory and oversight parts of
the FREE-AI report. config/model_risk.yaml is the inventory; this module checks it
against the facts on every read and keeps the validation record.

What it decides from the facts rather than takes from the YAML:
  * the risk tier, from materiality, complexity and autonomy, with the rule that a
    low score on one cannot pull a highly material model down (MRM 19-20, 52);
  * whether each model has a current independent validation of the version in use,
    by someone outside its development and ownership (MRM 7(8), 29-30), reported to
    the RMCB within three months (MRM 33);
  * whether anything is in use without an inventory entry (MRM 21);
  * whether drift monitoring is current (MRM 37, 54(7));
  * whether the inventory was reviewed within half a year (FREE-AI rec 23).
"""

from __future__ import annotations

import json
import secrets
from datetime import date, datetime, timedelta, timezone
from typing import Any

from ..config import load
from ..models import base as model_base
from . import audit, store

OUTCOMES = ("approved", "approved_with_exceptions", "rejected")


class ModelRiskError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def conf() -> dict[str, Any]:
    return load("model_risk")


def inventory() -> dict[str, dict[str, Any]]:
    return conf()["models"]


def tier(entry: dict[str, Any], levels: list[str]) -> str:
    """Materiality sets the floor; higher complexity or autonomy raises it one level.
    Nothing lowers it (MRM 20: factors must not offset each other)."""
    m = levels.index(entry["materiality"])
    other = max(levels.index(entry["complexity"]), levels.index(entry["autonomy"]))
    return levels[min(len(levels) - 1, m + 1)] if other > m else levels[m]


def _cards() -> dict[str, dict[str, Any]]:
    cards = {}
    for path in model_base.registry_dir().glob("*.card.json"):
        card = json.loads(path.read_text(encoding="utf-8"))
        cards[card.get("name")] = card
    return cards


def version(name: str, entry: dict[str, Any], cards: dict[str, dict[str, Any]]) -> str:
    """The version in use now: a validation applies to this and nothing later (MRM 30)."""
    if entry.get("card"):
        card = cards.get(entry["card"]) or {}
        return f"{card.get('version', '?')} trained {card.get('trained_at', '?')}"
    source = entry.get("version_from")
    if source == "policy":
        return str(load("policy").get("policy_version"))
    if source == "acceptance":
        return f"{load('acceptance').get('acceptance_config_version')} / " \
               f"{(load('acceptance').get('extraction') or {}).get('model') or 'configured model'}"
    if source == "agentic_os":
        aga = (load("agentic_os").get("training") or {}).get("autogluon_assistant") or {}
        return f"{load('agentic_os').get('agentic_os_config_version')} / {aga.get('model', '?')}"
    if source in ("market_research", "narrative"):
        from ..agents import market_research_agent
        from ..xai import narrative
        module = market_research_agent if source == "market_research" else narrative
        return str(getattr(module, "DEFAULT_MODEL", "?"))
    return "unversioned"


def validations(model: str | None = None) -> list[dict[str, Any]]:
    sql, params = "SELECT * FROM model_validations", ()
    if model:
        sql, params = sql + " WHERE model = ?", (model,)
    rows = store.query(sql + " ORDER BY validated_at DESC", params)
    for r in rows:
        r["validator_roles"] = store.decode(r.get("validator_roles"))
    return rows


def record_validation(model: str, *, outcome: str, findings: str, validator: str, roles: list[str],
                      exceptions: str | None = None, remediation_due: str | None = None) -> dict[str, Any]:
    inv = inventory()
    if model not in inv:
        raise ModelRiskError(404, f"{model} is not in the model inventory; add it before validating (MRM 21)")
    entry = inv[model]
    if outcome not in OUTCOMES:
        raise ModelRiskError(422, f"outcome must be one of {', '.join(OUTCOMES)}")
    if not (findings or "").strip():
        raise ModelRiskError(422, "a validation records its findings (MRM 32)")
    builders = set(entry.get("developers") or []) | {entry.get("owner")}
    if validator in builders or set(roles) & builders:
        raise ModelRiskError(403, f"{validator} builds or owns {model}: the validator must be independent of "
                                  "development and ownership (MRM 7(8))")
    if not set(roles) & set(entry.get("validators") or []):
        raise ModelRiskError(403, f"validating {model} needs one of the roles "
                                  f"{', '.join(entry.get('validators') or [])}")
    if outcome == "approved_with_exceptions" and not ((exceptions or "").strip() and remediation_due):
        raise ModelRiskError(422, "approval with exceptions names the exceptions and a remediation date (MRM 34-35)")
    row = {"validation_id": f"VAL-{secrets.token_hex(4).upper()}", "model": model,
           "model_version": version(model, entry, _cards()), "outcome": outcome, "findings": findings.strip(),
           "exceptions": (exceptions or "").strip() or None, "remediation_due": remediation_due,
           "validator": validator, "validator_roles": roles, "validated_at": store.now(),
           "reported_to_rmcb_on": None}
    store.insert("model_validations", row)
    audit.record(validator, "model_validated", "model", model,
                 {"validation_id": row["validation_id"], "outcome": outcome, "version": row["model_version"]})
    return row


def report_to_rmcb(validation_id: str, *, reported_on: str, recorded_by: str) -> dict[str, Any]:
    row = store.fetch_one("SELECT * FROM model_validations WHERE validation_id = ?", (validation_id,))
    if row is None:
        raise ModelRiskError(404, f"no validation {validation_id}")
    try:
        when = date.fromisoformat(reported_on)
    except ValueError as exc:
        raise ModelRiskError(422, "reported_on is a date, YYYY-MM-DD") from exc
    if when < date.fromisoformat(row["validated_at"][:10]):
        raise ModelRiskError(422, "a report cannot reach the RMCB before the validation was done")
    with store.connect() as conn:
        conn.execute("UPDATE model_validations SET reported_to_rmcb_on = ? WHERE validation_id = ?",
                     (when.isoformat(), validation_id))
    audit.record(recorded_by, "validation_reported_to_rmcb", "model", row["model"],
                 {"validation_id": validation_id, "reported_on": when.isoformat()})
    return validations(row["model"])[0]


def record_review(*, reviewer: str, notes: str) -> dict[str, Any]:
    """The periodic review of the inventory and its tiers (MRM 17, FREE-AI rec 23)."""
    if not (notes or "").strip():
        raise ModelRiskError(422, "a review records what was looked at and concluded")
    tiers = {n: tier(e, conf()["framework"]["levels"]) for n, e in inventory().items()}
    audit.record(reviewer, "model_inventory_reviewed", "model_inventory", "inventory",
                 {"notes": notes.strip(), "tiers": tiers, "version": conf().get("model_risk_config_version")})
    return {"reviewed_by": reviewer, "tiers": tiers}


def _days_since(stamp: str | None, today: date) -> int | None:
    if not stamp:
        return None
    return (today - date.fromisoformat(str(stamp)[:10])).days


def _drift(today: date, max_age: int) -> dict[str, Any]:
    from ..monitoring import drift
    latest = drift.reports_dir() / "latest.json"
    if not latest.exists():
        return {"ran": False, "finding": "drift detection has never run (MRM 37, 54(7))"}
    report = json.loads(latest.read_text(encoding="utf-8"))
    age = _days_since(report.get("generated_at"), today)
    out = {"ran": True, "run_id": report.get("run_id"), "generated_at": report.get("generated_at"),
           "scenario": report.get("scenario"), "status": report.get("status"), "age_days": age, "finding": None}
    if report.get("scenario") not in (None, "none"):
        out["finding"] = f"the latest drift run was a simulated '{report['scenario']}' scenario, not the live book"
    elif age is not None and age > max_age:
        out["finding"] = f"drift last checked {age} days ago (limit {max_age})"
    elif (report.get("status") or {}).get("overall") == "alert":
        out["finding"] = "the latest drift run raised an alert: review and act (MRM 14, 37)"
    return out


def assess(today: date | None = None) -> dict[str, Any]:
    today = today or datetime.now(timezone.utc).date()
    c = conf()
    fw, levels = c["framework"], c["framework"]["levels"]
    cards = _cards()
    inv = c["models"]
    vals: dict[str, list[dict[str, Any]]] = {}
    for v in validations():
        vals.setdefault(v["model"], []).append(v)

    rows, findings = [], []

    def flag(model: str | None, severity: str, text: str, ref: str) -> None:
        findings.append({"model": model, "severity": severity, "text": text, "ref": ref})

    listed_cards = {e.get("card") for e in inv.values() if e.get("card")}
    for name in sorted(set(cards) - listed_cards):
        flag(name, "gap", f"model card '{name}' is in the registry but not in the inventory", "MRM 21")

    for name, e in inv.items():
        t = tier(e, levels)
        current = version(name, e, cards)
        hist = vals.get(name) or []
        latest = hist[0] if hist else None
        missing = [k for k in ("owner", "developers", "validators", "intended_use", "materiality", "complexity",
                               "autonomy", "upstream", "downstream") if e.get(k) in (None, "")]
        if missing:
            flag(name, "gap", f"inventory fields missing: {', '.join(missing)}", "MRM 22")
        if set(e.get("validators") or []) & (set(e.get("developers") or []) | {e.get("owner")}):
            flag(name, "gap", "a listed validator also builds or owns the model", "MRM 7(8)")
        if e.get("card") and e["card"] not in cards and e.get("status") == "active":
            flag(name, "gap", f"listed as active but no model card '{e['card']}' is registered", "MRM 21")

        state = "never validated"
        if e.get("status") == "active":
            if latest is None:
                flag(name, "gap", "no independent validation on record", "MRM 29")
            else:
                age = _days_since(latest["validated_at"], today) or 0
                due = int(fw["validation_every_months"][t]) * 30
                if latest["outcome"] == "rejected":
                    state = "rejected"
                    flag(name, "gap", "its latest validation rejected it, and it is still in use", "MRM 14, 50")
                elif latest["model_version"] != current:
                    state = "changed since validation"
                    flag(name, "attention", f"validated as {latest['model_version']}, now {current}", "MRM 30")
                elif age > due:
                    state = "overdue"
                    flag(name, "attention", f"last validated {age} days ago; a {t} model is due every "
                                            f"{fw['validation_every_months'][t]} months", "MRM 30")
                else:
                    state = latest["outcome"].replace("_", " ")
                if latest["outcome"] == "approved_with_exceptions" and latest.get("remediation_due") \
                        and latest["remediation_due"] < today.isoformat():
                    flag(name, "attention", f"exception remediation was due {latest['remediation_due']}", "MRM 34")
                if not latest.get("reported_to_rmcb_on") and age > int(fw["validation_report_to_rmcb_days"]):
                    flag(name, "attention", f"validation report not yet with the RMCB {age} days on "
                                            f"(limit {fw['validation_report_to_rmcb_days']})", "MRM 33")
            if not e.get("kill_switch"):
                flag(name, "attention", "no way to suspend this model on its own", "MRM 60(ii)")
            if not e.get("fallback"):
                flag(name, "attention", "no fallback if it fails or degrades", "MRM 43, FREE-AI rec 21")
            elif not e.get("fallback_tested_on"):
                flag(name, "info", "fallback never tested in a drill", "FREE-AI rec 21")
        for risk in e.get("known_risks") or []:
            flag(name, "attention", risk, "MRM 58")

        rows.append({
            "name": name, "title": e["title"], "kind": e["kind"], "status": e.get("status"),
            "intended_use": e["intended_use"], "owner": e.get("owner"), "developers": e.get("developers"),
            "validators": e.get("validators"), "approver": fw["approvers"][t],
            "materiality": e["materiality"], "complexity": e["complexity"], "autonomy": e["autonomy"],
            "tier": t, "consumer_facing": bool(e.get("consumer_facing")), "third_party": e.get("third_party"),
            "enhanced_oversight": bool(e.get("third_party")) and bool(fw.get("third_party_enhanced_oversight")),
            "upstream": e.get("upstream") or [], "downstream": e.get("downstream") or [],
            "oversight": e.get("oversight"), "kill_switch": e.get("kill_switch"), "fallback": e.get("fallback"),
            "version": current, "validation": {"state": state, "latest": latest, "history": len(hist)},
        })

    review = next(iter(audit.trail(entity_type="model_inventory", limit=1)), None)
    since = _days_since(review["created_at"], today) if review else None
    if review is None:
        flag(None, "gap", "the inventory and its tiers have never been reviewed", "MRM 17, FREE-AI rec 23")
    elif since > int(fw["inventory_review_max_days"]):
        flag(None, "attention", f"inventory last reviewed {since} days ago (at least half-yearly)", "FREE-AI rec 23")
    if not fw.get("board_approved_on"):
        flag(None, "outside", "the model risk framework has no recorded Board approval", "MRM 9-11")
    drift_state = _drift(today, int(fw["drift_report_max_age_days"]))
    if drift_state["finding"]:
        flag(None, "attention" if drift_state["ran"] else "gap", drift_state["finding"], "MRM 37, 54(7)")

    by_tier = {lv: sum(1 for r in rows if r["tier"] == lv) for lv in levels}
    return {
        "version": c.get("model_risk_config_version"), "as_of": today.isoformat(),
        "framework": fw, "models": rows, "findings": findings,
        "summary": {"models": len(rows), "by_tier": by_tier,
                    "third_party": sum(1 for r in rows if r["third_party"]),
                    "validated_current": sum(1 for r in rows if r["validation"]["state"] in
                                             ("approved", "approved with exceptions")),
                    "gaps": sum(1 for f in findings if f["severity"] == "gap"),
                    "attention": sum(1 for f in findings if f["severity"] == "attention")},
        "inventory_review": {"last": review["created_at"] if review else None, "by": review["actor"] if review else None,
                             "days_since": since},
        "drift": drift_state,
    }
