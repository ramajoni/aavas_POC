"""Sign-off over changes to the system itself.

The engine already routes a *priced decision* to an authority. It had no way to
approve anything about itself: a config change was a file edit and a restart, a
model promotion was a re-run of training, and there was no published release to
run or roll back to. This is that missing half.

Four properties make it a control rather than a form:

* **Every required role must sign.** `config/governance.yaml` lists them per
  artefact type; a model promotion needs model risk, an agent-set publication
  needs technology *and* model risk. One signature on a promotion is not a
  control.
* **Gates run at request time and are recorded.** A model whose card fails the
  fairness gate cannot be promoted by approving it harder.
* **A failed gate can be overridden, never silently.** It takes a specific role, a
  written justification of real length, and the override is stored against the
  approval forever.
* **Nobody approves their own request.** Separation of duties is checked against
  the principal, not the role.

Everything here is inert until authentication is on. With
``AAVAS_AUTH_MODE=disabled`` the caller's identity is whatever they claim, and
every surface that reports an approval says so - a signature from an
unauthenticated caller is a name in a text field, and calling it a control would
be the more dangerous mistake.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from ..config import load
from ..logging_utils import get_logger
from . import audit, store

log = get_logger(__name__)

PENDING, APPROVED, REJECTED, WITHDRAWN, APPLIED = (
    "pending", "approved", "rejected", "withdrawn", "applied")


def config() -> dict[str, Any]:
    return load("governance")


def roles() -> dict[str, Any]:
    return dict(config().get("roles") or {})


def artefact_rules(artefact_type: str) -> dict[str, Any]:
    rules = (config().get("artefacts") or {}).get(artefact_type)
    if not rules:
        raise ValueError(f"unknown artefact type {artefact_type!r}; expected one of "
                         f"{sorted((config().get('artefacts') or {}))}")
    return rules


# ----------------------------------------------------------------------- gates
# Each returns (passed, detail). A gate that cannot be evaluated returns None for
# passed - "not measurable" is not "passed", and the request records which it was.

def _gate_fairness_pass(artefact_id: str, **_: Any) -> tuple[bool | None, str]:
    from . import model_registry

    name, _, version = artefact_id.partition(":")
    row = store.fetch_one("SELECT card FROM model_registry WHERE name = ? AND version = ?",
                          (name, version)) if version else model_registry.champion(name)
    if not row:
        return False, f"no registered card for {artefact_id!r}"
    card = store.decode(row.get("card")) or {}
    fairness = card.get("fairness")
    if fairness is None:
        return None, ("fairness was not measured for this model - the protected axis was "
                      "unavailable on the training frame. Not measured is not passed")
    if fairness.get("passed") is None:
        return None, "; ".join(fairness.get("reasons") or ["insufficient evidence to compare groups"])
    if fairness.get("passed"):
        gap = (fairness.get("gaps") or {}).get("calibration_gap")
        return True, f"calibration gap by {fairness.get('group_column')} is {gap:.4f}"
    return False, "; ".join(fairness.get("reasons") or ["fairness gate failed"])


def _gate_card_registered(artefact_id: str, **_: Any) -> tuple[bool | None, str]:
    name, _, version = artefact_id.partition(":")
    if not version:
        return False, "promotion must name a version, as 'model:version'"
    row = store.fetch_one("SELECT 1 FROM model_registry WHERE name = ? AND version = ?",
                          (name, version))
    return bool(row), (f"{artefact_id} is on the model registry" if row
                       else f"{artefact_id} is not on the model registry")


def _gate_challenger_beats_champion(artefact_id: str, **_: Any) -> tuple[bool | None, str]:
    from . import model_registry

    name, _, version = artefact_id.partition(":")
    champion = model_registry.champion(name)
    if not champion:
        return None, f"no incumbent champion for {name}; nothing to compare against"
    if champion["version"] == version:
        return None, f"{artefact_id} is already the champion"
    return None, ("shadow comparison has not been run. learning/champion_challenger.py "
                  "declares the gates; no replay has been recorded for this pair")


def _gate_manifest_registered(artefact_id: str, **_: Any) -> tuple[bool | None, str]:
    from . import agent_registry

    row = agent_registry.get(artefact_id)
    return bool(row), (f"manifest {artefact_id} is on file" if row
                       else f"manifest {artefact_id} has not been published")


def _gate_config_parses(artefact_id: str, **_: Any) -> tuple[bool | None, str]:
    try:
        load(artefact_id)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"{artefact_id} does not load: {exc}"
    return True, f"{artefact_id} parses"


def _gate_consent_store_configured(artefact_id: str, **_: Any) -> tuple[bool | None, str]:
    from ..connectors.registry import consent_backend

    backend = consent_backend()
    if backend in ("none", None, ""):
        return False, ("no consent store is configured. A connector that reads a customer's "
                       "own words may not be enabled without one")
    return True, f"consent store is {backend!r}"


def _proposal(artefact_id: str) -> dict[str, Any] | None:
    from . import feature_registry

    stage, _, name = artefact_id.partition(":")
    return feature_registry.by_name(name, stage) if name else feature_registry.by_name(stage)


def _gate_feature_proposed(artefact_id: str, **_: Any) -> tuple[bool | None, str]:
    row = _proposal(artefact_id)
    if not row:
        return False, (f"no tested proposal for {artefact_id!r}. A feature is admitted with "
                       f"its evidence or not at all - run discovery and propose it first")
    return True, f"proposed {row['proposed_at'][:10]} as {row['proposal_id']}"


def _gate_reason_label_authored(artefact_id: str, **_: Any) -> tuple[bool | None, str]:
    """The gate that is not statistical.

    A feature with no human-authored label cannot appear in a customer-facing
    explanation, so it cannot be admitted however strong it is. The fallback in
    ``feature_pipeline.reason_label`` title-cases the column name; that is a
    placeholder and must not satisfy this.
    """
    from ..features.discovery import has_reason_label

    _, _, name = artefact_id.partition(":")
    name = name or artefact_id
    if has_reason_label(name):
        return True, f"a reason label for {name!r} is authored in feature_config.yaml"
    return False, (f"no reason label for {name!r}. Add one to "
                   f"feature_config.reason_code_labels first - a machine-written "
                   f"explanation of a machine-written feature is a circle with a customer "
                   f"at the end of it")


def _gate_feature_no_leakage(artefact_id: str, **_: Any) -> tuple[bool | None, str]:
    from ..features.discovery import allowed_columns

    row = _proposal(artefact_id)
    if not row:
        return None, "no proposal on file, so its inputs cannot be checked"
    try:
        permitted = set(allowed_columns(row["stage"]))
    except Exception as exc:                                    # noqa: BLE001
        return None, f"could not read the {row['stage']!r} contract: {exc}"
    outside = sorted(set(row["inputs"]) - permitted)
    if outside:
        return False, f"built from column(s) {row['stage']} may not see: {outside}"
    return True, f"every input is inside the {row['stage']} contract"


def _gate_feature_not_redundant(artefact_id: str, **_: Any) -> tuple[bool | None, str]:
    from ..features.discovery import DEFAULTS

    row = _proposal(artefact_id)
    if not row:
        return None, "no proposal on file, so redundancy cannot be checked"
    value = (row["tests"] or {}).get("max_correlation_with_contract")
    if value is None:
        return None, "the proposal records no redundancy test"
    limit = DEFAULTS["max_abs_correlation"]
    against = (row["tests"] or {}).get("most_correlated_with") or "a contract feature"
    if value > limit:
        return False, f"correlates {value:.2f} with {against}, above {limit}"
    return True, f"peak correlation with the contract is {value:.2f} ({against}), under {limit}"


GATES: dict[str, Callable[..., tuple[bool | None, str]]] = {
    "feature_proposed": _gate_feature_proposed,
    "reason_label_authored": _gate_reason_label_authored,
    "feature_no_leakage": _gate_feature_no_leakage,
    "feature_not_redundant": _gate_feature_not_redundant,
    "fairness_pass": _gate_fairness_pass,
    "card_registered": _gate_card_registered,
    "challenger_beats_champion": _gate_challenger_beats_champion,
    "manifest_registered": _gate_manifest_registered,
    "config_parses": _gate_config_parses,
    "consent_store_configured": _gate_consent_store_configured,
}


def run_gates(artefact_type: str, artefact_id: str) -> list[dict[str, Any]]:
    results = []
    for name in artefact_rules(artefact_type).get("gates") or []:
        gate = GATES.get(name)
        if gate is None:
            results.append({"gate": name, "passed": None,
                            "detail": f"gate {name!r} is declared and not implemented"})
            continue
        try:
            passed, detail = gate(artefact_id)
        except Exception as exc:                                # noqa: BLE001
            passed, detail = None, f"gate raised {type(exc).__name__}: {exc}"
        results.append({"gate": name, "passed": passed, "detail": detail})
    return results


def gates_clear(results: list[dict[str, Any]]) -> bool:
    """A gate that failed blocks. A gate that could not be evaluated does not
    block, but is reported - blocking on unmeasurable would make every new model
    unpromotable, and passing it silently would be the opposite mistake."""
    return not any(r["passed"] is False for r in results)


# -------------------------------------------------------------------- requests


def new_id() -> str:
    return f"APR-{uuid.uuid4().hex[:10].upper()}"


def request(artefact_type: str, artefact_id: str, action: str, requested_by: str,
            summary: str = "", artefact_hash: str | None = None,
            override_justification: str | None = None) -> dict[str, Any]:
    rules = artefact_rules(artefact_type)
    results = run_gates(artefact_type, artefact_id)
    clear = gates_clear(results)

    if not clear:
        rule = config().get("gate_override") or {}
        if not rule.get("allowed", False):
            raise PermissionError("a failed gate cannot be overridden under this configuration")
        minimum = int(rule.get("min_justification_chars", 40))
        if rule.get("requires_justification", True) and len(override_justification or "") < minimum:
            failed = ", ".join(r["gate"] for r in results if r["passed"] is False)
            raise ValueError(
                f"gate(s) failed: {failed}. Raising this request anyway needs a written "
                f"justification of at least {minimum} characters, and it is recorded against "
                f"the approval permanently")

    approval_id = new_id()
    store.insert("approvals", {
        "approval_id": approval_id, "artefact_type": artefact_type, "artefact_id": artefact_id,
        "artefact_hash": artefact_hash, "action": action, "state": PENDING,
        "requested_by": requested_by, "requested_at": store.now(), "summary": summary,
        "required_roles": list(rules.get("requires") or []),
        "gate_results": results, "gates_passed": bool(clear),
        "override_justification": override_justification if not clear else None,
    })
    audit.record(requested_by, "request_approval", artefact_type, artefact_id,
                 {"approval_id": approval_id, "action": action, "gates_passed": clear})
    log.info("approval %s requested: %s %s by %s (gates %s)",
             approval_id, action, artefact_id, requested_by, "clear" if clear else "FAILED")
    return get(approval_id)


def sign(approval_id: str, role: str, principal: str, verdict: str,
         comment: str = "") -> dict[str, Any]:
    record = get(approval_id)
    if not record:
        raise KeyError(f"no approval {approval_id!r}")
    if record["state"] not in (PENDING,):
        raise ValueError(f"approval {approval_id} is {record['state']}, not open for signature")
    if role not in record["required_roles"]:
        raise PermissionError(
            f"role {role!r} is not required for a {record['artefact_type']} change; "
            f"required: {record['required_roles']}")
    if (config().get("separation_of_duties") or {}).get("forbid_self_approval", True) \
            and principal == record["requested_by"]:
        raise PermissionError(
            f"{principal} raised this request and may not also sign it. Separation of duties "
            f"is checked against the person, not the role")
    if any(s["role"] == role for s in record["signatures"]):
        raise ValueError(f"role {role!r} has already signed {approval_id}")
    if verdict not in ("approve", "reject"):
        raise ValueError("verdict must be 'approve' or 'reject'")

    store.insert("approval_signatures", {
        "signature_id": f"SIG-{uuid.uuid4().hex[:10].upper()}", "approval_id": approval_id,
        "role": role, "principal": principal, "verdict": verdict,
        "signed_at": store.now(), "comment": comment,
    })
    audit.record(principal, f"{verdict}_approval", record["artefact_type"],
                 record["artefact_id"], {"approval_id": approval_id, "role": role})

    refreshed = get(approval_id)
    if verdict == "reject":
        _set_state(approval_id, REJECTED)
    elif set(refreshed["signed_roles"]) >= set(refreshed["required_roles"]):
        _set_state(approval_id, APPROVED)
        log.info("approval %s is fully signed", approval_id)
    return get(approval_id)


def _set_state(approval_id: str, state: str, applied: bool = False) -> None:
    with store.connect() as conn:
        if applied:
            conn.execute("UPDATE approvals SET state = ?, applied_at = ? WHERE approval_id = ?",
                         (state, store.now(), approval_id))
        else:
            conn.execute("UPDATE approvals SET state = ?, decided_at = ? WHERE approval_id = ?",
                         (state, store.now(), approval_id))


def withdraw(approval_id: str, principal: str) -> dict[str, Any]:
    record = get(approval_id)
    if not record:
        raise KeyError(f"no approval {approval_id!r}")
    if record["requested_by"] != principal:
        raise PermissionError("only the requester may withdraw a request")
    _set_state(approval_id, WITHDRAWN)
    audit.record(principal, "withdraw_approval", record["artefact_type"], record["artefact_id"],
                 {"approval_id": approval_id})
    return get(approval_id)


def mark_applied(approval_id: str, principal: str, detail: dict[str, Any] | None = None) -> None:
    _set_state(approval_id, APPLIED, applied=True)
    with store.connect() as conn:
        conn.execute("UPDATE approvals SET detail = ? WHERE approval_id = ?",
                     (store._encode(detail or {}), approval_id))
    audit.record(principal, "apply_approval", "approval", approval_id, detail or {})


def get(approval_id: str) -> dict[str, Any] | None:
    row = store.fetch_one("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,))
    if not row:
        return None
    return _hydrate(row)


def _hydrate(row: dict[str, Any]) -> dict[str, Any]:
    row["required_roles"] = store.decode(row.get("required_roles")) or []
    row["gate_results"] = store.decode(row.get("gate_results")) or []
    row["detail"] = store.decode(row.get("detail"))
    row["gates_passed"] = bool(row.get("gates_passed"))
    signatures = store.query(
        "SELECT * FROM approval_signatures WHERE approval_id = ? ORDER BY signed_at",
        (row["approval_id"],))
    row["signatures"] = signatures
    row["signed_roles"] = [s["role"] for s in signatures if s["verdict"] == "approve"]
    row["awaiting_roles"] = [r for r in row["required_roles"] if r not in row["signed_roles"]]
    return row


def listing(state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    if state:
        rows = store.query("SELECT * FROM approvals WHERE state = ? ORDER BY requested_at DESC "
                           "LIMIT ?", (state, limit))
    else:
        rows = store.query("SELECT * FROM approvals ORDER BY requested_at DESC LIMIT ?", (limit,))
    return [_hydrate(row) for row in rows]


def approved_for(artefact_type: str, artefact_id: str, action: str) -> dict[str, Any] | None:
    """An approval that authorises this exact change and has not been applied."""
    rows = store.query(
        "SELECT * FROM approvals WHERE artefact_type = ? AND artefact_id = ? AND action = ? "
        "AND state = ? ORDER BY requested_at DESC LIMIT 1",
        (artefact_type, artefact_id, action, APPROVED))
    return _hydrate(rows[0]) if rows else None


def summary() -> dict[str, Any]:
    rows = listing(limit=500)
    by_state: dict[str, int] = {}
    for row in rows:
        by_state[row["state"]] = by_state.get(row["state"], 0) + 1
    return {
        "total": len(rows), "by_state": by_state,
        "pending": [r["approval_id"] for r in rows if r["state"] == PENDING],
        "blocked_by_gates": [r["approval_id"] for r in rows if not r["gates_passed"]],
        "roles": roles(), "artefacts": config().get("artefacts") or {},
        "version": config().get("governance_config_version"),
    }
