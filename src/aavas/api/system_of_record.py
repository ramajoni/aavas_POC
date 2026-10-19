"""The pricing system of record, and whether a decision can actually be rebuilt.

Twelve registries are proposed. This module reports which of them exist here and,
more usefully, tests the claim that rests on them: *a production pricing decision
can be reconstructed exactly - which agent, which version, which skill, which
tool, which model, which features, which data, which policy, which branch
personalization, which human decision, which final rate.*

That is a falsifiable claim, so ``/reconstruct/{decision_id}`` walks it link by
link against a real decision and reports each one as resolved, partial or broken
with the reason. A system of record that is described rather than exercised is a
diagram.

One distinction the audit turns on, because it decides what a registry is worth:

* A **registry** is queryable as of a past moment - the model registry stores the
  full card against (name, version), so a decision from March resolves to the
  card that was live in March.
* **Versioned config** is a file with a version string recorded on the decision.
  It reconstructs only if that version of the file still exists.
* A **code declaration** - the agent, skill and tool registries - is neither. It
  describes the system as it is now. Asking it what an agent looked like at the
  time of a decision returns today's answer with no way to know it differs.

Three of the twelve are absent outright, and saying so is the point.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..agents import capabilities as agent_capabilities
from ..agents import roster as agent_roster
from ..config import load, versions
from ..governance import (agent_registry, decision_logger, knowledge_registry,
                          model_registry, prompt_registry, store)
from ..logging_utils import get_logger
from .deps import get_orchestrator

log = get_logger(__name__)
router = APIRouter(prefix="/system-of-record", tags=["system of record"])

RESOLVED, PARTIAL, BROKEN = "resolved", "partial", "broken"


# ------------------------------------------------------------ the registries


def _registries() -> list[dict[str, Any]]:
    try:
        connectors = load("connectors")
        connector_version = connectors.get("connectors_config_version")
        connector_count = len(connectors.get("connectors") or {})
    except Exception:                                           # noqa: BLE001
        connector_version, connector_count = None, 0

    v = versions()
    registered = model_registry.registered()
    manifest = agent_registry.manifest()
    prompt_registry.register_declared(actor="api")
    prompts = prompt_registry.registered()
    indexes = knowledge_registry.registered()
    datasets = {(store.decode(r.get("card")) or {}).get("dataset", {}).get("dataset_id")
                for r in registered}
    datasets.discard(None)
    rounds = store.count("federated_rounds")

    return [
        {"key": "agent", "title": "Agent registry", "kind": "registry",
         "backing": "governance store, agent_manifests table", "entries": len(agent_roster.AGENTS),
         "versioned": True, "queryable_as_of": True,
         "note": f"Snapshotted as {manifest['manifest_id']} and stamped on every decision. The "
                 f"id is a content hash over the declaration, so it cannot drift from the code "
                 f"it names - a hand-bumped number is a promise someone has to keep on every "
                 f"edit, and the edit where they forget is the one that matters."},
        {"key": "skill", "title": "Skill registry", "kind": "registry",
         "backing": "governance store, inside the agent manifest",
         "entries": len(agent_capabilities.SKILLS), "versioned": True, "queryable_as_of": True,
         "note": "Carried in the same snapshot, so which skills an agent had is answerable as "
                 "of the decision. What each skill *did* still depends on the config versions "
                 "stamped alongside it."},
        {"key": "tool", "title": "Tool registry", "kind": "registry",
         "backing": "governance store, inside the agent manifest",
         "entries": len(agent_capabilities.TOOLS), "versioned": True, "queryable_as_of": True,
         "note": "Registered, not permissioned. Which tools an agent could call is on file; "
                 "there is no permission model saying which it was *allowed* to, and only "
                 "model artefacts carry their own version onto the decision."},
        {"key": "model", "title": "Model registry", "kind": "registry",
         "backing": "governance store, model_registry table", "entries": len(registered),
         "versioned": True, "queryable_as_of": True,
         "note": "The only true registry here. Stores the full card against (name, version), "
                 "so a decision resolves to the card that was live when it was priced."},
        {"key": "connector", "title": "Connector registry", "kind": "versioned config",
         "backing": "config/connectors.yaml", "entries": connector_count,
         "versioned": bool(connector_version), "queryable_as_of": False,
         "note": f"Version {connector_version}. Not recorded on the decision, and nothing "
                 f"is enabled, so no connector has yet influenced a price."},
        {"key": "knowledge", "title": "Knowledge registry", "kind": "registry",
         "backing": "governance store, knowledge_indexes table", "entries": len(indexes),
         "versioned": True, "queryable_as_of": True,
         "note": "Each built index records its corpus, backend, document count and a content "
                 "hash over the documents themselves - so re-indexing an unchanged corpus "
                 "produces the same id rather than a new row."},
        {"key": "prompt", "title": "Prompt / blueprint registry", "kind": "registry",
         "backing": "governance store, prompts table", "entries": len(prompts),
         "versioned": True, "queryable_as_of": True,
         "note": "Content-hashed, so an edited prompt gets a new id whether or not anyone "
                 "bumped one. Nothing here is on the priced path today - which is exactly "
                 "why it exists now: wiring the extraction agent in without it would put an "
                 "unversioned prompt in front of a customer's own words."},
        {"key": "policy", "title": "Policy registry", "kind": "versioned config",
         "backing": "config/pricing_policy.yaml", "entries": 1,
         "versioned": True, "queryable_as_of": False,
         "note": f"Version {v['policy_version']}, recorded on every decision. It "
                 f"reconstructs as long as that version of the file still exists; the file "
                 f"itself is not stored with the decision."},
        {"key": "memory", "title": "Memory registry", "kind": "absent",
         "backing": None, "entries": 0, "versioned": False, "queryable_as_of": False,
         "note": "Nothing. No agent carries state between decisions, which is why there is "
                 "none to register - an accurate description of the system rather than a "
                 "gap that needs filling before anything else works."},
        {"key": "feature", "title": "Feature registry", "kind": "versioned config",
         "backing": "config/feature_config.yaml", "entries": 1,
         "versioned": True, "queryable_as_of": False,
         "note": f"Version {v['feature_config_version']}, recorded on every decision. The "
                 f"per-model feature list also survives on each model card."},
        {"key": "trace", "title": "Execution trace registry", "kind": "per-decision data",
         "backing": "decision payload, trace block", "entries": _traced_count(),
         "versioned": False, "queryable_as_of": True,
         "note": "Persisted with the decision: which agent ran, in what position, how long "
                 "it took and one line about what it produced. Decisions priced before the "
                 "trace existed carry none."},
        {"key": "dataset", "title": "Dataset registry", "kind": "registry",
         "backing": "dataset identity on each model card", "entries": len(datasets),
         "versioned": True, "queryable_as_of": True,
         "note": f"{len(datasets)} distinct dataset(s) behind the registered cards, each a "
                 f"content hash over the frame the model was actually fitted on - not its "
                 f"parent. Two models fitted on differently filtered extracts of one window "
                 f"now carry different ids."},
    ]


def _traced_count() -> int:
    try:
        rows = decision_logger.recent(limit=200)
    except Exception:                                           # noqa: BLE001
        return 0
    return sum(1 for d in rows if (d.get("payload") or {}).get("trace"))


# ----------------------------------------------------------- reconstruction


def _link(step: int, question: str, status: str, value: Any = None, why: str = "") -> dict[str, Any]:
    return {"step": step, "question": question, "status": status, "value": value, "why": why}


def _reconstruct(decision: dict[str, Any]) -> list[dict[str, Any]]:
    payload = decision.get("payload") or {}
    trace = payload.get("trace") or {}
    decision_versions = payload.get("versions") or {}
    model_versions = payload.get("model_versions") or {}
    ran = list(dict.fromkeys(trace.get("agents") or []))

    links: list[dict[str, Any]] = []

    links.append(_link(1, "Which decision?", RESOLVED,
                       {"decision_id": decision["decision_id"], "loan_id": decision["loan_id"],
                        "stage": decision["stage"], "priced_at": decision["created_at"]}))

    snapshot = payload.get("application_snapshot") or {}
    links.append(_link(
        2, "Which file, exactly as priced?",
        RESOLVED if snapshot else BROKEN,
        {"fields": len(snapshot)} if snapshot else None,
        "" if snapshot else "no application snapshot was stored with this decision"))

    links.append(_link(
        3, "Which agents ran?", RESOLVED if ran else BROKEN, ran,
        "" if ran else "this decision was priced before the step trace existed"))

    manifest_id = decision_versions.get("agent_manifest")
    recorded = agent_registry.get(manifest_id) if manifest_id else None
    if recorded:
        drifted = manifest_id != agent_registry.current_id()
        links.append(_link(
            4, "Which version of the agent set?", RESOLVED,
            {"agent_manifest": manifest_id, "content_hash": recorded["content_hash"][:16],
             "agents": recorded["agents"], "wired": recorded["wired"],
             "skills": recorded["skills"], "tools": recorded["tools"],
             "is_current": not drifted},
            ("the roster has changed since this decision was priced; the snapshot above is "
             "the one that ran" if drifted else "")))
    elif manifest_id:
        links.append(_link(
            4, "Which version of the agent set?", PARTIAL, {"agent_manifest": manifest_id},
            "the decision names a manifest that is not on file - the snapshot was not "
            "published before the decision that referenced it"))
    else:
        links.append(_link(
            4, "Which version of the agent set?", BROKEN, None,
            "this decision was priced before the agent set was registered. app_version names "
            "the code build, not the roster, so its agent set cannot be recovered"))

    # Prefer the snapshot that ran over today's roster. That is the whole point of
    # registering it: "which skill" must be answerable as of the decision.
    as_of = {a["name"]: a for a in ((recorded or {}).get("manifest") or {}).get("agents", [])} \
        if recorded else {}
    source = "the agent manifest recorded on this decision" if as_of else "today's roster"
    lookup = as_of or agent_roster.BY_NAME

    skills = sorted({s for name in ran for s in (lookup.get(name) or {}).get("skills", [])})
    unknown = sorted(n for n in ran if n not in lookup)
    links.append(_link(
        5, "Which skills did they use?",
        RESOLVED if skills and as_of and not unknown else PARTIAL if skills else BROKEN,
        skills,
        (f"resolved against {source}" if as_of else
         "resolved against today's roster, not the roster as of this decision")
        + (f". Agents on this decision not in that set: {unknown}" if unknown else "")))

    model_tools = sorted(f"model.{name}" for name in model_versions)
    other = sorted({t for name in ran for t in (lookup.get(name) or {}).get("tools", [])
                    if not t.startswith("model.")})
    from ..agents import capabilities as agent_capabilities

    grants = {name: agent_capabilities.permissions_for(name) for name in ran}
    grants = {k: v for k, v in grants.items() if v}
    links.append(_link(
        6, "Which tools did they call, and under what permission?", PARTIAL,
        {"versioned": model_tools, "unversioned": other, "resolved_against": source,
         "permissions": grants},
        "model artefacts carry a version on the decision; the rest do not, so which shap "
        "build or which MCP server was in force is still unrecorded. The permission grant "
        "above is the current declaration, not one stamped on this decision"))

    links.append(_link(
        7, "Which model versions?", RESOLVED if model_versions else BROKEN, model_versions,
        "" if model_versions else "no model versions were recorded"))

    cards, missing = {}, []
    for name, version in model_versions.items():
        row = store.fetch_one(
            "SELECT card FROM model_registry WHERE name = ? AND version = ?", (name, version))
        if row:
            card = store.decode(row["card"]) or {}
            fairness = card.get("fairness")
            cards[name] = {"version": version, "learner": card.get("learner"),
                           "features": len(card.get("features") or []),
                           "trained_at": card.get("trained_at"),
                           "training_window": card.get("training_window") or {},
                           "n_train": card.get("n_train"),
                           "dataset": card.get("dataset") or {},
                           "fairness_passed": None if fairness is None else fairness.get("passed")}
        else:
            missing.append(f"{name}:{version}")
    links.append(_link(
        8, "Which model card, as of that version?",
        RESOLVED if cards and not missing else PARTIAL if cards else BROKEN,
        cards, f"not in the registry: {', '.join(missing)}" if missing else ""))

    links.append(_link(
        9, "Which feature contract?",
        RESOLVED if decision_versions.get("feature_config_version") else BROKEN,
        decision_versions.get("feature_config_version"),
        "the per-model feature list also survives on each card above"))

    datasets = {n: c["dataset"] for n, c in cards.items() if c.get("dataset")}
    distinct = {d.get("dataset_id") for d in datasets.values() if d.get("dataset_id")}
    without = sorted(n for n, c in cards.items() if not c.get("dataset"))
    links.append(_link(
        10, "Which data were the models fitted on?",
        RESOLVED if datasets and not without else PARTIAL if datasets else BROKEN,
        {"datasets": datasets, "distinct_datasets": sorted(distinct)},
        (f"cards without a dataset identity (fitted before it existed): {without}" if without
         else f"{len(distinct)} distinct dataset(s), each a content hash over the frame the "
              f"model was actually fitted on - not its parent, so a differently filtered "
              f"extract of the same window is a different id")))

    links.append(_link(
        11, "Which policy?",
        RESOLVED if decision_versions.get("policy_version") else BROKEN,
        {"policy_version": decision_versions.get("policy_version"),
         "model_config_version": decision_versions.get("model_config_version"),
         "app_version": decision_versions.get("app_version")},
        "the version is recorded; the file itself is not stored with the decision, so this "
        "reconstructs only while that version of the file still exists"))

    latest_round = store.fetch_one(
        "SELECT run_id, created_at, strategy, participants, global_version, final_mae_bps "
        "FROM federated_rounds ORDER BY created_at DESC LIMIT 1")
    if not latest_round:
        links.append(_link(
            12, "Which branch personalization?", BROKEN, None,
            "no federated round has been recorded. The implementation exists; until a run is "
            "persisted there is nothing for a decision to name"))
    elif latest_round["created_at"] > (decision.get("created_at") or ""):
        links.append(_link(
            12, "Which branch personalization?", PARTIAL, latest_round,
            "rounds are on file but this decision predates the most recent one, and the "
            "decision does not carry a round id. The regional adjustment is not applied on "
            "the priced path yet, so no decision has one to record"))
    else:
        links.append(_link(
            12, "Which branch personalization?", PARTIAL, latest_round,
            "the round that was current when this decision was priced. The regional "
            "adjustment is not applied on the priced path, so this is the personalisation "
            "that existed, not one that moved this price"))

    # Match the override on decision_id, not loan_id. A file priced more than
    # once - which the demo flow does - would otherwise attach an override made
    # against a different decision and report it as this one's.
    override = store.fetch_one(
        "SELECT * FROM overrides WHERE decision_id = ? ORDER BY created_at DESC LIMIT 1",
        (decision["decision_id"],))
    on_file = store.fetch_one(
        "SELECT COUNT(*) AS n FROM overrides WHERE loan_id = ?", (decision["loan_id"],))
    elsewhere = int((on_file or {}).get("n") or 0) - (1 if override else 0)

    if override:
        why = ""
    elif elsewhere > 0:
        why = (f"no override was recorded against this decision. {elsewhere} exist(s) against "
               f"other decisions on the same file and are deliberately not shown here - an "
               f"override belongs to the price it moved, not to the loan id")
    else:
        why = "routed to an authority; no override was recorded"
    links.append(_link(
        13, "Which human decision?", RESOLVED if override or not elsewhere else PARTIAL,
        {"required_approval": decision.get("required_approval"),
         "override": {k: override[k] for k in
                      ("override_id", "human_rate", "reason", "approved_by", "within_authority")}
         if override else None,
         "overrides_on_other_decisions_for_this_file": elsewhere},
        why))

    # Outcomes are keyed by loan_id alone, so a file priced twice has one outcome
    # and no way to say which decision it followed.
    outcome = store.fetch_one("SELECT * FROM outcomes WHERE loan_id = ?", (decision["loan_id"],))
    priced_times = int((store.fetch_one(
        "SELECT COUNT(*) AS n FROM decisions WHERE loan_id = ? AND stage = ?",
        (decision["loan_id"], decision["stage"])) or {}).get("n") or 1)
    if not outcome:
        outcome_why = "no outcome has been captured for this file yet"
        status = RESOLVED
    elif priced_times > 1:
        outcome_why = (f"the outcomes table is keyed by loan id alone, and this file was priced "
                       f"{priced_times} times at this stage. The sanctioned rate below is the "
                       f"file's, not demonstrably this decision's - which is the join the "
                       f"learning loop needs and does not have")
        status = PARTIAL
    else:
        outcome_why = ""
        status = RESOLVED
    links.append(_link(
        14, "Which final rate?", status,
        {"recommended": decision.get("recommended_rate"),
         "overridden_to": override["human_rate"] if override else None,
         "sanctioned": outcome.get("sanctioned_rate") if outcome else None,
         "customer_response": outcome.get("customer_response") if outcome else None,
         "times_this_file_was_priced": priced_times},
        outcome_why))

    return links


def _summary(links: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {RESOLVED: 0, PARTIAL: 0, BROKEN: 0}
    for link in links:
        counts[link["status"]] += 1
    return {"links": len(links), **counts,
            "reconstructible": counts[BROKEN] == 0 and counts[PARTIAL] == 0}


# ------------------------------------------------------------------ routes


@router.get("", summary="The twelve registries, and which of them exist here")
def index(orchestrator=Depends(get_orchestrator)) -> dict[str, Any]:
    rows = _registries()
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    return {
        "registries": rows,
        "summary": {"declared": len(rows), "by_kind": kinds,
                    "absent": [r["key"] for r in rows if r["kind"] == "absent"],
                    "queryable_as_of": [r["key"] for r in rows if r["queryable_as_of"]]},
        "versions": versions(),
        "note": ("A registry is queryable as of a past moment. Versioned config reconstructs "
                 "only while that version of the file survives. A code declaration describes "
                 "the system as it is now and cannot answer what it was - which is why the "
                 "reconstruction below is partial rather than complete."),
    }


@router.get("/manifests", summary="Every registered agent set, newest first")
def manifests(limit: int = 50) -> dict[str, Any]:
    agent_registry.publish(actor="api")
    rows = agent_registry.registered(limit=limit)
    return {
        "manifests": rows, "current": agent_registry.current_id(),
        "note": ("The id is a content hash over the agent, skill and tool declarations plus "
                 "the pipeline order. Editing the roster produces a new id on the next "
                 "decision; editing a comment does not."),
    }


@router.get("/manifest/{manifest_id}", summary="One agent set exactly as it was recorded")
def manifest(manifest_id: str) -> dict[str, Any]:
    if manifest_id == "current":
        agent_registry.publish(actor="api")
        manifest_id = agent_registry.current_id()
    row = agent_registry.get(manifest_id)
    if not row:
        raise HTTPException(404, f"no agent manifest {manifest_id!r} on file")
    return {**row, "is_current": manifest_id == agent_registry.current_id()}


@router.get("/manifest-diff", summary="What changed between two agent sets")
def manifest_diff(left: str, right: str) -> dict[str, Any]:
    try:
        return agent_registry.diff(left, right)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/reconstruct/latest", summary="Walk the chain for the most recent traced decision")
def reconstruct_latest() -> dict[str, Any]:
    # Registered before /reconstruct/{decision_id}: 'latest' would otherwise be
    # read as a decision id and 404.
    for decision in decision_logger.recent(limit=50):
        if (decision.get("payload") or {}).get("trace"):
            return reconstruct(decision["decision_id"])
    raise HTTPException(404, "no decision on record carries a trace; price one to produce it")


@router.get("/reconstruct/{decision_id}", summary="Can this decision be rebuilt, link by link?")
def reconstruct(decision_id: str) -> dict[str, Any]:
    decision = decision_logger.get_decision(decision_id)
    if not decision:
        raise HTTPException(404, f"no decision {decision_id!r} on record")
    links = _reconstruct(decision)
    summary = _summary(links)
    return {
        "decision_id": decision_id,
        "loan_id": decision.get("loan_id"),
        "chain": links,
        "summary": summary,
        "verdict": ("This decision cannot be reconstructed exactly. "
                    f"{summary[BROKEN]} link(s) have nothing behind them and "
                    f"{summary[PARTIAL]} resolve against the system as it is now rather than "
                    f"as it was."
                    if not summary["reconstructible"] else
                    "Every link resolves as of the moment this decision was priced."),
    }
