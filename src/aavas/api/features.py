"""Feature discovery: propose, test, refuse. Never admit.

``POST /features/discover`` runs the agent over the training table and returns
every candidate with its tests and its refusals. ``POST /features/propose``
records one against the governance store so that an admission request has
evidence behind it.

There is deliberately no endpoint that adds a feature to the contract. Widening
``feature_config.yaml`` is an approved change of artefact type ``feature``,
requiring model risk *and* credit policy, and one of its gates is not statistical
at all: a human-authored reason label must exist first.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from ..features import cards as feature_cards
from ..features import discovery, profiling
from ..governance import feature_registry
from ..ingestion import loader
from ..logging_utils import get_logger
from .security import Principal, current_principal, require_roles

log = get_logger(__name__)
router = APIRouter(prefix="/features", tags=["feature discovery"])

_CACHE: dict[str, Any] = {}


@router.get("", summary="What the discovery agent does, and what it may not do")
def index(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    from ..config import feature_config

    cfg = feature_config()
    return {
        "agent": discovery.summary(),
        "contracts": {name: {"numeric": len(block.get("numeric") or []),
                             "categorical": len(block.get("categorical") or [])}
                      for name, block in cfg.items() if isinstance(block, dict)
                      and ("numeric" in block or "categorical" in block)},
        "reason_labels": len(cfg.get("reason_code_labels") or {}),
        "proposals": feature_registry.summary(),
        "version": cfg.get("feature_config_version"),
        "principal": principal.as_dict(),
    }


@router.get("/proposals", summary="Candidates put forward, with their evidence")
def proposals(state: str | None = None, stage: str | None = None,
              principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    return {"proposals": feature_registry.listing(state=state, stage=stage),
            "summary": feature_registry.summary()}


@router.post("/discover", summary="Run the agent over the training table")
def discover(stage: str = Body("stage2", embed=True),
             generators: list[str] | None = Body(None, embed=True),
             limit: int = Body(400, embed=True),
             principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    key = f"{stage}:{','.join(sorted(generators or []))}:{limit}"
    if key in _CACHE:
        return {**_CACHE[key], "cached": True}
    try:
        frame = loader.build_training_table()
        report = discovery.discover(frame, stage=stage, generators=generators,
                                    max_candidates=limit)
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:                                    # noqa: BLE001
        log.exception("feature discovery failed")
        raise HTTPException(500, f"discovery failed: {exc}") from exc

    body = {"report": report.as_dict(), "cached": False}
    _CACHE[key] = {"report": body["report"]}
    return body


@router.get("/profile", summary="Understand the dataset before anything is engineered")
def profile(stage: str = "stage2",
            principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    frame = loader.build_training_table()
    columns = discovery.allowed_columns(stage)
    return {
        "stage": stage, "rows": len(frame),
        "columns": profiling.profile(frame, columns),
        "note": ("Only the columns this stage may see. A profile of a column the stage is "
                 "forbidden to use would be an invitation to use it."),
    }


@router.get("/examine/{name}", summary="The full statistical battery for one feature")
def examine(name: str, stage: str = "stage2", target: str | None = None,
            principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    from ..ingestion import schemas

    frame = loader.build_training_table()
    if name not in frame.columns:
        raise HTTPException(404, f"{name!r} is not a column in the training table")
    target = target or schemas.TARGETS.get(stage, "defaulted")
    if target not in frame.columns:
        raise HTTPException(422, f"target {target!r} is not in the training table")

    # Binary targets are only observable on disbursed files; using the whole table
    # would compute a default rate over loans that never existed.
    working = frame[frame["disbursed"].astype(bool)] if target in {"defaulted", "bt_out_24m"} \
        else frame
    contract_columns = [c for c in discovery.allowed_columns(stage) if c in working.columns]
    report = profiling.examine(working[name], working[target],
                               working[contract_columns], name=name)
    card = feature_cards.build_card(
        name, stage, report, recipe=name, inputs=[name],
        has_reason_label=discovery.has_reason_label(name),
        leaks=name not in contract_columns)
    return {"examination": report, "card": card.as_dict(),
            "research": discovery.research_needed(report), "target": target}


@router.get("/cards", summary="A feature card per contract feature, with its decision")
def cards(stage: str = "stage2", target: str | None = None,
          principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    from ..ingestion import schemas

    frame = loader.build_training_table()
    target = target or ("defaulted" if stage == "risk" else schemas.TARGETS.get(stage, "disbursed"))
    working = frame[frame["disbursed"].astype(bool)] if target in {"defaulted", "bt_out_24m"} \
        else frame
    columns = [c for c in discovery.allowed_columns(stage) if c in working.columns]
    contract = working[columns]

    built = []
    for column in columns:
        report = profiling.examine(working[column], working[target], contract, name=column)
        built.append(feature_cards.build_card(
            column, stage, report, recipe=column, inputs=[column], family="contract",
            has_reason_label=discovery.has_reason_label(column), created_by="feature_config.yaml"))

    by_decision: dict[str, int] = {}
    for card in built:
        by_decision[card.decision] = by_decision.get(card.decision, 0) + 1
    return {
        "stage": stage, "target": target,
        "cards": [c.as_dict() for c in built],
        "by_decision": by_decision,
        "lineage": feature_cards.lineage(built),
        "note": ("These are the features already in the contract, scored on the same evidence "
                 "a candidate would be. A contract feature reading 'review' or 'reject' is "
                 "not a bug - it is the standard being applied to what is already in "
                 "production. Two caveats a reader needs: the decision is **target-specific**, "
                 "so a feature weak for default can be strong for conversion and the counts "
                 "move when `target` changes; and VIF is **not** target-specific, so a high "
                 "VIF is a statement about the feature matrix that holds whichever target is "
                 "chosen."),
    }


@router.post("/propose", summary="Record a tested candidate against the governance store")
def propose(name: str = Body(..., embed=True),
            stage: str = Body("stage2", embed=True),
            principal: Principal = Depends(require_roles("model_risk", "technology"))
            ) -> dict[str, Any]:
    """Recording a proposal is not admitting a feature.

    It is the evidence an admission request is later judged against, and it
    deliberately stores the refusals too - a candidate blocked only for want of a
    label is the most interesting row in the table.
    """
    key = next((k for k in _CACHE if k.startswith(f"{stage}:")), None)
    if key is None:
        raise HTTPException(409, f"no discovery run for {stage!r} in this process. "
                                 f"POST /features/discover first - a feature is proposed with "
                                 f"its evidence or not at all")
    report = _CACHE[key]["report"]
    pool = report["top_by_signal"] + report["awaiting_a_reason_label"] + report["admissible"]
    candidate = next((c for c in pool if c["name"] == name), None)
    if candidate is None:
        raise HTTPException(404, f"{name!r} is not among the tested candidates for {stage!r}")

    dataset_id = None
    try:
        from ..governance import model_registry, store

        row = model_registry.champion(stage)
        if row:
            dataset_id = ((store.decode(row.get("card")) or {}).get("dataset") or {}).get("dataset_id")
    except Exception:                                           # noqa: BLE001
        pass

    record = feature_registry.propose(candidate, stage=stage, proposed_by=principal.name,
                                      dataset_id=dataset_id)
    return {"proposal": record,
            "next": (f"Admission is a separate, approved change. Raise it with "
                     f"POST /governance/approvals as artefact_type='feature', "
                     f"artefact_id='{stage}:{name}'."
                     if not candidate["refusals"] else
                     f"This candidate was refused: "
                     f"{', '.join(r['code'] for r in candidate['refusals'])}. "
                     f"The refusal is recorded with the proposal; an admission request will "
                     f"fail its gates until the cause is resolved.")}
