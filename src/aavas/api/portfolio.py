"""The deck's model portfolio, with the evidence behind each model.

    GET /models/portfolio

One entry per deck model, in deck order (``model_config.portfolio``): what it is
for, which repo model(s) implement it, how AutoML chose the learner (families,
trials, the selected configuration and its out-of-time validation score), what
the trained artefact scores on the untouched holdout, and which features carry
its weight (share of mean |SHAP|). The rule components (yield optimisation,
policy guardrails) and the supporting models sit beside them.

Everything is read from files the pipeline writes - the selection reports under
``reports/automl/portfolio/``, the model cards in the registry directory, and
``global_weights.json`` - so the page shows the latest real run. When one of them
is missing or out of step with another (a selection the artefact was not trained
on, weights computed on an older artefact) the entry says so in ``issues``
rather than filling the gap: a portfolio view that only ever showed green would
be decoration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter

from ..automl import portfolio as selection
from ..config import model_config, policy
from ..features.feature_pipeline import reason_label
from ..logging_utils import get_logger
from ..models import base as model_base

log = get_logger(__name__)
router = APIRouter(tags=["models"])

WEIGHTS_FILE = selection.REPORT_DIR / "global_weights.json"


def _read(path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _card(name: str) -> dict[str, Any] | None:
    return _read(model_base.artifact_path(name, ".card.json"))


def _headline(name: str, card: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, float]:
    if not card:
        return {}
    metrics = card.get("metrics") or {}
    wanted = [k for k in (cfg.get("headline_metrics") or {}).get(name, []) if k in metrics]
    keys = wanted or list(metrics)[:3]
    return {k: metrics[k] for k in keys}


def _automl(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    chosen = report.get("selected") or {}
    rule = report.get("rule") or {}
    metric = report.get("metric")
    return {
        "task": report.get("task"),
        "target": report.get("target"),
        "framework": report.get("framework"),
        "families": report.get("families_considered") or [],
        "families_skipped": report.get("families_skipped") or [],
        "trials": report.get("trials"),
        "trials_failed": report.get("trials_failed", 0),
        "selected": {"family": chosen.get("family"),
                     "label": chosen.get("label") or selection.label(chosen.get("family") or ""),
                     "params": chosen.get("params"), "origin": chosen.get("origin")},
        "metric": metric,
        "higher_is_better": report.get("higher_is_better"),
        "units": report.get("units"),
        "value": (chosen.get("validation") or {}).get(metric),
        "validation": chosen.get("validation") or {},
        "best_per_family": [{"family": b.get("family"), "label": selection.label(b.get("family", "")),
                             "value": (b.get("metrics") or {}).get(metric),
                             "explainability": b.get("explainability")}
                            for b in report.get("best_per_family") or []],
        "decided_by": rule.get("decided_by"),
        "within_tolerance": rule.get("within_tolerance") or [],
        "windows": report.get("windows") or {},
        "elapsed_seconds": report.get("elapsed_seconds"),
        "generated_at": report.get("generated_at"),
        "report": str(selection.report_path(report.get("model", "")).relative_to(
            selection.REPORT_DIR.parents[2])),
    }


def _weights(name: str, card: dict[str, Any] | None, weights: dict[str, Any] | None,
             top: int) -> tuple[dict[str, Any] | None, list[str]]:
    entry = ((weights or {}).get("models") or {}).get(name)
    if not entry:
        skipped = ((weights or {}).get("skipped") or {}).get(name)
        return None, [f"global weights not computed ({skipped})" if skipped else
                      "global weights not computed: run scripts/run_model_selection.py --weights"]
    issues = []
    artefact = entry.get("artefact") or {}
    if card and artefact.get("trained_at") != card.get("trained_at"):
        issues.append("global weights were computed on an earlier artefact "
                      f"({artefact.get('learner')}, {artefact.get('trained_at')}); rerun "
                      "scripts/run_model_selection.py --weights")
    rows = entry.get("top") or entry.get("features") or []
    return {"method": entry.get("method"), "engine": entry.get("engine"),
            "units": entry.get("units"), "fallback_reason": entry.get("fallback_reason"),
            "holdout": entry.get("holdout"), "artefact": artefact,
            "generated_at": (weights or {}).get("generated_at"),
            "top": [{"feature": r["feature"],
                     "label": r.get("label") or reason_label(r["feature"])["label"],
                     "share_pct": r.get("share_pct"), "rank": r.get("rank")}
                    for r in rows[:top]]}, issues


def _component(name: str, cfg: dict[str, Any], weights: dict[str, Any] | None,
               with_weights: bool) -> dict[str, Any]:
    report = selection.load_report(name)
    card = _card(name)
    automl = _automl(report)
    if automl is not None:
        automl["metric_text"] = _metric_text(automl)
    issues: list[str] = []
    if automl is None:
        issues.append("no AutoML selection report: run scripts/run_model_selection.py")
    if card is None:
        issues.append(f"no trained artefact ({model_base.artifact_path(name).name}): run make train")
    elif automl and card.get("learner") != automl["selected"]["family"]:
        issues.append(f"the trained artefact is {card.get('learner')} but the selection is "
                      f"{automl['selected']['family']}: retrain to fit the selection")
    top = int(cfg.get("top_weights", 5))
    weight_view = None
    if with_weights:
        weight_view, weight_issues = _weights(name, card, weights, top)
        issues += weight_issues
    status = "ready" if not issues else ("missing" if automl is None and card is None else "partial")
    return {
        "model": name,
        "role": (cfg.get("roles") or {}).get(name),
        "status": status,
        "issues": issues,
        "target": (automl or {}).get("target") or (card or {}).get("target"),
        "automl": automl,
        "card": None if card is None else {
            "version": card.get("version"), "learner": card.get("learner"),
            "learner_label": selection.label(card.get("learner") or ""),
            "trained_at": card.get("trained_at"), "n_train": card.get("n_train"),
            "n_test": card.get("n_test"), "holdout_metrics": _headline(name, card, cfg),
            "selection": card.get("selection"),
            "training_window": card.get("training_window")},
        "weights": weight_view,
    }


def _business() -> dict[str, Any]:
    """The cards in business terms: each segment's risk and starting rate, its
    acceptance curve, what a discount buys, and the fixed parts of the floor.
    One failing piece leaves the others standing."""
    from ..analytics import segment_view

    out: dict[str, Any] = {}
    for key, build in (("segments", segment_view.segments),
                       ("acceptance_curves", segment_view.acceptance_curves),
                       ("discount_effect", segment_view.discount_effect),
                       ("floor", segment_view.floor_components)):
        try:
            out[key] = build()
        except Exception as exc:                                   # noqa: BLE001
            log.warning("model cards: %s unavailable: %s", key, exc)
            out[key] = None
    return out


def _setting(path: str) -> Any:
    node: Any = policy()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _metric_text(automl: dict[str, Any] | None) -> str | None:
    if not automl or automl.get("value") is None:
        return None
    metric, units = automl["metric"] or "", automl.get("units") or ""
    name = metric[: -len(units) - 1] if units and metric.endswith(units) else metric
    text = f"{name.upper() if name in ('auc', 'mae', 'rmse') else name} {automl['value']:.4g}"
    return text + (f" {units}" if units and metric.endswith(units) else "")


@router.get("/models/portfolio",
            summary="The five deck models, with AutoML selection, holdout metrics and global weights")
def portfolio_view() -> dict[str, Any]:
    cfg = model_config().get("portfolio") or {}
    weights = _read(WEIGHTS_FILE)
    weighted = set(cfg.get("weights") or [])
    deck = []
    for entry in cfg.get("models") or []:
        components = [_component(name, cfg, weights, name in weighted)
                      for name in entry.get("models") or []]
        statuses = {c["status"] for c in components}
        deck.append({
            "key": entry.get("key"), "title": entry.get("title"), "stage": entry.get("stage"),
            "question": entry.get("question"), "models": entry.get("models") or [],
            "status": ("ready" if statuses == {"ready"} else
                       "missing" if statuses == {"missing"} else "partial"),
            "components": components,
        })

    table = [{"deck_model": m["title"], "model": c["model"], "role": c["role"],
              "target": c["target"], "families": len(c["automl"]["families"]),
              "family_names": [selection.label(f) for f in c["automl"]["families"]],
              "trials": c["automl"]["trials"], "selected": c["automl"]["selected"]["label"],
              "metric": c["automl"]["metric"], "value": c["automl"]["value"],
              "metric_text": c["automl"]["metric_text"], "decided_by": c["automl"]["decided_by"]}
             for m in deck for c in m["components"] if c["automl"]]

    rules = [{"key": r.get("key"), "title": r.get("title"), "question": r.get("question"),
              "kind": "rule", "source": r.get("source"),
              "settings": [{"key": k, "value": _setting(k)} for k in r.get("settings") or []]}
             for r in cfg.get("rules") or []]
    supporting = [{**_component(name, cfg, weights, name in weighted), "kind": "supporting"}
                  for name in cfg.get("supporting") or []]
    for item in supporting:
        # Supporting models are not part of the portfolio search; the missing
        # report is expected, not an issue.
        item["issues"] = [i for i in item["issues"] if "AutoML selection report" not in i]
        item["status"] = "ready" if not item["issues"] else "partial"

    any_report = next((c["automl"] for m in deck for c in m["components"] if c["automl"]), None)
    settings = selection.settings()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": deck,
        "automl_table": table,
        "rules": rules,
        "supporting": supporting,
        "selection_rule": {
            "statement": "Best validation score, then explainability.",
            "tolerance_rel": settings.get("tolerance_rel"),
            "explainability_ranking": list(selection.EXPLAINABILITY),
            "validation": (f"out of time: the later {settings.get('valid_size', 0.25):.0%} of each "
                           "model's training window; the trainers' holdout (last 20% by "
                           "application date) is never shown to the search"),
            "holdout": (any_report or {}).get("windows", {}).get("holdout"),
            "framework": (any_report or {}).get("framework"),
        },
        "weights_generated_at": (weights or {}).get("generated_at"),
        "business": _business(),
    }
