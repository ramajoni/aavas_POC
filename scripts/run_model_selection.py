#!/usr/bin/env python3
"""Run the AutoML portfolio selection for the deck models, or their global weights.

    python scripts/run_model_selection.py                     # every model, config budgets
    python scripts/run_model_selection.py --models stage2 risk
    python scripts/run_model_selection.py --budget 4          # trials per family
    python scripts/run_model_selection.py --fast              # selection.fast_budget per family
    python scripts/run_model_selection.py --weights           # global SHAP weights, no search

The search (``aavas.automl.portfolio``) writes one report per model to
``reports/automl/portfolio/<model>.json`` and points ``selection.json`` at the
winner. ``make train`` then fits exactly that family and those params - the
trainers read ``selection.json`` through ``portfolio.build``. This script does
not train anything into ``models/``.

``--weights`` is the other half of the deck (slide 12): each feature's share of
mean |SHAP| on the untouched holdout, computed from the *trained* artefacts in
``models/`` (or ``AAVAS_MODEL_DIR``), written to ``global_weights.json``. Run it
after retraining, so the weights describe the models that price.

Models: stage1, stage2, risk and conversion always; ``lgd`` when
``aavas.models.lgd.train`` exposes ``selection_data(df)``; ``uplift`` when
``aavas.models.uplift.train`` exposes ``search(df, budget)`` returning a report
in the same schema. A model whose module is not there yet is skipped with a
message, not a traceback.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Idle OpenMP threads sleep instead of spinning. On a shared machine a spinning
# pool waiting at a barrier for a descheduled sibling burns the cores the other
# fits need; measured here, it turned seconds into minutes. Set before numpy and
# the learners load their OpenMP runtimes; an explicit setting wins.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aavas.automl import portfolio  # noqa: E402
from aavas.config import model_config  # noqa: E402
from aavas.ingestion import loader  # noqa: E402
from aavas.ingestion.schemas import TARGETS  # noqa: E402
from aavas.logging_utils import get_logger  # noqa: E402

log = get_logger("run_model_selection")

MODELS = ("stage1", "stage2", "risk", "conversion", "lgd", "uplift")
WEIGHT_MODELS = tuple((model_config().get("portfolio") or {}).get(
    "weights", ("risk", "stage2", "conversion", "lgd")))
WEIGHTS_FILE = portfolio.REPORT_DIR / "global_weights.json"


# ------------------------------------------------------------------- data
def training_table(models: list[str]) -> pd.DataFrame:
    """The trainers' table. Rebuilt in memory (never written) when the cached
    processed table predates a target a requested model needs - realized_lgd
    arrived with the LGD model, after some caches were written."""
    table = loader.load_training_table()
    needed = [TARGETS[m] for m in models if m in TARGETS]
    missing = [c for c in needed if c not in table.columns]
    if missing:
        log.warning("cached training table lacks %s; rebuilding it in memory from the raw "
                    "data (the cache is left as it is)", missing)
        table = loader.build_training_table(save=False)
    return table


def _trainer(name: str):
    try:
        return importlib.import_module(f"aavas.models.{name}.train"), None
    except ImportError as exc:
        return None, f"aavas.models.{name}.train is not importable ({exc})"


# ----------------------------------------------------------------- search
def search_one(name: str, table: pd.DataFrame, args: argparse.Namespace) -> dict | str:
    """The report, or a string saying why the model was skipped."""
    module, why = _trainer(name)
    if module is None:
        return why
    if name == "uplift":
        if not hasattr(module, "search"):
            return "aavas.models.uplift.train exposes no search(df, budget) yet"
        budget = args.budget or (4 if args.fast else
                                 (model_config().get("uplift") or {}).get("search_budget"))
        report = module.search(table, budget)
    else:
        if not hasattr(module, "selection_data"):
            return f"aavas.models.{name}.train exposes no selection_data(df) yet"
        data = module.selection_data(table)
        log.info("[%s] fit %s rows to %s | valid %s rows %s..%s | holdout %s rows from %s "
                 "(not shown to the search)", name,
                 data.window["fit"]["rows"], data.window["fit"]["to"],
                 data.window["valid"]["rows"], data.window["valid"]["from"],
                 data.window["valid"]["to"], data.window.get("holdout", {}).get("rows"),
                 data.window.get("holdout", {}).get("from"))
        report = portfolio.search(data, budget=args.budget, fast=args.fast)
    portfolio.record(report)
    return report


def _metric_text(report: dict) -> str:
    selected = report.get("selected") or {}
    validation = selected.get("validation") or {}
    metric = report.get("metric", "")
    value = validation.get(metric)
    if value is None:
        return "-"
    units = report.get("units", "")
    name = metric.replace(f"_{units}", "") if units else metric
    text = f"{name.upper() if name in ('auc', 'mae', 'rmse') else name} {value:.4g}"
    if units and metric.endswith(units):
        text += f" {units}"
    if "coverage_pct" in validation:
        text += f" (coverage {validation['coverage_pct']:.1f}%)"
    return text


def deck_table(reports: dict[str, dict | str], elapsed: float) -> str:
    rows = [("Model", "Target", "Families", "Trials", "Selected model", "Validation metric")]
    for name, report in reports.items():
        if isinstance(report, str):
            rows.append((name, "-", "-", "-", "skipped", report[:60]))
            continue
        selected = report.get("selected") or {}
        rows.append((name, report.get("target") or "-",
                     str(len(report.get("families_considered") or [])),
                     str(report.get("trials", "-")),
                     selected.get("label") or portfolio.label(selected.get("family", "-")),
                     _metric_text(report)))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))  # noqa: E731
    out = ["", "=" * 100, "AUTOML MODEL SELECTION - out of time; selection rule: best validation "
           "score, then explainability", "=" * 100, line(rows[0]),
           line(tuple("-" * w for w in widths))]
    out += [line(r) for r in rows[1:]]
    done = [r for r in reports.values() if isinstance(r, dict)]
    if done:
        windows = done[0].get("windows") or {}
        holdout = windows.get("holdout") or {}
        out.append("")
        out.append(f"  untouched holdout: {holdout.get('from')} to {holdout.get('to')} "
                   f"(never shown to the search); validation = later "
                   f"{portfolio.settings()['valid_size']:.0%} of each training window")
        for report in done:
            rule = report.get("rule") or {}
            skipped = report.get("families_skipped") or []
            out.append(f"  {report['model']:<11} decided by {rule.get('decided_by', '-')}; "
                       f"within tolerance: {', '.join(rule.get('within_tolerance') or [])}"
                       + ("; skipped: " + ", ".join(f"{s['family']} ({s['kind']})" for s in skipped)
                          if skipped else ""))
    out.append(f"\n  total runtime {elapsed / 60:.1f} min. Reports: {portfolio.REPORT_DIR}")
    return "\n".join(out)


# ---------------------------------------------------------------- weights
def weights(table: pd.DataFrame, models: list[str], top: int, out_path: Path | None = None) -> dict:
    """Global SHAP weights for each trained artefact, on its untouched holdout."""
    from aavas.models import base
    from aavas.xai.shap_explainer import PricingExplainer, global_weights

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "definition": "share of mean |SHAP| per feature on the trainer's untouched holdout "
                         "(the last 20% by application date), from the trained artefact",
           "models": {}, "skipped": {}}
    for name in models:
        module, why = _trainer(name)
        if module is None:
            out["skipped"][name] = why
            continue
        if not base.exists(name):
            out["skipped"][name] = f"no trained artefact at {base.artifact_path(name)}"
            continue
        if not (hasattr(module, "split") and hasattr(module, "features")):
            out["skipped"][name] = (f"aavas.models.{name}.train exposes no split(df) / "
                                    "features(frame) to rebuild its holdout")
            continue
        payload, card = base.load(name)
        model, transformer, explained = _explainable(payload)
        if model is None or transformer is None:
            out["skipped"][name] = ("artefact payload carries neither 'model' + 'transformer' nor "
                                    "'surrogate' + 'estimator.transformer'")
            continue
        _, holdout = module.split(table)
        X = transformer.transform(module.features(holdout))
        started = time.perf_counter()
        explainer = PricingExplainer(model, transformer,
                                     is_classifier=hasattr(model, "predict_proba"))
        result = global_weights(explainer, X)
        dates = pd.to_datetime(holdout[model_config()["data"]["time_column"]])
        out["models"][name] = {
            "model": name,
            "artefact": {"version": card.version, "learner": card.learner,
                         "trained_at": card.trained_at,
                         "selection": getattr(card, "selection", None)},
            "holdout": {"rows": int(len(X)), "from": str(dates.min().date()),
                        "to": str(dates.max().date())},
            "explained": explained,
            **{k: v for k, v in result.items() if k != "features"},
            "top": result["features"][:top],
            "features": result["features"],
            "seconds": round(time.perf_counter() - started, 1),
        }
        log.info("[%s] %s on %s holdout rows: %s", name, result["engine"], len(X),
                 ", ".join(f"{f['feature']} {f['share_pct']:.1f}%" for f in result["features"][:top]))
    out["path"] = str(out_path or WEIGHTS_FILE)
    portfolio._write_json(Path(out["path"]), out)
    return out


def _explainable(payload: dict) -> tuple[object, object, str]:
    """(model, transformer, what is explained) from an artefact payload.

    Most artefacts carry the fitted model and its transformer. The uplift model
    is a meta-learner over a propensity adjustment with no single tree to
    explain; it ships a shallow surrogate of the served uplift (in acceptance
    points) for exactly this, fitted on its own transformer's features.
    """
    if payload.get("model") is not None and payload.get("transformer") is not None:
        return payload["model"], payload["transformer"], "the fitted model"
    estimator = payload.get("estimator")
    if payload.get("surrogate") is not None and getattr(estimator, "transformer", None) is not None:
        return (payload["surrogate"], estimator.transformer,
                "the surrogate of the served uplift at the reference discount (acceptance points)")
    return None, None, ""


def weights_table(result: dict, top: int) -> str:
    lines = ["", "=" * 100, f"GLOBAL WEIGHTS - top {top} features by share of mean |SHAP| on the "
             "untouched holdout", "=" * 100]
    for name, entry in result["models"].items():
        lines.append(f"  {name:<11} {entry['artefact']['learner']} ({entry['engine']}, "
                     f"{entry['units']}, {entry['holdout']['rows']} rows)")
        for row in entry["top"]:
            lines.append(f"      {row['rank']}. {row['label']:<32} {row['share_pct']:>5.1f}%")
    for name, why in result["skipped"].items():
        lines.append(f"  {name:<11} skipped: {why}")
    lines.append(f"\n  -> {result['path']}")
    return "\n".join(lines)


# ------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=None, choices=list(MODELS),
                        help="models to search (default: all); with --weights, models to weigh")
    parser.add_argument("--budget", type=int, default=None,
                        help="trials per family (default: model_config.selection)")
    parser.add_argument("--fast", action="store_true",
                        help="selection.fast_budget trials per family - a smoke run")
    parser.add_argument("--weights", action="store_true",
                        help="compute global SHAP weights from the trained artefacts instead of "
                             "searching")
    parser.add_argument("--out", type=Path, default=None,
                        help="with --weights: write here instead of global_weights.json (for a "
                             "candidate registry set with AAVAS_MODEL_DIR)")
    parser.add_argument("--top", type=int, default=None,
                        help="features per model in the weights table (default: "
                             "model_config.portfolio.top_weights)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    if args.weights:
        models = args.models or list(WEIGHT_MODELS)
        top = args.top or int((model_config().get("portfolio") or {}).get("top_weights", 5))
        result = weights(training_table(models), models, top, args.out)
        print(weights_table(result, top))
        return 0

    models = args.models or list(MODELS)
    table = training_table(models)
    reports: dict[str, dict | str] = {}
    failures = 0
    for name in models:
        t0 = time.perf_counter()
        try:
            reports[name] = search_one(name, table, args)
        except Exception as exc:                                  # noqa: BLE001
            failures += 1
            reports[name] = f"FAILED: {type(exc).__name__}: {exc}"
            log.error("[%s] selection failed: %s", name, exc, exc_info=True)
        if isinstance(reports[name], str):
            log.warning("[%s] %s", name, reports[name])
        else:
            log.info("[%s] done in %.1fs -> %s", name, time.perf_counter() - t0,
                     portfolio.report_path(name))
    print(deck_table(reports, time.perf_counter() - started))
    print("\n  selection.json:", json.dumps({k: v.get("family") for k, v in
                                             portfolio.selections().items()}))
    print("  Nothing was trained. `make train` fits these selections; then run "
          "`scripts/run_model_selection.py --weights`.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
