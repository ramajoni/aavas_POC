#!/usr/bin/env python3
"""Run the agentic AutoML search for each classification contract, into MLflow.

    python scripts/run_automl.py                        # all four, tracked
    python scripts/run_automl.py --contracts risk       # one
    python scripts/run_automl.py --budget 4 --fast      # a quick pass
    python scripts/run_automl.py --no-tracking          # no MLflow writes
    python scripts/run_automl.py --status               # where tracking points

What this does **not** do is promote anything. The search selects a champion and
records why; the artefacts under ``models/`` still come from
``scripts/train_all.py``. Promotion changes the priced path and is a deliberate,
separate decision - see docs/target-architecture-gap.md section 4.

Only the four classification contracts are searched. ``stage1`` and ``stage2``
predict a rate, and this search is built on AUC, KS, ECE, PSI and calibration -
every one of which is defined for a probability. A run labelled `stage2` in an
older store is a demonstration run against the conversion target, not a search of
the rate model.

Each contract is fed exactly what its trainer feeds its own model: the same row
filter, the same frame builder, the same feature contract. Anything else would
compare an AutoML candidate against a champion fitted on a different feature
space.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aavas.config import load  # noqa: E402
from aavas.features import sales_features, stage2_features  # noqa: E402
from aavas.features.feature_pipeline import FeatureTransformer, spec  # noqa: E402
from aavas.ingestion import loader  # noqa: E402
from aavas.ingestion.schemas import TARGETS  # noqa: E402
from aavas.logging_utils import get_logger  # noqa: E402
from aavas.paths import MLRUNS_DIR, REPORTS_DIR, ensure_dirs  # noqa: E402

log = get_logger("run_automl")

REPORT_DIR = REPORTS_DIR / "automl"

# Age bands for the fairness gate. Age is a numeric pricing feature in four of
# the six contracts (docs/target-architecture-gap.md section 1), which is exactly
# why the gate is computed over it rather than over a business segment.
AGE_BANDS = [(0, 30, "under 30"), (30, 40, "30-39"), (40, 50, "40-49"),
             (50, 60, "50-59"), (60, 200, "60 and over")]


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _disbursed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["disbursed"].astype(bool)].copy()


def _override_build(df: pd.DataFrame) -> pd.DataFrame:
    return sales_features.build(df, reference_rate_col="ai_recommended_rate")


# rows / build / target, copied from each trainer rather than re-derived.
CONTRACTS: dict[str, dict] = {
    "conversion": {"rows": _identity, "build": _identity, "target": TARGETS["conversion"]},
    "prepayment": {"rows": _disbursed, "build": _identity, "target": TARGETS["prepayment"]},
    "risk": {"rows": _disbursed, "build": stage2_features.build, "target": TARGETS["risk"]},
    "override": {"rows": _identity, "build": _override_build,
                 "target": TARGETS["override_flag"]},
}


def age_band(series: pd.Series) -> pd.Series:
    ages = pd.to_numeric(series, errors="coerce")
    out = pd.Series("unknown", index=series.index, dtype=object)
    for low, high, label in AGE_BANDS:
        out[(ages >= low) & (ages < high)] = label
    return out


def fairness_groups(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    """The axis the fairness gate is computed over."""
    if column == "age_band":
        if "applicant_age" not in frame.columns:
            return None
        return age_band(frame["applicant_age"]).to_numpy()
    return frame[column].astype(str).to_numpy() if column in frame.columns else None


def _clean_boundary(fit_df: pd.DataFrame, valid_df: pd.DataFrame,
                    time_column: str = "application_date"
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move every row sharing the boundary date onto the validation side.

    ``loader.train_test_split`` cuts by row position, and a busy day lands on
    both sides of that cut. For the trainers that is immaterial; for the split a
    champion is *selected* on it is a day of the future leaking into the fit, so
    the boundary date is made to belong to one side only.
    """
    if time_column not in fit_df.columns or valid_df.empty:
        return fit_df, valid_df
    boundary = valid_df[time_column].min()
    straddling = fit_df[time_column] >= boundary
    if not straddling.any():
        return fit_df, valid_df
    moved = fit_df[straddling]
    log.debug("moved %s row(s) dated %s onto the validation side", len(moved), boundary)
    return (fit_df[~straddling].copy(),
            pd.concat([moved, valid_df]).sort_values(time_column).copy())


def prepare(contract: str, table: pd.DataFrame, *, valid_size: float,
            fairness_group: str, unit_column: str) -> dict:
    """Fit / validation frames for one contract, split out of time twice.

    The holdout the trainers use is carved off first and never handed to the
    search: a champion chosen on the same rows it will later be measured on has
    not been measured at all. What remains is split again, earlier rows to fit,
    later rows to validate.
    """
    setup = CONTRACTS[contract]
    rows = setup["rows"](table)
    # The holdout keeps the trainers' own convention, so it is the same holdout
    # they held out. The selection split is stricter - see _clean_boundary.
    dev_df, holdout_df = loader.train_test_split(rows)
    fit_df, valid_df = _clean_boundary(*loader.train_test_split(dev_df, test_size=valid_size))

    feature_spec = spec(contract)
    transformer = FeatureTransformer(feature_spec)
    # Fitted on the fit slice only. Fitting it on dev would leak the validation
    # rows' medians and categories into the features the search is scored on.
    X_fit = transformer.fit_transform(setup["build"](fit_df))
    X_valid = transformer.transform(setup["build"](valid_df))

    y_fit = fit_df[setup["target"]].astype(int).to_numpy()
    y_valid = valid_df[setup["target"]].astype(int).to_numpy()

    units = (valid_df[unit_column].astype(str).to_numpy()
             if unit_column in valid_df.columns else None)

    return {"X_fit": X_fit, "y_fit": y_fit, "X_valid": X_valid, "y_valid": y_valid,
            "groups": fairness_groups(valid_df, fairness_group), "units": units,
            "n_holdout": len(holdout_df), "features": len(feature_spec.columns),
            "window": {"fit_to": str(fit_df["application_date"].max()),
                       "valid_from": str(valid_df["application_date"].min()),
                       "holdout_from": str(holdout_df["application_date"].min())}}


def run_one(contract: str, table: pd.DataFrame, args: argparse.Namespace) -> dict:
    from aavas.automl import AutoMLSearch, SearchConfig

    cfg = SearchConfig.from_yaml(contract)
    if args.budget:
        cfg.hpo_budget = args.budget
    if args.fast:
        cfg.hpo_budget = min(cfg.hpo_budget, 3)
        cfg.ensemble_methods = []
        cfg.calibration_methods = ["platt", "isotonic"]

    automl_defaults = load("automl").get("defaults", {})
    data = prepare(contract, table,
                   valid_size=args.valid_size,
                   fairness_group=args.fairness_group or automl_defaults.get(
                       "fairness_group", "age_band"),
                   unit_column=(automl_defaults.get("locality", {}) or {}).get(
                       "unit_column", "branch_id"))

    log.info("[%s] fit=%s valid=%s features=%s | fairness groups=%s | units=%s",
             contract, len(data["y_fit"]), len(data["y_valid"]), data["features"],
             "none" if data["groups"] is None else len(set(data["groups"])),
             "none" if data["units"] is None else len(set(data["units"])))

    result = AutoMLSearch(cfg).run(
        data["X_fit"], data["y_fit"], data["X_valid"], data["y_valid"],
        groups_valid=data["groups"], units_valid=data["units"])

    payload = result.as_dict()
    # The gate failures name a number ("calibration gap 0.054 exceeds 0.05") but
    # not which group carries it, and that is the half a reviewer acts on.
    leader = result.selection.champion or result.selection.best_referred
    if leader is not None:
        payload["diagnostics"] = {
            "candidate": leader.name,
            "admitted": result.selection.champion is not None,
            "fairness": leader.fairness,
            "locality": leader.locality,
        }
    payload["data"] = {"n_fit": int(len(data["y_fit"])), "n_valid": int(len(data["y_valid"])),
                       "n_holdout_untouched": data["n_holdout"],
                       "features": data["features"],
                       "positive_rate_fit": round(float(np.mean(data["y_fit"])), 4),
                       "positive_rate_valid": round(float(np.mean(data["y_valid"])), 4),
                       **data["window"]}
    payload["config"] = {"hpo_strategy": cfg.hpo_strategy, "hpo_budget": cfg.hpo_budget,
                         "imbalance_strategy": cfg.imbalance_strategy,
                         "monotone_required": cfg.monotone_required,
                         "calibration_methods": list(cfg.calibration_methods),
                         "ensemble_methods": list(cfg.ensemble_methods)}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{contract}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info("[%s] search report -> %s", contract, out)
    return payload


def summarise(contract: str, payload: dict) -> str:
    board = payload.get("leaderboard") or []
    lines = ["", "=" * 96,
             f"{contract.upper():<12} champion: {payload.get('champion') or 'NONE ADMITTED'}"
             f"   ({payload['candidates_considered']} considered, "
             f"{payload['elapsed_seconds']}s)",
             "=" * 96,
             f"  {'#':>2}  {'candidate':<28} {'family':<18} {'auc':>7} {'ks':>7} "
             f"{'ece':>7} {'psi':>7}  gates"]
    for i, row in enumerate(board[:8], 1):
        failures = [r["failures"] for r in payload.get("referred_to_review", [])
                    if r["name"] == row["name"]]
        verdict = "; ".join(failures[0]) if failures else "passed"
        lines.append(f"  {i:>2}  {row['name']:<28} {str(row.get('family','')):<18} "
                     f"{row.get('auc', float('nan')):>7.4f} {row.get('ks', float('nan')):>7.4f} "
                     f"{row.get('ece', float('nan')):>7.4f} {row.get('psi', float('nan')):>7.4f}"
                     f"  {verdict[:44]}")
    data = payload["data"]
    lines.append(f"  fit {data['n_fit']:,} rows to {data['fit_to'][:10]} | "
                 f"valid {data['n_valid']:,} from {data['valid_from'][:10]} | "
                 f"holdout {data['n_holdout_untouched']:,} untouched")
    skipped = payload.get("candidates_skipped") or []
    if skipped:
        lines.append(f"  skipped: " + ", ".join(f"{s['name']} ({s['reason']})"
                                                for s in skipped[:4]))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contracts", nargs="*", default=list(CONTRACTS),
                        choices=list(CONTRACTS), help="contracts to search (default: all)")
    parser.add_argument("--budget", type=int, default=None, help="override the HPO budget")
    parser.add_argument("--valid-size", type=float, default=0.25,
                        help="share of the development window held back to validate on")
    parser.add_argument("--fairness-group", default=None,
                        help="column the fairness gate is computed over "
                             "(age_band, segment, city, channel)")
    parser.add_argument("--fast", action="store_true",
                        help="small HPO budget, no ensembles - for a smoke run")
    parser.add_argument("--no-tracking", dest="tracking", action="store_false",
                        help="do not write to MLflow")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow URI (default: the absolute file store at mlruns/)")
    parser.add_argument("--status", action="store_true",
                        help="print where tracking points, then exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    ensure_dirs()

    # Tracking is off by default everywhere in this repo; this script is the one
    # place that turns it on, because a search nobody can audit later is a search
    # that did not happen. The URI is made absolute so the run lands in the same
    # store the admin view and `mlflow ui` read, whatever the working directory.
    if args.tracking:
        os.environ.setdefault("AAVAS_TRACKING_ENABLED", "1")
        os.environ.setdefault("MLFLOW_TRACKING_URI",
                              args.tracking_uri or f"file:{MLRUNS_DIR}")
    else:
        os.environ["AAVAS_TRACKING_ENABLED"] = "false"

    from aavas.automl import tracking

    status = tracking.status()
    log.info("tracking: enabled=%s uri=%s experiment=%s mlflow=%s",
             status["tracking_enabled"], status["tracking_uri"],
             status["experiment"], status["mlflow_version"] or "not installed")
    if args.status:
        print(json.dumps(status, indent=2))
        return 0
    if args.tracking and not status["tracking_enabled"]:
        log.warning("tracking was requested but is not active; the search will still run, "
                    "unrecorded. Install mlflow to record it.")

    table = loader.load_training_table()
    summaries, failures = {}, {}
    for contract in args.contracts:
        try:
            summaries[contract] = run_one(contract, table, args)
        except Exception as exc:                                # noqa: BLE001
            failures[contract] = f"{type(exc).__name__}: {exc}"
            log.error("[%s] search failed: %s", contract, exc, exc_info=True)

    for contract, payload in summaries.items():
        print(summarise(contract, payload))

    print("\n" + "=" * 96)
    print("AUTOML SEARCH")
    print("=" * 96)
    for contract in args.contracts:
        if contract in failures:
            print(f"  {contract:<12} FAILED   {failures[contract]}")
        else:
            payload = summaries[contract]
            print(f"  {contract:<12} champion {payload.get('champion') or '-':<26} "
                  f"report reports/automl/{contract}.json")
    if args.tracking and status["tracking_enabled"]:
        print(f"\n  Recorded in MLflow: {status['tracking_uri']} "
              f"(experiment {status['experiment']})")
        print("  Admin view: #/mlflow -> AutoML searches")
    print("\n  Nothing was promoted. models/ still holds what scripts/train_all.py trained.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
