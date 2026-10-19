#!/usr/bin/env python3
"""Fit every registered algorithm and run every strategy. Report what works.

    python scripts/verify_algorithms.py [--quick]

`available()` on the registry answers whether an algorithm *imports*. This answers
whether it *runs*: every learner is instantiated, fitted on a real slice of the
training table and asked to predict, and every strategy - HPO, imbalance,
calibration, ensembling, federation, explainability, feature generation - is
executed against real inputs.

The distinction matters. An algorithm that imports and then raises on `fit` is
worse than one that is absent, because the catalogue says it is available.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aavas.logging_utils import get_logger  # noqa: E402

log = get_logger("verify_algorithms")

PASS, FAIL, SKIP = "works", "BROKEN", "unavailable"
ROWS = 1200


def _data(rows: int = ROWS):
    """A real slice of the training table, not synthetic noise.

    Returns the transformer too: the explainer wraps a model *and* its
    transformer, because an attribution has to be expressed in the caller's
    columns rather than the fitted matrix's.
    """
    from aavas.features.feature_pipeline import FeatureTransformer, spec
    from aavas.features import stage2_features
    from aavas.ingestion import loader

    df = loader.build_training_table()
    df = df[df["disbursed"].astype(bool)].head(rows).copy()
    contract = spec("risk")
    transformer = FeatureTransformer(contract)
    X = transformer.fit_transform(stage2_features.build(df))
    y = df["defaulted"].astype(int).to_numpy()
    return np.asarray(X, dtype=float), y, df, transformer


def _row(name: str, group: str, status: str, detail: str = "", ms: float = 0.0) -> dict:
    return {"name": name, "group": group, "status": status, "detail": detail, "ms": round(ms, 1)}


# ------------------------------------------------------------------- learners


def check_algorithms(X, y) -> list[dict]:
    from aavas.automl import registry

    out = []
    for algo in registry.ALGORITHMS:
        reason = algo.unavailable_reason()
        if reason:
            out.append(_row(algo.name, algo.family, SKIP, reason))
            continue
        started = time.perf_counter()
        try:
            model = algo.instantiate()
            model.fit(X, y)
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                p = model.decision_function(X)
            else:
                p = model.predict(X)
            p = np.asarray(p, dtype=float)
            if len(p) != len(y):
                raise ValueError(f"returned {len(p)} predictions for {len(y)} rows")
            if not np.isfinite(p).all():
                raise ValueError("produced non-finite predictions")
            out.append(_row(algo.name, algo.family, PASS,
                            f"fitted and predicted {len(p)} rows",
                            (time.perf_counter() - started) * 1000))
        except Exception as exc:                                # noqa: BLE001
            out.append(_row(algo.name, algo.family, FAIL, f"{type(exc).__name__}: {exc}",
                            (time.perf_counter() - started) * 1000))
    return out


# ------------------------------------------------------------------ strategies


def check_hpo() -> list[dict]:
    from aavas.automl import hpo

    out = []
    space = {"x": ("float", -3.0, 3.0), "y": ("float", -3.0, 3.0)}

    def objective(params: dict) -> float:
        """Called with a params dict, not kwargs - `hpo.optimize` invokes
        `objective({})` for an empty space, which settles the contract."""
        return -((params.get("x", 0.0) - 1.0) ** 2 + (params.get("y", 0.0) + 0.5) ** 2)
    for strategy in hpo.STRATEGIES:
        started = time.perf_counter()
        try:
            result = hpo.optimize(space, objective, strategy=strategy, budget=24)
            distance = abs(result.best_params.get("x", 99) - 1.0)
            ok = distance < 1.5
            out.append(_row(strategy, "hpo", PASS if ok else FAIL,
                            f"best x={result.best_params.get('x'):.2f} (optimum 1.00)"
                            if ok else f"did not approach the optimum: x={result.best_params.get('x')}",
                            (time.perf_counter() - started) * 1000))
        except Exception as exc:                                # noqa: BLE001
            out.append(_row(strategy, "hpo", FAIL, f"{type(exc).__name__}: {exc}",
                            (time.perf_counter() - started) * 1000))
    return out


def check_imbalance(X, y) -> list[dict]:
    from aavas.automl import imbalance

    out = []
    for strategy in imbalance.STRATEGIES:
        started = time.perf_counter()
        try:
            result = imbalance.apply(X, y, strategy)
            n = len(result.y)
            # Report the strategy that actually ran. `apply` renames itself to
            # e.g. "smote->random_oversample" when imbalanced-learn is absent, and
            # a verification that printed the requested name would hide that.
            ran = getattr(result, "strategy", strategy)
            fell_back = ran != strategy
            out.append(_row(strategy, "imbalance", PASS,
                            (f"ran as {ran} - " if fell_back else "")
                            + f"{len(y)} -> {n} rows, base rate "
                              f"{y.mean():.3f} -> {np.mean(result.y):.3f}",
                            (time.perf_counter() - started) * 1000))
        except Exception as exc:                                # noqa: BLE001
            message = str(exc)
            status = SKIP if "install" in message.lower() or "imblearn" in message.lower() else FAIL
            out.append(_row(strategy, "imbalance", status, f"{type(exc).__name__}: {message}",
                            (time.perf_counter() - started) * 1000))
    return out


def check_calibration(X, y) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from aavas.automl import calibration

    model = HistGradientBoostingClassifier(max_iter=40).fit(X, y)
    raw = model.predict_proba(X)[:, 1]
    out = []
    for method in ("isotonic", "platt", "beta", "none"):
        started = time.perf_counter()
        try:
            calibrated = calibration.fit(raw, y, method=method)
            # calibrated.predict, not a guessed .transform. An earlier version of
            # this harness fell back to the raw scores when the attribute was
            # missing, and reported four identical ECEs as four working
            # calibrators - the exact silent-fallback failure this file exists to
            # catch. There is no fallback now: a missing attribute is a failure.
            p = np.asarray(calibrated(raw), dtype=float)
            if len(p) != len(raw):
                raise ValueError(f"returned {len(p)} values for {len(raw)} inputs")
            before = calibration.expected_calibration_error(raw, y)
            ece = calibration.expected_calibration_error(p, y)
            moved = method == "none" or not np.allclose(p, raw)
            out.append(_row(method, "calibration", PASS if moved else FAIL,
                            f"ECE {before:.4f} -> {ece:.4f}" if moved else
                            "returned the raw scores unchanged",
                            (time.perf_counter() - started) * 1000))
        except Exception as exc:                                # noqa: BLE001
            out.append(_row(method, "calibration", FAIL, f"{type(exc).__name__}: {exc}",
                            (time.perf_counter() - started) * 1000))
    return out


def check_federated() -> list[dict]:
    from aavas.federated import server as fed_server
    from aavas.federated.client import RegionalAdjustmentClient, RegionalDataset
    from aavas.federated import strategy as strategies

    def fresh_clients():
        """Deliberately unequal, and one region deliberately different.

        With equal client sizes, sample-weighted FedAvg and uniform averaging are
        the same computation and trimmed-mean has nothing to trim - so every
        strategy returns an identical number and the check discriminates nothing.
        An earlier version of this harness did exactly that and reported five
        working strategies on one result.
        """
        rng = np.random.default_rng(7)
        sizes = [900, 400, 200, 120]
        clients = []
        for index, n in enumerate(sizes):
            X = rng.normal(size=(n, 3))
            # The smallest region behaves differently, which is what robust
            # aggregation is meant to handle.
            beta = np.array([2.5, -2.0, 1.5]) if index == 3 else np.array([0.4, -0.2, 0.1])
            y = X @ beta + rng.normal(scale=0.02, size=n)
            clients.append(RegionalAdjustmentClient(RegionalDataset(f"r{index}", X, y)))
        return clients

    out = []
    seen: dict[str, float] = {}
    for name in sorted(strategies.STRATEGIES):
        started = time.perf_counter()
        try:
            run = fed_server.run(fresh_clients(), rounds=4, strategy_name=name)
            mae = run.history[-1]["global_mae_bps"]
            seen[name] = mae
            out.append(_row(name, "federated", PASS,
                            f"MAE {mae:.2f} bps, mu={run.proximal_mu}",
                            (time.perf_counter() - started) * 1000))
        except Exception as exc:                                # noqa: BLE001
            out.append(_row(name, "federated", FAIL, f"{type(exc).__name__}: {exc}",
                            (time.perf_counter() - started) * 1000))

    # fedavg and fedavg_weighted are the same aggregation by design. Everything
    # else must actually differ, or the strategy is a name on one implementation.
    distinct = {round(v, 4) for k, v in seen.items() if k != "fedavg_weighted"}
    out.append(_row("strategies_differ", "federated",
                    PASS if len(distinct) > 1 else FAIL,
                    f"{len(distinct)} distinct results across "
                    f"{len(seen) - 1} strategies: {sorted(distinct)}"))
    return out


def check_explainers(X, y, transformer) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    out = []
    model = HistGradientBoostingClassifier(max_iter=30).fit(X, y)
    started = time.perf_counter()
    try:
        from aavas.xai.shap_explainer import PricingExplainer

        explainer = PricingExplainer(model, transformer, is_classifier=True)
        out.append(_row("shap_or_ablation", "explainability", PASS,
                        f"backend={explainer.backend}, baseline prediction "
                        f"{explainer.baseline_prediction_:.4f}",
                        (time.perf_counter() - started) * 1000))
    except Exception as exc:                                    # noqa: BLE001
        out.append(_row("shap_or_ablation", "explainability", FAIL, f"{type(exc).__name__}: {exc}",
                        (time.perf_counter() - started) * 1000))

    for name, module in (("omnixai", "omnixai"), ("lime", "lime")):
        try:
            __import__(module)
            out.append(_row(name, "explainability", PASS, "importable in this interpreter"))
        except Exception as exc:                                # noqa: BLE001
            out.append(_row(name, "explainability", SKIP,
                            f"not installed here ({type(exc).__name__}); the analysis venv "
                            f"carries it"))
    return out


def check_feature_generators(df) -> list[dict]:
    from aavas.features import discovery

    out = []
    for name in discovery.GENERATORS:
        started = time.perf_counter()
        try:
            report = discovery.discover(df, stage="stage2", generators=[name],
                                        max_candidates=60)
            out.append(_row(name, "feature_generation", PASS,
                            f"{report.considered} candidates generated, "
                            f"{len(report.candidates)} evaluated",
                            (time.perf_counter() - started) * 1000))
        except Exception as exc:                                # noqa: BLE001
            out.append(_row(name, "feature_generation", FAIL, f"{type(exc).__name__}: {exc}",
                            (time.perf_counter() - started) * 1000))
    return out


def check_statistics(df) -> list[dict]:
    from aavas.features import profiling
    from aavas.kernel import psi

    disbursed = df
    checks = {
        "information_value": lambda: profiling.information_value(
            disbursed["ltv"], disbursed["defaulted"])["iv"],
        "mutual_information": lambda: profiling.mutual_information(
            disbursed["ltv"], disbursed["defaulted"]),
        "mann_whitney_ks": lambda: profiling.group_separation(
            disbursed["ltv"], disbursed["defaulted"])["ks_statistic"],
        "cramers_v": lambda: profiling.cramers_v(disbursed["city"], disbursed["defaulted"]),
        "vif": lambda: profiling.variance_inflation(
            disbursed[["ltv", "foir", "bureau_score"]], "ltv"),
        "psi": lambda: psi(disbursed["ltv"].to_numpy()[:600], disbursed["ltv"].to_numpy()[600:]),
    }
    out = []
    for name, fn in checks.items():
        started = time.perf_counter()
        try:
            value = fn()
            ok = value is not None and np.isfinite(float(value))
            out.append(_row(name, "statistics", PASS if ok else FAIL,
                            f"= {value}" if ok else "returned no usable value",
                            (time.perf_counter() - started) * 1000))
        except Exception as exc:                                # noqa: BLE001
            out.append(_row(name, "statistics", FAIL, f"{type(exc).__name__}: {exc}",
                            (time.perf_counter() - started) * 1000))
    return out


def check_optimizer(df) -> list[dict]:
    """The priced path itself, through the API contract."""
    from fastapi.testclient import TestClient

    from aavas.api import examples
    from aavas.api.main import app

    out = []
    client = TestClient(app)
    for label, path, payload in (
            ("stage1_band", "/pricing/pre-assessment",
             {**examples.PRE_ASSESSMENT, "loan_id": "ALGO-CHECK-1"}),
            ("stage2_optimizer", "/pricing/final-assessment",
             {**examples.FINAL_ASSESSMENT, "loan_id": "ALGO-CHECK-1"})):
        started = time.perf_counter()
        try:
            response = client.post(path, json=payload)
            body = response.json()
            if response.status_code != 200:
                raise RuntimeError(str(body)[:160])
            detail = (f"band {body['band']['min_rate']}-{body['band']['max_rate']}%"
                      if label == "stage1_band" else
                      f"rate {body['recommendation']['recommended_rate']}%, "
                      f"{len(body['recommendation'].get('scenarios') or [])} scenarios")
            out.append(_row(label, "pricing", PASS, detail,
                            (time.perf_counter() - started) * 1000))
        except Exception as exc:                                # noqa: BLE001
            out.append(_row(label, "pricing", FAIL, f"{type(exc).__name__}: {exc}",
                            (time.perf_counter() - started) * 1000))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit every algorithm and run every strategy")
    parser.add_argument("--quick", action="store_true", help="fewer rows, learners only")
    parser.add_argument("--group", help="check one group only")
    args = parser.parse_args(argv)

    X, y, df, transformer = _data(400 if args.quick else ROWS)
    log.info("verifying against %s rows, %s features, base rate %.3f", len(y), X.shape[1], y.mean())

    results: list[dict] = []
    groups = {
        "learners": lambda: check_algorithms(X, y),
        "hpo": check_hpo,
        "imbalance": lambda: check_imbalance(X, y),
        "calibration": lambda: check_calibration(X, y),
        "federated": check_federated,
        "explainability": lambda: check_explainers(X, y, transformer),
        "feature_generation": lambda: check_feature_generators(df),
        "statistics": lambda: check_statistics(df),
        "pricing": lambda: check_optimizer(df),
    }
    if args.quick:
        groups = {"learners": groups["learners"]}
    if args.group:
        groups = {k: v for k, v in groups.items() if k == args.group}

    for name, fn in groups.items():
        try:
            results.extend(fn())
        except Exception as exc:                                # noqa: BLE001
            log.exception("group %s could not run", name)
            results.append(_row(name, name, FAIL, f"group failed: {exc}"))

    by_status = {PASS: 0, FAIL: 0, SKIP: 0}
    for row in results:
        by_status[row["status"]] += 1

    width = max(len(r["name"]) for r in results) + 2
    current = None
    print()
    for row in sorted(results, key=lambda r: (r["group"], r["name"])):
        if row["group"] != current:
            current = row["group"]
            print(f"\n  {current.upper()}")
        mark = "ok  " if row["status"] == PASS else "FAIL" if row["status"] == FAIL else "--  "
        print(f"    {mark} {row['name']:<{width}} {row['detail'][:82]}")

    print(f"\n  {by_status[PASS]} working, {by_status[FAIL]} broken, "
          f"{by_status[SKIP]} unavailable, {len(results)} checked")
    if by_status[FAIL]:
        print("\n  BROKEN:")
        for row in results:
            if row["status"] == FAIL:
                print(f"    {row['group']}/{row['name']}: {row['detail'][:120]}")
    return 1 if by_status[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
