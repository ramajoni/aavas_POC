#!/usr/bin/env python3
"""Train (optionally), then run the OmniXAI global + instance analysis and serve
the OmniXAI dashboard for each model.

    python scripts/run_omnixai.py                      # analyse all six, serve the UI
    python scripts/run_omnixai.py --retrain            # train first, then analyse
    python scripts/run_omnixai.py --models stage2 risk # a subset
    python scripts/run_omnixai.py --no-ui              # reports only, no server
    python scripts/run_omnixai.py --fast               # smaller samples, no counterfactuals

OmniXAI's dashboard keeps its state in a module-level singleton, so one process
can only ever show one model. Each dashboard therefore gets its own process and
its own port, and each child computes its own analysis - so the served models are
analysed in parallel while the parent works through whatever is left over.

Requires the explainability extras. They are pinned away from the rest of the
environment in .venv-xai, so run this with .venv-xai/bin/python (or `make xai`).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aavas.logging_utils import get_logger  # noqa: E402
from aavas.models import base as model_base  # noqa: E402
from aavas.paths import ensure_dirs  # noqa: E402

log = get_logger("run_omnixai")

DEFAULT_PORT = 8765
# Each dashboard process carries its own copy of the data, the explainers and
# (via omnixai) torch: about 1.5 GB resident. Three is what a 16 GB laptop takes
# without swapping; --max-dashboards raises it deliberately.
DEFAULT_MAX_DASHBOARDS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from aavas.xai.omnixai_analysis import MODELS, REPORT_DIR

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS),
                        help="models to analyse (default: all)")
    parser.add_argument("--retrain", action="store_true", help="run scripts/train_all.py first")
    parser.add_argument("--no-ui", dest="ui", action="store_false", help="write reports only")
    parser.add_argument("--ui-models", nargs="*", default=None,
                        help="models to serve a dashboard for (default: the analysed models, "
                             "capped at --max-dashboards)")
    parser.add_argument("--max-dashboards", type=int, default=DEFAULT_MAX_DASHBOARDS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="first port; one per model")
    parser.add_argument("--instances", type=int, default=7,
                        help="files explained in detail: both predicted tails, the median, "
                             "both ends of the credit score, then random")
    parser.add_argument("--train-sample", type=int, default=500,
                        help="reference population each explainer is defined against")
    parser.add_argument("--global-sample", type=int, default=100,
                        help="holdout rows KernelSHAP attributes for the global ranking")
    parser.add_argument("--permutation-sample", type=int, default=1000)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--pdp-features", type=int, default=12,
                        help="number of features to compute PDP/ALE curves for (0 = all)")
    parser.add_argument("--no-counterfactuals", dest="counterfactuals", action="store_false",
                        help="skip MACE counterfactuals (the slowest explainer)")
    parser.add_argument("--fast", action="store_true", help="small samples, no counterfactuals")
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    return parser.parse_args(argv)


def limit_threads(share_between: int) -> int:
    """Cap each process's share of the machine.

    scikit-learn's boosted trees are OpenMP-parallel and default to every core.
    With one process per dashboard that is N x cores threads fighting over
    cores, and the analysis runs an order of magnitude slower than it does on
    its own - measured, not theoretical. Both knobs are set: the environment
    variables cover libraries that read them at first use, threadpoolctl covers
    the pools numpy has already created.
    """
    threads = max(1, (os.cpu_count() or 4) // max(1, share_between))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads)
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(threads)
    except Exception:                                          # noqa: BLE001
        pass
    return threads


def analysis_kwargs(args: argparse.Namespace) -> dict:
    if args.fast:
        return {"train_sample": 200, "global_sample": 50, "n_instances": min(args.instances, 5),
                "permutation_sample": 500, "permutation_repeats": 3, "counterfactuals": False,
                "pdp_features": 8}
    return {"train_sample": args.train_sample, "global_sample": args.global_sample,
            "n_instances": args.instances, "permutation_sample": args.permutation_sample,
            "permutation_repeats": args.permutation_repeats,
            "counterfactuals": args.counterfactuals,
            "pdp_features": args.pdp_features or None}


def run_one(name: str, args: argparse.Namespace):
    """Analyse one model and write its report. Returns (summary, report_dir)."""
    from aavas.xai import omnixai_analysis as oa

    analysis = oa.analyse(name, **analysis_kwargs(args))
    directory = oa.write_report(analysis, Path(args.report_dir) / name)
    return analysis, oa.console_summary(analysis), directory


def serve(name: str, args: argparse.Namespace, port: int, queue, siblings: int) -> None:
    """Child process: analyse, write the report, then hold the dashboard open."""
    try:
        limit_threads(siblings + 1)          # the other children, plus the parent
        analysis, summary, directory = run_one(name, args)
        queue.put(("ready", name, port, summary, str(directory)))

        from omnixai.visualization.dashboard import Dashboard

        Dashboard(
            instances=analysis.instances,
            local_explanations=analysis.local_explanations,
            global_explanations=analysis.global_explanations,
            data_explanations=analysis.data_explanations,
            prediction_explanations=analysis.prediction_explanations,
            class_names=analysis.bundle.class_names,
            # Plot the features PDP was actually computed for, not the first N of
            # the contract - the analysis ranks before it curves.
            params={"pdp": {"features": list(analysis.report["pdp"].keys())}},
            explainer=analysis.whatif_explainer,   # powers the What-if tab
        ).show(host=args.host, port=str(port))
    except Exception as exc:                                   # noqa: BLE001
        queue.put(("failed", name, port, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", ""))


def write_manifest(ports: dict[str, int], host: str, report_dir: str) -> Path | None:
    """Record which model is served on which port.

    A Dash process does not advertise what it holds, so without this the admin
    view can only report that *something* answers on 8766. Written on start and
    removed on a clean exit; the API probes each entry anyway, so a manifest left
    behind by a crash reads as "not running" rather than as a lie.
    """
    import json
    from datetime import datetime, timezone

    if not ports:
        return None
    path = Path(report_dir) / "ui" / "dashboards.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "pid": os.getpid(),
               "dashboards": [{"model": name, "port": port,
                               "url": f"http://{host}:{port}"}
                              for name, port in sorted(ports.items(), key=lambda kv: kv[1])]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("dashboard manifest -> %s", path)
    return path


def check_trained(models: list[str]) -> list[str]:
    return [m for m in models if not model_base.exists(m)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_dirs()

    if args.retrain:
        from train_all import main as train_all

        log.info("training the full model suite before analysing it")
        train_all([])
    missing = check_trained(args.models)
    if missing:
        log.error("no trained artefact for: %s. Train first: python scripts/train_all.py "
                  "(or pass --retrain)", ", ".join(missing))
        return 1

    ui_models: list[str] = []
    if args.ui:
        ui_models = args.ui_models if args.ui_models is not None else list(args.models)
        ui_models = [m for m in ui_models if m in args.models]
        if len(ui_models) > args.max_dashboards:
            log.warning("serving dashboards for the first %s of %s models (%s); raise "
                        "--max-dashboards to serve more, each costs ~1.5 GB",
                        args.max_dashboards, len(ui_models), ", ".join(ui_models[:args.max_dashboards]))
            ui_models = ui_models[:args.max_dashboards]

    ports = {name: args.port + i for i, name in enumerate(ui_models)}
    manifest = write_manifest(ports, args.host, args.report_dir)
    queue, children = None, []
    if ui_models:
        # spawn, not fork: the parent has sklearn/OpenMP thread pools live, and
        # forking those into a long-running server is how you get a silent hang.
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        for name in ui_models:
            child = ctx.Process(target=serve, args=(name, args, ports[name], queue, len(ui_models)),
                                name=f"omnixai-{name}", daemon=False)
            child.start()
            children.append(child)
        log.info("analysing %s in parallel for the dashboard", ", ".join(ui_models))

    # Whatever is not being served is analysed here, while the children work.
    summaries: dict[str, str] = {}
    directories: dict[str, str] = {}
    failures: dict[str, str] = {}
    own_work = [m for m in args.models if m not in ui_models]
    if own_work:
        log.info("each process limited to %s threads", limit_threads(len(ui_models) + 1))
    for name in own_work:
        try:
            _, summary, directory = run_one(name, args)
            summaries[name], directories[name] = summary, str(directory)
        except Exception as exc:                               # noqa: BLE001
            failures[name] = f"{type(exc).__name__}: {exc}"
            log.error("[%s] analysis failed: %s", name, exc, exc_info=True)

    for _ in range(len(ui_models)):
        status, name, port, payload, directory = queue.get()
        if status == "ready":
            summaries[name], directories[name] = payload, directory
        else:
            failures[name] = payload
            log.error("[%s] dashboard process failed:\n%s", name, payload)

    for name in args.models:
        if name in summaries:
            print(summaries[name])

    print("\n" + "=" * 88)
    print("OMNIXAI ANALYSIS")
    print("=" * 88)
    for name in args.models:
        if name in failures:
            print(f"  {name:<12} FAILED  {failures[name].splitlines()[0]}")
        else:
            print(f"  {name:<12} report {directories.get(name, '-')}")
    if ui_models:
        print("\n  Dashboards (Data / Local / Global / Prediction / What-if tabs):")
        for name in ui_models:
            if name not in failures:
                print(f"    {name:<12} http://{args.host}:{ports[name]}")
        print("\n  Ctrl-C to stop.")
        # The parent then blocks in join() forever. Without an explicit flush the
        # URL table sits in stdout's block buffer whenever output is redirected
        # rather than attached to a terminal - i.e. exactly when someone is
        # tailing a log to find the URLs.
        sys.stdout.flush()
        try:
            for child in children:
                child.join()
        except KeyboardInterrupt:
            print("\n  stopping dashboards")
            for child in children:
                child.terminate()
            for child in children:
                child.join(timeout=5)
        finally:
            if manifest and manifest.exists():
                manifest.unlink(missing_ok=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
