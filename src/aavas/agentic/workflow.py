"""The agentic workflow: data -> policy -> training -> serving.

Each step is an agent: it checks its pre-conditions, does its work in the run's
workspace, and checks its post-conditions. A failed check raises ``StepError``,
which stops the run with the reason; nothing after it runs. Nothing touches the
live dataset or the champion models except ``promote``, which needs an approver.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config import load as load_config
from ..config import model_config
from ..governance import model_registry, policy_cases, policy_checks
from ..ingestion.loader import data_path
from ..models import base as model_base
from ..paths import ROOT, resolve
from . import book_checks, runs

HEADLINE = {
    "stage1": ["coverage_pct", "mean_width_bps", "median_mae_bps"],
    "stage2": ["mae_bps", "within_25bps_pct", "r2"],
    "conversion": ["auc", "calibration_gap"],
    "prepayment": ["auc"],
    "risk": ["auc", "base_rate"],
    "override": ["auc", "magnitude_mae_bps"],
    "lgd": ["rmse_pts", "policy_rmse_pts"],
    "uplift": ["qini", "ate_pts"],
}


class StepError(Exception):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


# ------------------------------------------------------------------ workspace


def data_files(run: runs.Run) -> dict[str, Path]:
    d = run.workspace / "data"
    return {"raw": d / "home_loan_data.csv", "notes": d / "sales_notes.csv", "processed": d / "training_data.csv"}


def workspace_env(run: runs.Run) -> dict[str, str]:
    files = data_files(run)
    env = dict(os.environ)
    env.update({
        "AAVAS_RAW_PATH": str(files["raw"]), "AAVAS_NOTES_PATH": str(files["notes"]),
        "AAVAS_PROCESSED_PATH": str(files["processed"]), "AAVAS_MODEL_DIR": str(run.workspace / "models"),
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        "PYTHONIOENCODING": "utf-8",
    })
    return env


def run_logged(run: runs.Run, cmd: list[str], env: dict[str, str], timeout: int = 3600,
               cwd: Path | None = None) -> None:
    """Run a command, streaming every line into the run log; a non-zero exit fails the step."""
    run.log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(cmd, cwd=cwd or ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    try:
        for line in proc.stdout:
            if line.strip() and "Warning" not in line:
                run.log("  " + line.rstrip())
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise StepError(f"{Path(cmd[1]).name} ran past {timeout // 60} minutes and was stopped")
    if code != 0:
        raise StepError(f"{Path(cmd[1]).name} exited with code {code}; see the run log")


# ------------------------------------------------------------------ 1. data


def data_step(run: runs.Run, env: dict[str, str]) -> tuple[str, dict]:
    cfg = run.config["data"]
    files = data_files(run)
    files["raw"].parent.mkdir(parents=True, exist_ok=True)
    if cfg["source"] == "synthetic":
        run_logged(run, [sys.executable, str(ROOT / "data/synthetic/generate_dataset.py"),
                         "-n", str(int(cfg["rows"])), "--seed", str(int(cfg["seed"]))], env, timeout=900)
    elif cfg["source"] == "live":
        live_raw = data_path("raw_path")
        if not live_raw.exists():
            raise StepError(f"no live dataset at {live_raw}")
        shutil.copy2(live_raw, files["raw"])
        if data_path("notes_path").exists():
            shutil.copy2(data_path("notes_path"), files["notes"])
    else:
        raise StepError(f"unknown data source '{cfg['source']}' (synthetic | live)")

    df = pd.read_csv(files["raw"])
    if df.empty:
        raise StepError("the dataset has no rows")
    results = book_checks.run(df, cfg.get("checks"))
    details = {"profile": book_checks.profile(df), "checks": results}
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise StepError("the dataset breaks the policy: " + "; ".join(
            f"{r['violations']} {r['what']}" for r in failed), details)
    p = details["profile"]
    return (f"{p['rows']:,} loans, {p['dates'][0]} to {p['dates'][1]}, rates {p['sanctioned_rate']['min']:.2f}-"
            f"{p['sanctioned_rate']['max']:.2f}%; all {len(results)} policy checks pass", details)


# ------------------------------------------------------------------ 2. policy


def policy_report() -> dict[str, Any]:
    pol_path = ROOT / "config" / "pricing_policy.yaml"
    policy = yaml.safe_load(pol_path.read_text(encoding="utf-8"))
    checks = policy_checks.run(policy, model_config()["risk"]["grade_cutoffs"])
    cases = policy_cases.run()
    return {
        "policy_version": policy.get("policy_version"), "status": policy.get("status"),
        "effective_from": policy.get("effective_from"), "decisions": policy.get("decisions") or {},
        "consistency": policy_checks.summary(checks),
        "sample_cases": {"cases": len(cases), "passed": sum(c["ok"] for c in cases),
                         "failed": [c for c in cases if not c["ok"]]},
    }


def policy_step(run: runs.Run, env: dict[str, str]) -> tuple[str, dict]:
    cfg = run.config["policy"]
    report = policy_report()
    problems = []
    if cfg.get("require_consistency_checks", True) and not report["consistency"]["ok"]:
        problems.append(f"{len(report['consistency']['failed'])} consistency check(s) fail")
    if cfg.get("require_sample_cases", True) and report["sample_cases"]["failed"]:
        problems.append(f"{len(report['sample_cases']['failed'])} sample case(s) disagree with the engine")
    if problems:
        raise StepError(f"{report['policy_version']}: " + "; ".join(problems), report)
    return (f"{report['policy_version']}: {report['consistency']['passed']} of {report['consistency']['checks']} "
            f"consistency checks and {report['sample_cases']['passed']} of {report['sample_cases']['cases']} "
            f"sample cases pass", report)


# ------------------------------------------------------------------ 3. training


def model_cards(directory: Path) -> dict[str, dict]:
    cards = {}
    for path in sorted(directory.glob("*.card.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        cards[card.get("name") or path.name.split("__")[0]] = card
    return cards


def compare(new: dict[str, dict], champions: dict[str, dict]) -> list[dict[str, Any]]:
    rows = []
    for name, card in new.items():
        old = (champions.get(name) or {}).get("metrics", {})
        metrics = card.get("metrics", {})
        rows.append({"model": name, "learner": card.get("learner"), "n_train": card.get("n_train"),
                     "metrics": {k: {"new": metrics.get(k), "champion": old.get(k)}
                                 for k in HEADLINE.get(name, []) if k in metrics}})
    return rows


def training_step(run: runs.Run, env: dict[str, str]) -> tuple[str, dict]:
    engine = run.config["training"]["engine"]
    if engine == "local":
        from .engines import local as eng
    elif engine == "sagemaker":
        from .engines import sagemaker as eng
    elif engine == "autogluon_assistant":
        from .engines import mlzero as eng
    else:
        raise StepError(f"unknown training engine '{engine}' (local | sagemaker | autogluon_assistant)")
    details = eng.train(run, env)
    champions = model_cards(model_base.registry_dir())
    details["comparison"] = compare(details.get("cards", {}), champions)
    return details.pop("summary"), details


# ------------------------------------------------------------------ 4. serving


def serving_step(run: runs.Run, env: dict[str, str]) -> tuple[str, dict]:
    cfg = run.config["serving"]
    engine = run.config["training"]["engine"]
    training = run.step("training")["details"]
    if cfg["mode"] == "sagemaker_serverless":
        if engine != "sagemaker":
            raise StepError("a SageMaker Serverless endpoint serves a SageMaker-trained bundle (the endpoint "
                            "image must load what it serves); set training.engine to sagemaker")
        from . import serving
        details = serving.deploy(run, training)
        return details.pop("summary"), details
    if cfg["mode"] != "in_process":
        raise StepError(f"unknown serving mode '{cfg['mode']}' (in_process | sagemaker_serverless)")
    if engine != "local":
        return (f"{engine} models are kept as challengers for comparison; in-process serving keeps the "
                f"current champions", {"awaiting_promotion": False})
    if cfg.get("promote_on_success"):
        details = promote(run, "workflow (promote_on_success)")
        return f"promoted {len(details['promoted'])} models to champion", details
    n = len(training.get("cards", {}))
    return (f"{n} challenger models ready; an approver promotes them from the Agentic OS screen",
            {"awaiting_promotion": True})


def promote(run: runs.Run, approved_by: str) -> dict[str, Any]:
    """Make a local run's models the champions: back up, copy in, register, reload."""
    if run.record.get("engine") != "local":
        raise StepError("only locally trained models are promoted into the API; SageMaker models are served "
                        "by their endpoint and AutoGluon Assistant models stay challengers")
    if run.record.get("status") != "succeeded":
        raise StepError("only a run that finished every step can be promoted")
    source = run.workspace / "models"
    live = resolve(model_config()["registry"]["dir"])          # the champions, whatever this process overrides
    backup = run.workspace / "replaced_champions"
    backup.mkdir(parents=True, exist_ok=True)
    promoted = []
    for card_path in sorted(source.glob("*.card.json")):
        stem = card_path.name[: -len(".card.json")]
        for suffix in (".joblib", ".card.json"):
            current = live / f"{stem}{suffix}"
            if current.exists():
                shutil.copy2(current, backup / current.name)
            shutil.copy2(source / f"{stem}{suffix}", current)
        card = json.loads(card_path.read_text(encoding="utf-8"))
        known = set(model_base.ModelCard.__dataclass_fields__)
        model_registry.register(model_base.ModelCard(**{k: v for k, v in card.items() if k in known}),
                                role="champion")
        promoted.append(card.get("name") or stem)
    model_registry.load_bundle(refresh=True)
    try:
        from ..api import deps
        deps.reset()
    except Exception:                                                # noqa: BLE001 - no API in this process
        pass
    record = {"approved_by": approved_by, "at": runs.now(), "promoted": promoted, "backup": str(backup)}
    run.record["promotion"] = record
    run.log(f"[promotion] {approved_by} promoted {', '.join(promoted)}")
    run.save()
    return record


STEPS = {"data": data_step, "policy": policy_step, "training": training_step, "serving": serving_step}


def execute(run: runs.Run) -> None:
    run.record.update(status="running", started_at=runs.now())
    run.save()
    env = workspace_env(run)
    for name in run.config["workflow"]:
        run.start_step(name)
        try:
            summary, details = STEPS[name](run, env)
        except StepError as exc:
            run.fail_step(name, str(exc), exc.details)
            return
        except Exception as exc:                                     # noqa: BLE001 - recorded on the run
            run.log(traceback.format_exc())
            run.fail_step(name, f"{type(exc).__name__}: {exc}")
            return
        run.finish_step(name, summary, details)
    run.record.update(status="succeeded", finished_at=runs.now())
    run.save()
    run.log("run finished: every step succeeded")


def resume(run: runs.Run) -> None:
    """Re-run a failed run from its first unfinished step; finished steps keep their results."""
    run.record.update(status="running", finished_at=None, resumed_at=runs.now())
    run.save()
    run.log("run resumed")
    env = workspace_env(run)
    for name in run.config["workflow"]:
        if run.step(name)["status"] == "succeeded":
            continue
        run.start_step(name)
        try:
            summary, details = STEPS[name](run, env)
        except StepError as exc:
            run.fail_step(name, str(exc), exc.details)
            return
        except Exception as exc:                                     # noqa: BLE001 - recorded on the run
            run.log(traceback.format_exc())
            run.fail_step(name, f"{type(exc).__name__}: {exc}")
            return
        run.finish_step(name, summary, details)
    run.record.update(status="succeeded", finished_at=runs.now())
    run.save()
    run.log("run finished: every step succeeded")


def start(started_by: str, config: dict[str, Any] | None = None) -> runs.Run:
    cfg = config or load_config("agentic_os")
    run = runs.create(cfg, started_by)
    if not runs.launch(run, execute):
        run.record.update(status="refused", finished_at=runs.now())
        run.save()
        raise StepError(f"run {runs.active()} is still executing; one run at a time")
    return run
