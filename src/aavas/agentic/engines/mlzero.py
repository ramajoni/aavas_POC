"""AutoGluon Assistant (MLZero): an LLM agent that writes and runs its own training code.

The engine gives the agent the same task a champion was trained on - the contract's
features, the same chronological split - with the test answers held back. The agent
(Claude on Amazon Bedrock) reads the data, writes AutoGluon code, runs it, debugs it,
and writes predictions for the test file. The engine then scores those predictions
against the held-back answers and records a challenger card beside the champion.

Its models are challengers only: AutoGluon predictors are not artefacts the pricing
engine loads, so they are compared, never promoted.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...config import feature_config
from ...governance import model_registry
from ...ingestion import loader
from ...models import base as model_base
from ...paths import ROOT
from .. import runs
from ..workflow import StepError, run_logged

TARGETS = {
    "stage2": {"column": "sanctioned_rate", "kind": "regression", "metric": "mae_bps",
               "task": "predict sanctioned_rate (the loan's final interest rate, % p.a.)"},
    "conversion": {"column": "disbursed", "kind": "binary", "metric": "auc",
                   "task": "predict the probability that disbursed is true (the customer took the loan)"},
}


def _dataset(run: runs.Run, target: str):
    spec = TARGETS[target]
    files = {"raw": run.workspace / "data" / "home_loan_data.csv",
             "processed": run.workspace / "data" / "training_data.csv"}
    # Explicit paths, not os.environ: the API serves other requests in this process
    # while a run executes, and they must keep reading the live dataset.
    table = loader.load_training_table(raw_path=files["raw"], processed_path=files["processed"])
    contract = feature_config()[target]
    wanted = list(contract["numeric"]) + list(contract["categorical"])
    columns = [c for c in wanted if c in table.columns]
    train, test = loader.train_test_split(table)
    return spec, columns, sorted(set(wanted) - set(columns)), train, test


def train(run: runs.Run, env: dict[str, str]) -> dict:
    cfg = run.config["training"]["autogluon_assistant"]
    if cfg.get("allow_unsandboxed") is not True:
        raise StepError("the AutoGluon Assistant runs code it writes itself, unsandboxed on this host, so it "
                        "is refused here. It needs an isolated container first; on a developer machine "
                        "only, set training.autogluon_assistant.allow_unsandboxed: true.")
    target = cfg.get("target", "stage2")
    if target not in TARGETS:
        raise StepError(f"AutoGluon Assistant target must be one of {sorted(TARGETS)}")
    cli_python = ROOT / Path(cfg["cli"]).parent / "python.exe"
    if not cli_python.exists():
        cli_python = ROOT / Path(cfg["cli"]).parent / "python"
    if not cli_python.exists():
        raise StepError(f"AutoGluon Assistant is not installed at {cfg['cli']}")

    spec, columns, missing, train_df, test_df = _dataset(run, target)
    work = run.workspace / "mlzero"
    inp, out = work / "input", work / "output"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    id_col = "loan_id"
    train_df[[id_col, *columns, spec["column"]]].to_csv(inp / "train.csv", index=False)
    test_df[[id_col, *columns]].to_csv(inp / "test.csv", index=False)
    answers = test_df[[id_col, spec["column"]]].copy()
    answers.to_csv(work / "answers.csv", index=False)          # never in the agent's folder
    (inp / "description.txt").write_text(
        f"Home-loan pricing data. Task: {spec['task']} for every row of test.csv.\n"
        f"train.csv has {len(train_df):,} rows with the target column '{spec['column']}'; test.csv has "
        f"{len(test_df):,} later rows without it (a chronological split).\n"
        f"'{id_col}' identifies a row and is not a feature. Write the predictions for test.csv as "
        f"results.csv with the columns {id_col} and {spec['column']}.\n"
        f"Evaluation: {'mean absolute error' if spec['kind'] == 'regression' else 'ROC AUC'} on test.csv.\n",
        encoding="utf-8")
    run.log(f"[training] MLZero task: {target} ({spec['kind']}), {len(columns)} features, "
            f"{len(train_df):,} train / {len(test_df):,} test rows" + (f"; contract columns not in the "
                                                                     f"table: {missing}" if missing else ""))

    # The agent's LLM: every role on the configured Bedrock model.
    default = ROOT / ".venv-mlzero" / "Lib" / "site-packages" / "autogluon" / "assistant" / "configs" / "default.yaml"
    if not default.exists():
        found = list((ROOT / Path(cfg["cli"]).parent.parent).rglob("assistant/configs/default.yaml"))
        default = found[0] if found else None
    if default is None:
        raise StepError("could not find the AutoGluon Assistant default config")
    conf = yaml.safe_load(default.read_text(encoding="utf-8"))
    conf["per_execution_timeout"] = int(cfg["time_limit_minutes"]) * 60
    for key, value in conf.items():
        if isinstance(value, dict) and value.get("provider"):
            value.update(provider=cfg["provider"], model=cfg["model"])
    (work / "config.yaml").write_text(yaml.safe_dump(conf, sort_keys=False), encoding="utf-8")

    agent_env = agent_environment(env, cfg, cli_python)
    instruction = (cfg.get("instructions") or "").strip() + (
        f" Use AutoGluon TabularPredictor with time_limit={max(60, int(cfg['time_limit_minutes']) * 60 // 2)} "
        f"seconds. Save results.csv in the output folder.")
    run_logged(run, [str(cli_python), str(ROOT / "deploy" / "mlzero" / "launch.py"), "-i", str(inp),
                     "-o", str(out), "-c", str(work / "config.yaml"), "-n", str(int(cfg["max_iterations"])),
                     "-t", instruction, "-v", "1"],
               agent_env, cwd=out, timeout=int(cfg["time_limit_minutes"]) * 60 * max(1, int(cfg["max_iterations"])) + 900)

    results = sorted(out.rglob("results*.csv"), key=lambda p: p.stat().st_mtime)
    if not results:
        raise StepError("the agent finished without writing a results.csv for the test rows", {"output": str(out)})
    scored = [(path, _score(path, answers, id_col, spec)) for path in results]
    usable = [(p, s) for p, s in scored if s is not None]
    if not usable:
        raise StepError(f"none of the agent's {len(results)} results files could be matched to the test rows")
    chosen, metrics = usable[-1]                                  # the agent's last (final) answer
    champion = json.loads((model_base.registry_dir() / f"{target}__v1.card.json").read_text(encoding="utf-8")) \
        if (model_base.registry_dir() / f"{target}__v1.card.json").exists() else {}
    card = {"name": f"{target}_autogluon_assistant", "version": run.run_id,
            "learner": f"autogluon_assistant ({cfg['model']})", "target": spec["column"],
            "trained_at": runs.now(), "n_train": int(len(train_df)), "n_test": int(len(test_df)),
            "features": columns, "metrics": metrics,
            "limitations": ["trained by the AutoGluon Assistant agent from code it wrote; a challenger for "
                            "comparison only - the pricing engine does not load AutoGluon predictors",
                            f"scored on {chosen.relative_to(out)} against held-back answers"]}
    known = set(model_base.ModelCard.__dataclass_fields__)
    model_registry.register(model_base.ModelCard(**{k: v for k, v in card.items() if k in known}), role="challenger")
    key = spec["metric"]
    vs = champion.get("metrics", {}).get(key)
    better = None if vs is None else (metrics[key] < vs if key == "mae_bps" else metrics[key] > vs)
    return {"summary": (f"AutoGluon Assistant trained a {target} challenger: {key} {metrics[key]:.4g}"
                        + (f" vs champion {vs:.4g} ({'better' if better else 'not better'})" if vs is not None else "")),
            "engine": "autogluon_assistant", "target": target, "cards": {target: card},
            "champion_metric": vs, "results_files": [str(p.relative_to(out)) for p in results],
            "scored_file": str(chosen.relative_to(out)), "output": str(out)}


def _score(path: Path, answers: pd.DataFrame, id_col: str, spec: dict) -> dict | None:
    try:
        pred = pd.read_csv(path)
    except Exception:                                                # noqa: BLE001 - not a predictions file
        return None
    col = spec["column"]
    if id_col in pred.columns and col in pred.columns:
        merged = answers.merge(pred[[id_col, col]], on=id_col, suffixes=("", "_pred"))
        if len(merged) != len(answers):
            return None
        y, p = merged[col].to_numpy(float), merged[f"{col}_pred"].to_numpy(float)
    elif len(pred) == len(answers):
        y = answers[col].to_numpy(float)
        p = pred[col].to_numpy(float) if col in pred.columns else pred.iloc[:, -1].to_numpy(float)
    else:
        return None
    if spec["kind"] == "regression":
        err = np.abs(p - y) * 100
        return {"mae_bps": round(float(err.mean()), 3), "within_25bps_pct": round(float((err <= 25).mean() * 100), 2)}
    from sklearn.metrics import roc_auc_score
    return {"auc": round(float(roc_auc_score(y, p)), 4)}


# The agent runs code it wrote itself, unsandboxed, as this user. It gets only what it
# needs: the OS basics, its LLM's credentials and proxy settings. Never the aavas store,
# OpenObserve or webhook secrets, nor the run's data paths. Real isolation (a container
# with Bedrock-only credentials and no host mounts but input/output) is still required
# before this engine runs anywhere but a developer's own machine.
_OS_KEYS = ("PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
            "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
            "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS", "LANG", "LC_ALL",
            "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
_PROVIDER_KEYS = {"bedrock": ("AWS_",), "anthropic": ("ANTHROPIC_API_KEY",), "openai": ("OPENAI_API_KEY",)}


def agent_environment(env: dict[str, str], cfg: dict, cli_python: Path) -> dict[str, str]:
    allowed = _PROVIDER_KEYS.get(str(cfg.get("provider", "bedrock")).lower(), ())
    out = {k: v for k, v in env.items()
           if k.upper() in _OS_KEYS or k.upper().startswith("HF_") or any(k.upper().startswith(a) for a in allowed)}
    out.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                "PATH": str(cli_python.parent) + os.pathsep + out.get("PATH", "")})
    if allowed == ("AWS_",):
        out.update({"AWS_DEFAULT_REGION": cfg["region"], "AWS_REGION": cfg["region"]})
    return out
