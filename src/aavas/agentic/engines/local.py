"""Train in a subprocess on this machine, into the run's own model directory."""

from __future__ import annotations

import sys

from ...paths import ROOT
from .. import runs
from ..workflow import StepError, model_cards, run_logged


def train(run: runs.Run, env: dict[str, str]) -> dict:
    cfg = run.config["training"]
    models = list(cfg["models"])
    run_logged(run, [sys.executable, str(ROOT / "scripts/train_all.py"), "--only", *models,
                     "--role", cfg.get("register_as", "challenger")], env, timeout=3600)
    cards = model_cards(run.workspace / "models")
    missing = sorted(set(models) - set(cards))
    if missing:
        raise StepError(f"training finished without a model card for {', '.join(missing)}")
    return {"summary": f"trained {len(cards)} models locally as {cfg.get('register_as', 'challenger')}s",
            "engine": "local", "cards": cards}
