"""Filesystem layout. Everything resolves from the repo root, so scripts,
notebooks, tests and the API all agree on where artefacts live."""

from __future__ import annotations

import os
from pathlib import Path

# src/aavas/paths.py -> src/aavas -> src -> repo root
ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"
REPORTS_DIR = ROOT / "reports"
OMNIXAI_DIR = REPORTS_DIR / "omnixai"


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config-supplied path against the repo root when relative."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, MLRUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
