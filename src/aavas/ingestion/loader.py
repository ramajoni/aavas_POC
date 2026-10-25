"""Loading and the train/test split.

Pricing is non-stationary - competitor behaviour, targets and market rates drift -
so the default split is chronological. A random split lets the model peek at the
future and flatters every metric in this repo.
"""

from __future__ import annotations

import pandas as pd

from ..config import model_config
from ..logging_utils import get_logger
from ..paths import resolve
from . import validator

log = get_logger(__name__)


def load_raw(path: str | None = None) -> pd.DataFrame:
    cfg = model_config()["data"]
    p = resolve(path or cfg["raw_path"])
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Generate it first: python data/synthetic/generate_dataset.py"
        )
    df = pd.read_csv(p)
    return validator.coerce_types(df)


def load_notes(path: str | None = None) -> pd.DataFrame:
    cfg = model_config()["data"]
    p = resolve(path or cfg["notes_path"])
    if not p.exists():
        return pd.DataFrame(columns=["loan_id", "note_text", "override_reason"])
    return pd.read_csv(p)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derived fields shared by several models.

    ``offered_rate`` is deliberately explicit: the conversion and prepayment models
    are trained on *the rate the customer was actually shown*, which is what makes
    them usable to score a hypothetical price during scenario simulation.
    """
    out = df.copy()
    if "sanctioned_rate" in out:
        out["offered_rate"] = out["sanctioned_rate"]
        out["competitor_gap_bps"] = ((out["sanctioned_rate"] - out["competitor_rate"]) * 100).round(1)
    if {"ai_recommended_rate", "competitor_rate"} <= set(out.columns):
        out["recommended_gap_bps"] = ((out["ai_recommended_rate"] - out["competitor_rate"]) * 100).round(1)
    if "override_bps" in out:
        out["was_overridden"] = out["override_bps"].fillna(0) != 0
        out["override_direction"] = pd.cut(
            out["override_bps"].fillna(0), bins=[-1e9, -0.5, 0.5, 1e9],
            labels=["discount", "none", "premium"],
        ).astype(str)
    if {"policy_card_rate", "sanctioned_rate"} <= set(out.columns):
        out["total_discount_bps"] = ((out["policy_card_rate"] - out["sanctioned_rate"]) * 100).round(1)
    if "application_date" in out:
        out["application_month"] = pd.to_datetime(out["application_date"]).dt.to_period("M").astype(str)
    return out


def build_training_table(save: bool = True) -> pd.DataFrame:
    cfg = model_config()["data"]
    df = add_derived_columns(load_raw())
    report = validator.validate(df)
    log.info("training table: %s", report.summary())
    df, bad = validator.quarantine(df, report)
    if len(bad):
        log.warning("quarantined %s rows", len(bad))
    if save:
        out = resolve(cfg["processed_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and rename over it. A write cut short (a full
        # disk, a killed process) must leave the previous table, not half of a
        # new one with a fresher timestamp than the raw file it came from.
        partial = out.with_name(out.name + ".partial")
        try:
            df.to_csv(partial, index=False)
            partial.replace(out)
        finally:
            partial.unlink(missing_ok=True)
        log.info("wrote processed training table -> %s (%s rows)", out, len(df))
    return df


def load_training_table(rebuild: bool = False) -> pd.DataFrame:
    cfg = model_config()["data"]
    p = resolve(cfg["processed_path"])
    raw = resolve(cfg["raw_path"])
    # The processed table is a cache of the raw file. Regenerating the raw data
    # and then reading a cache built from the previous version would train every
    # model on columns the raw file no longer has, or without ones it gained.
    stale = p.exists() and raw.exists() and p.stat().st_mtime < raw.stat().st_mtime
    if stale:
        log.info("processed training table is older than %s; rebuilding", raw.name)
    if rebuild or stale or not p.exists():
        return build_training_table()
    return validator.coerce_types(pd.read_csv(p))


def train_test_split(df: pd.DataFrame, test_size: float | None = None,
                     strategy: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = model_config()["data"]
    test_size = cfg["test_size"] if test_size is None else test_size
    strategy = strategy or cfg.get("split_strategy", "time")

    if strategy == "time" and cfg["time_column"] in df.columns:
        ordered = df.sort_values(cfg["time_column"])
        cut = int(len(ordered) * (1 - test_size))
        train, test = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
        log.info("time split at %s: train=%s test=%s",
                 test[cfg["time_column"]].min(), len(train), len(test))
        return train, test

    from sklearn.model_selection import train_test_split as sk_split

    from ..config import seed

    return sk_split(df, test_size=test_size, random_state=seed())
