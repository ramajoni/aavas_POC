"""The historical loan book, held in the database.

The behaviour analyses query this table rather than reading the processed CSV on
every call. It lives in the governance store (``AAVAS_STORE_URL``) alongside the
decisions and applications, so one database holds both what happened in the past
and what is happening to the file on screen.

The table is a copy of ``data/processed/training_data.csv``. It is loaded on first
use and reloaded whenever that file (or the raw file it is built from) changes; the
``datasets`` table records which version is loaded, from where, and when.
"""

from __future__ import annotations

import threading
from typing import Any

import pandas as pd

from ..config import model_config
from ..governance import store
from ..ingestion import loader
from ..logging_utils import get_logger
from ..paths import resolve

log = get_logger(__name__)
_lock = threading.Lock()
NAME = "loan_book"

# The columns the behaviour analyses read, and their SQL types.
COLUMNS = {
    "loan_id": "TEXT PRIMARY KEY", "application_month": "TEXT", "city": "TEXT", "state": "TEXT",
    "segment": "TEXT", "channel": "TEXT", "property_type": "TEXT", "bureau_score": "REAL",
    "requested_loan_amount": "REAL", "competitor_name": "TEXT", "competitor_rate": "REAL",
    "competitor_evidence_verified": "INTEGER", "competitor_gap_bps": "REAL",
    "prior_negotiation_rounds": "INTEGER", "customer_negotiating": "INTEGER",
    "days_to_month_end": "INTEGER", "sales_target_pct": "REAL", "existing_relationship": "INTEGER",
    "ai_recommended_rate": "REAL", "sales_offered_rate": "REAL", "offered_rate": "REAL",
    "sanctioned_rate": "REAL", "customer_response": "TEXT", "disbursed": "INTEGER",
    "bt_out_24m": "INTEGER", "months_to_exit": "REAL",
}
SCHEMA = (
    f"CREATE TABLE IF NOT EXISTS {NAME} (\n    "
    + ",\n    ".join(f"{c} {t}" for c, t in COLUMNS.items()) + "\n);\n"
    f"CREATE INDEX IF NOT EXISTS idx_{NAME}_segment ON {NAME}(segment);\n"
    f"CREATE INDEX IF NOT EXISTS idx_{NAME}_channel ON {NAME}(channel);\n"
    "CREATE TABLE IF NOT EXISTS datasets (\n"
    "    name TEXT PRIMARY KEY, source TEXT NOT NULL, fingerprint TEXT NOT NULL,\n"
    "    rows INTEGER NOT NULL, loaded_at TEXT NOT NULL\n);\n"
)


def _sources() -> tuple[Any, Any]:
    cfg = model_config()["data"]
    return resolve(cfg["processed_path"]), resolve(cfg["raw_path"])


def fingerprint() -> str:
    """Size and modification time of the processed file and the raw file behind it."""
    parts = []
    for path in _sources():
        stat = path.stat() if path.exists() else None
        parts.append(f"{stat.st_size}:{stat.st_mtime_ns}" if stat else "missing")
    return "|".join(parts)


def status() -> dict[str, Any] | None:
    with store.connect() as conn:
        conn.executescript(SCHEMA)
        row = conn.execute("SELECT * FROM datasets WHERE name = ?", (NAME,)).fetchone()
        return dict(row) if row else None


def ensure_loaded() -> dict[str, Any]:
    """Load the book if the table is empty or the source has changed. Returns the
    ``datasets`` row describing what is loaded."""
    with _lock:
        current = status()
        if current and current["fingerprint"] == fingerprint() and current["rows"] > 0:
            return current
        frame = loader.load_training_table()     # rebuilds the CSV first if the raw file is newer
        rows = _rows(frame)
        names = list(COLUMNS)
        loaded = {"name": NAME, "source": str(_sources()[0]), "fingerprint": fingerprint(),
                  "rows": len(rows), "loaded_at": store.now()}
        with store.connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute(f"DELETE FROM {NAME}")
            conn.executemany(
                f"INSERT OR REPLACE INTO {NAME} ({', '.join(names)}) "
                f"VALUES ({', '.join('?' for _ in names)})", rows)
            conn.execute("INSERT OR REPLACE INTO datasets VALUES (:name, :source, :fingerprint, "
                         ":rows, :loaded_at)", loaded)
        log.info("loaded %s loans into the %s table", len(rows), NAME)
        return loaded


def _rows(frame: pd.DataFrame) -> list[tuple]:
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"the loan book is missing columns {missing}")
    table = frame[list(COLUMNS)].copy()
    for name, kind in COLUMNS.items():
        if kind == "INTEGER":
            table[name] = table[name].astype("boolean" if table[name].dtype == bool else "Int64")
    table = table.astype(object).where(table.notna(), None)
    return [tuple(int(v) if isinstance(v, bool) else v for v in row)
            for row in table.itertuples(index=False, name=None)]


def frame() -> pd.DataFrame:
    """The whole book as a DataFrame, read from the database."""
    ensure_loaded()
    with store.connect() as conn:
        return pd.read_sql_query(f"SELECT * FROM {NAME}", conn)


def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    ensure_loaded()
    return store.query(sql, params)
