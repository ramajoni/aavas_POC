"""Durable store for decisions, overrides, outcomes and the audit trail.

SQLite by default so the POC runs with nothing installed; the DSN is read from
``AAVAS_STORE_URL`` so a shared deployment points at Postgres instead. The schema
is deliberately append-only for decisions and audit entries: a pricing decision
that can be quietly edited afterwards is not an audit trail.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config import env
from ..logging_utils import get_logger
from ..paths import ROOT

log = get_logger(__name__)
_LOCK = threading.Lock()

SCHEMA = """
-- Borrowers who registered online. Staff accounts live in config/access.yaml.
CREATE TABLE IF NOT EXISTS borrower_accounts (
    username        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    mobile          TEXT NOT NULL,
    email           TEXT,
    password        TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

-- A borrower login a relationship manager issued with an assisted application.
-- The temporary password is never stored in clear; until the customer sets their
-- own (password_changed_at), the account can do nothing but change it.
CREATE TABLE IF NOT EXISTS customer_logins (
    username            TEXT PRIMARY KEY,
    issued_by           TEXT NOT NULL,
    loan_id             TEXT,
    issued_at           TEXT NOT NULL,
    password_changed_at TEXT
);

-- A reminder an RM sent a customer whose file has stalled (documents due, or a
-- quote never applied for). ``ref`` is the loan id, or a sample lead's id.
CREATE TABLE IF NOT EXISTS customer_nudges (
    nudge_id        TEXT PRIMARY KEY,
    ref             TEXT NOT NULL,
    channel         TEXT NOT NULL,
    message         TEXT NOT NULL,
    sent_by         TEXT NOT NULL,
    sent_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nudges_ref ON customer_nudges(ref);

-- What a competitor advertises, found online by the market research agent and
-- reused per lender until it is old (config/market_research.yaml).
CREATE TABLE IF NOT EXISTS market_rate_research (
    research_id     TEXT PRIMARY KEY,
    lender          TEXT NOT NULL,           -- registry name, or 'market'
    status          TEXT NOT NULL,           -- ok | not_found | failed
    researched_at   TEXT NOT NULL,
    published_min   REAL,
    published_max   REAL,
    payload         TEXT NOT NULL,           -- rates with their sources, notes, searches
    requested_by    TEXT
);
CREATE INDEX IF NOT EXISTS idx_market_rate_research ON market_rate_research(lender, researched_at);

-- An offer the relationship manager sent, and what the customer then decided.
CREATE TABLE IF NOT EXISTS offers (
    offer_id        TEXT PRIMARY KEY,
    loan_id         TEXT NOT NULL,
    decision_id     TEXT NOT NULL,
    rate            REAL NOT NULL,
    recommended_rate REAL NOT NULL,
    route           TEXT NOT NULL,
    reason          TEXT,
    note            TEXT,
    request_id      TEXT,
    sent_by         TEXT NOT NULL,
    sent_at         TEXT NOT NULL,
    status          TEXT NOT NULL,
    customer_decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_offers_loan ON offers(loan_id);

-- A lower rate the relationship manager asked the area manager to approve.
CREATE TABLE IF NOT EXISTS rate_requests (
    request_id      TEXT PRIMARY KEY,
    loan_id         TEXT NOT NULL,
    decision_id     TEXT NOT NULL,
    applicant_name  TEXT,
    loan_amount     REAL,
    requested_rate  REAL NOT NULL,
    recommended_rate REAL NOT NULL,
    reason          TEXT NOT NULL,
    note            TEXT,
    evidence_verified INTEGER DEFAULT 0,
    analysis        TEXT,
    requested_by    TEXT NOT NULL,
    requested_at    TEXT NOT NULL,
    status          TEXT NOT NULL,
    decided_by      TEXT,
    decided_at      TEXT,
    approved_rate   REAL,
    decision_reason TEXT,
    decision_note   TEXT
);
CREATE INDEX IF NOT EXISTS idx_rate_requests_status ON rate_requests(status);

CREATE TABLE IF NOT EXISTS applications (
    loan_id         TEXT PRIMARY KEY,
    owner           TEXT NOT NULL,
    applicant_name  TEXT,
    applicant       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_applications_owner ON applications(owner);

CREATE TABLE IF NOT EXISTS application_documents (
    document_id     TEXT PRIMARY KEY,
    loan_id         TEXT NOT NULL,
    document_type   TEXT NOT NULL,
    filename        TEXT NOT NULL,
    content_type    TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    sha256          TEXT NOT NULL,
    uploaded_by     TEXT NOT NULL,
    uploaded_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_loan ON application_documents(loan_id);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id     TEXT PRIMARY KEY,
    loan_id         TEXT NOT NULL,
    stage           TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    recommended_rate REAL,
    card_rate       REAL,
    min_rate        REAL,
    max_rate        REAL,
    conversion_probability REAL,
    expected_value_inr REAL,
    required_approval TEXT,
    policy_version  TEXT,
    model_versions  TEXT,
    elapsed_ms      REAL,
    payload         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_loan ON decisions(loan_id);
CREATE INDEX IF NOT EXISTS idx_decisions_created ON decisions(created_at);

CREATE TABLE IF NOT EXISTS overrides (
    override_id     TEXT PRIMARY KEY,
    decision_id     TEXT,
    loan_id         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    ai_rate         REAL NOT NULL,
    human_rate      REAL NOT NULL,
    override_bps    REAL NOT NULL,
    reason          TEXT NOT NULL,
    reason_detail   TEXT,
    evidence_verified INTEGER DEFAULT 0,
    approved_by     TEXT,
    required_approval TEXT,
    within_authority INTEGER,
    sales_context   TEXT,
    payload         TEXT
);
CREATE INDEX IF NOT EXISTS idx_overrides_loan ON overrides(loan_id);

-- Keyed by outcome_id, not loan_id. A file priced more than once used to have one
-- outcome and no way to say which decision it followed, which is precisely the
-- join the learning loop needs. `decision_id` is the join; `loan_id` is kept
-- because an outcome can arrive for a file the engine never priced.
CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id      TEXT PRIMARY KEY,
    loan_id         TEXT NOT NULL,
    decision_id     TEXT,
    updated_at      TEXT NOT NULL,
    customer_response TEXT,
    disbursed       INTEGER,
    sanctioned_rate REAL,
    bt_out          INTEGER,
    delinquent      INTEGER,
    months_to_exit  INTEGER,
    realized_yield_pct REAL,
    realized_value_pct REAL,
    -- Why the file was lost. "rejected" says nothing an underwriter can act on.
    lost_to         TEXT,
    lost_to_rate    REAL,
    loss_reason     TEXT,
    payload         TEXT
);
CREATE INDEX IF NOT EXISTS idx_outcomes_loan ON outcomes(loan_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_decision ON outcomes(decision_id);

CREATE TABLE IF NOT EXISTS audit_log (
    entry_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,
    entity_type     TEXT,
    entity_id       TEXT,
    detail          TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);

-- A discovered feature candidate put forward for admission. A proposal is not an
-- admission: the contract is only widened by an approved change to
-- feature_config.yaml, and a feature with no human-authored reason label cannot
-- be admitted however strong it is.
CREATE TABLE IF NOT EXISTS feature_proposals (
    proposal_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    stage           TEXT NOT NULL,
    recipe          TEXT NOT NULL,
    inputs          TEXT NOT NULL,
    generator       TEXT,
    proposed_at     TEXT NOT NULL,
    proposed_by     TEXT,
    dataset_id      TEXT,
    tests           TEXT,
    refusals        TEXT,
    state           TEXT NOT NULL      -- proposed | admitted | declined
);
CREATE INDEX IF NOT EXISTS idx_feature_proposals ON feature_proposals(stage, state);

-- Prompts as versioned objects. A prompt is model input: an unversioned one on a
-- priced path means a decision cannot say what the model was asked.
CREATE TABLE IF NOT EXISTS prompts (
    prompt_id       TEXT PRIMARY KEY,   -- name + content hash
    name            TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    registered_at   TEXT NOT NULL,
    role            TEXT,               -- system | user | tool
    used_by         TEXT,
    body            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompts(name);

-- What the knowledge indexes were built from. Rebuilt at every start, so without
-- this a policy answer cannot be traced to the text that produced it.
CREATE TABLE IF NOT EXISTS knowledge_indexes (
    index_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    built_at        TEXT NOT NULL,
    backend         TEXT,
    documents       INTEGER,
    corpus          TEXT,
    content_hash    TEXT NOT NULL
);

-- A federated round, its participants and the adjustment it produced. Without
-- this no decision can name the branch model that was active.
CREATE TABLE IF NOT EXISTS federated_rounds (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    proximal_mu     REAL,
    rounds          INTEGER,
    participants    INTEGER,
    global_version  TEXT,
    final_mae_bps   REAL,
    client_sizes    TEXT,
    per_region      TEXT,
    global_parameters TEXT
);

-- A monitoring pass that actually ran. Drift computed when someone opens a page
-- is a report; a monitor is something that records having run and what it found.
CREATE TABLE IF NOT EXISTS monitor_runs (
    run_id          TEXT PRIMARY KEY,
    monitor         TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT,               -- ok | warn | alert | failed
    triggered       INTEGER DEFAULT 0,
    findings        TEXT,
    actor           TEXT
);
CREATE INDEX IF NOT EXISTS idx_monitor_runs ON monitor_runs(monitor, started_at);

-- Change requests against the system itself, and their sign-off. Distinct from
-- the authority check on a single priced decision: that one asks who signs this
-- rate, this one asks who signs a change to the thing that produces rates.
CREATE TABLE IF NOT EXISTS approvals (
    approval_id     TEXT PRIMARY KEY,
    artefact_type   TEXT NOT NULL,
    artefact_id     TEXT NOT NULL,
    artefact_hash   TEXT,
    action          TEXT NOT NULL,
    state           TEXT NOT NULL,      -- pending | approved | rejected | withdrawn | applied
    requested_by    TEXT NOT NULL,
    requested_at    TEXT NOT NULL,
    summary         TEXT,
    required_roles  TEXT NOT NULL,
    gate_results    TEXT,
    gates_passed    INTEGER,
    override_justification TEXT,
    decided_at      TEXT,
    applied_at      TEXT,
    detail          TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_state ON approvals(state);
CREATE INDEX IF NOT EXISTS idx_approvals_artefact ON approvals(artefact_type, artefact_id);

-- One signature. An approval needs every required role, so the signatures are
-- rows rather than a column - and who signed what is the audit trail.
CREATE TABLE IF NOT EXISTS approval_signatures (
    signature_id    TEXT PRIMARY KEY,
    approval_id     TEXT NOT NULL,
    role            TEXT NOT NULL,
    principal       TEXT NOT NULL,
    verdict         TEXT NOT NULL,      -- approve | reject
    signed_at       TEXT NOT NULL,
    comment         TEXT
);
CREATE INDEX IF NOT EXISTS idx_signatures_approval ON approval_signatures(approval_id);

-- Every request that reached a priced endpoint, written BEFORE the pipeline runs
-- and updated when it finishes. The decisions table only holds requests that
-- succeeded, so a success rate computed over it is 100% by construction.
CREATE TABLE IF NOT EXISTS requests (
    request_id      TEXT PRIMARY KEY,
    route           TEXT NOT NULL,
    method          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status_code     INTEGER,
    outcome         TEXT,            -- ok | client_error | failed | abandoned
    elapsed_ms      REAL,
    decision_id     TEXT,
    loan_id         TEXT,
    error_type      TEXT,
    error_detail    TEXT,
    actor           TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_started ON requests(started_at);
CREATE INDEX IF NOT EXISTS idx_requests_outcome ON requests(outcome);

-- The agent set as a registered object. The id is a content hash over the
-- agent, skill and tool declarations, so it cannot drift from the code it names.
CREATE TABLE IF NOT EXISTS agent_manifests (
    manifest_id     TEXT PRIMARY KEY,
    content_hash    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    agents          INTEGER,
    wired           INTEGER,
    skills          INTEGER,
    tools           INTEGER,
    manifest        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'champion',
    registered_at   TEXT NOT NULL,
    card            TEXT NOT NULL,
    PRIMARY KEY (name, version)
);
"""


def database_path() -> Path:
    url = env("AAVAS_STORE_URL", "sqlite:///aavas_governance.db") or ""
    if url.startswith("sqlite:///"):
        raw = url.replace("sqlite:///", "", 1)
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    # Postgres and friends are a phase-2 swap; the POC keeps a local file.
    log.warning("store URL %r is not sqlite; falling back to a local sqlite file", url)
    return ROOT / "aavas_governance.db"


LEGACY_OUTCOME_COLUMNS = ["loan_id", "updated_at", "customer_response", "disbursed",
                          "sanctioned_rate", "bt_out", "delinquent", "months_to_exit",
                          "realized_yield_pct", "realized_value_pct", "payload"]


def _rename_legacy(conn: sqlite3.Connection) -> list[str] | None:
    """Move an out-of-date ``outcomes`` table aside before the schema is applied.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op against a table that already exists,
    so a store written before the outcome join keeps its old shape forever unless
    it is migrated - and the new index over ``decision_id`` then fails against a
    table that has no such column. The rename has to happen first.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(outcomes)")}
    if not columns or "outcome_id" in columns:
        return None
    log.info("migrating outcomes to the decision-scoped schema")
    conn.execute("ALTER TABLE outcomes RENAME TO outcomes_legacy")
    return [c for c in LEGACY_OUTCOME_COLUMNS if c in columns]


def _carry_legacy(conn: sqlite3.Connection, carried: list[str]) -> None:
    """Copy legacy rows across. They keep their evidence and lose nothing.

    ``decision_id`` is NULL rather than guessed: an outcome recorded when the join
    did not exist genuinely cannot say which decision it followed, and inventing
    one would be worse than admitting it.
    """
    names = ", ".join(carried)
    conn.execute(
        f"INSERT OR IGNORE INTO outcomes (outcome_id, decision_id, {names}) "
        f"SELECT 'OUT-LEGACY-' || loan_id, NULL, {names} FROM outcomes_legacy")
    moved = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    conn.execute("DROP TABLE outcomes_legacy")
    log.info("carried %s legacy outcome(s) across with decision_id NULL", moved)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = database_path()
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        with _LOCK:
            carried = _rename_legacy(conn)
            conn.executescript(SCHEMA)
            if carried:
                _carry_legacy(conn, carried)
        yield conn
        conn.commit()
    finally:
        conn.close()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def insert(table: str, row: dict[str, Any]) -> None:
    payload = {k: _encode(v) for k, v in row.items()}
    columns = ", ".join(payload)
    placeholders = ", ".join("?" for _ in payload)
    with connect() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})",
            list(payload.values()),
        )


def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def fetch_one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def count(table: str) -> int:
    row = fetch_one(f"SELECT COUNT(*) AS n FROM {table}")
    return int(row["n"]) if row else 0


def reset() -> None:
    """Drop every table. Tests and demos only."""
    with connect() as conn:
        for table in ("decisions", "overrides", "outcomes", "audit_log", "model_registry",
                      "agent_manifests", "requests", "approvals", "approval_signatures",
                      "monitor_runs",
                      "federated_rounds", "knowledge_indexes", "prompts",
                      "feature_proposals", "applications", "application_documents",
                      "loan_book", "datasets", "config_changes", "borrower_accounts",
                      "offers", "rate_requests", "market_rate_research", "customer_logins",
                      "customer_nudges"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.executescript(SCHEMA)
    log.info("governance store reset (%s)", database_path())


def _encode(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    if isinstance(value, bool):
        return int(value)
    return value


def decode(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in "{[":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
