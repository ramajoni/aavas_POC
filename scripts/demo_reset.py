"""Clear the demo's loan files so it starts again under the policy in force.

Removes every submitted application and what hangs off it - documents (and their
uploaded files), priced decisions, rate requests, offers, overrides, outcomes,
nudges and the customer logins issued for them - and the policy edit history, so
only the current pricing policy is left. The audit log is kept: it is the record
of what happened, and this reset is written into it.

A copy of the store is taken first, beside it, unless ``--no-backup`` is given.

    python scripts/demo_reset.py            # back up, then clear
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import time

from aavas.config import policy
from aavas.governance import audit, store
from aavas.intake import applications

TABLES = ("application_documents", "applications", "rate_requests", "offers", "overrides", "outcomes",
          "decisions", "customer_nudges", "customer_logins", "borrower_accounts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    path = store.database_path()
    if not args.no_backup:
        copy = path.with_name(f"{path.stem}.before-demo-reset-{time.strftime('%Y%m%d%H%M')}.db")
        source, target = sqlite3.connect(path), sqlite3.connect(copy)
        source.backup(target)
        target.close(), source.close()
        print(f"backed up to {copy}")
    counts = {t: store.count(t) for t in TABLES}
    with store.connect() as conn:
        for table in TABLES:
            conn.execute(f"DELETE FROM {table}")
        try:
            conn.execute("DELETE FROM config_changes WHERE name = 'policy'")
        except sqlite3.OperationalError:                           # no edit history yet
            pass
    folder = applications.upload_dir()
    if folder.exists():
        shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    audit.record("demo_reset", "cleared_demo_files", "store", "demo",
                 {"removed": counts, "policy_version": policy().get("policy_version")})
    print("removed", counts)


if __name__ == "__main__":
    main()
