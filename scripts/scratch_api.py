"""A throwaway API on a copy of the governance store, for `demo_catalogue.py probe`.

    PYTHONPATH=src python scripts/scratch_api.py start --source aavas_governance.db --port 8011 --dir /tmp/scratch
    ... run the probe against http://127.0.0.1:8011 ...
    PYTHONPATH=src python scripts/scratch_api.py stop --dir /tmp/scratch

start copies the sqlite store with the backup API (never the live file in place), points
AAVAS_STORE_URL and AAVAS_UPLOAD_DIR at the copy, starts uvicorn on 127.0.0.1 and waits for
/health. stop kills that API and deletes the directory. Use the Python that runs the app
(on the server, its venv's python) with PYTHONPATH set to the app's src directory. Nothing
here touches the source store or any server already running.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def start(args) -> None:
    work = Path(args.dir).resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"{work} is not empty; run stop first or pick another --dir")
    (work / "uploads").mkdir(parents=True, exist_ok=True)
    copy = work / "store.db"
    src = sqlite3.connect(f"file:{Path(args.source).resolve().as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(copy)
    src.backup(dst)
    dst.close()
    src.close()
    env = {**os.environ, "AAVAS_STORE_URL": f"sqlite:///{copy.as_posix()}", "AAVAS_UPLOAD_DIR": str(work / "uploads")}
    log = open(work / "api.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "aavas.api.main:app", "--host", "127.0.0.1",
                             "--port", str(args.port)], env=env, stdout=log, stderr=subprocess.STDOUT)
    (work / "api.pid").write_text(str(proc.pid))
    url = f"http://127.0.0.1:{args.port}/health"
    for _ in range(args.wait):
        if proc.poll() is not None:
            raise SystemExit(f"the API exited; see {work / 'api.log'}")
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"scratch API up on port {args.port} (pid {proc.pid}), store copy {copy}")
            return
        except OSError:
            time.sleep(1)
    proc.terminate()
    raise SystemExit(f"the API did not answer on {url} within {args.wait}s; see {work / 'api.log'}")


def stop(args) -> None:
    work = Path(args.dir).resolve()
    pid_file = work / "api.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text())
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
        time.sleep(1)
    if work.exists() and (work / "store.db").exists():  # only ever delete a directory this helper made
        shutil.rmtree(work, ignore_errors=True)
    print("stopped and removed", work)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("start")
    st.add_argument("--source", required=True, help="the sqlite store to copy")
    st.add_argument("--port", type=int, default=8011)
    st.add_argument("--dir", required=True)
    st.add_argument("--wait", type=int, default=90)
    sp = sub.add_parser("stop")
    sp.add_argument("--dir", required=True)
    args = ap.parse_args()
    if args.cmd == "start":
        if args.port == 8000:
            raise SystemExit("refusing port 8000, the live local server")
        start(args)
    else:
        stop(args)


if __name__ == "__main__":
    main()
