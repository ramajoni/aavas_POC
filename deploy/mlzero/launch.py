"""Run the AutoGluon Assistant (MLZero) CLI with two adjustments, then the unmodified CLI.

1. Bedrock model validation also accepts the region's inference profiles (see
   ``_bedrock_models_and_profiles``).
2. An executor that also works on Windows (below).

MLZero runs the code it writes with ``python -c`` / ``bash -c`` and streams the
output with ``select.select`` on the process pipes. Windows' select accepts sockets
only, so every execution would fail there and the agent would debug code that was
never wrong. On Windows this launcher swaps in a thread-based reader with the same
contract - (success, stdout, stderr), de-duplicated lines, the same time limit - and
then hands over to the unmodified CLI. Elsewhere it changes nothing.

    python deploy/mlzero/launch.py -i <input folder> -o <output folder> -c <config.yaml> -n 3 -t "<instruction>"
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections import deque


def execute_code(code: str, language: str, timeout: float):
    if language.lower() == "python":
        cmd = [sys.executable, "-c", code]          # the interpreter MLZero itself runs in
    elif language.lower() == "bash":
        cmd = ["bash", "-c", code]
    else:
        return False, "", f"Unsupported language: {language}. Use 'python' or 'bash'."
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1)
    except Exception as exc:                                         # noqa: BLE001 - reported to the agent
        return False, "", f"Error executing {language} code: {exc}"

    chunks = {"out": [], "err": []}

    def pump(stream, key):
        recent = deque(maxlen=100)                                   # MLZero's de-duplication
        for line in stream:
            if line in recent:
                continue
            recent.append(line)
            chunks[key].append(line)
            print(line, end="", flush=True)

    readers = [threading.Thread(target=pump, args=(process.stdout, "out"), daemon=True),
               threading.Thread(target=pump, args=(process.stderr, "err"), daemon=True)]
    for r in readers:
        r.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        time.sleep(3)
        if process.poll() is None:
            process.kill()
        chunks["out"].append(f"\nProcess reached time limit after {timeout} seconds.\n")
    for r in readers:
        r.join(timeout=5)
    return process.returncode == 0, "".join(chunks["out"]), "".join(chunks["err"])


def _bedrock_models_and_profiles():
    """MLZero validates the model id against Bedrock's foundation models only, with a
    3-character prefix strip for "us." profiles. Cross-region profiles such as
    "global." or "apac." - the way ap-south-1 reaches current Claude models - would be
    refused. Accept the region's inference profiles as well: still a real check."""
    import os

    import boto3

    bedrock = boto3.client("bedrock", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    models = [m["modelId"] for m in bedrock.list_foundation_models()["modelSummaries"]]
    profiles, token = [], None
    while True:
        page = bedrock.list_inference_profiles(**({"nextToken": token} if token else {}))
        profiles += [p["inferenceProfileId"] for p in page.get("inferenceProfileSummaries", [])]
        token = page.get("nextToken")
        if not token:
            return models + profiles


def main() -> None:
    if sys.platform == "win32":
        from autogluon.assistant.agents import executer_agent
        executer_agent.execute_code = execute_code
    from autogluon.assistant.llm import llm_factory
    llm_factory.get_bedrock_models = _bedrock_models_and_profiles
    from autogluon.assistant.cli.app import app
    app(prog_name="mlzero")


if __name__ == "__main__":
    main()
