"""Serve a SageMaker-trained bundle from a SageMaker Serverless Inference endpoint.

The endpoint runs the scikit-learn image the bundle was trained in, with the
inference handler and aavas source the training job packed into the artefact
(deploy/sagemaker/inference.py). Serverless: billed per request, nothing while idle;
the first call after a quiet spell waits while a container starts.

Deploying is checked, not assumed: after the endpoint is InService, it is called with
the reference rows the training job recorded, and every model's answer must match
what the job predicted on the same rows.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from . import runs

MODELS = ("stage1", "stage2", "conversion", "prepayment", "lgd", "risk", "override", "uplift")


class ServingError(Exception):
    pass


def _clients(region: str):
    import boto3
    from botocore.config import Config
    slow = Config(read_timeout=180, connect_timeout=20, retries={"max_attempts": 2})
    return boto3.client("sagemaker", region_name=region), boto3.client("sagemaker-runtime", region_name=region,
                                                                       config=slow)


def status(cfg: dict[str, Any]) -> dict[str, Any]:
    scfg = cfg["serving"]["sagemaker_serverless"]
    region = cfg["training"]["sagemaker"]["region"]
    out = {"mode": cfg["serving"]["mode"], "endpoint_name": scfg["endpoint_name"], "region": region, "exists": False}
    try:
        sm, _ = _clients(region)
        ep = sm.describe_endpoint(EndpointName=scfg["endpoint_name"])
    except Exception as exc:                                         # noqa: BLE001 - absent or unreachable
        text = str(exc)
        out["detail"] = "not deployed yet" if "Could not find endpoint" in text else text
        return out
    out.update(exists=True, status=ep["EndpointStatus"], created_at=str(ep.get("CreationTime")),
               last_modified=str(ep.get("LastModifiedTime")), endpoint_config=ep.get("EndpointConfigName"),
               failure_reason=ep.get("FailureReason"))
    try:
        conf = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
        variant = conf["ProductionVariants"][0]
        out["model_name"] = variant["ModelName"]
        out["serverless"] = variant.get("ServerlessConfig")
        model = sm.describe_model(ModelName=variant["ModelName"])
        out["model_data"] = model["PrimaryContainer"].get("ModelDataUrl")
    except Exception as exc:                                         # noqa: BLE001
        out["detail"] = str(exc)
    return out


def invoke(cfg: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    scfg = cfg["serving"]["sagemaker_serverless"]
    _, rt = _clients(cfg["training"]["sagemaker"]["region"])
    started = time.perf_counter()
    try:
        resp = rt.invoke_endpoint(EndpointName=scfg["endpoint_name"], ContentType="application/json",
                                  Accept="application/json", Body=json.dumps(payload))
    except Exception as exc:                                         # noqa: BLE001
        raise ServingError(f"endpoint {scfg['endpoint_name']} did not answer: {exc}") from exc
    body = json.loads(resp["Body"].read())
    body["latency_ms"] = round((time.perf_counter() - started) * 1000)
    body["endpoint"] = scfg["endpoint_name"]
    return body


def deploy(run: runs.Run, training: dict[str, Any]) -> dict[str, Any]:
    from .workflow import StepError
    cfg = run.config
    from .engines.sagemaker import aws_settings
    scfg, tcfg = cfg["serving"]["sagemaker_serverless"], aws_settings(cfg["training"]["sagemaker"])
    artifact = training.get("artifact_uri")
    if not artifact:
        raise StepError("no SageMaker model artefact to serve: the training step did not produce one")
    sm, _ = _clients(tcfg["region"])
    name = scfg["endpoint_name"]
    attempt = time.strftime("%H%M%S")                    # a retry must not collide with a failed attempt's names
    model_name = f"aavas-{run.run_id}-{attempt}"[:63]
    config_name = f"{model_name}-cfg"[:63]
    tags = [{"Key": "project", "Value": "aavas-pricing"}, {"Key": "agentic-run", "Value": run.run_id}]
    code_uri = _upload_serving_code(run, tcfg)

    sm.create_model(
        ModelName=model_name, ExecutionRoleArn=tcfg["role_arn"], Tags=tags,
        PrimaryContainer={"Image": tcfg["image"], "ModelDataUrl": artifact, "Environment": {
            # The scikit-learn serving container loads its code from an S3 sourcedir.tar.gz,
            # not from inside the model artefact.
            "SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": code_uri,
            # The container pip-installs that code into a user site created after Python
            # started, and /opt/ml/code is created after start-up too, so Python caches it
            # as missing. The model artefact is unpacked to /opt/ml/model before the
            # container starts and carries the same code/ folder: import from there.
            "PYTHONPATH": "/opt/ml/model/code:/opt/ml/code",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20", "SAGEMAKER_REGION": tcfg["region"],
            "AAVAS_STORE_URL": "sqlite:////tmp/aavas_governance.db"}})
    sm.create_endpoint_config(
        EndpointConfigName=config_name, Tags=tags,
        ProductionVariants=[{"VariantName": "AllTraffic", "ModelName": model_name,
                             "ServerlessConfig": {"MemorySizeInMB": int(scfg["memory_mb"]),
                                                  "MaxConcurrency": int(scfg["max_concurrency"])}}])
    current = status(cfg)
    if current.get("exists") and current.get("status") == "Failed":
        sm.delete_endpoint(EndpointName=name)            # a failed endpoint cannot be updated
        run.log(f"[serving] deleted failed endpoint {name}")
        for _ in range(60):
            if not status(cfg).get("exists"):
                break
            time.sleep(10)
        current = status(cfg)
    if current.get("exists"):
        sm.update_endpoint(EndpointName=name, EndpointConfigName=config_name)
        run.log(f"[serving] updating endpoint {name} to {config_name}")
    else:
        sm.create_endpoint(EndpointName=name, EndpointConfigName=config_name, Tags=tags)
        run.log(f"[serving] creating serverless endpoint {name} ({scfg['memory_mb']} MB, "
                f"concurrency {scfg['max_concurrency']})")

    deadline, last = time.time() + 30 * 60, None
    while True:
        ep = sm.describe_endpoint(EndpointName=name)
        if ep["EndpointStatus"] != last:
            run.log(f"[serving] {name}: {ep['EndpointStatus']}")
            last = ep["EndpointStatus"]
        if ep["EndpointStatus"] == "InService":
            break
        if ep["EndpointStatus"] in ("Failed", "OutOfService") or time.time() > deadline:
            raise StepError(f"endpoint {name} {ep['EndpointStatus'].lower()}: "
                            f"{ep.get('FailureReason', 'did not come into service in 30 minutes')}")
        time.sleep(30)

    parity = check_parity(run, cfg, float(scfg.get("parity_tolerance", 1e-6)))
    matched = [m for m, p in parity["models"].items() if p["ok"]]
    details = {"endpoint_name": name, "status": "InService", "model_name": model_name,
               "endpoint_config": config_name, "artifact_uri": artifact, "parity": parity}
    if len(matched) != len(parity["models"]):
        raise StepError(f"endpoint {name} is up but disagrees with training on "
                        f"{', '.join(sorted(set(parity['models']) - set(matched)))}", details)
    details["summary"] = (f"serverless endpoint {name} InService ({scfg['memory_mb']} MB); all "
                          f"{len(matched)} models answer exactly as training recorded "
                          f"(first call {parity['first_call_ms']} ms, then {parity['warm_call_ms']} ms)")
    return details


def _upload_serving_code(run: runs.Run, tcfg: dict[str, Any]) -> str:
    """Repack the artefact's code/ folder (inference.py, aavas source, config, lib) as a
    sourcedir.tar.gz in S3, where the scikit-learn serving container fetches its code."""
    import io
    import tarfile

    import boto3

    local = run.workspace / "sagemaker" / "model.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(local) as src, tarfile.open(fileobj=buf, mode="w:gz") as out:
        for member in src.getmembers():
            name = member.name.lstrip("./")
            if not name.startswith("code/") or not member.isfile() or "__pycache__" in name:
                continue
            member.name = name[len("code/"):]
            out.addfile(member, src.extractfile(member))
    key = f"{tcfg['prefix'].strip('/')}/runs/{run.run_id}/serving/sourcedir.tar.gz"
    boto3.client("s3", region_name=tcfg["region"]).put_object(Bucket=tcfg["bucket"], Key=key, Body=buf.getvalue())
    run.log(f"[serving] uploaded the serving code to s3://{tcfg['bucket']}/{key}")
    return f"s3://{tcfg['bucket']}/{key}"


def check_parity(run: runs.Run, cfg: dict[str, Any], tolerance: float) -> dict[str, Any]:
    reference = json.loads((run.workspace / "sagemaker" / "reference.json").read_text(encoding="utf-8"))
    rows = reference["rows"]
    out: dict[str, Any] = {"rows": len(rows), "models": {}}
    timings = []
    for model, expected in reference["predictions"].items():
        if isinstance(expected, dict) and "error" in expected:
            out["models"][model] = {"ok": False, "detail": f"training could not predict: {expected['error']}"}
            continue
        answer = invoke(cfg, {"model": model, "rows": rows})
        timings.append(answer["latency_ms"])
        diff = _max_diff(answer.get("predictions"), expected)
        out["models"][model] = {"ok": diff is not None and diff <= tolerance, "max_abs_diff": diff,
                                "latency_ms": answer["latency_ms"]}
        run.log(f"[serving] parity {model}: max difference {diff} in {answer['latency_ms']} ms")
    out["first_call_ms"] = timings[0] if timings else None
    out["warm_call_ms"] = round(sum(timings[1:]) / len(timings[1:])) if len(timings) > 1 else None
    return out


def _max_diff(got, want) -> float | None:
    """Largest numeric difference between two JSON values of the same shape; None if shapes differ."""
    if isinstance(want, dict) and isinstance(got, dict):
        if set(want) != set(got):
            return None
        diffs = [_max_diff(got[k], want[k]) for k in want]
        return None if None in diffs else max(diffs, default=0.0)
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return None
        diffs = [_max_diff(g, w) for g, w in zip(got, want)]
        return None if None in diffs else max(diffs, default=0.0)
    if isinstance(want, (int, float)) and isinstance(got, (int, float)) and not isinstance(want, bool):
        if math.isnan(want) and math.isnan(got):
            return 0.0
        return abs(float(got) - float(want))
    return 0.0 if got == want else None
