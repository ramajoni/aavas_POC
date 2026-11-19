"""Train the model suite as an Amazon SageMaker training job.

Packs the aavas source and config with deploy/sagemaker/train_entry.py, uploads the
run's dataset, starts a script-mode job in the scikit-learn image (the same image the
serverless endpoint serves with), follows it - status and CloudWatch log lines into
the run log - and brings back the model cards for the comparison with the champions.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
from pathlib import Path

from ...governance import model_registry
from ...models import base as model_base
from ...paths import ROOT
from .. import runs
from ..workflow import StepError, data_files

SOURCE = [("src/aavas", "src/aavas"), ("config", "config"), ("scripts/train_all.py", "scripts/train_all.py"),
          ("reports/automl/portfolio", "reports/automl/portfolio"),
          ("deploy/sagemaker/train_entry.py", "train_entry.py"),
          ("deploy/sagemaker/inference.py", "inference.py"),
          ("deploy/sagemaker/requirements.txt", "requirements.txt")]
SKIP = {"__pycache__", ".pytest_cache"}


def _clients(cfg):
    import boto3
    region = cfg["region"]
    return boto3.client("sagemaker", region_name=region), boto3.client("s3", region_name=region), \
        boto3.client("logs", region_name=region)


def source_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src, arc in SOURCE:
            path = ROOT / src
            if not path.exists():
                continue
            tar.add(path, arcname=arc, filter=lambda info: None if any(p in SKIP for p in Path(info.name).parts)
                    or info.name.endswith(".pyc") else info)
    return buf.getvalue()


ROLE_ENV, BUCKET_ENV = "AAVAS_SAGEMAKER_ROLE_ARN", "AAVAS_SAGEMAKER_BUCKET"


def aws_settings(cfg: dict) -> dict:
    """The SageMaker settings with the account-specific values filled in.

    The execution role ARN and the bucket name carry the AWS account number, and the
    repository is public, so they are not committed: leave them empty in
    config/agentic_os.yaml and set AAVAS_SAGEMAKER_ROLE_ARN and AAVAS_SAGEMAKER_BUCKET
    in the environment or the git-ignored .env. A value written in the config wins.
    """
    from ...config import env_or_file
    out = dict(cfg)
    for key, var in (("role_arn", ROLE_ENV), ("bucket", BUCKET_ENV)):
        out[key] = out.get(key) or env_or_file(var)
        if not out[key]:
            raise StepError(f"no SageMaker {key.replace('_', ' ')}: set {var} in the environment or in .env")
    return out


def job_name(run: runs.Run) -> str:
    return f"aavas-{run.run_id}"[:63]


def train(run: runs.Run, env: dict[str, str]) -> dict:
    cfg = aws_settings(run.config["training"]["sagemaker"])
    models = list(run.config["training"]["models"])
    sm, s3, logs = _clients(cfg)
    bucket, prefix = cfg["bucket"], f"{cfg['prefix'].strip('/')}/runs/{run.run_id}"

    files = data_files(run)
    for key in ("raw", "notes"):
        if files[key].exists():
            s3.upload_file(str(files[key]), bucket, f"{prefix}/data/train/{files[key].name}")
    code_key = f"{prefix}/code/sourcedir.tar.gz"
    s3.put_object(Bucket=bucket, Key=code_key, Body=source_tarball())
    run.log(f"[training] uploaded the dataset and source to s3://{bucket}/{prefix}/")

    name = job_name(run)
    sm.create_training_job(
        TrainingJobName=name,
        RoleArn=cfg["role_arn"],
        AlgorithmSpecification={"TrainingImage": cfg["image"], "TrainingInputMode": "File"},
        HyperParameters={
            "sagemaker_program": json.dumps("train_entry.py"),
            "sagemaker_submit_directory": json.dumps(f"s3://{bucket}/{code_key}"),
            "sagemaker_region": json.dumps(cfg["region"]),
            "models": json.dumps(",".join(models)),
            "role": json.dumps(run.config["training"].get("register_as", "challenger")),
        },
        InputDataConfig=[{"ChannelName": "train", "ContentType": "text/csv",
                          "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                                                          "S3DataDistributionType": "FullyReplicated",
                                                          "S3Uri": f"s3://{bucket}/{prefix}/data/train/"}}}],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/{prefix}/output/"},
        ResourceConfig={"InstanceType": cfg["instance_type"], "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": int(cfg["max_runtime_minutes"]) * 60},
        EnableNetworkIsolation=False,
        Tags=[{"Key": "project", "Value": "aavas-pricing"}, {"Key": "agentic-run", "Value": run.run_id}],
    )
    run.log(f"[training] started SageMaker training job {name} on {cfg['instance_type']}")
    run.record["sagemaker_job"] = name
    run.save()

    job = _follow(run, sm, logs, name, int(cfg.get("poll_seconds", 30)))
    if job["TrainingJobStatus"] != "Completed":
        raise StepError(f"SageMaker job {name} {job['TrainingJobStatus'].lower()}: "
                        f"{job.get('FailureReason', 'see the run log')}", {"job_name": name})

    artifact = job["ModelArtifacts"]["S3ModelArtifacts"]
    local = run.workspace / "sagemaker"
    local.mkdir(parents=True, exist_ok=True)
    a_bucket, a_key = artifact.replace("s3://", "").split("/", 1)
    s3.download_file(a_bucket, a_key, str(local / "model.tar.gz"))
    cards, environment, reference = _unpack(local)
    missing = sorted(set(models) - set(cards))
    if missing:
        raise StepError(f"the SageMaker job finished without a model card for {', '.join(missing)}")
    known = set(model_base.ModelCard.__dataclass_fields__)
    for card in cards.values():
        card.setdefault("limitations", []).append(
            f"trained by SageMaker job {name} (Python {environment.get('python')}, scikit-learn "
            f"{environment.get('scikit-learn')}); served by its endpoint, not loaded in-process")
        model_registry.register(model_base.ModelCard(**{k: v for k, v in card.items() if k in known}),
                                role="challenger")
    seconds = job.get("BillableTimeInSeconds") or 0
    return {"summary": f"SageMaker job {name} trained {len(cards)} models on {cfg['instance_type']} "
                       f"in {seconds // 60} min {seconds % 60} s billable",
            "engine": "sagemaker", "job_name": name, "artifact_uri": artifact, "cards": cards,
            "environment": environment, "billable_seconds": seconds, "instance_type": cfg["instance_type"],
            "reference_rows": len(reference.get("rows", []))}


def _unpack(local: Path):
    cards, environment, reference = {}, {}, {}
    with tarfile.open(local / "model.tar.gz") as tar:
        for member in tar.getmembers():
            name = member.name.lstrip("./")
            if name.startswith("models/") and name.endswith(".card.json"):
                card = json.load(tar.extractfile(member))
                cards[card.get("name") or Path(name).name.split("__")[0]] = card
            elif name == "environment.json":
                environment = json.load(tar.extractfile(member))
            elif name == "reference.json":
                reference = json.load(tar.extractfile(member))
    (local / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    (local / "environment.json").write_text(json.dumps(environment, indent=1), encoding="utf-8")
    return cards, environment, reference


def _follow(run, sm, logs, name: str, poll: int) -> dict:
    """Poll the job; copy its status changes and CloudWatch log lines into the run log."""
    last, token, stream = None, None, None
    while True:
        job = sm.describe_training_job(TrainingJobName=name)
        state = (job["TrainingJobStatus"], job.get("SecondaryStatus"))
        if state != last:
            run.log(f"[training] {name}: {state[0]} / {state[1]}")
            last = state
        try:
            if stream is None:
                found = logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs",
                                                  logStreamNamePrefix=name).get("logStreams", [])
                stream = found[0]["logStreamName"] if found else None
            if stream:
                kwargs = {"logGroupName": "/aws/sagemaker/TrainingJobs", "logStreamName": stream,
                          "startFromHead": True}
                if token:
                    kwargs["nextToken"] = token
                page = logs.get_log_events(**kwargs)
                for event in page.get("events", []):
                    message = event["message"].rstrip()
                    if message and "Warning" not in message:
                        run.log("  | " + message[:400])
                token = page.get("nextForwardToken", token)
        except Exception as exc:                                    # noqa: BLE001 - logs are best effort
            run.log(f"  (log read failed: {exc})")
        if job["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            return job
        time.sleep(poll)
