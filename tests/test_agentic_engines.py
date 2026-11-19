"""The AutoGluon Assistant agent runs code it wrote itself: it must not inherit secrets."""

from __future__ import annotations

from pathlib import Path

from aavas.agentic.engines import mlzero

SECRETS = {"AAVAS_O2_PASSWORD": "x", "AAVAS_O2_WEBHOOK_TOKEN": "y", "ZO_ROOT_USER_PASSWORD": "z",
           "AAVAS_STORE_URL": "sqlite:///store.db", "AAVAS_RAW_PATH": "raw.csv", "OPENAI_API_KEY": "k"}


def test_the_agent_gets_its_llm_credentials_and_nothing_of_ours() -> None:
    env = {"PATH": "C:/bin", "SYSTEMROOT": "C:/Windows", "AWS_PROFILE": "dev", **SECRETS}
    out = mlzero.agent_environment(env, {"provider": "bedrock", "region": "us-west-2"}, Path("C:/v/python.exe"))
    assert not set(SECRETS) & set(out)
    assert out["AWS_PROFILE"] == "dev" and out["AWS_REGION"] == "us-west-2"
    assert out["PATH"].startswith(str(Path("C:/v")))


def test_another_provider_gets_only_its_own_key() -> None:
    env = {"PATH": "C:/bin", "AWS_SECRET_ACCESS_KEY": "a", "ANTHROPIC_API_KEY": "b", **SECRETS}
    out = mlzero.agent_environment(env, {"provider": "anthropic", "region": "x"}, Path("C:/v/python.exe"))
    assert out["ANTHROPIC_API_KEY"] == "b"
    assert "AWS_SECRET_ACCESS_KEY" not in out and not set(SECRETS) & set(out)


# ------------------------------------------- SageMaker account values (not committed)

def test_sagemaker_account_values_come_from_the_environment(monkeypatch, tmp_path) -> None:
    import pytest

    from aavas import config
    from aavas.agentic.engines import sagemaker
    from aavas.agentic.workflow import StepError

    monkeypatch.setattr(config, "env_or_file", lambda key, path=None: {
        sagemaker.ROLE_ENV: "arn:aws:iam::000000000000:role/test", sagemaker.BUCKET_ENV: "test-bucket"}.get(key))
    filled = sagemaker.aws_settings({"role_arn": None, "bucket": None, "region": "ap-south-1"})
    assert filled["role_arn"].endswith("role/test") and filled["bucket"] == "test-bucket"
    assert sagemaker.aws_settings({"role_arn": "arn:x", "bucket": "b"})["bucket"] == "b"   # config wins
    monkeypatch.setattr(config, "env_or_file", lambda key, path=None: None)
    with pytest.raises(StepError, match="AAVAS_SAGEMAKER_ROLE_ARN"):
        sagemaker.aws_settings({"role_arn": None, "bucket": "b"})


def test_env_or_file_reads_the_dotenv_when_the_environment_is_silent(monkeypatch, tmp_path) -> None:
    from aavas.config import env_or_file

    dotenv = tmp_path / ".env"
    dotenv.write_text("# AAVAS_X=commented\nAAVAS_X='from-file'\n", encoding="utf-8")
    monkeypatch.delenv("AAVAS_X", raising=False)
    assert env_or_file("AAVAS_X", dotenv) == "from-file"
    monkeypatch.setenv("AAVAS_X", "from-env")
    assert env_or_file("AAVAS_X", dotenv) == "from-env"
    assert env_or_file("AAVAS_MISSING", dotenv) is None


def test_no_aws_account_number_is_committed_in_config() -> None:
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "config"
    for path in root.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        own = [m for m in re.findall(r"arn:aws:iam::(\d{12})|-(\d{12})\b", text) if any(m)]
        assert not own, f"{path.name} carries an AWS account number; use the environment"
