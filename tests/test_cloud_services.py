"""Cloud-server behaviour: the browser reaches MLflow, OmniXAI and the MLZero CLI
differently from the API host (probe address vs public path)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from aavas.agentic.engines import mlzero
from aavas.api import agentic, tracking

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("run_omnixai_script", ROOT / "scripts" / "run_omnixai.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- MLflow

def test_mlflow_public_url_from_env(monkeypatch):
    monkeypatch.setenv("AAVAS_MLFLOW_UI_URL", "http://127.0.0.1:5000/")
    monkeypatch.setenv("AAVAS_MLFLOW_PUBLIC_URL", "/mlflow/")
    monkeypatch.setattr(tracking, "_listening", lambda *_a, **_k: True)
    svc = next(s for s in tracking._services() if s["kind"] == "mlflow")
    assert svc["url"] == "http://127.0.0.1:5000"
    assert svc["public_url"] == "/mlflow/"


def test_mlflow_public_url_falls_back_to_url(monkeypatch):
    monkeypatch.setenv("AAVAS_MLFLOW_UI_URL", "http://127.0.0.1:5000")
    monkeypatch.delenv("AAVAS_MLFLOW_PUBLIC_URL", raising=False)
    monkeypatch.setattr(tracking, "_listening", lambda *_a, **_k: True)
    svc = next(s for s in tracking._services() if s["kind"] == "mlflow")
    assert svc["public_url"] == svc["url"] == "http://127.0.0.1:5000"


def test_store_dir_accepts_absolute_file_uris(monkeypatch, tmp_path):
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "meta.yaml").write_text("experiment_id: '1'\nname: x\n", encoding="utf-8")
    posix = tmp_path.as_posix()
    for uri in (f"file:{posix}", f"file:///{posix.lstrip('/')}", posix):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        assert tracking._store_dir() == tmp_path, uri


# ----------------------------------------------------------------- OmniXAI

def test_omnixai_manifest_public_url(monkeypatch, tmp_path):
    manifest = tmp_path / "dashboards.json"
    manifest.write_text(json.dumps({"dashboards": [
        {"model": "stage2", "port": 8765, "url": "http://127.0.0.1:8765", "public_url": "/omnixai/8765/"},
        {"model": "risk", "port": 8766, "url": "http://127.0.0.1:8766"}]}), encoding="utf-8")
    monkeypatch.setattr(tracking, "DASHBOARD_MANIFEST", manifest)
    monkeypatch.setattr(tracking, "_listening", lambda *_a, **_k: False)
    out = [s for s in tracking._services() if s["kind"] == "omnixai"]
    assert out[0]["url"] == "http://127.0.0.1:8765" and out[0]["public_url"] == "/omnixai/8765/"
    assert out[1]["public_url"] == out[1]["url"] == "http://127.0.0.1:8766"


def test_run_omnixai_child_environment_sets_dash_prefix():
    script = _load_script()
    env = script.child_environment("/omnixai", 8766, {"PATH": "x"})
    assert env["DASH_REQUESTS_PATHNAME_PREFIX"] == "/omnixai/8766/"
    assert script.child_environment("omnixai/", 8765, {})["DASH_REQUESTS_PATHNAME_PREFIX"] == "/omnixai/8765/"
    assert "DASH_REQUESTS_PATHNAME_PREFIX" not in script.child_environment("", 8765, {"PATH": "x"})


def test_run_omnixai_manifest_and_options(tmp_path):
    script = _load_script()
    args = script.parse_args(["--public-base", "/omnixai", "--port-base", "9000"])
    assert args.public_base == "/omnixai" and args.port == 9000
    path = script.write_manifest({"stage2": 9000}, "127.0.0.1", str(tmp_path), "/omnixai")
    entry = json.loads(path.read_text(encoding="utf-8"))["dashboards"][0]
    assert entry["url"] == "http://127.0.0.1:9000" and entry["public_url"] == "/omnixai/9000/"
    path = script.write_manifest({"stage2": 9000}, "127.0.0.1", str(tmp_path))
    assert json.loads(path.read_text(encoding="utf-8"))["dashboards"][0]["public_url"] == "http://127.0.0.1:9000"


# ----------------------------------------------------------------- MLZero

def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_mlzero_env_wins(monkeypatch, tmp_path):
    exe = _touch(tmp_path / "custom" / "mlzero")
    _touch(tmp_path / ".venv-mlzero" / "Scripts" / "mlzero.exe")
    monkeypatch.setattr(mlzero, "ROOT", tmp_path)
    monkeypatch.setenv("AAVAS_MLZERO_CLI", str(exe))
    assert mlzero.resolve_cli(".venv-mlzero/Scripts/mlzero.exe") == exe


def test_mlzero_env_must_be_absolute_and_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(mlzero, "ROOT", tmp_path)
    monkeypatch.setenv("AAVAS_MLZERO_CLI", "relative/mlzero")
    assert mlzero.resolve_cli(".venv-mlzero/Scripts/mlzero.exe") is None


def test_mlzero_configured_path_then_platform_equivalent(monkeypatch, tmp_path):
    monkeypatch.delenv("AAVAS_MLZERO_CLI", raising=False)
    monkeypatch.setattr(mlzero, "ROOT", tmp_path)
    win = _touch(tmp_path / ".venv-mlzero" / "Scripts" / "mlzero.exe")
    assert mlzero.resolve_cli(".venv-mlzero/Scripts/mlzero.exe") == win
    win.unlink()
    linux = _touch(tmp_path / ".venv-mlzero" / "bin" / "mlzero")
    assert mlzero.resolve_cli(".venv-mlzero/Scripts/mlzero.exe") == linux
    linux.unlink()
    win = _touch(tmp_path / ".venv-mlzero" / "Scripts" / "mlzero.exe")
    assert mlzero.resolve_cli(".venv-mlzero/bin/mlzero") == win


def test_mlzero_not_found_and_python_next_to_cli(monkeypatch, tmp_path):
    monkeypatch.delenv("AAVAS_MLZERO_CLI", raising=False)
    monkeypatch.setattr(mlzero, "ROOT", tmp_path)
    assert mlzero.resolve_cli(".venv-mlzero/Scripts/mlzero.exe") is None
    cli = _touch(tmp_path / "v" / "bin" / "mlzero")
    assert mlzero.find_python(cli) is None
    py = _touch(tmp_path / "v" / "bin" / "python")
    assert mlzero.find_python(cli) == py


@pytest.mark.parametrize("allowed", [True, False])
def test_readiness_carries_allowed(monkeypatch, tmp_path, allowed):
    monkeypatch.delenv("AAVAS_MLZERO_CLI", raising=False)
    monkeypatch.setattr(mlzero, "ROOT", tmp_path)
    _touch(tmp_path / "v" / "bin" / "mlzero")
    cfg = {"training": {"autogluon_assistant": {"cli": "v/Scripts/mlzero.exe", "allow_unsandboxed": allowed}}}
    out = agentic._readiness(cfg)["autogluon_assistant"]
    assert out["ok"] is True and out["allowed"] is allowed
    cfg["training"]["autogluon_assistant"]["cli"] = "missing/bin/mlzero"
    out = agentic._readiness(cfg)["autogluon_assistant"]
    assert out["ok"] is False and out["allowed"] is allowed
