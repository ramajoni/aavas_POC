"""Shared fixtures.

Tests that need trained artefacts skip cleanly when the repo has not been
trained yet, so `pytest` is useful immediately after clone. Anything that
exercises pure policy or maths logic runs unconditionally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session", autouse=True)
def isolated_store(tmp_path_factory):
    """Never write to the developer's governance database during a test run."""
    path = tmp_path_factory.mktemp("governance") / "test.db"
    os.environ["AAVAS_STORE_URL"] = f"sqlite:///{path}"
    yield path


@pytest.fixture(scope="session")
def training_data():
    from aavas.ingestion import loader

    try:
        return loader.load_training_table()
    except FileNotFoundError:
        pytest.skip("dataset not generated; run: python data/synthetic/generate_dataset.py")


@pytest.fixture(scope="session")
def bundle():
    from aavas.governance import model_registry

    loaded = model_registry.load_bundle(refresh=True)
    if not loaded.ready:
        pytest.skip(f"models not trained ({loaded.missing}); run: python scripts/train_all.py")
    return loaded


@pytest.fixture(scope="session")
def sample_application(training_data):
    return training_data.iloc[42].to_dict()


@pytest.fixture(scope="session")
def orchestrator(bundle):
    from aavas.agents.orchestrator import PricingOrchestrator

    return PricingOrchestrator(bundle=bundle)


@pytest.fixture(scope="session")
def decision(orchestrator, sample_application):
    return orchestrator.process(sample_application)
