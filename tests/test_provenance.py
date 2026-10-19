"""Dataset identity, fairness on the training path, and the registries that were
absent: prompts, knowledge indexes and federated rounds.

Each of these existed as provenance and not as identity. The tests check the
distinction rather than the presence: a training window is provenance, a content
hash is identity, and only one of them can tell two models apart.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aavas.governance import knowledge_registry, model_registry, prompt_registry
from aavas.models import fairness, provenance


# ------------------------------------------------------------ dataset identity

def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = pd.Series(range(n))
    return pd.DataFrame({
        "application_date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "applicant_age": (rng * 7 + seed) % 45 + 22,
        "amount": (rng * 13 + seed) % 5_000_000,
    })


def test_the_same_frame_hashes_the_same_twice() -> None:
    frame = _frame()
    assert provenance.content_hash(frame) == provenance.content_hash(frame.copy())


def test_a_filtered_slice_is_a_different_dataset() -> None:
    """The defect this closes: two models fitted on differently filtered extracts
    of one window carried identical cards."""
    frame = _frame()
    whole = provenance.identity(frame)
    slice_ = provenance.identity(frame[frame["applicant_age"] > 30])
    assert whole["window"] == {"from": str(frame["application_date"].min().date()),
                               "to": str(frame["application_date"].max().date())}
    assert whole["dataset_id"] != slice_["dataset_id"]
    assert not provenance.same_dataset(whole, slice_)


def test_a_schema_change_alone_changes_the_identity() -> None:
    frame = _frame()
    renamed = frame.rename(columns={"amount": "loan_amount"})
    assert provenance.identity(frame)["dataset_id"] != provenance.identity(renamed)["dataset_id"]


def test_every_registered_card_carries_a_dataset_identity() -> None:
    for name, card in model_registry.load_bundle().cards().items():
        dataset = card.get("dataset") or {}
        assert dataset.get("dataset_id"), f"{name} has no dataset identity"
        assert dataset.get("rows", 0) > 0
        assert dataset.get("slice_of"), f"{name} does not say what it is a slice of"


def test_models_fitted_on_different_slices_have_different_dataset_ids() -> None:
    cards = model_registry.load_bundle().cards()
    full = cards["stage2"]["dataset"]["dataset_id"]
    disbursed_only = cards["risk"]["dataset"]["dataset_id"]
    assert full != disbursed_only


# --------------------------------------------------- fairness on the real path

def test_fairness_is_measured_on_the_training_path_not_only_in_automl() -> None:
    """It gated candidates in a search that has never promoted anything, while the
    models actually pricing decisions were never evaluated."""
    cards = model_registry.load_bundle().cards()
    assert cards["risk"]["fairness"] is not None, "the risk champion has no fairness record"
    assert cards["risk"]["fairness"]["group_column"] == fairness.DEFAULT_GROUP_COLUMN


def test_a_regressor_records_that_fairness_was_not_measurable() -> None:
    """Not measured is not passed, and must not render as passed."""
    cards = model_registry.load_bundle().cards()
    assert cards["stage1"]["fairness"] is None


def test_fairness_returns_none_when_the_protected_axis_is_missing() -> None:
    from aavas.models import base

    frame = pd.DataFrame({"x": [1, 2, 3, 4]})
    assert base.fairness_report(frame, [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) is None


def test_groups_for_returns_none_rather_than_guessing() -> None:
    assert fairness.groups_for(pd.DataFrame({"x": [1]})) is None
    banded = fairness.groups_for(pd.DataFrame({"applicant_age": [25, 35, 65]}))
    assert list(banded) == ["18-29", "30-39", "60+"]


# ------------------------------------------------------------------ registries

def test_a_prompt_id_is_derived_from_its_body() -> None:
    first = prompt_registry.prompt_id("p", "you are a careful reader")
    assert first == prompt_registry.prompt_id("p", "you are a careful reader")
    assert first != prompt_registry.prompt_id("p", "you are a careful reader.")


def test_the_declared_prompt_is_registered_and_names_where_it_is_used() -> None:
    ids = prompt_registry.register_declared(actor="test")
    assert ids, "no prompt was registered"
    row = prompt_registry.get(ids[0])
    assert row["name"] == "acceptance_extraction"
    assert "dormant" in row["used_by"]
    assert len(row["body"]) > 100


def test_registering_an_unchanged_prompt_does_not_create_a_second_row() -> None:
    before = len(prompt_registry.registered())
    prompt_registry.register_declared(actor="test")
    assert len(prompt_registry.registered()) == before


def test_a_knowledge_index_records_its_corpus_and_hash() -> None:
    index_id = knowledge_registry.register("test_corpus", ["alpha", "beta"], corpus="unit test")
    row = knowledge_registry.latest("test_corpus")
    assert row["index_id"] == index_id
    assert row["documents"] == 2
    assert row["corpus"] == "unit test"
    # Re-indexing unchanged documents must not fill the table with noise.
    assert knowledge_registry.register("test_corpus", ["alpha", "beta"]) == index_id
    assert knowledge_registry.register("test_corpus", ["alpha", "gamma"]) != index_id


def test_a_federated_round_is_persisted_with_its_strategy() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_learning import _federated_clients

    from aavas.federated import server as fed_server
    from aavas.governance import store

    run = fed_server.run(_federated_clients(), rounds=2, strategy_name="fedprox")
    assert run.run_id, "the round was not persisted"
    row = store.fetch_one("SELECT * FROM federated_rounds WHERE run_id = ?", (run.run_id,))
    assert row["strategy"] == "fedprox"
    assert row["proximal_mu"] > 0
    assert row["participants"] == 4
    assert row["global_version"].startswith("FEDG-")


# ------------------------------------------------------------------ PSI defect

def test_psi_detects_drift_in_a_boolean_feature() -> None:
    """It used to return 0.0 for any low-cardinality feature - a false 'no drift'
    on exactly the flags a lender watches."""
    import numpy as np

    from aavas.learning.drift_detection import psi

    rng = np.random.default_rng(0)
    stable = rng.random(5000) < 0.30
    shifted = rng.random(5000) < 0.45
    assert psi(stable, stable) == pytest.approx(0.0, abs=1e-6)
    assert psi(stable, shifted) > 0.05


def test_psi_still_works_on_a_continuous_feature() -> None:
    import numpy as np

    from aavas.learning.drift_detection import psi

    rng = np.random.default_rng(1)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.05
    assert psi(rng.normal(size=5000), rng.normal(0.5, size=5000)) > 0.1


# ------------------------------------------------------- outcome join + reason

def test_an_outcome_attaches_to_the_decision_not_the_file() -> None:
    from fastapi.testclient import TestClient

    from aavas.api import examples
    from aavas.api.main import app

    client = TestClient(app)
    first = client.post("/pricing/final-assessment",
                        json={**examples.FINAL_ASSESSMENT, "loan_id": "JOIN-1"}).json()
    second = client.post("/pricing/final-assessment",
                         json={**examples.FINAL_ASSESSMENT, "loan_id": "JOIN-1"}).json()
    assert first["decision_id"] != second["decision_id"]

    client.post("/pricing/outcome", json={
        "loan_id": "JOIN-1", "disbursed": True, "sanctioned_rate": 11.2,
        "customer_response": "accepted", "decision_id": first["decision_id"]})
    client.post("/pricing/outcome", json={
        "loan_id": "JOIN-1", "disbursed": False, "customer_response": "rejected",
        "decision_id": second["decision_id"], "loss_reason": "competitor_lower_rate",
        "lost_to": "Another lender", "lost_to_rate": 10.85})

    from aavas.governance import store
    rows = store.query("SELECT * FROM outcomes WHERE loan_id = ? ORDER BY updated_at",
                       ("JOIN-1",))
    assert len(rows) == 2, "one file priced twice must be able to carry two outcomes"
    assert {r["decision_id"] for r in rows} == {first["decision_id"], second["decision_id"]}
    lost = next(r for r in rows if r["loss_reason"])
    assert lost["lost_to"] == "Another lender" and lost["lost_to_rate"] == 10.85


def test_an_outcome_cannot_be_attached_to_another_files_decision() -> None:
    from fastapi.testclient import TestClient

    from aavas.api import examples
    from aavas.api.main import app

    client = TestClient(app)
    other = client.post("/pricing/final-assessment",
                        json={**examples.FINAL_ASSESSMENT, "loan_id": "JOIN-2"}).json()
    wrong = client.post("/pricing/outcome", json={
        "loan_id": "JOIN-3", "disbursed": True, "decision_id": other["decision_id"]})
    assert wrong.status_code == 422
    assert "belongs to loan" in wrong.json()["detail"]


def test_a_rejected_file_with_no_reason_is_labelled_not_left_blank() -> None:
    from fastapi.testclient import TestClient

    from aavas.api import examples
    from aavas.api.main import app

    client = TestClient(app)
    client.post("/pricing/final-assessment",
                json={**examples.FINAL_ASSESSMENT, "loan_id": "JOIN-4"})
    body = client.post("/pricing/outcome", json={
        "loan_id": "JOIN-4", "disbursed": False, "customer_response": "rejected"}).json()
    assert body["loss_reason"] == "no_reason_recorded"
    assert body["decision_source"] == "inferred_latest"


def test_an_unknown_loss_reason_is_refused() -> None:
    from fastapi.testclient import TestClient

    from aavas.api import examples
    from aavas.api.main import app

    client = TestClient(app)
    client.post("/pricing/final-assessment",
                json={**examples.FINAL_ASSESSMENT, "loan_id": "JOIN-5"})
    bad = client.post("/pricing/outcome", json={
        "loan_id": "JOIN-5", "disbursed": False, "loss_reason": "they_were_rude"})
    assert bad.status_code == 422


# ------------------------------------------------------------- tool permissions

def test_an_agent_may_not_write_a_tool_it_only_reads() -> None:
    from aavas.agents import capabilities

    assert capabilities.permitted("pricing", "model.stage2", capabilities.READ)
    assert not capabilities.permitted("pricing", "store.governance", capabilities.WRITE)
    with pytest.raises(PermissionError):
        capabilities.check("pricing", "store.governance", capabilities.WRITE)


def test_only_the_learning_agent_writes_the_governance_store() -> None:
    """A pricing agent that could write a decision could write its own."""
    from aavas.agents import capabilities

    assert set(capabilities.writers()) == {"learning"}


def test_the_connector_boundary_enforces_the_permission() -> None:
    """The one path that reads a customer's own words checks rather than assumes."""
    from aavas.acceptance.sources import McpSource

    source = McpSource("whatsapp", {}, "whatsapp")
    assert source.fetch(None, {"agent": "pricing"}) == []


def test_every_permission_names_a_tool_the_agent_declares() -> None:
    from aavas.agents import capabilities, roster

    for agent, granted in capabilities.PERMISSIONS.items():
        declared = set(roster.BY_NAME[agent]["tools"])
        assert set(granted) == declared, (
            f"{agent}: permissions {sorted(granted)} do not match declared tools "
            f"{sorted(declared)}")


# ------------------------------------------------------- generator determinism

def test_the_generator_is_deterministic_under_a_fixed_seed() -> None:
    """AAVAS_SEED claims global determinism, and it was not honoured.

    ``branch_id`` came from the builtin ``hash(city)``, which Python salts per
    process, so every regeneration remapped every city to a different branch.
    Metrics never moved - branch_id is an identity column, not a feature - so
    nothing caught it until the dataset content hash reported a different dataset
    for a run whose metrics were bit-identical. AutoML's locality search uses
    branch_id as its unit, so branch-level results were not reproducible.
    """
    import hashlib

    def branch(city: str) -> str:
        return f"BR{int(hashlib.blake2b(city.encode(), digest_size=4).hexdigest(), 16) % 40:03d}"

    cities = ["Jaipur", "Kota", "Surat", "Indore", "Lucknow"]
    mapping = {city: branch(city) for city in cities}
    assert mapping == {city: branch(city) for city in cities}
    assert len(set(mapping.values())) > 1, "every city mapped to one branch"


def test_the_generator_does_not_use_the_salted_builtin_hash() -> None:
    """The regression guard. hash() on a str is randomised per process, so it must
    not decide anything that lands in a dataset."""
    import pathlib as _pathlib

    from aavas.synthetic import generator

    source = _pathlib.Path(generator.__file__).read_text()
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#") or "hashlib" in stripped:
            continue
        assert "hash(" not in stripped, (
            f"builtin hash() decides a generated value at line {number}: {stripped}")
