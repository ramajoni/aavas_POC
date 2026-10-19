"""The AutoML runner's contract with the rest of the repo.

The search itself is tested in test_automl.py. What this file protects is the
part that is easy to get quietly wrong: which contracts are searched, what each
one is fed, and the discipline of the splits. A search fed a different feature
space than the champion it is compared against produces a number that means
nothing, and one scored on the holdout has not been scored at all.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location("run_automl", ROOT / "scripts" / "run_automl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _runner()


def test_only_classification_contracts_are_searched(runner):
    """The search ranks on AUC, KS, ECE and PSI. stage1 and stage2 predict a
    rate, so putting them through it would produce a number with no meaning."""
    from aavas.ingestion.schemas import TARGETS

    assert set(runner.CONTRACTS) == {"conversion", "prepayment", "risk", "override"}
    assert runner.CONTRACTS["conversion"]["target"] == TARGETS["conversion"]
    assert runner.CONTRACTS["risk"]["target"] == TARGETS["risk"]
    assert runner.CONTRACTS["override"]["target"] == TARGETS["override_flag"]


def test_each_contract_is_fed_what_its_own_trainer_feeds_it(runner, training_data):
    """Same row filter, same frame builder, same feature contract - otherwise an
    AutoML candidate is being compared against a champion fitted on other data."""
    from aavas.features.feature_pipeline import spec

    for contract, setup in runner.CONTRACTS.items():
        frame = setup["build"](setup["rows"](training_data).head(200))
        missing = [c for c in spec(contract).columns if c not in frame.columns]
        assert not missing, f"{contract}: {missing} absent after the builder ran"


def test_prepayment_and_risk_see_disbursed_files_only(runner, training_data):
    """Neither target is defined on money that never went out of the door."""
    for contract in ("prepayment", "risk"):
        rows = runner.CONTRACTS[contract]["rows"](training_data)
        assert rows["disbursed"].astype(bool).all()
        assert len(rows) < len(training_data)


def test_age_bands_cover_the_boundaries(runner):
    banded = runner.age_band(pd.Series([21, 29, 30, 39, 40, 59, 60, 91, np.nan]))
    assert list(banded[:8]) == ["under 30", "under 30", "30-39", "30-39",
                                "40-49", "50-59", "60 and over", "60 and over"]
    assert banded.iloc[8] == "unknown", "a missing age must not fall into a band"


def test_fairness_defaults_to_age_because_age_is_a_pricing_feature(runner, training_data):
    from aavas.config import load

    assert load("automl")["defaults"]["fairness_group"] == "age_band"
    groups = runner.fairness_groups(training_data.head(500), "age_band")
    assert groups is not None and len(set(groups)) > 1


def test_fairness_group_is_none_when_the_column_is_absent(runner):
    """A missing group column disables the gate rather than inventing one."""
    assert runner.fairness_groups(pd.DataFrame({"ltv": [0.6]}), "age_band") is None
    assert runner.fairness_groups(pd.DataFrame({"ltv": [0.6]}), "segment") is None


def test_the_holdout_is_never_handed_to_the_search(runner, training_data):
    """Two chronological splits: the trainers' holdout first, then fit/valid out
    of what is left. A champion selected on the rows it will later be measured on
    has not been measured."""
    data = runner.prepare("conversion", training_data, valid_size=0.25,
                          fairness_group="age_band", unit_column="branch_id")

    assert data["n_holdout_untouched" if "n_holdout_untouched" in data else "n_holdout"] > 0
    assert len(data["y_fit"]) > len(data["y_valid"])
    assert data["X_fit"].shape[1] == data["X_valid"].shape[1]
    # Out of time, in the right order, with no overlap.
    assert data["window"]["fit_to"] < data["window"]["valid_from"]
    assert data["window"]["valid_from"] < data["window"]["holdout_from"]


def test_prepared_frames_carry_the_fairness_and_locality_axes(runner, training_data):
    data = runner.prepare("risk", training_data, valid_size=0.25,
                          fairness_group="age_band", unit_column="branch_id")
    assert len(data["groups"]) == len(data["y_valid"])
    assert len(data["units"]) == len(data["y_valid"])
    assert not np.isnan(data["X_fit"].to_numpy(dtype=float)).any(), \
        "a NaN would break half the learners in the registry"
