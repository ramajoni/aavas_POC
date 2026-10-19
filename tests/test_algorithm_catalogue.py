"""Every registered algorithm either runs or says why it cannot.

`available()` answers whether an algorithm imports. These tests answer whether it
runs - and, more importantly, that the catalogue never claims something works when
it does not. An algorithm that imports and then raises on `fit` is worse than one
that is absent, because the inventory says it is available.

The full sweep lives in `scripts/verify_algorithms.py` (`make verify-algorithms`)
because fitting 41 learners is too slow for every test run. What is asserted here
is the contract the sweep depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from aavas.automl import calibration, hpo, imbalance, registry


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = (X @ np.array([0.9, -0.6, 0.3, 0.0, 0.0, 0.0]) + rng.normal(scale=0.5, size=400) > 0.4)
    return X, y.astype(int)


def test_an_unavailable_algorithm_always_gives_a_reason() -> None:
    """The catalogue is complete on purpose - it registers families that cannot
    run on this schema. An omission would say nothing; a reason says something."""
    for algo in registry.ALGORITHMS:
        reason = algo.unavailable_reason()
        if reason is None:
            continue
        assert len(reason) > 10, f"{algo.name} is unavailable and gives no usable reason"
        assert any(k in reason for k in ("pip install", "no adapter", "no sequence", "no graph"))


def test_every_available_algorithm_instantiates() -> None:
    """Between `available()` and `fit` sits `instantiate`, which is where a bad
    default parameter shows up."""
    for algo in registry.ALGORITHMS:
        if algo.unavailable_reason():
            continue
        assert algo.instantiate() is not None, algo.name


@pytest.mark.parametrize("name", sorted(hpo.STRATEGIES))
def test_every_hpo_strategy_moves_toward_the_optimum(name: str) -> None:
    """A strategy that returns a random point is not a search. The objective takes
    a params dict - `optimize` calls `objective({})` for an empty space."""
    space = {"x": ("float", -3.0, 3.0), "y": ("float", -3.0, 3.0)}
    result = hpo.optimize(space, lambda p: -((p["x"] - 1.0) ** 2 + (p["y"] + 0.5) ** 2),
                          strategy=name, budget=24)
    assert abs(result.best_params["x"] - 1.0) < 1.5, f"{name} did not approach the optimum"


@pytest.mark.parametrize("name", sorted(imbalance.STRATEGIES))
def test_every_imbalance_strategy_runs_or_declines(name: str, data) -> None:
    X, y = data
    try:
        result = imbalance.apply(X, y, name)
    except Exception as exc:                                    # noqa: BLE001
        assert "install" in str(exc).lower() or "imblearn" in str(exc).lower(), (
            f"{name} failed for a reason other than a missing dependency: {exc}")
        return
    assert len(result.y) > 0


@pytest.mark.parametrize("method", ["isotonic", "platt", "beta", "none"])
def test_a_calibrator_actually_transforms(method: str, data) -> None:
    """The check this replaces silently fell back to the raw scores when it could
    not find a `transform` attribute, and reported four identical ECEs as four
    working calibrators. The interface is `Calibrated.__call__`."""
    X, y = data
    raw = np.clip(0.5 + 0.3 * (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() or 1), 0.01, 0.99)
    calibrated = calibration.fit(raw, y, method=method)
    p = np.asarray(calibrated(raw), dtype=float)
    assert len(p) == len(raw)
    assert np.isfinite(p).all()
    if method == "none":
        assert np.allclose(p, raw)
    else:
        assert not np.allclose(p, raw), f"{method} returned the raw scores unchanged"


def test_federated_strategies_are_not_one_implementation_under_five_names() -> None:
    """With equal client sizes, sample-weighted FedAvg and uniform averaging are
    the same computation - so a check on equal clients proves nothing. Unequal
    sizes with one divergent region is what separates them."""
    from aavas.federated import server as fed_server
    from aavas.federated.client import RegionalAdjustmentClient, RegionalDataset

    def clients():
        rng = np.random.default_rng(7)
        out = []
        for index, n in enumerate([900, 400, 200, 120]):
            X = rng.normal(size=(n, 3))
            beta = np.array([2.5, -2.0, 1.5]) if index == 3 else np.array([0.4, -0.2, 0.1])
            out.append(RegionalAdjustmentClient(RegionalDataset(
                f"r{index}", X, X @ beta + rng.normal(scale=0.02, size=n))))
        return out

    weighted = fed_server.run(clients(), rounds=4, strategy_name="fedavg")
    uniform = fed_server.run(clients(), rounds=4, strategy_name="uniform")
    assert not np.allclose(weighted.global_parameters, uniform.global_parameters), (
        "sample-weighted and uniform aggregation produced the same parameters on "
        "deliberately unequal clients")
