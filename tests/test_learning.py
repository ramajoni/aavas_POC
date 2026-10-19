"""The learning loop: override capture, the confounding correction, and drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aavas.governance import audit, decision_logger, store
from aavas.learning import drift_detection, override_learning


# ------------------------------------------------------------------- capture


def test_override_capture_keeps_both_numbers():
    """The AI recommendation is never overwritten - without both numbers there is
    nothing to learn from."""
    record = override_learning.capture(
        loan_id="TEST-OVR-1", ai_rate=11.50, human_rate=11.25, reason="competitor_lower_rate",
        approved_by="a.singh:branch_manager", evidence_verified=True)
    assert record["ai_rate"] == 11.50
    assert record["human_rate"] == 11.25
    assert record["override_bps"] == -25.0

    stored = store.query("SELECT * FROM overrides WHERE loan_id = ?", ("TEST-OVR-1",))
    assert len(stored) == 1
    assert stored[0]["ai_rate"] == 11.50 and stored[0]["human_rate"] == 11.25


def test_override_is_audited():
    override_learning.capture(loan_id="TEST-OVR-2", ai_rate=11.0, human_rate=10.8,
                              reason="strong_negotiation")
    trail = audit.trail(entity_id="TEST-OVR-2")
    assert any(entry["action"] == "capture_override" for entry in trail)


def test_reason_must_be_on_the_approved_list():
    assert override_learning.validate_reason("competitor_lower_rate")
    assert not override_learning.validate_reason("because_i_felt_like_it")


def test_agent_rejects_an_unapproved_reason(orchestrator):
    with pytest.raises(ValueError, match="not an allowed override reason"):
        orchestrator.learning.capture_override("TEST-OVR-3", 11.0, 10.5, "gut_feel")


# --------------------------------------------------------------- confounding


def test_propensity_adjustment_changes_the_answer(bundle, training_data):
    """The central methodological claim: overrides are not randomly assigned, so
    the naive comparison and the adjusted one must not agree."""
    propensity = bundle.override.predict(training_data)["override_probability"].to_numpy()
    naive = override_learning.naive_comparison(training_data)
    adjusted = override_learning.propensity_adjusted(training_data, propensity)

    assert adjusted.n_override > 0 and adjusted.n_control > 0
    assert abs(adjusted.s2d_lift_pts - naive["s2d_lift_pts"]) > 1.0, (
        "if adjusting for selection changes nothing, the correction is not doing its job")
    assert adjusted.verdict in {"value creating", "value destroying",
                                "no measurable effect", "insufficient evidence"}


def test_verdict_requires_statistical_evidence():
    assert override_learning._verdict(10_000.0, 1_000.0) == "value creating"
    assert override_learning._verdict(-10_000.0, 1_000.0) == "value destroying"
    assert override_learning._verdict(500.0, 5_000.0) == "no measurable effect"
    assert override_learning._verdict(float("nan"), float("nan")) == "insufficient evidence"


def test_league_table_is_ranked_and_reports_sample_size(bundle, training_data):
    propensity = bundle.override.predict(training_data)["override_probability"].to_numpy()
    table = override_learning.reason_league_table(training_data, propensity)
    assert "all overrides" in table["label"].tolist()
    assert (table["n_override"] > 0).all()
    ranked = table["value_lift_inr"].dropna().tolist()
    assert ranked == sorted(ranked, reverse=True)


def test_suggestions_are_bounded_and_advisory(bundle, training_data):
    propensity = bundle.override.predict(training_data)["override_probability"].to_numpy()
    table = override_learning.reason_league_table(training_data, propensity)
    for suggestion in override_learning.suggested_adjustments(table, max_bps=15.0):
        assert abs(suggestion["suggested_bps"]) <= 15.0
        assert "committee" in suggestion["requires"]


def test_a_thin_slice_returns_insufficient_evidence(training_data):
    tiny = training_data.head(30)
    effect = override_learning.propensity_adjusted(tiny, np.linspace(0, 1, len(tiny)))
    assert effect.verdict == "insufficient evidence"


# --------------------------------------------------------------------- drift


def test_psi_is_zero_for_an_unchanged_distribution():
    values = np.random.default_rng(0).normal(size=5000)
    assert drift_detection.psi(values, values) < 0.01


def test_psi_flags_a_shifted_distribution():
    rng = np.random.default_rng(0)
    assert drift_detection.psi(rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)) > 0.25


def test_conversion_drift_detects_a_market_move(training_data):
    """The dangerous drift: the same files convert differently because
    competitors moved, while input distributions look unchanged."""
    reference = training_data.head(6000)
    shifted = training_data.tail(6000).copy()
    # Simulate customers becoming far harder to convert at the same price.
    rng = np.random.default_rng(0)
    shifted["disbursed"] = shifted["disbursed"].astype(bool) & (rng.random(len(shifted)) < 0.55)
    result = drift_detection.conversion_drift(reference, shifted)
    assert result["status"] in {"warn", "alert"}
    assert result["max_shift_pts"] > 4


def test_outcome_capture_links_back_to_the_decision(orchestrator, sample_application):
    decision = orchestrator.process(sample_application)
    outcome = orchestrator.capture_outcome(
        loan_id=decision["loan_id"], disbursed=True,
        sanctioned_rate=decision["recommendation"]["recommended_rate"],
        customer_response="accepted", months_to_exit=60)
    comparison = outcome["vs_expected"]
    assert comparison["decision_id"] == decision["decision_id"]
    assert comparison["converted"] is True
    assert comparison["rate_delta_bps"] == 0.0


# ----------------------------------------------------------------- federated

def _federated_clients(n_clients: int = 4, rows: int = 120):
    import numpy as np
    from aavas.federated.client import RegionalAdjustmentClient, RegionalDataset
    rng = np.random.default_rng(7)
    clients = []
    for i in range(n_clients):
        X = rng.normal(size=(rows, 3))
        y = X @ np.array([0.4, -0.2, 0.1]) + 0.05 * i + rng.normal(scale=0.02, size=rows)
        clients.append(RegionalAdjustmentClient(RegionalDataset(f"region-{i}", X, y)))
    return clients


def test_fedprox_passed_as_an_argument_actually_turns_on_the_proximal_term():
    """FedProx differs from FedAvg in the client objective, not the aggregation.
    Selecting it by argument has to reach the fit config, or the run is plain
    FedAvg under another name."""
    from aavas.federated import server as fed_server
    run = fed_server.run(_federated_clients(), rounds=3, strategy_name="fedprox")
    assert run.strategy == "fedprox"
    assert run.proximal_mu > 0
    assert run.summary()["proximal_mu"] > 0


def test_fedavg_leaves_the_proximal_term_off():
    from aavas.federated import server as fed_server
    run = fed_server.run(_federated_clients(), rounds=3, strategy_name="fedavg")
    assert run.strategy == "fedavg"
    assert run.proximal_mu == 0.0


def test_a_run_reports_which_strategy_produced_it():
    """Two runs with different strategies are not comparable unless each says
    which one it was."""
    from aavas.federated import server as fed_server
    clients = _federated_clients()
    assert fed_server.run(clients, rounds=2, strategy_name="uniform").summary()["strategy"] == "uniform"
