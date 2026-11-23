"""Best offer: the search over rate, loan amount, tenure and processing fee.

What must hold: the grid follows config/best_offer.yaml; the fee is never above the published
fee and the amount never above the request; every offer is inside 13-17 and, when eligible,
inside its own floor..card; each objective honours its at_least constraint against the
references; ties go to the offer with fewer changes; reduced amounts are flagged; the ask is
scored even when it is outside policy; no qualifying offer gives null with a reason; the lever
table starts from the recommendation; the route is for the area manager; and the config is
registered and its version reaches the release notes.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aavas import config
from aavas.analytics import best_offer as bo
from aavas.analytics import offer_evaluator as oe
from aavas.analytics import risk_pricing
from aavas.api import access
from aavas.api.main import app
from aavas.config import best_offer_config, policy
from test_risk_pricing import FILE, PAYLOAD, REQUEST, stub_bundle, who

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stubbed(monkeypatch):
    from aavas.governance import model_registry

    held = stub_bundle()
    monkeypatch.setattr(model_registry, "load_bundle", lambda *a, **k: held)
    monkeypatch.setattr(risk_pricing, "_shift", lambda city: (0.0, None))
    bo._cache.clear()
    return held


@pytest.fixture
def result(stubbed):
    """One run on stub models for the read-only checks below."""
    return bo.best_offers(REQUEST, PAYLOAD, {})


def with_config(monkeypatch, **levers) -> dict:
    cfg = copy.deepcopy(best_offer_config())
    for lever, values in levers.items():
        cfg["levers"][lever].update(values)
    monkeypatch.setattr(bo, "best_offer_config", lambda: cfg)
    return cfg


# ------------------------------------------------------------------ the grid

def test_the_grid_follows_the_config(stubbed, monkeypatch) -> None:
    cfg = with_config(monkeypatch, tenure={"min_months": 48, "max_months": 96, "step_months": 24},
                      loan_amount={"reductions_pct": [10]})
    g = bo.grid(PAYLOAD, 2800000.0, 84, 13.92, cfg)
    assert sorted(g["tenure_months"].unique()) == [48, 72, 84, 96]          # the range plus the file's own
    assert sorted(g["loan_amount"].unique()) == [2520000.0, 2800000.0]
    assert sorted(g["fee_pct"].unique()) == [0.0, 0.25, 0.5]
    assert len(g) == 4 * 2 * 3
    narrow = bo.grid(PAYLOAD, 2800000.0, 84, 13.92, {**cfg, "levers": {**cfg["levers"], "tenure": {**cfg["levers"]["tenure"], "include_requested": False}}})
    assert 84 not in set(narrow["tenure_months"])


def test_the_policy_is_stricter_than_the_config_range(stubbed, monkeypatch) -> None:
    cfg = with_config(monkeypatch, tenure={"min_months": 12, "max_months": 240, "step_months": 12})
    g = bo.grid(PAYLOAD, 2800000.0, 84, 13.92, cfg)
    pol = policy()["constraints"]
    assert g["tenure_months"].min() == pol["min_tenure_months"] and g["tenure_months"].max() == pol["max_tenure_months"]


def test_the_fee_is_never_above_the_published_fee_and_the_amount_never_above_the_request(stubbed, monkeypatch) -> None:
    cfg = with_config(monkeypatch, processing_fee={"levels_pct": [0.9, 0.5, 0.0]}, loan_amount={"reductions_pct": [10, 20, -5]})
    g = bo.grid(PAYLOAD, 2800000.0, 84, 13.92, cfg)
    assert g["fee_pct"].max() == policy()["economics"]["processing_fee_pct"]
    assert g["loan_amount"].max() <= 2800000.0


def test_the_eligible_maximum_is_below_the_request_when_the_file_cannot_carry_it(stubbed) -> None:
    thin = {**PAYLOAD, "application_snapshot": {**PAYLOAD["application_snapshot"], "verified_monthly_income": 60000.0}}
    cap = bo.eligible_amount(thin, 2800000.0, 13.92)
    assert cap is not None and cap < 2800000.0 and cap % 1000 == 0
    assert bo.eligible_amount(PAYLOAD, 2800000.0, 13.92) is None


def test_reduced_amounts_are_flagged_with_the_limitation(result) -> None:
    assert any("no customers who were offered less" in t for t in result["limitations"])
    reduced = [p for p in result["points"] if p["amount_reduced"]]
    assert reduced and all(p["loan_amount"] < REQUEST["loan_amount"] for p in reduced)
    assert all(not p["amount_reduced"] for p in result["points"] if p["loan_amount"] == REQUEST["loan_amount"])
    assert any(l["lever"] == "amount" and l["amount_reduced"] for l in result["levers"])


# ------------------------------------------------------------------ the offers

def test_every_offer_is_inside_the_band_and_eligible_ones_inside_their_own_floor_and_card(result, stubbed) -> None:
    assert result["points"] and all(13.0 <= p["rate"] <= 17.0 for p in result["points"] + result["table"])
    rows = pd.DataFrame(result["table"])[["rate", "loan_amount", "tenure_months", "fee_pct"]]
    again = oe.evaluate_offers(PAYLOAD, rows, bundle=stubbed)
    assert again["in_band"].all() and again["eligible"].all()
    assert (again["rate"] >= again["floor_rate"] - 1e-9).all() and (again["rate"] <= again["card_rate"] + 1e-9).all()
    assert result["counts"]["eligible"] <= result["counts"]["scored"]


def test_the_ask_is_scored_even_when_it_is_outside_policy(result) -> None:
    ask = result["references"]["ask"]
    assert ask["rate"] == 13.62 and not ask["eligible"] and "rate_below_floor" in ask["failures"]
    assert ask["acceptance"] is not None and ask["value_inr"] is not None
    assert result["references"]["recommendation"]["eligible"]


def test_objectives_honour_their_constraints_against_the_references(result) -> None:
    cfg = best_offer_config()
    refs = result["references"]
    measure = {"acceptance": "acceptance", "value": "value_inr"}
    assert [b["key"] for b in result["best"]] == [o["key"] for o in cfg["objectives"]]
    found = [b for b in result["best"] if b["offer"]]
    assert found, "the stub market should leave at least one objective with an offer"
    for b in found:
        need = b["at_least"]
        assert b["offer"][measure[need["measure"]]] >= refs[need["reference"]][measure[need["measure"]]] - 1e-9
        assert b["offer"]["eligible"] and b["offer"]["sentence"]
        assert b["offer"]["vs_ask"]["value_inr"] == b["offer"]["value_inr"] - refs["ask"]["value_inr"]
        assert b["offer"]["changes"]["rate_bps"] == round((b["offer"]["rate"] - refs["recommendation"]["rate"]) * 100)
    likely = next(b for b in result["best"] if b["key"] == "most_likely")
    assert likely["at_least"] == {"measure": "value", "reference": "recommendation"}, "not benchmarked on the ask"
    if likely["offer"]:
        assert likely["offer"]["value_inr"] >= refs["recommendation"]["value_inr"]


def _offers(rows):
    return pd.DataFrame(rows).assign(eligible=True)


def test_the_tie_break_prefers_fewer_changes() -> None:
    offers = _offers([
        {"value": 100000.0, "acceptance": 0.80, "changes": 3},
        {"value": 99900.0, "acceptance": 0.81, "changes": 1},       # within the tie tolerance, one change
        {"value": 100100.0, "acceptance": 0.70, "changes": 2},
        {"value": 50000.0, "acceptance": 0.99, "changes": 0}])
    refs = {"recommendation": pd.Series({"acceptance": 0.5, "value": 0.0}), "ask": pd.Series({"acceptance": 0.5, "value": 0.0})}
    objective = {"maximise": "value", "at_least": {"measure": "acceptance", "reference": "recommendation"}}
    chosen, why = bo.pick(offers, objective, refs, ["fewer_changes", "value", "acceptance"], 0.5)
    assert why is None and chosen["changes"] == 1
    # without the tolerance the plain maximum wins; the tie-break order is configurable
    chosen, _ = bo.pick(offers, objective, refs, ["fewer_changes", "value", "acceptance"], 0.0)
    assert chosen["value"] == 100100.0
    chosen, _ = bo.pick(offers, objective, refs, ["value", "fewer_changes"], 5.0)
    assert chosen["value"] == 100100.0


def test_no_qualifying_offer_returns_null_with_a_reason() -> None:
    offers = _offers([{"value": 100.0, "acceptance": 0.5, "changes": 1}])
    refs = {"recommendation": pd.Series({"acceptance": 0.9, "value": 0.0}), "ask": pd.Series({"acceptance": 0.9, "value": 0.0})}
    chosen, why = bo.pick(offers, {"maximise": "value", "at_least": {"measure": "acceptance", "reference": "ask"}}, refs,
                          ["fewer_changes"], 0.5)
    assert chosen is None and "ask" in why and "acceptance" in why
    ineligible = offers.assign(eligible=False)
    refs["recommendation"]["acceptance"] = 0.1
    assert bo.pick(ineligible, {"maximise": "value", "at_least": {"measure": "acceptance", "reference": "recommendation"}},
                   refs, [], 0.5)[0] is None


def test_the_whole_run_reports_null_best_with_the_reason(stubbed) -> None:
    broke = {**PAYLOAD, "application_snapshot": {**PAYLOAD["application_snapshot"], "verified_monthly_income": 20000.0, "declared_obligations": 20000.0}}
    out = bo.best_offers({**REQUEST, "request_id": "RRQ-NULL"}, broke, {})
    assert out["counts"]["eligible"] == 0 and out["points"] == [] and out["frontier"] == []
    assert all(b["offer"] is None and b["reason"] for b in out["best"])
    assert any(r["code"] == "stressed_foir_over_cap" for r in out["ineligible"])


# ------------------------------------------------------------------ levers, frontier, cache

def test_the_levers_table_starts_from_the_recommendation(result) -> None:
    rec = result["references"]["recommendation"]
    floor = oe.evaluate_offers(PAYLOAD, pd.DataFrame({"rate": [rec["rate"]]}), bundle=stub_bundle())["floor_rate"].iloc[0]
    own = result["own"]
    by = {}
    for l in result["levers"]:
        by.setdefault(l["lever"], []).append(l)
        moved = [abs(l["rate"] - rec["rate"]) > 0.004, l["tenure_months"] != own["tenure_months"],
                 abs(l["fee_pct"] - own["fee_pct"]) > 1e-9, abs(l["loan_amount"] - own["loan_amount"]) > 0.5]
        assert sum(moved) == 1, f"{l['label']} changes more than one lever"
        assert l["acceptance_change_pts"] == pytest.approx((l["acceptance"] - rec["acceptance"]) * 100, abs=0.06)
        assert l["value_change_inr"] == l["value_inr"] - rec["value_inr"]
    assert set(by) == {"rate", "tenure", "fee", "amount"}
    assert by["rate"][0]["rate"] == floor
    assert own["tenure_months"] not in [l["tenure_months"] for l in by["tenure"]]
    assert 0.0 in [l["fee_pct"] for l in by["fee"]] and own["fee_pct"] not in [l["fee_pct"] for l in by["fee"]]


def test_the_frontier_is_the_pareto_set_of_eligible_offers(result) -> None:
    f = result["frontier"]
    assert f == sorted(f, key=lambda p: p["acceptance"])
    values = [p["value_inr"] for p in f]
    assert values == sorted(values, reverse=True), "more acceptance can only come with less value on the frontier"
    pts = [(p["acceptance"], p["value_inr"]) for p in result["points"]]
    assert len(result["points"]) <= bo.POINTS_MAX and len(result["table"]) <= best_offer_config()["display"]["table_rows"]
    assert [t["value_inr"] for t in result["table"]] == sorted((t["value_inr"] for t in result["table"]), reverse=True)
    top = max(p["value_inr"] for p in result["points"])
    assert max(values) >= top - 1, "the most valuable offer is on the frontier"
    assert pts, "the chart has points"


def test_the_pareto_function() -> None:
    d = pd.DataFrame({"acceptance": [0.9, 0.8, 0.7, 0.85], "value": [100.0, 200.0, 150.0, 120.0]})
    assert pareto_pairs(bo.pareto(d)) == [(0.8, 200.0), (0.85, 120.0), (0.9, 100.0)]


def pareto_pairs(df):
    return [(float(a), float(v)) for a, v in zip(df["acceptance"], df["value"])]


def test_results_are_cached_per_request_and_fingerprint(stubbed) -> None:
    a = bo.best_offers({**REQUEST, "request_id": "RRQ-C"}, PAYLOAD, {})
    assert bo.best_offers({**REQUEST, "request_id": "RRQ-C"}, PAYLOAD, {}) is a
    changed = {**PAYLOAD, "application_snapshot": {**PAYLOAD["application_snapshot"], "verified_monthly_income": 150000.0}}
    assert bo.best_offers({**REQUEST, "request_id": "RRQ-C"}, changed, {}) is not a


# ------------------------------------------------------------------ config and route

def test_the_config_is_registered_and_its_version_reaches_the_release_notes() -> None:
    assert config.config_path("best_offer").name == "best_offer.yaml"
    assert config.best_offer_config()["best_offer_config_version"].startswith("OFFER-")
    spec = importlib.util.spec_from_file_location("release_script", ROOT / "scripts" / "release.py")
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    assert release.config_versions()["best_offer.yaml"] == config.best_offer_config()["best_offer_config_version"]


@pytest.fixture
def client(monkeypatch, bundle) -> TestClient:
    monkeypatch.setenv("AAVAS_AUTH_MODE", "session")
    access.reset_state()
    yield TestClient(app)
    access.reset_state()


def test_the_route_is_for_the_area_manager(client) -> None:
    rohit = who(client, "rohit")
    priced = client.post("/pricing/final-assessment", json={"loan_id": "BST-1", **FILE}, headers=rohit)
    assert priced.status_code == 200, priced.text
    rec = priced.json()["recommendation"]["recommended_rate"]
    asked = client.post("/pricing/rate-requests", headers=rohit,
                        json={"loan_id": "BST-1", "rate": round(rec - 0.30, 2), "reason": "competitor_lower_rate", "note": "HDFC at 11.0"})
    assert asked.status_code == 200, asked.text
    request_id = asked.json()["request_id"]
    meera = who(client, "meera")
    got = client.get(f"/pricing/rate-requests/{request_id}/best-offer", headers=meera)
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["references"]["recommendation"]["rate"] == rec and len(body["best"]) == len(best_offer_config()["objectives"])
    # this file asks for a 20-year tenure, outside policy, so the honest answer may be that nothing qualifies
    assert body["counts"]["scored"] > 0 and body["limitations"]
    assert all(b["offer"] is not None or b["reason"] for b in body["best"]) and body["note"].startswith("Decision support")
    assert all(13.0 <= p["rate"] <= 17.0 for p in body["points"])
    assert client.get(f"/pricing/rate-requests/{request_id}/best-offer", headers=rohit).status_code == 403
    assert client.get("/pricing/rate-requests/RRQ-NOPE/best-offer", headers=meera).status_code == 404


# ------------------------------------------------------------------ comfort limit, flags, tolerance, APR

def test_the_emi_comfort_limit_makes_heavier_offers_ineligible(result) -> None:
    limit = best_offer_config()["levers"]["emi"]["max_increase_pct"]
    own_emi = result["references"]["recommendation"]["emi"]
    ceiling = own_emi * (1 + limit / 100) + 0.5
    assert all(o["emi"] <= ceiling for o in result["table"] + [b["offer"] for b in result["best"] if b["offer"]])
    comfort = next(r for r in result["ineligible"] if r["code"] == "emi_over_comfort")
    assert comfort["kind"] == "comfort" and comfort["offers"] > 0 and f"{limit:g}%" in comfort["words"]
    assert all(r["kind"] == "policy" for r in result["ineligible"] if r["code"] != "emi_over_comfort")
    short = next(l for l in result["levers"] if l["lever"] == "tenure" and l["tenure_months"] == 48)
    assert not short["eligible"] and "emi_over_comfort" in short["failures"]


def test_a_different_tenure_is_flagged_with_the_limitation(result) -> None:
    own = result["own"]["tenure_months"]
    assert any("different tenure" in t for t in result["limitations"])
    pts = [p for p in result["points"] if p["tenure_months"] != own]
    assert pts and all(p["tenure_changed"] for p in pts)
    assert all(not p["tenure_changed"] for p in result["points"] if p["tenure_months"] == own)
    assert result["references"]["ask"]["tenure_changed"] is False


def test_the_tie_tolerance_comes_from_the_config(stubbed, monkeypatch) -> None:
    assert best_offer_config()["tie_tolerance_pct"] == 0.5
    seen = []
    real = bo.pick
    monkeypatch.setattr(bo, "pick", lambda *a, **k: (seen.append(a[-1]), real(*a, **k))[1])
    cfg = copy.deepcopy(best_offer_config())
    cfg["tie_tolerance_pct"] = 2.5
    monkeypatch.setattr(bo, "best_offer_config", lambda: cfg)
    bo.best_offers({**REQUEST, "request_id": "RRQ-TOL"}, PAYLOAD, {})
    assert seen and set(seen) == {2.5}


def test_the_vectorised_apr_matches_the_kfs_calculator() -> None:
    from aavas.intake import kfs

    rng = np.random.default_rng(7)
    for _ in range(40):
        amount = float(round(rng.uniform(5e5, 7.5e6)))
        rate = float(round(rng.uniform(13, 17), 2))
        months = int(rng.choice([36, 48, 60, 84, 96, 120]))
        fee = float(rng.choice([0.0, 0.25, 0.5]))
        upfront = oe._upfront(np.array([amount]), np.array([fee]))[0]
        want = kfs.apr(amount, rate, months, upfront)["apr_pct"]
        got = oe.apr_vector(np.array([amount]), np.array([rate]), np.array([months]), np.array([upfront]))[0]
        # the KFS rounds to 2 dp (up to 0.005) and its last instalment clears the balance (a few rupees)
        assert abs(got - want) <= 0.0055, (amount, rate, months, fee, got, want)
    many = oe.apr_vector(np.full(500, 2.8e6), np.linspace(13, 17, 500), np.full(500, 120), np.full(500, 14000.0))
    assert many.shape == (500,) and (np.diff(many) > 0).all()
