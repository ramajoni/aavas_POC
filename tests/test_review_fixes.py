"""Fixes from the independent review of the risk-pricing and best-offer back end.

Each test names the finding it holds: income for the stress test, a missing recommendation, the
shared cache under threads, what the cache is keyed on and what clears it, the ask that is not
a choice but is a fact, marked rates that are never clamped, no floor for a declined grade, the
config check, the light route loader, and the stress add-on written from the policy.
"""

from __future__ import annotations

import copy
import json
import math
import threading

import numpy as np
import pandas as pd
import pytest

from aavas.analytics import best_offer as bo
from aavas.analytics import decision_support
from aavas.analytics import offer_evaluator as oe
from aavas.analytics import risk_pricing as rp
from aavas.api import configuration
from aavas.config import best_offer_config, load, policy
from aavas.optimization import constraints
from test_best_offer import client, stubbed          # noqa: F401 - fixtures
from test_risk_pricing import FILE, PAYLOAD, REQUEST, SNAP, offers, stub_bundle, who


def with_snapshot(**changes) -> dict:
    return {**PAYLOAD, "application_snapshot": {**SNAP, **changes}}


# ------------------------------------------------------------------ 1. income and the stress test

def test_the_stress_test_reads_declared_income_when_verified_income_is_missing() -> None:
    plain = float(constraints._emi(np.array([2.8e6]), np.array([14.5]), np.array([7.0]))[0])
    payload = with_snapshot(verified_monthly_income=float("nan"), declared_monthly_income=plain / 0.485)
    row = oe.evaluate_offers(payload, offers(rate=[14.5]), bundle=stub_bundle("A")).iloc[0]
    assert np.isfinite(row["stressed_foir"]) and row["stressed_foir"] > row["foir_cap"] >= row["foir"]
    assert "stressed_foir_over_cap" in row["failures"] and not row["stressed_foir_ok"]


def test_an_offer_that_cannot_be_stress_tested_is_not_eligible_and_both_views_agree(stubbed) -> None:
    payload = with_snapshot(verified_monthly_income=float("nan"), declared_monthly_income=float("nan"))
    out = oe.evaluate_offers(payload, offers(rate=[14.0]), bundle=stubbed)
    row = out.iloc[0]
    assert np.isnan(row["stressed_foir"]) and "stressed_foir_unavailable" in row["failures"] and not row["eligible"]
    assert not row["stressed_foir_ok"]
    rows, _ = rp.rates_table(payload, [(14.0, ["recommended"])], "Rajkot", 2.8e6, 84)
    assert rows[0]["stressed_foir_ok"] is False
    text = rp.verdict(ask=14.0, floor=13.0, grade="A", band_info={"borderline": False}, rows=rows, ask_row=None, rec_row=None,
                      book_margin=None, recommended=14.0)["text"]
    assert "nan" not in text.lower() and "cannot be run" in text


def test_the_risk_table_and_the_evaluator_agree_on_the_stress_test(stubbed) -> None:
    rows, _ = rp.rates_table(PAYLOAD, [(13.5, []), (15.0, [])], "Rajkot", 2.8e6, 84)
    out = oe.evaluate_offers(PAYLOAD, offers(rate=[13.5, 15.0]), bundle=stubbed)
    assert [r["stressed_foir_ok"] for r in rows] == [bool(v) for v in out["stressed_foir_ok"]]


# ------------------------------------------------------------------ 2. a payload key that is None

def test_a_decision_with_no_recommendation_block_does_not_crash_the_analysis(stubbed) -> None:
    payload = {**PAYLOAD, "recommendation": None}
    out = rp.risk_pricing(REQUEST, payload)
    assert out["score"]["expected_loss_inr_over_life"] is None and out["rates"]
    assert bo.best_offers({**REQUEST, "request_id": "RRQ-NOREC"}, {**payload, "risk": None}, None)["references"]


# ------------------------------------------------------------------ 3. the shared cache under threads

def test_the_cache_survives_concurrent_requests(stubbed, monkeypatch) -> None:
    monkeypatch.setattr(bo, "CACHE_SIZE", 3)
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(3):
                bo.best_offers({**REQUEST, "request_id": f"RRQ-T{n}-{i}"}, PAYLOAD)
        except BaseException as exc:                                   # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(bo._cache) <= 3


# ------------------------------------------------------------------ 4. what the cache is keyed on, and what clears it

def test_a_policy_change_or_a_different_model_is_a_cache_miss(stubbed, monkeypatch) -> None:
    from aavas.governance import model_registry

    first = bo.best_offers({**REQUEST, "request_id": "RRQ-KEY"}, PAYLOAD)
    assert bo.best_offers({**REQUEST, "request_id": "RRQ-KEY"}, PAYLOAD) is first
    changed = copy.deepcopy(policy())
    changed["economics"]["funding_cost_pct"] += 0.5
    monkeypatch.setattr(bo, "policy", lambda: changed)
    assert bo.best_offers({**REQUEST, "request_id": "RRQ-KEY"}, PAYLOAD) is not first
    monkeypatch.undo()
    monkeypatch.setattr(model_registry, "load_bundle", lambda *a, **k: stub_bundle(risk=False))
    assert bo.best_offers({**REQUEST, "request_id": "RRQ-KEY"}, PAYLOAD) is not first


def test_a_config_reload_clears_both_caches(stubbed) -> None:
    bo.best_offers({**REQUEST, "request_id": "RRQ-RELOAD"}, PAYLOAD)
    assert bo._cache
    configuration._reload()
    assert not bo._cache and rp._book.cache_info().currsize == 0


# ------------------------------------------------------------------ 5. the ask is a fact even when it is not a choice

def _row(rate, outside=False, v=100000, acc=0.8):
    return {"rate": rate, "acceptance_model": acc, "acceptance_branch": acc, "value_branch_inr": v, "value_model_inr": v,
            "outside_policy": outside, "mark": None}


def test_the_suggestion_states_the_asks_acceptance_and_value_but_never_picks_it() -> None:
    scen = {"recommended_rate": 14.0, "table": [_row(13.6, True, v=900000, acc=0.95), _row(13.9), _row(14.0, v=95000, acc=0.7)]}
    out = decision_support.suggest({"requested_rate": 13.6, "reason": "competitor_lower_rate"}, scen, floor_rate=13.9,
                                   similar=None, evidence_verified=True)
    assert out["rate"] == 13.9 and out["action"] == "counter"
    text = " ".join(r["text"] for r in out["reasons"])
    assert "from 70% to 95%" in text and "against" in text and "13.60%" in text.replace("13.6%", "13.60%")


# ------------------------------------------------------------------ 6. a marked rate is never clamped

def test_an_ask_below_the_band_is_scored_at_its_own_rate_and_flagged(stubbed) -> None:
    grid = rp.rate_grid({"min_rate": 13.9, "max_rate": 14.5}, 12.75, 13.92, 14.5)
    assert dict(grid)[12.75] == ["ask"]
    assert all(13.0 <= r <= 17.0 for r, marks in grid if not marks), "steps stay inside the band"
    rows, _ = rp.rates_table(PAYLOAD, grid, "Rajkot", 2.8e6, 84)
    ask = next(r for r in rows if "ask" in r["marks"])
    assert ask["rate"] == 12.75 and ask["in_band"] is False and "rate_outside_13_17" in ask["failures"]


# ------------------------------------------------------------------ 7. no floor, no lever, no NaN

def _nan_free(value) -> bool:
    if isinstance(value, float):
        return not math.isnan(value)
    if isinstance(value, dict):
        return all(_nan_free(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nan_free(v) for v in value)
    return True


def test_a_declined_grade_has_no_floor_lever_and_no_nan_rate_reaches_the_output(monkeypatch) -> None:
    from aavas.analytics import risk_pricing
    from aavas.governance import model_registry

    held = stub_bundle("D")
    monkeypatch.setattr(model_registry, "load_bundle", lambda *a, **k: held)
    monkeypatch.setattr(risk_pricing, "_shift", lambda city: (0.0, None))
    bo.clear_caches()
    out = bo.best_offers({**REQUEST, "request_id": "RRQ-DECLINED"}, PAYLOAD)
    assert not any(l["lever"] == "rate" for l in out["levers"])
    assert "nan" not in json.dumps(out, default=str).lower().replace("financ", "")
    for l in out["levers"] + out["table"] + out["points"]:
        assert math.isfinite(l["rate"])
    assert all(b["offer"] is None and b["reason"] for b in out["best"])
    assert out["counts"]["eligible"] == 0


# ------------------------------------------------------------------ 8. the best_offer.yaml check

def _bad(mutate) -> dict:
    cfg = copy.deepcopy(best_offer_config())
    mutate(cfg)
    return cfg


BAD_EDITS = {
    "rate step 0": lambda c: c["levers"]["rate"].update(step_pct=0),
    "tenure step 0": lambda c: c["levers"]["tenure"].update(step_months=0),
    "tenure range backwards": lambda c: c["levers"]["tenure"].update(min_months=120, max_months=36),
    "fee above the published fee": lambda c: c["levers"]["processing_fee"].update(levels_pct=[0.9, 0.0]),
    "negative fee": lambda c: c["levers"]["processing_fee"].update(levels_pct=[-0.1]),
    "reduction of 100%": lambda c: c["levers"]["loan_amount"].update(reductions_pct=[100]),
    "unknown measure to maximise": lambda c: c["objectives"][0].update(maximise="speed"),
    "unknown at_least measure": lambda c: c["objectives"][0]["at_least"].update(measure="margin"),
    "unknown reference": lambda c: c["objectives"][0]["at_least"].update(reference="competitor"),
    "missing table_rows": lambda c: c["display"].pop("table_rows"),
    "missing tie_break": lambda c: c.pop("tie_break"),
    "unknown tie_break": lambda c: c.update(tie_break=["luck"]),
    "negative comfort limit": lambda c: c["levers"]["emi"].update(max_increase_pct=-1),
    "no objectives": lambda c: c.update(objectives=[]),
}


def test_the_current_best_offer_file_passes() -> None:
    bo.validate_config(best_offer_config())


@pytest.mark.parametrize("name", sorted(BAD_EDITS))
def test_each_bad_edit_is_rejected(name) -> None:
    with pytest.raises(ValueError, match="best_offer.yaml"):
        bo.validate_config(_bad(BAD_EDITS[name]))


def test_the_config_smoke_check_rejects_a_bad_best_offer_edit(monkeypatch) -> None:
    bad = _bad(BAD_EDITS["rate step 0"])
    monkeypatch.setattr(configuration, "load", lambda name: bad if name == "best_offer" else load(name))
    with pytest.raises(ValueError, match="step_pct"):
        configuration._smoke()


# ------------------------------------------------------------------ 9. the light loader

def test_the_two_routes_do_not_load_the_portfolio(client, monkeypatch) -> None:        # noqa: F811
    from aavas.api import offers as offers_api

    rohit, meera = who(client, "rohit"), who(client, "meera")
    priced = client.post("/pricing/final-assessment", json={"loan_id": "LGT-1", **FILE}, headers=rohit)
    assert priced.status_code == 200, priced.text
    rec = priced.json()["recommendation"]["recommended_rate"]
    made = client.post("/pricing/rate-requests", headers=rohit,
                       json={"loan_id": "LGT-1", "rate": round(rec - 0.30, 2), "reason": "competitor_lower_rate", "note": "HDFC"})
    request_id = made.json()["request_id"]

    def heavy(*a, **k):
        raise AssertionError("the portfolio and branch propensity are not needed here")

    monkeypatch.setattr(offers_api, "_scenario_inputs", heavy)
    monkeypatch.setattr(offers_api.offers, "headroom", heavy)
    for name in ("risk-pricing", "best-offer"):
        assert client.get(f"/pricing/rate-requests/{request_id}/{name}", headers=meera).status_code == 200, name
        assert client.get(f"/pricing/rate-requests/{request_id}/{name}", headers=rohit).status_code == 403
        assert client.get(f"/pricing/rate-requests/RRQ-NOPE/{name}", headers=meera).status_code == 404


# ------------------------------------------------------------------ 10. the stress add-on is the policy's

def test_the_stress_add_on_in_words_comes_from_the_policy(monkeypatch) -> None:
    monkeypatch.setattr(constraints, "stress_add_pct", lambda: 3.5)
    assert "rose 3.5 points" in bo.reason_words()["stressed_foir_over_cap"]
    rows = [{"rate": 15.0, "stressed_foir_ok": False, "stressed_foir": 0.52, "foir_cap": 0.5, "margin_over_risk_cost": 4.0, "pd_annual": 0.01}]
    text = rp.verdict(ask=15.0, floor=13.0, grade="A", band_info={"borderline": False}, rows=rows, ask_row=None, rec_row=None,
                      book_margin=None, recommended=15.0)["text"]
    assert "rose 3.5 points" in text and "rose 2 points" not in text
    assert rp.risk_pricing  # the API carries the figure for the screen
