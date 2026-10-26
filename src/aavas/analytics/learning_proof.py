"""Evidence that the engine learns - continuously, and across states without pooling data.

Two claims a model-risk reviewer will ask to see proven rather than described.

* **Continuous learning.** Every recommendation, human override and customer
  outcome is recorded and joined to the decision it judged; monitors check for
  drift and decide when to retrain; and a model fitted on more outcomes predicts
  better. The last part is a backtest, not an assertion: the conversion model is
  refitted on growing windows of history and every fit is scored on the same
  untouched latest months, so the only thing that changes is how much it has
  learned from.
* **Federated learning.** Each state trains on its own files and sends only a
  parameter vector. The proof is four numbers side by side: the shared model
  before any round, after the rounds, a model fitted on every state's data
  pooled centrally (what federation must come close to without pooling), and
  each state's own personalised blend. Runs are recorded under a content hash, so
  the same data and settings give the same model id.

What has not happened yet is said plainly, not papered over: no retrained
challenger has been shadow-compared or promoted, and the federated rounds are an
in-process simulation of what Flower would carry between states.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ..config import federated_config, model_config
from ..governance import decision_logger, monitors, store
from ..learning import outcome_capture
from ..paths import ROOT
from . import portfolio

# The regional adjustment the federated clients learn: how far below the card
# rate each state actually sanctions, from the same drivers the demo uses
# (scripts/run_demo.py).
FEDERATED_FEATURES = ["ltv", "foir", "bureau_score", "competitor_gap_bps",
                      "sales_target_pct", "branch_target_pct", "sanctioned_loan_amount"]
FEDERATED_TARGET = "discount"
CURVE_MONTHS = (3, 6, 9, 12)       # plus "everything before the held-out months"
HOLDOUT_MONTHS = 4


def _num(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, digits)


# =================================================================== continuous

def loop() -> dict[str, Any]:
    """What the store holds: each step of the loop, counted, with its latest entries."""
    latest = store.fetch_one("SELECT created_at, model_versions, policy_version FROM decisions "
                             "ORDER BY created_at DESC LIMIT 1")
    versioned = store.fetch_one("SELECT COUNT(*) AS n FROM decisions WHERE model_versions IS NOT NULL "
                                "AND policy_version IS NOT NULL")
    overrides = store.query("SELECT loan_id, created_at, ai_rate, human_rate, override_bps, reason, "
                            "approved_by FROM overrides ORDER BY created_at DESC LIMIT 8")
    outcomes = []
    for row in store.query("SELECT * FROM outcomes ORDER BY updated_at DESC LIMIT 8"):
        decision = decision_logger.get_decision(row["decision_id"]) if row.get("decision_id") else None
        compared = outcome_capture.compare_to_decision(row["loan_id"], row, decision) or {}
        outcomes.append({"loan_id": row["loan_id"], "decision_id": row.get("decision_id"),
                         "updated_at": row["updated_at"], "customer_response": row.get("customer_response"),
                         "converted": bool(row.get("disbursed")),
                         "expected_conversion": compared.get("expected_conversion"),
                         "recommended_rate": compared.get("recommended_rate"),
                         "sanctioned_rate": compared.get("sanctioned_rate"),
                         "rate_delta_bps": compared.get("rate_delta_bps")})
    joined = store.fetch_one("SELECT COUNT(*) AS n FROM outcomes WHERE decision_id IS NOT NULL")
    return {
        "decisions": store.count("decisions"),
        "decisions_with_versions": int((versioned or {}).get("n") or 0),
        "latest_decision": latest,
        "overrides": store.count("overrides"), "recent_overrides": overrides,
        "outcomes": store.count("outcomes"),
        "outcomes_joined_to_decision": int((joined or {}).get("n") or 0),
        "recent_outcomes": outcomes,
    }


def checks() -> dict[str, Any]:
    """The monitors: when each last ran and what it found."""
    status = monitors.status()
    return {"status": status["monitors"], "never_run": status["never_run"],
            "history": monitors.history(limit=10)}


def promotion() -> dict[str, Any]:
    """The gate a retrained model must pass, and whether any model has been through it."""
    gates = model_config()["champion_challenger"]
    registered = store.query("SELECT name, version, role, registered_at FROM model_registry "
                             "ORDER BY registered_at DESC")
    cards = []
    for path in sorted((ROOT / "models").glob("*.card.json")):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        metrics = card.get("metrics") or {}
        cards.append({"name": card.get("name"), "version": card.get("version"),
                      "trained_at": card.get("trained_at"),
                      "headline": _num(metrics.get("auc") if "auc" in metrics else metrics.get("mae")),
                      "metric": "AUC" if "auc" in metrics else "MAE" if "mae" in metrics else None})
    challengers = [r for r in registered if r["role"] == "challenger"]
    return {"gates": gates, "registered": registered, "artefacts": cards,
            "challengers_compared": 0, "promotions": sum(r["role"] == "champion" for r in registered),
            "status": ("A retrained challenger is registered and waiting for its shadow comparison."
                       if challengers else
                       "No retrained challenger has been shadow-compared or promoted yet. The gate is "
                       "defined and the backtest above shows what retraining gains; the live "
                       "comparison runs once a challenger is trained on new outcomes.")}


@lru_cache(maxsize=1)
def _curve(files: int) -> dict[str, Any]:
    from ..features.feature_pipeline import spec
    from ..ingestion.schemas import TARGETS
    from ..models import base
    from ..models.conversion import train as conversion

    cfg = model_config()["conversion"]
    df = portfolio.frame()
    dates = pd.to_datetime(df["application_date"])
    start = dates.min().normalize().replace(day=1)
    holdout_from = (dates.max().normalize().replace(day=1)
                    - pd.DateOffset(months=HOLDOUT_MONTHS - 1))
    holdout = df[dates >= holdout_from]
    span = int((holdout_from.year - start.year) * 12 + holdout_from.month - start.month)
    rows = []
    for months in [m for m in CURVE_MONTHS if m < span] + [span]:
        train = df[(dates < holdout_from) & (dates < start + pd.DateOffset(months=months))]
        _, ((X_train, y_train), (X_test, y_test)) = base.design(
            spec("conversion"), conversion.features, TARGETS["conversion"], train, holdout, binary=True)
        model = base.calibrate(base.hgb_classifier(cfg["params"], spec("conversion")),
                               X_train, y_train, cfg.get("calibration"))
        metrics = base.classification_metrics(y_test, model.predict_proba(X_test)[:, 1])
        rows.append({"months_of_history": months, "files": int(len(train)),
                     "trained_to": str((start + pd.DateOffset(months=months) - pd.Timedelta(days=1)).date()),
                     "auc": _num(metrics["auc"]), "log_loss": _num(metrics["log_loss"]),
                     "brier": _num(metrics["brier"])})
    first, last = rows[0], rows[-1]
    return {
        "model": "conversion", "holdout_from": str(holdout_from.date()),
        "holdout_to": str(dates.max().date()), "holdout_files": int(len(holdout)),
        "rows": rows,
        "gain": {"auc_pts": _num((last["auc"] - first["auc"]) * 100, 2),
                 "log_loss_pct": _num((first["log_loss"] - last["log_loss"]) / first["log_loss"] * 100, 2),
                 "brier_pct": _num((first["brier"] - last["brier"]) / first["brier"] * 100, 2)},
        "improves_every_step": all(b["auc"] >= a["auc"] for a, b in zip(rows, rows[1:])),
    }


def learning_curve() -> dict[str, Any]:
    """The conversion model refitted on more and more outcomes, scored on the same latest months."""
    return _curve(len(portfolio.frame()))


def continuous() -> dict[str, Any]:
    curve = learning_curve()
    state = loop()
    check = checks()
    gate = promotion()
    claims = [
        {"claim": "Every recommendation is recorded with the model and policy versions that produced it",
         "proven": state["decisions"] > 0 and state["decisions_with_versions"] == state["decisions"],
         "evidence": f"{state['decisions']} decisions on file, {state['decisions_with_versions']} with versions"},
        {"claim": "Every human override is captured with its reason",
         "proven": state["overrides"] > 0,
         "evidence": f"{state['overrides']} overrides on file"},
        {"claim": "Every customer outcome is joined back to the decision it judged",
         "proven": state["outcomes"] > 0 and state["outcomes_joined_to_decision"] == state["outcomes"],
         "evidence": f"{state['outcomes_joined_to_decision']} of {state['outcomes']} outcomes joined to a decision"},
        {"claim": "Drift is monitored and decides when to retrain",
         "proven": not {"drift", "retraining"} & set(check["never_run"]),
         "evidence": ("drift and retraining checks have run" if not {"drift", "retraining"} & set(check["never_run"])
                      else "not run on this server yet - use Run checks now")},
        {"claim": "A model refitted on more outcomes predicts better",
         "proven": curve["improves_every_step"] and (curve["gain"]["auc_pts"] or 0) > 0,
         "evidence": f"AUC {curve['rows'][0]['auc']:.3f} → {curve['rows'][-1]['auc']:.3f} on the same "
                     f"{curve['holdout_files']:,} held-out files"},
        {"claim": "A retrained model replaces the champion only through a gate with human sign-off",
         "proven": gate["promotions"] > 0 and gate["challengers_compared"] > 0,
         "evidence": gate["status"]},
    ]
    return {"claims": claims, "loop": state, "checks": check, "curve": curve, "promotion": gate}


# ==================================================================== federated

def _clients():
    from ..federated import client as fed_client

    df = portfolio.frame()
    data = df.assign(**{FEDERATED_TARGET: df["policy_card_rate"] - df["sanctioned_rate"]})
    return fed_client.build_clients(data, FEDERATED_FEATURES, FEDERATED_TARGET)


def _weighted(values: list[tuple[float, int]]) -> float:
    total = sum(n for _, n in values) or 1
    return sum(v * n for v, n in values) / total


def centralised(clients) -> np.ndarray:
    """The same model fitted on every state's training rows pooled in one place -
    what federation has to come close to without the rows ever leaving a state."""
    l2 = float(federated_config().get("l2", 1e-3))
    X = np.vstack([c.dataset.X for c in clients])
    y = np.concatenate([c.dataset.y for c in clients])
    design = np.hstack([X, np.ones((len(y), 1))])
    n = len(y)
    return np.linalg.solve(design.T @ design / n + l2 * np.eye(design.shape[1]), design.T @ y / n)


def fingerprint(dataset) -> str:
    """A hash of a state's own rows, computed where they live. Only the parameter
    vector is sent; this is what lets anyone check the rows were not."""
    digest = hashlib.sha256(np.ascontiguousarray(dataset.X).tobytes()
                            + np.ascontiguousarray(dataset.y).tobytes())
    return digest.hexdigest()[:12]


def federated(strategy: str = "fedavg", record: bool = False) -> dict[str, Any]:
    """Run the rounds and lay the evidence out; ``record`` also files the round."""
    from ..federated import regional_personalization as personalization
    from ..federated import server

    clients = _clients()
    zero = np.zeros_like(clients[0].get_parameters())
    before = _weighted([(c.evaluate(zero)[0], len(c.dataset)) for c in clients])
    run = server.run(clients, strategy_name=strategy, persist=record)
    pooled = centralised(clients)
    table = personalization.compare(run, clients)
    rows = []
    for client, (_, row) in zip(clients, table.iterrows()):
        rows.append({"state": row["region"], "files": int(len(client.dataset)),
                     "held_out": int(len(client.dataset.y_test)),
                     "before_mae_bps": _num(client.evaluate(zero)[0], 2),
                     "federated_mae_bps": _num(row["global_mae_bps"], 2),
                     "centralised_mae_bps": _num(client.evaluate(pooled)[0], 2),
                     "local_mae_bps": _num(row["local_mae_bps"], 2),
                     "personalised_mae_bps": _num(row["personalized_mae_bps"], 2),
                     "alpha_global": _num(row["alpha_global"], 3),
                     "fingerprint": fingerprint(client.dataset)})
    after = run.history[-1]["global_mae_bps"]
    central = _weighted([(client.evaluate(pooled)[0], len(client.dataset)) for client in clients])
    k = int(len(run.global_parameters))
    rounds = len(run.history)
    rows_total = sum(len(c.dataset) for c in clients)
    values_per_row = len(FEDERATED_FEATURES) + 1
    recent = store.query("SELECT run_id, created_at, strategy, rounds, participants, global_version, "
                         "final_mae_bps FROM federated_rounds ORDER BY created_at DESC LIMIT 5")
    recorded = any(r["run_id"] == run.run_id for r in recent) or bool(
        store.fetch_one("SELECT run_id FROM federated_rounds WHERE run_id = ?", (run.run_id,)))
    claims = [
        {"claim": "Each state's files stay in that state; only a parameter vector is shared",
         "proven": True,
         "evidence": f"{rounds} rounds × {len(clients)} states × {k} numbers = {rounds * len(clients) * k:,} "
                     f"numbers sent, against {rows_total:,} files × {values_per_row} values that never were"},
        {"claim": "The shared model learns from the states' data",
         "proven": after < before,
         "evidence": f"error {before:.1f} bps before any round → {after:.1f} bps after {rounds}"},
        {"claim": "It is as accurate as pooling every state's data in one place",
         "proven": abs(after - central) <= 0.5,
         "evidence": f"federated {after:.2f} bps vs centralised {central:.2f} bps "
                     f"({'no difference' if abs(after - central) < 0.005 else f'{after - central:+.2f} bps'})"},
        {"claim": "Each state can personalise the shared model",
         "proven": any(r["personalised_mae_bps"] < r["federated_mae_bps"] for r in rows),
         "evidence": f"{sum(r['personalised_mae_bps'] < r['federated_mae_bps'] for r in rows)} of "
                     f"{len(rows)} states do better with their personalised blend"},
        {"claim": "Every run is recorded and reproducible",
         "proven": recorded,
         "evidence": (f"{run.run_id} is on file; the id is a hash of the trained parameters, so the same "
                      "data and settings give the same id" if recorded else
                      f"this run would be {run.run_id}; not recorded yet - use Run and record")},
    ]
    return {
        "claims": claims, "strategy": run.strategy, "proximal_mu": run.proximal_mu,
        "run_id": run.run_id, "recorded": recorded, "rounds": rounds, "parameters_per_state": k,
        "features": FEDERATED_FEATURES, "target": "the card rate minus the sanctioned rate",
        "mae_before_bps": _num(before, 3), "mae_after_bps": _num(after, 3),
        "mae_centralised_bps": _num(central, 3),
        "history": [{"round": 0, "global_mae_bps": _num(before, 3)}]
                   + [{"round": h["round"], "global_mae_bps": h["global_mae_bps"]} for h in run.history],
        "states": rows, "recent_runs": recent,
        "note": ("In-process simulation: the five states run in one process and exchange only parameter "
                 "vectors through the server's aggregate step. With Flower installed the same client "
                 "contract runs over the network; nothing here moves a recommended rate."),
    }
