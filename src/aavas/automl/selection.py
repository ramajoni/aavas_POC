"""Model selection - CV ranking, multi-objective scoring, Pareto fronts, champion choice.

Selection here is explicitly not `argmax AUC`. A candidate is scored on a weighted
blend of predictive quality, calibration, stability and business value, and then
has to clear hard gates. A candidate with the best blended score that fails a gate
does not win; it goes to review.

That ordering - score to rank, gates to admit - is what stops a strong average
hiding an unacceptable specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

# Higher is better for all of these once signs are applied in `blended_score`.
DEFAULT_WEIGHTS = {
    "auc": 0.25, "ks": 0.15, "pr_auc": 0.10,
    "calibration": 0.20,          # 1 - ECE
    "stability": 0.10,            # 1 - PSI
    "profit": 0.20,               # normalised expected profit
}

DEFAULT_GATES = {
    "min_auc": 0.60,
    "max_ece": 0.05,
    "max_psi": 0.25,
    "require_monotone": False,
    "require_fairness_pass": True,
    "require_locality_pass": True,
}


@dataclass
class Candidate:
    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    monotone: bool = False
    interpretable: bool = False
    fairness: dict[str, Any] | None = None
    locality: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    estimator: Any = None

    def as_row(self) -> dict[str, Any]:
        row = {"name": self.name, "family": self.family, "monotone": self.monotone,
               "interpretable": self.interpretable}
        row.update({k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in self.metrics.items()})
        if self.fairness is not None:
            row["fairness_passed"] = self.fairness.get("passed")
        if self.locality is not None:
            row["locality_passed"] = self.locality.get("passed")
            row["worst_unit"] = (self.locality.get("variance", {}).get("auc") or {}).get("worst_unit")
        return row


# ------------------------------------------------------------------- metrics
def ks_statistic(y_true, p) -> float:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    pos, neg = np.sort(p[y_true == 1]), np.sort(p[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    grid = np.unique(p)
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def evaluate_predictions(y_true, p, *, profit_fn=None) -> dict[str, float]:
    """The metric block every candidate is judged on."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from .calibration import brier, expected_calibration_error

    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    metrics: dict[str, float] = {}
    try:
        metrics["auc"] = float(roc_auc_score(y_true, p))
        metrics["pr_auc"] = float(average_precision_score(y_true, p))
    except ValueError:  # single-class fold
        metrics["auc"] = metrics["pr_auc"] = float("nan")
    metrics["ks"] = ks_statistic(y_true, p)
    metrics["ece"] = expected_calibration_error(p, y_true)
    metrics["brier"] = brier(p, y_true)
    metrics["base_rate"] = float(np.mean(y_true))
    if profit_fn is not None:
        try:
            metrics["profit"] = float(profit_fn(y_true, p))
        except Exception as exc:  # noqa: BLE001
            log.warning("automl: profit function failed (%s)", exc)
    return metrics


def psi(expected, actual, buckets: int = 10) -> float:
    """Population stability between two score distributions."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ------------------------------------------------------- multi-objective score
def blended_score(metrics: dict[str, float], weights: dict[str, float] | None = None,
                  profit_scale: float | None = None) -> float:
    """Weighted blend, with every component oriented so higher is better."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    parts = {
        "auc": metrics.get("auc", float("nan")),
        "ks": metrics.get("ks", float("nan")),
        "pr_auc": metrics.get("pr_auc", float("nan")),
        "calibration": 1.0 - metrics.get("ece", 0.0),
        "stability": 1.0 - min(metrics.get("psi", 0.0), 1.0),
        "profit": (metrics.get("profit", 0.0) / profit_scale) if profit_scale else 0.0,
    }
    total_weight = 0.0
    score = 0.0
    for key, weight in w.items():
        value = parts.get(key)
        if value is None or value != value:      # skip NaN rather than poison the score
            continue
        score += weight * float(value)
        total_weight += weight
    return score / total_weight if total_weight else float("nan")


def pareto_front(candidates: Sequence[Candidate],
                 objectives: Sequence[str] = ("auc", "calibration", "profit")) -> list[Candidate]:
    """Non-dominated set. A candidate is on the front if nothing beats it everywhere."""
    def vector(c: Candidate) -> list[float]:
        out = []
        for name in objectives:
            if name == "calibration":
                out.append(1.0 - c.metrics.get("ece", 0.0))
            elif name == "stability":
                out.append(1.0 - c.metrics.get("psi", 0.0))
            else:
                out.append(c.metrics.get(name, float("nan")))
        return out

    # Drop objectives no candidate reports (e.g. profit with no profit function),
    # otherwise every vector contains NaN and the front comes back empty.
    objectives = [o for o in objectives
                  if any(v == v for v in (vector(c)[list(objectives).index(o)]
                                          for c in candidates))] or list(objectives)

    front: list[Candidate] = []
    for candidate in candidates:
        v = vector(candidate)
        if any(v_i != v_i for v_i in v):
            continue
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            o = vector(other)
            if any(o_i != o_i for o_i in o):
                continue
            if all(a >= b for a, b in zip(o, v)) and any(a > b for a, b in zip(o, v)):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


# -------------------------------------------------------------------- gating
def check_gates(candidate: Candidate, gates: dict[str, Any] | None = None) -> list[str]:
    """Hard governance criteria. Returns the reasons a candidate may not ship."""
    g = {**DEFAULT_GATES, **(gates or {})}
    failures: list[str] = []

    auc = candidate.metrics.get("auc")
    if auc is not None and auc == auc and auc < g["min_auc"]:
        failures.append(f"AUC {auc:.3f} below the {g['min_auc']} floor")

    ece = candidate.metrics.get("ece")
    if ece is not None and ece == ece and ece > g["max_ece"]:
        failures.append(f"calibration error {ece:.3f} exceeds {g['max_ece']}")

    psi_value = candidate.metrics.get("psi")
    if psi_value is not None and psi_value == psi_value and psi_value > g["max_psi"]:
        failures.append(f"PSI {psi_value:.3f} exceeds {g['max_psi']}")

    if g["require_monotone"] and not candidate.monotone:
        failures.append("contract requires monotone constraints; this learner cannot carry them")

    if g["require_fairness_pass"] and candidate.fairness is not None:
        if candidate.fairness.get("passed") is False:
            failures.extend(f"fairness: {r}" for r in candidate.fairness.get("reasons", []))

    # A global average that passes while a branch fails is not a pass.
    if g["require_locality_pass"] and candidate.locality is not None:
        if candidate.locality.get("passed") is False:
            for unit in candidate.locality.get("failing_units", [])[:3]:
                failures.append(f"branch {unit['unit']}: {unit['reasons'][0]}")
    return failures


@dataclass
class Selection:
    champion: Candidate | None
    leaderboard: list[dict[str, Any]]
    pareto: list[str]
    referred_to_review: list[dict[str, Any]]
    weights: dict[str, float]
    gates: dict[str, Any]
    # The highest-ranked candidate when no candidate cleared every gate. "What
    # would have won, and which gate stopped it" is the review question; without
    # it a fully-referred search reports only that nothing was admitted.
    best_referred: Candidate | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"champion": self.champion.name if self.champion else None,
                "best_referred": self.best_referred.name if self.best_referred else None,
                "leaderboard": self.leaderboard, "pareto_front": self.pareto,
                "referred_to_review": self.referred_to_review,
                "weights": self.weights, "gates": self.gates}


def select(candidates: Sequence[Candidate], *, weights: dict[str, float] | None = None,
           gates: dict[str, Any] | None = None, profit_scale: float | None = None) -> Selection:
    """Rank on the blended score, admit on the gates."""
    scored: list[tuple[Candidate, float, list[str]]] = []
    for candidate in candidates:
        score = blended_score(candidate.metrics, weights, profit_scale)
        candidate.metrics["blended_score"] = round(score, 5)
        scored.append((candidate, score, check_gates(candidate, gates)))

    scored.sort(key=lambda item: (-item[1] if item[1] == item[1] else 0.0))
    eligible = [c for c, _, failures in scored if not failures]
    referred = [{"name": c.name, "blended_score": c.metrics.get("blended_score"),
                 "failures": failures} for c, _, failures in scored if failures]

    champion = eligible[0] if eligible else None
    if champion is None and scored:
        log.warning("automl: no candidate cleared every gate; all referred to review")

    return Selection(
        champion=champion,
        best_referred=None if champion is not None or not scored else scored[0][0],
        leaderboard=[c.as_row() for c, _, _ in scored],
        pareto=[c.name for c in pareto_front([c for c, _, _ in scored])],
        referred_to_review=referred,
        weights={**DEFAULT_WEIGHTS, **(weights or {})},
        gates={**DEFAULT_GATES, **(gates or {})},
    )
