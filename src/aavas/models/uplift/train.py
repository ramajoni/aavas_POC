"""Pricing uplift: for this file, how many points of acceptance does a discount of
``d`` bps actually buy, and is it worth what it costs?

The question the RM and the approving manager ask in every negotiation, and the
one the historical book answers worst. Discounts are not random: salespeople give
them to the files that were hardest to close - large competitor gaps, active
negotiation, month-end pressure - and those files convert worse *with* the
discount than an easy file does without one. Read naively, the override history
says a discount *lowers* acceptance (``learning/override_learning.py``; on the
holdout here, by about 3.5 points). Any estimator that does not adjust for that
selection learns the salesperson's judgement, not the customer's response.

Design (and why):

* **Rows.** Files with a discount override (``override_bps < 0``) against files
  left at the recommendation (``override_bps == 0``). Premiums are a different
  decision with a different selection, and are excluded rather than averaged in.
  The outcome is ``disbursed``.
* **Features.** ``feature_config.uplift``: only what the RM sees when deciding.
  ``competitor_gap_bps`` is re-measured at the recommended rate; the loader's
  own column sits at the sanctioned rate, after the discount, and would leak it.
* **Selection adjustment.** A propensity model - P(discounted | context),
  cross-fitted - is shared by every learner: as inverse weights, as the
  doubly-robust correction, and as a covariate. Overlap is real here (no file is
  certain to be discounted or not), which is what makes this possible at all;
  scores are clipped to ``propensity_clip`` and the share clipped is reported.
* **Candidates.** Learners {dose, rdose, t, x, dr} (``learners.py``) x the
  portfolio's monotone-capable boosted-tree families. ``dose`` is the
  dose-aware learner the deck needs: discount size is an input with a monotone
  +1 constraint, so a per-file curve over ``d`` exists and can never bend down.
  The others draw their curve as linear in ``d``.
* **Selection metric.** IPW-weighted Qini on rolling out-of-time windows carved
  from the end of the training window - the metric a bank could compute on its
  own book, because it needs only observed outcomes. The trainer's holdout is
  never shown to the search. Rule: best mean Qini, then explainability (dose,
  t, x, dr, rdose) among candidates within ``tie_tolerance_pts``.
* **Served model.** The selected learner, refitted on the whole training
  window, bagged with bootstrap refits: the estimate is their median and the
  interval their quantiles, so the interval always brackets the estimate and
  both stay monotone in ``d``. A shallow surrogate of the uplift at the
  reference discount gives, through TreeSHAP, the global and per-file drivers of
  who responds to a discount.
* **Truth check, synthetic-only.** The generator knows each file's elasticity, so
  PEHE and rank correlation against the true lift are computed on the holdout -
  as evaluation, labelled as such, never as an input.

What this does not claim: unconfoundedness is an assumption - anything that moved
both the salesperson and the customer and is not in the contract biases every
learner here alike. A randomised discount holdout is the only clean test.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd

from ...automl import portfolio
from ...config import model_config, seed
from ...features import sales_features, stage2_features
from ...features.feature_pipeline import FeatureTransformer, spec
from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger, log_metrics, run
from .. import base, provenance
from ..lgd.train import global_importance, table
from . import evaluation as ev
from . import learners as lr

log = get_logger(__name__)
NAME = "uplift"
TASK = "uplift"
TARGET = TARGETS["conversion"]
POINTS = 100.0
# Monotone-capable boosted-tree families from automl.portfolio, in report order.
SEARCH_FAMILIES = ("hist_gradient_boosting", "lightgbm", "xgboost", "catboost")


def cfg() -> dict[str, Any]:
    return model_config()[NAME]


# ------------------------------------------------------------------ frames


def rows(df: pd.DataFrame) -> pd.DataFrame:
    """Discounted files and files left alone; premiums excluded."""
    bps = pd.to_numeric(df["override_bps"], errors="coerce")
    out = df[bps.notna() & (bps <= 0)].copy()
    out["treated"] = (out["override_bps"] < 0).astype(int)
    out[lr.DOSE] = -out["override_bps"].astype(float)
    return out


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """What the RM sees: the assessed file, and the commercial context read
    against the *recommended* rate (before any discount)."""
    built = stage2_features.build(frame)
    return sales_features.build(built, reference_rate_col="ai_recommended_rate")


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The loader's chronological split, with the boundary day moved to the holdout.

    ``loader.train_test_split`` cuts by row position, so a busy day can sit on both
    sides. The search validates on the end of the training window, so a shared day
    would put the holdout's first day into model selection.
    """
    dev_df, holdout_df = loader.train_test_split(rows(df))
    time_column = model_config()["data"]["time_column"]
    if holdout_df.empty or time_column not in dev_df.columns:
        return dev_df, holdout_df
    straddling = dev_df[time_column] >= holdout_df[time_column].min()
    if straddling.any():
        holdout_df = pd.concat([dev_df[straddling], holdout_df])
        dev_df = dev_df[~straddling]
    return dev_df.copy(), holdout_df.copy()


def reference_discount(dev_df: pd.DataFrame) -> float:
    configured = cfg().get("reference_discount_bps")
    if configured:
        return float(configured)
    treated = dev_df.loc[dev_df["treated"] == 1, lr.DOSE]
    return float(np.median(treated)) if len(treated) else 10.0


def windows(dev_df: pd.DataFrame, n: int, fraction: float) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling-origin validation windows over the last ``fraction`` of the
    training window: window k fits on everything before block k and validates on
    block k. Rows sharing a boundary date go to the validation side."""
    time_column = model_config()["data"]["time_column"]
    ordered = dev_df.sort_values(time_column, kind="stable")
    start = int(len(ordered) * (1 - fraction))
    cuts = np.linspace(start, len(ordered), n + 1).astype(int)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        boundary = ordered[time_column].iloc[a]
        fit = ordered.iloc[:a]
        fit = fit[fit[time_column] < boundary]
        valid = ordered.iloc[len(fit):b]
        out.append((fit.copy(), valid.copy()))
    return out




# ------------------------------------------------------------------ windows


class Prepared:
    """One fit / validation pair, featurised and propensity-scored once and then
    shared by every trial - the comparison is between learners, not between
    nuisance fits."""

    def __init__(self, fit_df: pd.DataFrame, valid_df: pd.DataFrame, feature_spec):
        c = cfg()
        clip = tuple(float(v) for v in c["propensity_clip"])
        self.fit_df, self.valid_df = fit_df, valid_df
        self.transformer = FeatureTransformer(feature_spec)
        self.X_fit = self.transformer.fit_transform(features(fit_df))
        self.X_valid = self.transformer.transform(features(valid_df))
        self.t_fit, self.t_valid = (f["treated"].to_numpy(int) for f in (fit_df, valid_df))
        self.d_fit, self.d_valid = (f[lr.DOSE].to_numpy(float) for f in (fit_df, valid_df))
        self.y_fit, self.y_valid = (f[TARGET].astype(int).to_numpy() for f in (fit_df, valid_df))
        self.propensity_model, self.e_fit = lr.propensity(
            self.X_fit, self.t_fit, c["propensity_params"], int(c["cross_fit_folds"]), clip)
        self.e_valid_raw = self.propensity_model.predict_proba(self.X_valid)[:, 1]
        self.e_valid = np.clip(self.e_valid_raw, *clip)
        self.w_valid = lr.ipw(self.t_valid, self.e_valid, stabilised=False)

    def propensity_summary(self) -> dict[str, float]:
        from sklearn.metrics import roc_auc_score

        clip = cfg()["propensity_clip"]
        raw = self.e_valid_raw
        return {"auc": round(float(roc_auc_score(self.t_valid, raw)), 4),
                "clipped_pct": round(float(((raw < clip[0]) | (raw > clip[1])).mean() * 100), 2),
                "naive_difference_pts": round(ev.weighted_difference(self.y_valid, self.t_valid)
                                              * POINTS, 2),
                "ipw_difference_pts": round(ev.weighted_difference(self.y_valid, self.t_valid,
                                                                   self.w_valid) * POINTS, 2)}


def score(learner: lr.Learner, win: Prepared, reference_bps: float) -> dict[str, float]:
    """Fit on the window's fit rows; rank its validation rows at the reference discount."""
    learner.fit(win.X_fit, win.t_fit, win.d_fit, win.y_fit, win.e_fit)
    s = learner.uplift(win.X_valid, reference_bps, win.e_valid)
    y, t, w = win.y_valid, win.t_valid, win.w_valid
    return {"qini": ev.qini(y, t, s, w) * POINTS, "auuc": ev.auuc(y, t, s, w) * POINTS,
            "qini_unweighted": ev.qini(y, t, s) * POINTS, "ate_pts": float(s.mean() * POINTS)}


# ------------------------------------------------------------------ search


def _draw(space_spec: tuple, rng: np.random.Generator) -> Any:
    """One value from a portfolio search-space spec (same format as automl.portfolio)."""
    kind = space_spec[0]
    if kind == "int":
        return int(rng.integers(int(space_spec[1]), int(space_spec[2]) + 1))
    if kind == "float":
        return round(float(rng.uniform(space_spec[1], space_spec[2])), 4)
    if kind == "log":
        return float(f"{np.exp(rng.uniform(np.log(space_spec[1]), np.log(space_spec[2]))):.4g}")
    if kind == "choice":
        options = list(space_spec[1])
        return options[int(rng.integers(len(options)))]
    raise ValueError(f"unknown search-space spec {space_spec!r}")


def incumbent() -> dict[str, Any]:
    c = cfg()
    return {"learner": c["fallback_learner"], "family": c["fallback_family"],
            "params": dict(c["params"])}


def _trials(learner: str, family: str, n: int) -> list[tuple[str, dict]]:
    """Trial 0 is the incumbent's params on the incumbent's family - for every
    learner, so that family's column compares learners like for like - or the
    family's portfolio defaults elsewhere; the rest are seeded draws from the
    family's portfolio space."""
    import zlib

    fam = portfolio.family("binary", family)
    inc = incumbent()
    first = (("incumbent", dict(inc["params"])) if family == inc["family"]
             else ("defaults", dict(fam.defaults)))
    rng = np.random.default_rng([seed(), zlib.crc32(f"{learner}/{family}".encode())])
    trials, seen, attempts = [first], {repr(sorted(first[1].items()))}, 0
    while len(trials) < n and attempts < n * 20:
        attempts += 1
        params = {k: _draw(v, rng) for k, v in fam.space.items()}
        if repr(sorted(params.items())) not in seen:
            seen.add(repr(sorted(params.items())))
            trials.append(("random", params))
    return trials


def _allocate(cells: list[tuple[str, str]], budget: int) -> dict[tuple[str, str], int]:
    """Spread ``budget`` trials over the cells: one each, the remainder to the
    most explainable cells first. A budget below the cell count still runs one
    trial per cell (reported), because an unsearched cell is not a comparison."""
    base_n, extra = divmod(max(budget, len(cells)), len(cells))
    return {cell: base_n + (1 if i < extra else 0) for i, cell in enumerate(cells)}


def search(df: pd.DataFrame | None = None, budget: int | None = None, *,
           learners: list[str] | None = None, families: list[str] | None = None,
           record: bool = True) -> dict[str, Any]:
    """Learners x families, ranked by IPW-Qini on rolling out-of-time windows.

    Returns the report (A1's portfolio schema) and, with ``record``, writes it to
    ``reports/automl/portfolio/uplift.json`` and merges an ``uplift`` entry into
    ``selection.json`` through ``automl.portfolio.record`` (locked read-modify-
    write; other models' entries are untouched).
    """
    c = cfg()
    budget = int(c.get("search_budget", 30) if budget is None else budget)
    started = time.perf_counter()
    dev_df, holdout_df = split(table(df))
    feature_spec = spec(NAME)
    reference_bps = reference_discount(dev_df)
    order = list(c.get("explainability_order") or lr.LEARNERS)
    learners = [name for name in order if name in (learners or lr.LEARNERS)]

    considered, skipped, libraries = [], [], {}
    for family in families or SEARCH_FAMILIES:
        problem = lr.family_unavailable(family)
        if problem:
            skipped.append({"family": family, "kind": "unavailable", "reason": problem})
            continue
        considered.append(family)
        fam = portfolio.family("binary", family)
        libraries[fam.library] = getattr(__import__(fam.library), "__version__", None)
    for fam in portfolio.families("binary"):
        if fam.name not in considered and not any(s["family"] == fam.name for s in skipped):
            skipped.append({"family": fam.name, "kind": "excluded", "reason": (
                f"cannot carry the monotone discount constraint: {fam.no_monotone_reason}"
                if fam.monotone is None else
                "bagged trees: unshrunk leaf probabilities make a difference of two predictions "
                "mostly noise, and the deck's explainability rule ranks them last anyway"
                if fam.explainability == "bagged_trees" else "not in the uplift search")})
    if not considered:
        raise RuntimeError("no monotone-capable family is available for the uplift search")

    prepared = [Prepared(fit, valid, feature_spec)
                for fit, valid in windows(dev_df, int(c["validation_windows"]),
                                          float(c["validation_fraction"]))]
    cells = [(learner, family) for learner in learners for family in considered]
    allocation = _allocate(cells, budget)
    folds = int(c["cross_fit_folds"])

    leaderboard = []
    for learner, family in cells:
        fam = portfolio.family("binary", family)
        for i, (origin, params) in enumerate(_trials(learner, family, allocation[(learner, family)])):
            row = {"learner": learner, "family": family, "candidate": f"{learner}/{family}",
                   "trial": i, "origin": origin, "params": {"learner": learner, **params},
                   "explainability": fam.explainability,
                   "explainability_tier": order.index(learner) + 1}
            t0 = time.perf_counter()
            try:
                per_window = [score(lr.make(learner, family, params, folds=folds), win,
                                    reference_bps) for win in prepared]
                metrics = {k: float(np.mean([w[k] for w in per_window])) for k in per_window[0]}
                metrics["qini_by_window"] = [round(w["qini"], 4) for w in per_window]
                row.update(status="ok", metrics=_round(metrics),
                           fit_seconds=round(time.perf_counter() - t0, 2))
            except Exception as exc:                                # noqa: BLE001 - recorded, not raised
                row.update(status="failed", metrics={}, fit_seconds=None,
                           error=f"{type(exc).__name__}: {exc}"[:300])
                log.warning("[uplift] %s/%s trial %s failed: %s", learner, family, i, exc)
            log.info("[uplift] %-28s trial %s qini %s (%.1fs)", row["candidate"], i,
                     row["metrics"].get("qini"), time.perf_counter() - t0)
            leaderboard.append(row)

    ok = [r for r in leaderboard if r["status"] == "ok"]
    if not ok:
        raise RuntimeError("every uplift trial failed; see the log")
    winner, resolved = choose(ok, float(c.get("tie_tolerance_pts", 0.05)))
    leaderboard.sort(key=lambda r: (r["status"] != "ok", -(r["metrics"].get("qini") or -np.inf)))
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank

    def best_by(key: str) -> list[dict]:
        out = {}
        for r in ok:
            if r[key] not in out or r["metrics"]["qini"] > out[r[key]]["metrics"]["qini"]:
                out[r[key]] = r
        return sorted(({k: r[k] for k in ("learner", "family", "trial", "origin", "params",
                                            "metrics", "fit_seconds", "explainability")}
                       for r in out.values()), key=lambda r: -r["metrics"]["qini"])

    rolling = [portfolio.window(w.fit_df, w.valid_df) for w in prepared]
    validation = portfolio.window(prepared[0].fit_df,
                                  pd.concat([w.valid_df for w in prepared]))["valid"]
    report = {
        "model": NAME,
        "task": TASK,
        "target": TARGET,
        "framework": (f"aavas.models.uplift meta-learner search over aavas.automl.portfolio "
                      f"families ({', '.join(f'{k} {v}' for k, v in sorted(libraries.items()))})"),
        "metric": "qini",
        "higher_is_better": True,
        "units": "pts",
        "features": feature_spec.columns,
        "treatment": {"column": lr.DOSE, "monotone": lr.dose_contract()[1],
                      "definition": "discount below the recommendation, bps (override_bps < 0); "
                                    "control: override_bps == 0; premiums excluded"},
        "monotone_constraints": {lr.DOSE: lr.dose_contract()[1]},
        "learners_considered": learners,
        "families_considered": considered,
        "families_skipped": skipped,
        "budget": budget,
        "trials": len(leaderboard),
        "trials_failed": sum(1 for r in leaderboard if r["status"] != "ok"),
        "leaderboard": leaderboard,
        "best_per_family": best_by("family"),
        "best_per_learner": best_by("learner"),
        "incumbent": incumbent(),
        "selected": {"family": winner["family"], "learner": winner["learner"],
                     "label": f"{lr.LABELS[winner['learner']]} on {portfolio.label(winner['family'])}",
                     "params": winner["params"], "trial": winner["trial"],
                     "origin": winner["origin"], "explainability": winner["explainability"],
                     "curve": lr.CURVES[winner["learner"]], "validation": winner["metrics"]},
        "rule": {
            "statement": "Best validation score, then explainability.",
            "primary": "highest mean IPW-weighted Qini across the rolling out-of-time windows, "
                       f"every file scored at the reference discount ({reference_bps:g} bps)",
            "tolerance_abs_pts": float(c.get("tie_tolerance_pts", 0.05)),
            "tie_break": "candidates within the tolerance of the best are tied; among them the "
                         "most explainable learner wins, then the better score",
            "explainability_ranking": order,
            **resolved,
        },
        "reference_discount_bps": reference_bps,
        # A1's shape (fit / valid / holdout spans); the rolling detail beside it.
        "windows": {"fit": rolling[-1]["fit"], "valid": validation,
                    "holdout": {**portfolio.window(dev_df, dev_df, holdout_df)["holdout"]},
                    "rolling": rolling},
        "propensity": [w.propensity_summary() for w in prepared],
        "seed": seed(),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "notes": [
            "The trainer's holdout (the last 20% by application date) was never passed to this "
            "search; validation is rolling windows over the end of the training window.",
            "Qini is computed on observed outcomes only, reweighted by inverse propensity "
            "because discounts were not randomised; qini_unweighted is the unadjusted read.",
            "Qini depends only on how files are ranked, so learners whose curve is linear in the "
            "discount are compared with the fitted dose-response learner on equal terms.",
            "No synthetic ground truth is used here; the truth check runs on the holdout, in the "
            "trainer, as evaluation only.",
        ],
    }
    if record:
        portfolio.record(report)
    log.info("[uplift] selected %s (qini %.3f pts) from %s trials in %.0fs",
             report["selected"]["label"], winner["metrics"]["qini"], len(leaderboard),
             report["elapsed_seconds"])
    return report


def choose(rows_: list[dict], tolerance_pts: float) -> tuple[dict, dict]:
    """Best Qini, then explainability among those within ``tolerance_pts``."""
    top = max(rows_, key=lambda r: r["metrics"]["qini"])
    tied = [r for r in rows_ if top["metrics"]["qini"] - r["metrics"]["qini"] <= tolerance_pts + 1e-12]
    winner = min(tied, key=lambda r: (r["explainability_tier"], -r["metrics"]["qini"]))
    return winner, {
        "best_score_candidate": top["candidate"], "best_score": top["metrics"]["qini"],
        "within_tolerance": sorted({r["candidate"] for r in tied}),
        "decided_by": ("validation score" if winner is top
                       else "explainability (within tolerance of the best score)"),
    }


def _round(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(float(v), 6) if isinstance(v, (int, float)) and np.isfinite(v)
                else (v if isinstance(v, list) else None)) for k, v in metrics.items()}


# ------------------------------------------------------------------- train


def choice() -> dict[str, Any]:
    """What the trainer fits: the recorded selection when usable, else the fallback,
    with the reason recorded for the card."""
    c = cfg()
    fallback = incumbent()
    if c.get("learner") != "automl":
        return {**fallback, "selection": {"status": "disabled",
                                          "detail": f"model_config.uplift.learner is "
                                                    f"{c.get('learner')!r}, not 'automl'"}}
    entry = portfolio.selected(NAME)
    if entry is None:
        return {**fallback, "selection": {
            "status": "no_selection",
            "detail": "no uplift selection recorded in reports/automl/portfolio/selection.json; "
                      "the configured fallback learner was fitted"}}
    params = dict(entry.get("params") or {})
    learner = params.pop("learner", None)
    problem = (None if learner in lr.LEARNERS else f"unknown learner {learner!r}") \
        or lr.family_unavailable(entry["family"])
    if problem:
        log.warning("[uplift] recorded selection unusable (%s); fitting the fallback", problem)
        return {**fallback, "selection": {"status": "fallback", "selected": f"{learner}/{entry['family']}",
                                          "detail": problem, "report": entry.get("report")}}
    return {"learner": learner, "family": entry["family"], "params": params,
            "selection": {"status": "selected", **entry}}


def _bag(chosen: dict, X, t, d, y, e, n: int, folds: int) -> list[lr.Learner]:
    rng = np.random.default_rng(seed())
    members = []
    for _ in range(n):
        idx = rng.integers(0, len(X), len(X))
        members.append(lr.make(chosen["learner"], chosen["family"], chosen["params"], folds=folds)
                       .fit(X.iloc[idx].reset_index(drop=True), t[idx], d[idx], y[idx], e[idx]))
    return members


def _support(dev_df: pd.DataFrame) -> dict[str, float]:
    """Where the discount evidence is. Beyond ``max_bps`` the curve is extrapolation."""
    doses = dev_df.loc[dev_df["treated"] == 1, lr.DOSE].to_numpy(float)
    return {"min_bps": float(np.percentile(doses, 1)), "max_bps": float(np.percentile(doses, 99)),
            "observed_max_bps": float(doses.max()), "n_discounted": int(len(doses))}


def train(df: pd.DataFrame | None = None, save: bool = True) -> base.ModelCard:
    from sklearn.metrics import r2_score, roc_auc_score

    c = cfg()
    dev_df, holdout_df = split(table(df))
    feature_spec = spec(NAME)
    chosen = choice()
    clip = tuple(float(v) for v in c["propensity_clip"])
    folds = int(c["cross_fit_folds"])
    grid = [float(v) for v in c["curve_bps"]]
    reference_bps = reference_discount(dev_df)

    transformer = FeatureTransformer(feature_spec)
    X_dev = transformer.fit_transform(features(dev_df))
    X_hold = transformer.transform(features(holdout_df))
    t_dev, t_hold = (f["treated"].to_numpy(int) for f in (dev_df, holdout_df))
    d_dev, d_hold = (f[lr.DOSE].to_numpy(float) for f in (dev_df, holdout_df))
    y_dev, y_hold = (f[TARGET].astype(int).to_numpy() for f in (dev_df, holdout_df))

    with run(f"{NAME}-train", {"learner": f"{chosen['learner']}/{chosen['family']}"}) as active:
        prop_model, e_dev = lr.propensity(X_dev, t_dev, c["propensity_params"], folds, clip)
        e_raw = prop_model.predict_proba(X_hold)[:, 1]
        e_hold = np.clip(e_raw, *clip)
        learner = lr.make(chosen["learner"], chosen["family"], chosen["params"], folds=folds)
        learner.fit(X_dev, t_dev, d_dev, y_dev, e_dev)
        members = _bag(chosen, X_dev, t_dev, d_dev, y_dev, e_dev, int(c["bootstrap_members"]), folds)
        estimator = lr.UpliftEstimator(transformer, prop_model, learner, members, clip,
                                       tuple(float(v) for v in c["interval"]), reference_bps,
                                       _support(dev_df))

        s_ref = estimator.uplift(X_hold, e_hold, reference_bps)
        w = lr.ipw(t_hold, e_hold, stabilised=False)
        treated = t_hold == 1
        at_own = estimator.uplift(X_hold[treated], e_hold[treated], d_hold[treated])
        base_hold = estimator.baseline(X_hold, e_hold)
        curve = np.vstack([estimator.uplift(X_hold, e_hold, d) for d in grid])
        low, high = estimator.band(X_hold, e_hold, reference_bps)
        raw = learner.raw_effect(X_hold, e_hold)
        metrics = {
            "qini": ev.qini(y_hold, t_hold, s_ref, w) * POINTS,
            "auuc": ev.auuc(y_hold, t_hold, s_ref, w) * POINTS,
            "qini_unweighted": ev.qini(y_hold, t_hold, s_ref) * POINTS,
            "ate_pts": float(s_ref.mean() * POINTS),
            "att_pts": float(at_own.mean() * POINTS),
            "ipw_difference_pts": ev.weighted_difference(y_hold, t_hold, w) * POINTS,
            "naive_difference_pts": ev.weighted_difference(y_hold, t_hold) * POINTS,
            "reference_discount_bps": reference_bps,
            "propensity_auc": float(roc_auc_score(t_hold, e_raw)),
            "propensity_clipped_pct": float(((e_raw < clip[0]) | (e_raw > clip[1])).mean() * 100),
            "baseline_calibration_gap_pts": float(abs(base_hold[~treated].mean()
                                                      - y_hold[~treated].mean()) * POINTS),
            "monotonicity_violations_pct": float((np.diff(curve, axis=0) < -1e-9).any(axis=0).mean()
                                                 * 100),
            "negative_effect_clipped_pct": float((raw < 0).mean() * 100) if raw is not None else 0.0,
            "interval_width_pts": float((high - low).mean() * POINTS),
            "treated_share_pct": float(treated.mean() * 100),
        }
        metrics.update(ev.truth_check(holdout_df, lambda d: estimator.uplift(X_hold, e_hold, d),
                                      reference_bps, grid, band=(low, high)))
        log_metrics(active, metrics)

    # Who responds to a discount: a shallow surrogate of the served uplift at the
    # reference discount, explained by TreeSHAP.
    target_dev = estimator.uplift(X_dev, e_dev, reference_bps) * POINTS
    surrogate = base.hgb_regressor(c["surrogate_params"]).fit(X_dev, target_dev)
    metrics["surrogate_r2"] = float(r2_score(s_ref * POINTS, surrogate.predict(X_hold)))
    importance = global_importance(
        surrogate, X_hold, scale=1.0,
        top=int((model_config().get("portfolio") or {}).get("top_weights", 5)))

    p_observed = np.clip(base_hold + np.where(treated, estimator.uplift(X_hold, e_hold, d_hold), 0),
                         0, 1)
    fairness = base.fairness_report(holdout_df, y_hold, p_observed)
    dataset = provenance.identity(
        dev_df, source="loader.build_training_table",
        slice_of="chronological train split of files with a discount override or no override "
                 "(premiums excluded)")

    # The card's learner is the base family, the portfolio's convention (the
    # models page checks it against the selected family); which meta-learner
    # runs on it is in `selection` and in the target text.
    selection_record = {**chosen["selection"], "learner": chosen["learner"],
                        "family": chosen["family"], "label": lr.LABELS[chosen["learner"]]}
    card = base.ModelCard(
        name=NAME, version=model_config()["registry"]["version"], learner=chosen["family"],
        target=(f"{TARGET}: acceptance points bought by a discount, against no override "
                f"({lr.LABELS[chosen['learner']]})"),
        n_train=len(X_dev), n_test=len(X_hold), features=feature_spec.columns, metrics=metrics,
        training_window={"from": str(dev_df["application_date"].min()),
                         "to": str(dev_df["application_date"].max())},
        limitations=_limitations(metrics, chosen, estimator),
        dataset=dataset, fairness=fairness, selection=selection_record,
    ).finalize()

    log.info("uplift %s | qini %.3f pts | ATE@%gbps %.2f pts (naive %+.2f, IPW %+.2f) | "
             "synthetic PEHE %.2f vs constant %.2f, rank corr %.2f", estimator.label,
             metrics["qini"], reference_bps, metrics["ate_pts"], metrics["naive_difference_pts"],
             metrics["ipw_difference_pts"], metrics.get("synthetic_truth_pehe_pts", np.nan),
             metrics.get("synthetic_truth_constant_pehe_pts", np.nan),
             metrics.get("synthetic_truth_rank_corr", np.nan))
    if save:
        base.save(NAME, {"estimator": estimator, "surrogate": surrogate,
                         "evaluation": {"importance": importance, "curve_bps": grid,
                                        "average_curve_pts": [round(float(v), 3) for v in
                                                              curve.mean(axis=1) * POINTS]}}, card)
    return card


def _limitations(m: dict[str, float], chosen: dict, estimator: lr.UpliftEstimator) -> list[str]:
    s = estimator.support
    out = [
        "Observational: discounts were chosen by salespeople, not randomised. The estimate "
        "adjusts for the selection the contract can see (propensity weighting and covariate); "
        "anything that moved both the salesperson and the customer and is not in the contract "
        "biases it. A randomised discount holdout is the only clean test.",
        f"The naive read on the holdout is {m['naive_difference_pts']:+.1f} pts (discounted files "
        f"convert worse); inverse-propensity weighting alone gives {m['ipw_difference_pts']:+.1f} "
        f"pts; the model's average lift of the discounts actually given is {m['att_pts']:+.1f} pts.",
        f"Learned from discounts of {s.get('min_bps', 0):.0f}-{s.get('max_bps', 0):.0f} bps "
        f"(1st-99th percentile of {s.get('n_discounted', 0)} discounted files). Beyond "
        f"{s.get('max_bps', 0):.0f} bps the curve is extrapolation and is flagged as such.",
        "Premium overrides are excluded: this model says nothing about pricing above the "
        "recommendation.",
        "The outcome includes credit's clawback of aggressive discounts on weak files, so the "
        "lift is of a discount *offered*, not one necessarily kept.",
        f"Curve shape: {lr.CURVES[chosen['learner']]}. A discount never lowers acceptance in "
        "this model; an estimate below zero is reported as zero"
        + (f" ({m['negative_effect_clipped_pct']:.0f}% of holdout files)."
           if m.get("negative_effect_clipped_pct") else "."),
        f"The estimate is the median, and the interval the {estimator.interval[0]:.0%}-"
        f"{estimator.interval[1]:.0%} range, of the full-data fit and {len(estimator.members)} "
        "bootstrap refits; it reflects outcome-model variance, not propensity-model variance or "
        "unmeasured confounding"
        + (f" (synthetic-only coverage of the true lift: "
           f"{m['synthetic_truth_interval_coverage_pct']:.0f}%)."
           if "synthetic_truth_interval_coverage_pct" in m else "."),
    ]
    if "synthetic_truth_pehe_pts" in m:
        beats = m["synthetic_truth_pehe_pts"] < m["synthetic_truth_constant_pehe_pts"]
        out.append(
            f"Synthetic-only truth check (holdout, {m['reference_discount_bps']:g} bps): PEHE "
            f"{m['synthetic_truth_pehe_pts']:.2f} pts against {m['synthetic_truth_constant_pehe_pts']:.2f} "
            f"for an oracle that knows only the true average, rank correlation "
            f"{m['synthetic_truth_rank_corr']:.2f}; true average lift "
            f"{m['synthetic_truth_mean_pts']:.2f} pts vs predicted "
            f"{m['synthetic_truth_predicted_mean_pts']:.2f}. "
            + ("Per-file numbers beat the average." if beats else
               "Per-file numbers do NOT beat knowing the average: the ranking carries signal, "
               "the per-file magnitudes are noisy - read them as a direction, not a quote."))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search for and train the pricing uplift model")
    parser.add_argument("--search", action="store_true", help="run the learner x family search first")
    parser.add_argument("--budget", type=int, default=None, help="total search trials")
    parser.add_argument("--no-train", action="store_true", help="search only")
    args = parser.parse_args(argv)
    df = table()
    if args.search:
        search(df, budget=args.budget)
    if not args.no_train:
        train(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
