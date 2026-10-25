"""How an uplift model is judged.

Two kinds of evidence, kept apart on purpose:

* **Observable** - what the bank could compute on its own book. Individual
  uplift is never observed (a file is discounted or it is not), so the ranking
  is judged instead: the Qini curve sorts files by predicted uplift and asks
  whether the discounted files near the top out-convert the untouched files near
  the top by more than at random. Because discounts here were *chosen*, the
  comparison is reweighted by inverse propensity (``weights``); unweighted, the
  hardest files - the ones salespeople discount - would drag every curve down.
* **Synthetic-only truth** - the generator knows each file's price elasticity
  and its acceptance probability, so the true lift of any discount is
  computable. That is the only way to measure PEHE, and it exists only because
  the data is synthetic. Everything computed from it is labelled
  ``synthetic_truth_*``. The latent columns it reads never reach a feature, a
  fit or the search's ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The generator's response mechanics (synthetic/generator.py, "customer response"
# and the credit clawback). Mirrored here for one reason only - to compute ground
# truth on synthetic data. Nothing that predicts reads them.
_TRUTH_COLUMNS = ("true_conversion_probability", "price_elasticity", "override_bps",
                  "credit_clawback", "risk_grade")
_CLAWBACK_RATE = 0.15            # % points credit adds back
_CLAWBACK_PROBABILITY = 0.45
_CLAWBACK_GRADES = ("D", "E")
_CLAWBACK_ABOVE_BPS = 30.0


# ---------------------------------------------------------------- Qini / AUUC


def _ordered(y, treated, score, weights):
    y = np.asarray(y, dtype=float)
    t = np.asarray(treated).astype(float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    s = np.asarray(score, dtype=float)[order]
    # Evaluate only where the score changes: inside a run of tied scores the
    # order is arbitrary, and the trapezoid across the run is the average over
    # every ordering of it.
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    y, t, w = y[order], t[order], w[order]
    cum = lambda v: np.cumsum(v)[ends]                              # noqa: E731
    n = np.r_[0.0, cum(w)]
    wt, wc = np.r_[0.0, cum(w * t)], np.r_[0.0, cum(w * (1 - t))]
    yt, yc = np.r_[0.0, cum(w * y * t)], np.r_[0.0, cum(w * y * (1 - t))]
    return n / n[-1], wt, wc, yt, yc


def qini_curve(y, treated, score, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """(share of files targeted, incremental acceptances per treated file).

    At share x the curve is Y_t(x) - Y_c(x) * N_t(x) / N_c(x), divided by the
    total treated weight, so its end point is the overall (weighted) difference
    in acceptance rates. Where no control file is in the prefix yet, the control
    term is zero (the convention of Radcliffe 2007 and scikit-uplift).
    """
    x, wt, wc, yt, yc = _ordered(y, treated, score, weights)
    ratio = np.divide(wt, wc, out=np.zeros_like(wt), where=wc > 0)
    gain = (yt - yc * ratio) / max(wt[-1], 1e-12)
    return x, gain


def qini(y, treated, score, weights=None) -> float:
    """Qini coefficient: area between the Qini curve and the random-targeting
    line, as a fraction (x100 for acceptance points). Positive = better than
    random; zero = no better."""
    x, gain = qini_curve(y, treated, score, weights)
    return float(np.trapz(gain, x) - gain[-1] / 2.0)


def auuc(y, treated, score, weights=None) -> float:
    """Area under the uplift (cumulative gain) curve above random, as a fraction.

    U(x) = (acceptance rate treated - rate control within the top x) * x - the
    Gutierrez & Gerardy form, which counts a prefix's uplift on all its files."""
    x, wt, wc, yt, yc = _ordered(y, treated, score, weights)
    rate_t = np.divide(yt, wt, out=np.zeros_like(yt), where=wt > 0)
    rate_c = np.divide(yc, wc, out=np.zeros_like(yc), where=wc > 0)
    both = (wt > 0) & (wc > 0)
    curve = np.where(both, (rate_t - rate_c) * x, 0.0)
    return float(np.trapz(curve, x) - curve[-1] / 2.0)


def weighted_difference(y, treated, weights=None) -> float:
    """Acceptance rate treated minus control (Hajek-normalised under weights)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(treated).astype(bool)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    return float(np.average(y[t], weights=w[t]) - np.average(y[~t], weights=w[~t]))


# --------------------------------------------------------- synthetic truth


def has_truth(frame: pd.DataFrame) -> bool:
    return all(c in frame.columns for c in _TRUTH_COLUMNS)


def true_uplift(frame: pd.DataFrame, discount_bps) -> np.ndarray | None:
    """SYNTHETIC-ONLY. True acceptance lift (a probability) of offering a discount
    of ``discount_bps`` below the no-override rate, for each row.

    The generator's acceptance is sigmoid(logit) with the logit falling by
    elasticity x (rate - competitor). ``true_conversion_probability`` is at the
    rate the file actually got, so the no-override logit is recovered by undoing
    that file's own override (and any clawback). Credit claws 15 bps back off
    discounts above 30 bps on D/E files 45% of the time, so a discount offered is
    not always a discount kept; the truth is the expectation over that.
    None when the latent columns are absent - i.e. on real data.
    """
    if not has_truth(frame):
        return None
    p = np.clip(pd.to_numeric(frame["true_conversion_probability"], errors="coerce")
                .to_numpy(float), 1e-6, 1 - 1e-6)
    elasticity = frame["price_elasticity"].to_numpy(float)
    applied = (frame["override_bps"].fillna(0).to_numpy(float) / 100.0
               + _CLAWBACK_RATE * frame["credit_clawback"].fillna(False).astype(float).to_numpy())
    logit0 = np.log(p / (1 - p)) + elasticity * applied
    d = np.broadcast_to(np.asarray(discount_bps, dtype=float), logit0.shape)
    clawed = frame["risk_grade"].isin(_CLAWBACK_GRADES).to_numpy() & (d > _CLAWBACK_ABOVE_BPS)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))                         # noqa: E731
    kept = sig(logit0 + elasticity * d / 100.0)
    partly = sig(logit0 + elasticity * (d / 100.0 - _CLAWBACK_RATE))
    offered = np.where(clawed, (1 - _CLAWBACK_PROBABILITY) * kept + _CLAWBACK_PROBABILITY * partly,
                       kept)
    return offered - sig(logit0)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def truth_check(frame: pd.DataFrame, predict, reference_bps: float, grid: list[float],
                band: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, float]:
    """SYNTHETIC-ONLY. Predicted against true uplift on the holdout, in points.

    ``predict(d)`` returns the model's uplift (probability) for every row of
    ``frame`` at discount ``d``. ``constant_pehe_pts`` is the error of an oracle
    that knows the true average lift and nothing else - the bar a heterogeneous
    model has to clear before its per-file numbers mean anything.
    """
    truth = true_uplift(frame, reference_bps)
    if truth is None:
        return {}
    pred = np.asarray(predict(reference_bps), dtype=float)
    t_pts, p_pts = truth * 100, pred * 100
    q = pd.qcut(pd.Series(pred).rank(method="first"), 5, labels=False).to_numpy()
    out = {
        "synthetic_truth_pehe_pts": float(np.sqrt(np.mean((p_pts - t_pts) ** 2))),
        "synthetic_truth_constant_pehe_pts": float(np.std(t_pts)),
        "synthetic_truth_rank_corr": _spearman(p_pts, t_pts),
        "synthetic_truth_mean_pts": float(t_pts.mean()),
        "synthetic_truth_predicted_mean_pts": float(p_pts.mean()),
        "synthetic_truth_top_quintile_pts": float(t_pts[q == 4].mean()),
        "synthetic_truth_bottom_quintile_pts": float(t_pts[q == 0].mean()),
    }
    errors, oracle = [], []
    for d in grid:
        true_d = true_uplift(frame, d) * 100
        errors.append(np.sqrt(np.mean((np.asarray(predict(d)) * 100 - true_d) ** 2)))
        oracle.append(np.std(true_d))
    out["synthetic_truth_grid_pehe_pts"] = float(np.mean(errors))
    out["synthetic_truth_grid_constant_pehe_pts"] = float(np.mean(oracle))
    if band is not None:
        low, high = band
        inside = (truth >= np.asarray(low) - 1e-12) & (truth <= np.asarray(high) + 1e-12)
        out["synthetic_truth_interval_coverage_pct"] = float(inside.mean() * 100)
    return out
