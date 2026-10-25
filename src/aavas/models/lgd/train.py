"""Loss given default: the severity half of the risk model.

Pricing subtracts expected loss = PD x LGD from the spread. Until now the LGD
side was a policy table (``model_config.risk.lgd_by_property_type`` nudged by
LTV): a statement about collateral that is blind to what actually moves a
recovery on this book - a title exception, a technical exception, how fast the
state's courts move a SARFAESI case - and to the defaults that cure before
enforcement and lose almost nothing.

This model is fitted on the files that defaulted and carry an observed loss
severity (``realized_lgd``). That is thin data, as LGD data always is: about
1,300 defaults in the whole book, about 1,050 of them in the training window. So:

* the learner is shallow, and monotone where the economics says it must be
  (more LTV, a legal or a technical exception never recover *more*);
* every claim is made against the benchmark the model has to beat - the policy
  table, scored on exactly the same holdout rows - with a paired bootstrap
  interval on the difference, because on ~260 holdout defaults a one-point
  "improvement" can be noise;
* the target is the conditional *mean*, because expected loss needs E[LGD].
  Observed LGD is bimodal (cures near zero, the rest around the collateral
  loss), so MAE - the deck's metric, which rewards the median - can prefer a
  flat table that sits near the uncured mode even when the model is the better
  input to expected loss. Both are reported; neither is hidden.

The regulatory floor (``risk.lgd_floor``) is applied by ``RiskModel``, not here,
so this card's metrics describe the model rather than the floor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...config import model_config, seed
from ...features import stage2_features
from ...features.feature_pipeline import reason_label, spec
from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger, log_metrics, run
from .. import base, provenance

log = get_logger(__name__)
NAME = "lgd"
TASK = "regression"
POINTS = 100.0               # LGD fraction -> LGD points


# ------------------------------------------------------------------ frames


def table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The training table (the loader rebuilds its cache when the raw file is newer).
    Shared with the uplift trainer."""
    return loader.load_training_table() if df is None else df


def rows(df: pd.DataFrame) -> pd.DataFrame:
    """Loss severity is only observable on disbursed files that defaulted."""
    target = TARGETS[NAME]
    if target not in df.columns:
        raise KeyError(f"{target!r} is not in the training table; rebuild it from the raw file "
                       "(loader.build_training_table)")
    mask = (df["disbursed"].astype(bool) & df["defaulted"].astype(bool)
            & pd.to_numeric(df[target], errors="coerce").notna())
    return df[mask].copy()


def features(frame: pd.DataFrame) -> pd.DataFrame:
    return stage2_features.build(frame)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The trainer's chronological train / holdout split, of defaulted files."""
    return loader.train_test_split(rows(df))


def _fallback():
    return (base.hgb_regressor(model_config()[NAME]["params"], spec(NAME)),
            base.fallback_learner(NAME))


def incumbent() -> dict:
    return {"family": "hist_gradient_boosting", "params": dict(model_config()[NAME]["params"])}


def selection_data(df: pd.DataFrame | None = None):
    """Fit / validation frames for the AutoML search: this trainer's own split,
    row filter, builder and contract. The target is the LGD fraction; the search
    reports MAE in LGD points (``mae_pts``, scale 100)."""
    dev_df, holdout_df = split(table(df))
    return base.selection_data(
        NAME, TASK, dev_df, holdout_df, feature_spec=spec(NAME), build=features,
        target=TARGETS[NAME], metric="mae_pts", incumbent=incumbent())


def policy_table(frame: pd.DataFrame) -> np.ndarray:
    """The benchmark: what the risk model used before this one, on the same rows."""
    return stage2_features.lgd(features(frame)).to_numpy(float)


# ----------------------------------------------------------------- metrics


def accuracy(y: np.ndarray, pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """MAE / RMSE / bias in LGD points, and R2 - for the model or the table."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    err = pred - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        f"{prefix}mae_pts": float(np.abs(err).mean() * POINTS),
        f"{prefix}rmse_pts": float(np.sqrt((err ** 2).mean()) * POINTS),
        f"{prefix}r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        f"{prefix}bias_pts": float(err.mean() * POINTS),
    }


def calibration(y: np.ndarray, pred: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Mean predicted against mean observed, by decile of the prediction.

    A table has only as many distinct values as it has cells, so its "deciles"
    collapse into fewer bins; that is reported as it is rather than padded.
    """
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    try:
        bucket = pd.qcut(pred, bins, labels=False, duplicates="drop")
    except ValueError:
        bucket = np.zeros(len(pred), dtype=int)
    out = []
    for b in sorted(pd.unique(bucket)):
        mask = bucket == b
        out.append({"bin": int(b) + 1, "n": int(mask.sum()),
                    "predicted_pts": round(float(pred[mask].mean() * POINTS), 2),
                    "observed_pts": round(float(y[mask].mean() * POINTS), 2)})
    return out


def calibration_gap(table_: list[dict[str, float]]) -> float:
    """Count-weighted mean |predicted - observed| across the bins, in points."""
    n = sum(row["n"] for row in table_)
    return float(sum(row["n"] * abs(row["predicted_pts"] - row["observed_pts"])
                     for row in table_) / max(n, 1))


def paired_bootstrap(y: np.ndarray, model: np.ndarray, table_: np.ndarray, *,
                     resamples: int = 1000, level: float = 0.90) -> dict[str, float]:
    """Interval on (table error - model error), resampling holdout rows in pairs.

    Positive means the model is better. Paired, because both predictions are
    made on the same files: an unpaired comparison would charge the model for the
    noise in which defaults happen to be in the holdout.
    """
    rng = np.random.default_rng(seed())
    y, model, table_ = (np.asarray(a, dtype=float) for a in (y, model, table_))
    abs_m, abs_t = np.abs(model - y), np.abs(table_ - y)
    sq_m, sq_t = (model - y) ** 2, (table_ - y) ** 2
    idx = rng.integers(0, len(y), size=(resamples, len(y)))
    mae_gain = (abs_t[idx].mean(axis=1) - abs_m[idx].mean(axis=1)) * POINTS
    rmse_gain = (np.sqrt(sq_t[idx].mean(axis=1)) - np.sqrt(sq_m[idx].mean(axis=1))) * POINTS
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {
        "mae_gain_low_pts": float(np.percentile(mae_gain, lo)),
        "mae_gain_high_pts": float(np.percentile(mae_gain, hi)),
        "rmse_gain_low_pts": float(np.percentile(rmse_gain, lo)),
        "rmse_gain_high_pts": float(np.percentile(rmse_gain, hi)),
        "mae_gain_prob_positive": float((mae_gain > 0).mean()),
        "rmse_gain_prob_positive": float((rmse_gain > 0).mean()),
    }


def global_importance(model, X: pd.DataFrame, *, scale: float = 1.0, top: int | None = None,
                      sample: int = 2000) -> dict[str, Any]:
    """Mean |attribution| per contract feature, in the model's output units x ``scale``.

    TreeSHAP when shap is installed and the learner is a tree ensemble; the exact
    linear decomposition for the portfolio's linear families; permutation
    importance otherwise. The method that actually ran is returned with the
    numbers, because the three are not the same quantity. Shared with the uplift
    model's surrogate.
    """
    X = X.iloc[:sample]
    values, method = None, None
    inner = base.base_estimator(model)
    if hasattr(inner, "exact_contributions"):
        values, method = np.asarray(inner.exact_contributions(X)[0]), "exact linear contributions"
    if values is None:
        try:
            import shap

            raw = shap.TreeExplainer(inner).shap_values(X)
            raw = raw[1] if isinstance(raw, list) else np.asarray(raw)
            if raw.ndim == 3:
                raw = raw[..., -1]
            values, method = raw, "TreeSHAP (mean |SHAP|)"
        except Exception as exc:                                    # noqa: BLE001 - fall back, say so
            log.info("TreeSHAP unavailable for %s (%s); using permutation importance",
                     type(inner).__name__, exc)
    if values is not None:
        mean_abs = np.abs(values).mean(axis=0) * scale
    else:
        from sklearn.inspection import permutation_importance

        predict = getattr(model, "predict")
        baseline = predict(X)
        result = permutation_importance(model, X, baseline, n_repeats=3, random_state=seed(),
                                        scoring="neg_mean_absolute_error")
        mean_abs, method = result.importances_mean * scale, "permutation importance (MAE increase)"
    order = np.argsort(-mean_abs)
    if top:
        order = order[:top]
    return {"method": method, "features": [
        {"feature": X.columns[i], "label": reason_label(X.columns[i])["label"],
         "mean_abs": round(float(mean_abs[i]), 4)} for i in order]}


# ------------------------------------------------------------------- train


def train(df: pd.DataFrame | None = None, save: bool = True) -> base.ModelCard:
    cfg = model_config()[NAME]
    train_df, test_df = split(table(df))
    if len(train_df) < int(cfg.get("min_rows", 200)):
        raise ValueError(f"only {len(train_df)} defaulted files with an observed LGD in the "
                         f"training window; refusing to fit below {cfg.get('min_rows', 200)}")

    feature_spec = spec(NAME)
    transformer, ((X_train, y_train), (X_test, y_test)) = base.design(
        feature_spec, features, TARGETS[NAME], train_df, test_df)
    low, high = (float(v) for v in cfg.get("clip", [0.0, 0.95]))

    with run(f"{NAME}-train", {"learner": cfg["learner"]}) as active:
        estimator, learner_name = base.learner(NAME, TASK, feature_spec, fallback=_fallback)
        estimator.fit(X_train, y_train)
        pred = np.clip(estimator.predict(X_test), low, high)
        table_pred = policy_table(test_df)

        metrics = accuracy(y_test, pred)
        metrics.update(accuracy(y_test, table_pred, prefix="policy_"))
        metrics["mae_gain_pts"] = metrics["policy_mae_pts"] - metrics["mae_pts"]
        metrics["rmse_gain_pts"] = metrics["policy_rmse_pts"] - metrics["rmse_pts"]
        metrics.update(paired_bootstrap(y_test, pred, table_pred,
                                        resamples=int(cfg.get("bootstrap_resamples", 1000))))
        metrics["beats_policy_mae"] = float(metrics["mae_gain_pts"] > 0)
        metrics["beats_policy_rmse"] = float(metrics["rmse_gain_pts"] > 0)

        bins = int(cfg.get("calibration_bins", 10))
        model_calibration = calibration(y_test, pred, bins)
        table_calibration = calibration(y_test, table_pred, bins)
        metrics["calibration_gap_pts"] = calibration_gap(model_calibration)
        metrics["policy_calibration_gap_pts"] = calibration_gap(table_calibration)
        metrics["observed_mean_pts"] = float(y_test.mean() * POINTS)
        metrics["low_severity_share_pct"] = float((y_test < 0.10).mean() * 100)
        floor = float(model_config()["risk"].get("lgd_floor", 0.0))
        metrics["below_floor_pct"] = float((pred < floor).mean() * 100)
        log_metrics(active, metrics)

    importance = global_importance(
        estimator, X_test, scale=POINTS,
        top=int((model_config().get("portfolio") or {}).get("top_weights", 5)))

    dataset = provenance.identity(
        train_df, source="loader.build_training_table",
        slice_of="chronological train split of disbursed loans that defaulted with an observed "
                 "realized_lgd")

    card = base.ModelCard(
        name=NAME, version=model_config()["registry"]["version"], learner=learner_name,
        target=TARGETS[NAME], n_train=len(X_train), n_test=len(X_test),
        features=feature_spec.columns, metrics=metrics,
        training_window={"from": str(train_df["application_date"].min()),
                         "to": str(train_df["application_date"].max())},
        limitations=_limitations(metrics, len(X_train), len(X_test), floor),
        dataset=dataset,
        # A regressor: group fairness as the card measures it (calibration of a
        # probability by group) is not defined. None says "not measured".
        fairness=None,
        selection=base.selection_record(NAME, learner_name),
    ).finalize()

    log.info("lgd MAE %.2f pts (table %.2f) | RMSE %.2f (table %.2f) | bias %+.2f (table %+.2f)",
             metrics["mae_pts"], metrics["policy_mae_pts"], metrics["rmse_pts"],
             metrics["policy_rmse_pts"], metrics["bias_pts"], metrics["policy_bias_pts"])
    if save:
        base.save(NAME, {
            "model": estimator, "transformer": transformer, "clip": [low, high],
            "evaluation": {"calibration": {"model": model_calibration,
                                           "policy_table": table_calibration},
                           "importance": importance,
                           "holdout_rows": int(len(X_test))},
        }, card)
    return card


def _limitations(m: dict[str, float], n_train: int, n_test: int, floor: float) -> list[str]:
    def verdict(metric: str, label: str) -> str:
        better = m[f"{metric}_gain_pts"] > 0
        lo, hi = m[f"{metric}_gain_low_pts"], m[f"{metric}_gain_high_pts"]
        settled = lo > 0 or hi < 0
        return (f"{label} {m[f'{metric}_pts']:.1f} vs {m[f'policy_{metric}_pts']:.1f} pts "
                f"({'better' if better else 'worse'}; 90% interval on the gain {lo:+.1f} to "
                f"{hi:+.1f}{'' if settled else ', not distinguishable from no difference'})")

    return [
        f"Thin data, as LGD data always is: fitted on {n_train} defaulted files and judged on "
        f"{n_test}. Every number on this card carries a wide interval.",
        "Against the policy table on the same holdout rows: " + verdict("mae", "MAE") + "; "
        + verdict("rmse", "RMSE") + f"; bias {m['bias_pts']:+.1f} vs {m['policy_bias_pts']:+.1f} "
        "pts.",
        f"Fitted for the mean, which is what expected loss needs. Observed LGD is bimodal - "
        f"{m['low_severity_share_pct']:.0f}% of holdout defaults lost under 10% - so MAE, which "
        "rewards the median, can favour a flat table near the uncured mode; RMSE and bias are "
        "the comparison that matters for pricing.",
        f"risk.lgd_floor ({floor:.0%}) is applied by the risk model on top of this output; "
        f"{m['below_floor_pct']:.0f}% of holdout predictions sit below it. The metrics above are "
        "the model's own, before the floor.",
        "No downturn LGD: the training window contains no stress period, so this is a "
        "through-the-cycle-of-the-sample estimate, not a downturn one.",
        "Recoveries are taken as reported in realized_lgd: no workout-time discounting and no "
        "cost-of-recovery adjustment is modelled.",
        "Location enters as city, which nests inside state; a state's enforcement speed is "
        "learned only through its cities. `state` has no authored reason label, so it is not "
        "a feature.",
        "A change in collections strategy or in a court's pace changes LGD without changing any "
        "feature: monitor realised severity, not only input drift.",
    ]


def main() -> int:
    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
