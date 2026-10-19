"""Conversion (sanction-to-disbursement) model.

This is the model that makes yield optimization possible: it turns a candidate
price into a probability that the customer actually takes the loan, so the
engine can see where an extra 10 bps stops paying for itself.

Two properties are enforced rather than hoped for:

* ``offered_rate`` and ``competitor_gap_bps`` carry monotone decreasing
  constraints, so the demand curve can never bend the wrong way on a thin slice
  and hand the optimizer a nonsense recommendation.
* probabilities are isotonically calibrated, because they are multiplied by
  rupees downstream.
"""

from __future__ import annotations

import numpy as np

from ...config import model_config
from ...features.feature_pipeline import FeatureTransformer, spec
from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger, log_metrics, run
from .. import base, provenance

log = get_logger(__name__)
NAME = "conversion"


def train(df=None, save: bool = True) -> base.ModelCard:
    cfg = model_config()["conversion"]
    df = loader.load_training_table() if df is None else df
    train_df, test_df = loader.train_test_split(df)

    feature_spec = spec(NAME)
    transformer = FeatureTransformer(feature_spec)
    X_train = transformer.fit_transform(train_df)
    X_test = transformer.transform(test_df)
    y_train = train_df[TARGETS["conversion"]].astype(int).to_numpy()
    y_test = test_df[TARGETS["conversion"]].astype(int).to_numpy()

    with run(f"{NAME}-train", {"learner": cfg["learner"]}) as active:
        estimator = base.hgb_classifier(cfg["params"], feature_spec)
        model = base.calibrate(estimator, X_train, y_train, cfg.get("calibration"))
        proba = model.predict_proba(X_test)[:, 1]
        metrics = base.classification_metrics(y_test, proba)
        metrics.update(_elasticity_checks(model, transformer, test_df))
        log_metrics(active, metrics)

    # Measured on the held-out split, on the same path that produces the
    # champion. Fairness that only runs inside an AutoML search never reaches
    # the model actually pricing decisions.
    fairness = base.fairness_report(test_df, y_test, proba)

    # The frame the model was actually fitted on, not its parent: a hash of the
    # full table would be identical for two models fitted on different slices.
    dataset = provenance.identity(train_df, source="loader.build_training_table",
                                  slice_of='chronological train split of the full training table')

    card = base.ModelCard(
        name=NAME, version=model_config()["registry"]["version"], learner=cfg["learner"],
        target=TARGETS["conversion"], n_train=len(X_train), n_test=len(X_test),
        features=feature_spec.columns, metrics=metrics,
        training_window={"from": str(train_df["application_date"].min()),
                         "to": str(train_df["application_date"].max())},
        limitations=[
            "Observational data: the historical rate was not randomised, so the fitted price "
            "response is a correlation the monotone constraint makes safe to use, not a causal "
            "effect. A price experiment or holdout is needed before trusting the slope exactly.",
            "Conditional on sanction; pre-sanction drop-off is out of scope for this model.",
        ],
        dataset=dataset,
        fairness=fairness,
    ).finalize()

    log.info("conversion AUC %.3f | calibration gap %.4f | mean dS2D per +25bps %.1f pts",
             metrics["auc"], metrics["calibration_gap"], metrics["s2d_drop_per_25bps_pts"])

    # Record where the model actually has data. A monotone gradient-boosted model
    # is piecewise constant outside its training range: ask it about a rate no
    # customer was ever offered and it confidently returns the last value it saw.
    # The optimizer needs to know where that edge is.
    support = {
        "offered_rate": [float(np.percentile(train_df["offered_rate"], 1)),
                         float(np.percentile(train_df["offered_rate"], 99))],
        "competitor_gap_bps": [float(np.percentile(train_df["competitor_gap_bps"], 1)),
                               float(np.percentile(train_df["competitor_gap_bps"], 99))],
    }
    card.limitations.append(
        f"Price response is only supported between {support['offered_rate'][0]:.2f}% and "
        f"{support['offered_rate'][1]:.2f}% (and a competitor gap of "
        f"{support['competitor_gap_bps'][0]:.0f} to {support['competitor_gap_bps'][1]:.0f} bps). "
        "Predictions outside that range are flat extrapolation, not evidence."
    )

    if save:
        base.save(NAME, {"model": model, "transformer": transformer, "support": support}, card)
    return card


def _elasticity_checks(model, transformer, test_df) -> dict[str, float]:
    """Does the fitted demand curve behave, and does it recover the known
    heterogeneity in price sensitivity?"""
    sample = test_df.head(2000).copy()
    base_X = transformer.transform(sample)
    p0 = model.predict_proba(base_X)[:, 1]

    up = sample.copy()
    up["offered_rate"] = up["offered_rate"] + 0.25
    up["competitor_gap_bps"] = up["competitor_gap_bps"] + 25
    p1 = model.predict_proba(transformer.transform(up))[:, 1]

    drop = (p0 - p1) * 100
    out = {
        "s2d_drop_per_25bps_pts": float(drop.mean()),
        "monotonicity_violations_pct": float((p1 > p0 + 1e-9).mean() * 100),
    }
    if "price_elasticity" in sample.columns:
        # Do customers the generator made price-sensitive actually show a steeper
        # fitted response? This is only checkable because the data is synthetic.
        corr = np.corrcoef(drop, sample["price_elasticity"].to_numpy())[0, 1]
        out["elasticity_rank_corr"] = float(corr)
    return out


def main() -> int:
    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
