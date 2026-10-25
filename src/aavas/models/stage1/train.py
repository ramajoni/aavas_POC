"""Stage-1 training: a quantile trio, not a point estimate.

The customer is told a band because a single rate quoted before verification is a
promise the file cannot keep. Low/high quantiles give the band; the median gives
the anchor the Stage-2 recommendation is later compared against.
"""

from __future__ import annotations

import numpy as np

from ...config import model_config, policy
from ...features import stage1_features
from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger, log_metrics, run
from .. import base, provenance

log = get_logger(__name__)
NAME = "stage1"
TASK = "quantile"


def rows(df):
    """Every file: the band is quoted to everyone who enquires."""
    return df


def features(frame):
    return stage1_features.build(frame)


def split(df):
    """The trainer's chronological train / holdout split."""
    return loader.train_test_split(rows(df))


def incumbent() -> dict:
    """The learner this trainer fitted before the portfolio search existed."""
    return {"family": base.fallback_learner(NAME), "params": dict(model_config()[NAME]["params"])}


def selection_data(df):
    """Fit / validation frames for the AutoML search, from this trainer's own
    split, row filter, builder and contract (see ``base.selection_data``)."""
    dev_df, holdout_df = split(df)
    return base.selection_data(
        NAME, TASK, dev_df, holdout_df, feature_spec=stage1_features.stage1_spec(),
        build=features, target=TARGETS[NAME], metric="pinball_bps",
        quantiles=list(model_config()[NAME]["quantiles"]), incumbent=incumbent())


def train(df=None, save: bool = True) -> base.ModelCard:
    cfg = model_config()[NAME]
    df = loader.load_training_table() if df is None else df
    train_df, test_df = split(df)

    spec = stage1_features.stage1_spec()
    transformer, ((X_train, y_train), (X_test, y_test)) = base.design(
        spec, features, TARGETS[NAME], train_df, test_df)

    quantiles = cfg["quantiles"]
    models = {}
    learner_name = base.fallback_learner(NAME)
    with run(f"{NAME}-train", {"learner": cfg["learner"], "quantiles": quantiles}) as active:
        for q in quantiles:
            # One estimator per quantile, all of the selected family and params.
            model, learner_name = base.learner(
                NAME, TASK, spec, alpha=q,
                fallback=lambda q=q: (base.quantile_regressor(q, cfg["params"]),
                                      base.fallback_learner(NAME)))
            model.fit(X_train, y_train)
            models[str(q)] = model
            log.info("stage1 q=%.2f %s fitted on %s rows", q, learner_name, len(X_train))

        low, mid, high = (models[str(q)].predict(X_test) for q in quantiles)
        metrics = base.interval_metrics(y_test, low, high)
        metrics.update({f"median_{k}": v for k, v in base.regression_metrics(y_test, mid).items()})
        # Pinball loss is the honest scoring rule for a quantile model.
        for q in quantiles:
            pred = models[str(q)].predict(X_test)
            diff = y_test - pred
            metrics[f"pinball_q{int(q * 100)}"] = float(np.mean(np.maximum(q * diff, (q - 1) * diff)) * 100)
        log_metrics(active, metrics)

    # The frame the model was actually fitted on, not its parent: a hash of the
    # full table would be identical for two models fitted on different slices.
    dataset = provenance.identity(train_df, source="loader.build_training_table",
                                  slice_of='chronological train split of the full training table')

    card = base.ModelCard(
        name=NAME, version=model_config()["registry"]["version"],
        learner=learner_name, target=TARGETS["stage1"],
        n_train=len(X_train), n_test=len(X_test), features=spec.columns, metrics=metrics,
        training_window={
            "from": str(train_df["application_date"].min()),
            "to": str(train_df["application_date"].max()),
        },
        limitations=[
            "Trained on sanctioned files only: applications that never reached sanction are "
            "not represented, so the band describes prices that cleared, not prices that were quoted.",
            "Pre-assessment features are self-declared; verification haircuts are unknown at this stage.",
        ],
        dataset=dataset,
        selection=base.selection_record(NAME, learner_name),
    ).finalize()

    log.info("stage1 coverage %.1f%% | mean width %.0f bps | median MAE %.1f bps",
             metrics["coverage_pct"], metrics["mean_width_bps"], metrics["median_mae_bps"])

    if save:
        base.save(NAME, {"models": models, "transformer": transformer,
                         "quantiles": quantiles, "policy_version": policy()["policy_version"]}, card)
    return card


def main() -> int:
    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
