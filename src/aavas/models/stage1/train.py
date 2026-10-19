"""Stage-1 training: a quantile trio, not a point estimate.

The customer is told a band because a single rate quoted before verification is a
promise the file cannot keep. Low/high quantiles give the band; the median gives
the anchor the Stage-2 recommendation is later compared against.
"""

from __future__ import annotations

import numpy as np

from ...config import model_config, policy
from ...features import stage1_features
from ...features.feature_pipeline import FeatureTransformer
from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger, log_metrics, run
from .. import base, provenance

log = get_logger(__name__)
NAME = "stage1"


def train(df=None, save: bool = True) -> base.ModelCard:
    cfg = model_config()["stage1"]
    df = loader.load_training_table() if df is None else df
    train_df, test_df = loader.train_test_split(df)

    spec = stage1_features.stage1_spec()
    transformer = FeatureTransformer(spec)
    X_train = transformer.fit_transform(stage1_features.build(train_df))
    X_test = transformer.transform(stage1_features.build(test_df))
    y_train = train_df[TARGETS["stage1"]].to_numpy(float)
    y_test = test_df[TARGETS["stage1"]].to_numpy(float)

    quantiles = cfg["quantiles"]
    models = {}
    with run(f"{NAME}-train", {"learner": cfg["learner"], "quantiles": quantiles}) as active:
        for q in quantiles:
            model = base.quantile_regressor(q, cfg["params"])
            model.fit(X_train, y_train)
            models[str(q)] = model
            log.info("stage1 q=%.2f fitted on %s rows", q, len(X_train))

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
        learner=cfg["learner"], target=TARGETS["stage1"],
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
