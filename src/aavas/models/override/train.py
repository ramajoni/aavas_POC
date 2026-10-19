"""Override model: will a human move this price, and by how much?

This is not a policing tool. It has two jobs:

1. Tell the pricing agent, before it recommends, that a file sits in territory
   where humans routinely discount - a recommendation that is certain to be
   overridden is a recommendation that failed.
2. Give the learning loop a baseline expectation, so an *unusual* override stands
   out from the routine ones.
"""

from __future__ import annotations

import numpy as np

from ...config import model_config
from ...features import sales_features
from ...features.feature_pipeline import FeatureTransformer, spec
from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger, log_metrics, run
from .. import base, provenance

log = get_logger(__name__)
NAME = "override"


def train(df=None, save: bool = True) -> base.ModelCard:
    cfg = model_config()["override"]
    df = loader.load_training_table() if df is None else df
    df = sales_features.build(df, reference_rate_col="ai_recommended_rate")
    train_df, test_df = loader.train_test_split(df)

    feature_spec = spec(NAME)
    transformer = FeatureTransformer(feature_spec)
    X_train = transformer.fit_transform(train_df)
    X_test = transformer.transform(test_df)
    y_flag_train = train_df[TARGETS["override_flag"]].astype(int).to_numpy()
    y_flag_test = test_df[TARGETS["override_flag"]].astype(int).to_numpy()

    with run(f"{NAME}-train", {"learner": cfg["learner"]}) as active:
        classifier = base.calibrate(
            base.hgb_classifier(cfg["classifier_params"]), X_train, y_flag_train, "isotonic"
        )
        proba = classifier.predict_proba(X_test)[:, 1]
        metrics = base.classification_metrics(y_flag_test, proba)

        # Magnitude is only defined where an override actually happened.
        mask_train = y_flag_train == 1
        mask_test = y_flag_test == 1
        regressor = base.hgb_regressor(cfg["regressor_params"])
        regressor.fit(X_train[mask_train], train_df.loc[mask_train, TARGETS["override_magnitude"]].to_numpy(float))
        pred_bps = regressor.predict(X_test[mask_test])
        actual_bps = test_df.loc[mask_test, TARGETS["override_magnitude"]].to_numpy(float)
        metrics["magnitude_mae_bps"] = float(np.mean(np.abs(actual_bps - pred_bps)))
        metrics["magnitude_n"] = float(mask_test.sum())
        log_metrics(active, metrics)

    # Measured on the held-out split, on the same path that produces the
    # champion. Fairness that only runs inside an AutoML search never reaches
    # the model actually pricing decisions.
    fairness = base.fairness_report(test_df, y_flag_test, proba)

    # The frame the model was actually fitted on, not its parent: a hash of the
    # full table would be identical for two models fitted on different slices.
    dataset = provenance.identity(train_df, source="loader.build_training_table",
                                  slice_of='chronological train split of the full training table')

    card = base.ModelCard(
        name=NAME, version=model_config()["registry"]["version"], learner=cfg["learner"],
        target=f"{TARGETS['override_flag']} + {TARGETS['override_magnitude']}",
        n_train=len(X_train), n_test=len(X_test), features=feature_spec.columns, metrics=metrics,
        training_window={"from": str(train_df["application_date"].min()),
                         "to": str(train_df["application_date"].max())},
        limitations=[
            "Predicts override behaviour against the *historical* recommendation. Once this "
            "engine's recommendations replace it, the relationship shifts and the model must "
            "be refit on decisions made against the new baseline.",
            "Intended to anticipate negotiation, not to score individual salespeople.",
        ],
        dataset=dataset,
        fairness=fairness,
    ).finalize()

    log.info("override AUC %.3f | override rate %.1f%% | magnitude MAE %.1f bps",
             metrics["auc"], 100 * metrics["base_rate"], metrics["magnitude_mae_bps"])
    if save:
        base.save(NAME, {"classifier": classifier, "regressor": regressor, "transformer": transformer}, card)
    return card


def main() -> int:
    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
