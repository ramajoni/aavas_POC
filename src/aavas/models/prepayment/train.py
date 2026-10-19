"""Balance-transfer / prepayment model.

A price that converts today but loses the customer in 14 months is not a good
price. This model supplies the expected loan life that turns a nominal spread
into a realized one.

Trained on disbursed files only - a loan that never existed cannot transfer out -
which is stated in the model card rather than hidden.
"""

from __future__ import annotations

from ...config import model_config
from ...features.feature_pipeline import FeatureTransformer, spec
from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger, log_metrics, run
from .. import base, provenance

log = get_logger(__name__)
NAME = "prepayment"


def train(df=None, save: bool = True) -> base.ModelCard:
    cfg = model_config()["prepayment"]
    df = loader.load_training_table() if df is None else df
    disbursed = df[df["disbursed"].astype(bool)].copy()
    train_df, test_df = loader.train_test_split(disbursed)

    feature_spec = spec(NAME)
    transformer = FeatureTransformer(feature_spec)
    X_train = transformer.fit_transform(train_df)
    X_test = transformer.transform(test_df)
    y_train = train_df[TARGETS["prepayment"]].astype(int).to_numpy()
    y_test = test_df[TARGETS["prepayment"]].astype(int).to_numpy()

    with run(f"{NAME}-train", {"learner": cfg["learner"]}) as active:
        estimator = base.hgb_classifier(cfg["params"], feature_spec)
        model = base.calibrate(estimator, X_train, y_train, cfg.get("calibration"))
        proba = model.predict_proba(X_test)[:, 1]
        metrics = base.classification_metrics(y_test, proba)
        metrics["horizon_months"] = float(cfg["horizon_months"])
        log_metrics(active, metrics)

    # Measured on the held-out split, on the same path that produces the
    # champion. Fairness that only runs inside an AutoML search never reaches
    # the model actually pricing decisions.
    fairness = base.fairness_report(test_df, y_test, proba)

    # The frame the model was actually fitted on, not its parent: a hash of the
    # full table would be identical for two models fitted on different slices.
    dataset = provenance.identity(train_df, source="loader.build_training_table",
                                  slice_of='chronological train split of disbursed loans only')

    card = base.ModelCard(
        name=NAME, version=model_config()["registry"]["version"], learner=cfg["learner"],
        target=TARGETS["prepayment"], n_train=len(X_train), n_test=len(X_test),
        features=feature_spec.columns, metrics=metrics,
        training_window={"from": str(train_df["application_date"].min()),
                         "to": str(train_df["application_date"].max())},
        limitations=[
            f"Predicts balance-transfer-out within {cfg['horizon_months']} months only; "
            "later-life transfers are not modelled.",
            "Disbursed files only. Recent cohorts are censored - they have not had the full "
            "window to transfer - so the base rate is understated for the newest months.",
        ],
        dataset=dataset,
        fairness=fairness,
    ).finalize()

    log.info("prepayment AUC %.3f | base rate %.1f%%", metrics["auc"], 100 * metrics["base_rate"])
    if save:
        base.save(NAME, {"model": model, "transformer": transformer,
                         "horizon_months": cfg["horizon_months"]}, card)
    return card


def main() -> int:
    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
