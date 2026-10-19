"""Credit-risk model: probability of default, graded and turned into expected loss.

Pricing needs risk in money terms, not a score. This model supplies PD; LGD stays
a policy table (``model_config.risk.lgd_by_property_type``) because loss severity
is a collateral and recovery question, not something to fit on 20k rows.
"""

from __future__ import annotations

import numpy as np

from ...config import model_config
from ...features import stage2_features
from ...features.feature_pipeline import FeatureTransformer, spec
from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger, log_metrics, run
from .. import base, provenance

log = get_logger(__name__)
NAME = "risk"


def train(df=None, save: bool = True) -> base.ModelCard:
    cfg = model_config()["risk"]
    df = loader.load_training_table() if df is None else df
    # Default is only observable on loans that were actually disbursed.
    disbursed = df[df["disbursed"].astype(bool)].copy()
    train_df, test_df = loader.train_test_split(disbursed)

    feature_spec = spec(NAME)
    transformer = FeatureTransformer(feature_spec)
    X_train = transformer.fit_transform(stage2_features.build(train_df))
    X_test = transformer.transform(stage2_features.build(test_df))
    y_train = train_df[TARGETS["risk"]].astype(int).to_numpy()
    y_test = test_df[TARGETS["risk"]].astype(int).to_numpy()

    with run(f"{NAME}-train", {"learner": cfg["learner"]}) as active:
        estimator = base.hgb_classifier(cfg["params"])
        model = base.calibrate(estimator, X_train, y_train, "isotonic")
        proba = model.predict_proba(X_test)[:, 1]
        metrics = base.classification_metrics(y_test, proba)
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
        target=TARGETS["risk"], n_train=len(X_train), n_test=len(X_test),
        features=feature_spec.columns, metrics=metrics,
        training_window={"from": str(train_df["application_date"].min()),
                         "to": str(train_df["application_date"].max())},
        limitations=[
            "Survivorship: fitted on disbursed loans only, so it describes the risk of files "
            "that passed underwriting, not of all applicants.",
            "LGD is a policy table, not a fitted model.",
            "risk_grade is excluded from the features by construction - it is this "
            "model's own output, and feeding it back would be circular.",
            "Default flag is a lifetime-to-date view, annualised by dividing by "
            "risk.pd_horizon_years; no vintage or seasoning adjustment.",
        ],
        dataset=dataset,
        fairness=fairness,
    ).finalize()

    log.info("risk AUC %.3f | observed default rate %.2f%%", metrics["auc"], 100 * metrics["base_rate"])
    if save:
        base.save(NAME, {"model": model, "transformer": transformer,
                         "grade_cutoffs": cfg["grade_cutoffs"],
                         "pd_horizon_years": cfg["pd_horizon_years"]}, card)
    return card


def main() -> int:
    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
