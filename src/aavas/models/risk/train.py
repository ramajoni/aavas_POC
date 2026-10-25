"""Credit-risk model: probability of default, graded and turned into expected loss.

Pricing needs risk in money terms, not a score. This model supplies PD. Loss
severity is a separate model (``models/lgd``, fitted on defaulted files only);
``risk.predict`` combines the two into expected loss, falls back to the policy
LGD table (``model_config.risk.lgd_by_property_type``) when the LGD artefact is
missing, and floors LGD at ``risk.lgd_floor`` either way.
"""

from __future__ import annotations

from ...config import model_config
from ...features import stage2_features
from ...features.feature_pipeline import spec
from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger, log_metrics, run
from .. import base, provenance

log = get_logger(__name__)
NAME = "risk"
TASK = "binary"


def rows(df):
    """Default is only observable on loans that were actually disbursed."""
    return df[df["disbursed"].astype(bool)].copy()


def features(frame):
    return stage2_features.build(frame)


def split(df):
    """The trainer's chronological train / holdout split, of disbursed loans."""
    return loader.train_test_split(rows(df))


def _fallback():
    return base.hgb_classifier(model_config()[NAME]["params"]), base.fallback_learner(NAME)


def incumbent() -> dict:
    return {"family": "hist_gradient_boosting", "params": dict(model_config()[NAME]["params"])}


def selection_data(df):
    """Fit / validation frames for the AutoML search, from this trainer's own
    split, row filter, builder and contract (see ``base.selection_data``)."""
    dev_df, holdout_df = split(df)
    return base.selection_data(
        NAME, TASK, dev_df, holdout_df, feature_spec=spec(NAME), build=features,
        target=TARGETS[NAME], metric="auc", incumbent=incumbent())


def train(df=None, save: bool = True) -> base.ModelCard:
    cfg = model_config()[NAME]
    df = loader.load_training_table() if df is None else df
    train_df, test_df = split(df)

    feature_spec = spec(NAME)
    transformer, ((X_train, y_train), (X_test, y_test)) = base.design(
        feature_spec, features, TARGETS[NAME], train_df, test_df, binary=True)

    with run(f"{NAME}-train", {"learner": cfg["learner"]}) as active:
        estimator, learner_name = base.learner(NAME, TASK, feature_spec, fallback=_fallback)
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
        name=NAME, version=model_config()["registry"]["version"], learner=learner_name,
        target=TARGETS["risk"], n_train=len(X_train), n_test=len(X_test),
        features=feature_spec.columns, metrics=metrics,
        training_window={"from": str(train_df["application_date"].min()),
                         "to": str(train_df["application_date"].max())},
        limitations=[
            "Survivorship: fitted on disbursed loans only, so it describes the risk of files "
            "that passed underwriting, not of all applicants.",
            "PD only: loss severity comes from the separate LGD model (or the policy LGD "
            "table when its artefact is missing), combined in risk.predict.",
            "risk_grade is excluded from the features by construction - it is this "
            "model's own output, and feeding it back would be circular.",
            "Default flag is a lifetime-to-date view, annualised by dividing by "
            "risk.pd_horizon_years; no vintage or seasoning adjustment.",
        ],
        dataset=dataset,
        fairness=fairness,
        selection=base.selection_record(NAME, learner_name),
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
