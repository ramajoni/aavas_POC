"""Stage-2 training: the post-assessment price anchor.

This model answers "given everything we now know, what price does a file like
this actually clear at?". It is explicitly *not* the final recommendation - it
centres the scenario band that the yield optimizer then searches. Treating its
output as the answer would reintroduce the exact problem the business statement
describes: a price chosen without reference to conversion or loan life.
"""

from __future__ import annotations

import numpy as np

from ...config import model_config
from ...features import stage2_features
from ...features.feature_pipeline import FeatureTransformer
from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger, log_metrics, run
from .. import base, provenance

log = get_logger(__name__)
NAME = "stage2"


def train(df=None, save: bool = True) -> base.ModelCard:
    cfg = model_config()["stage2"]
    df = loader.load_training_table() if df is None else df
    train_df, test_df = loader.train_test_split(df)

    spec = stage2_features.stage2_spec()
    transformer = FeatureTransformer(spec)
    X_train = transformer.fit_transform(stage2_features.build(train_df))
    X_test = transformer.transform(stage2_features.build(test_df))
    y_train = train_df[TARGETS["stage2"]].to_numpy(float)
    y_test = test_df[TARGETS["stage2"]].to_numpy(float)

    params = _tune(X_train, y_train) if cfg.get("tuning", {}).get("enabled") else None

    with run(f"{NAME}-train", {"learner": cfg["learner"]}) as active:
        model, learner_name = base.stage2_learner()
        if params:
            model.set_params(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = base.regression_metrics(y_test, pred)
        # Residual spread becomes the Stage-2 uncertainty band the optimizer searches.
        residual_std = float(np.std(y_train - model.predict(X_train)))
        metrics["residual_std_bps"] = residual_std * 100
        log_metrics(active, metrics)

    # The frame the model was actually fitted on, not its parent: a hash of the
    # full table would be identical for two models fitted on different slices.
    dataset = provenance.identity(train_df, source="loader.build_training_table",
                                  slice_of='chronological train split of the full training table')

    card = base.ModelCard(
        name=NAME, version=model_config()["registry"]["version"], learner=learner_name,
        target=TARGETS["stage2"], n_train=len(X_train), n_test=len(X_test),
        features=spec.columns, metrics=metrics,
        training_window={"from": str(train_df["application_date"].min()),
                         "to": str(train_df["application_date"].max())},
        limitations=[
            "Target is the historically sanctioned rate, so the model reproduces historical "
            "commercial behaviour including its biases. It is an anchor, never the final price.",
            "Files that were declined or lapsed before sanction are absent (selection bias).",
            "Sales-context features are included so their contribution can be measured and "
            "reported separately; they are not a justification for pricing a customer.",
        ],
        dataset=dataset,
    ).finalize()

    log.info("stage2 MAE %.1f bps | within 25bps %.1f%% | R2 %.3f",
             metrics["mae_bps"], metrics["within_25bps_pct"], metrics["r2"])

    if save:
        base.save(NAME, {"model": model, "transformer": transformer,
                         "residual_std": residual_std, "learner": learner_name}, card)
    return card


def _tune(X, y) -> dict | None:
    """Optuna search when installed and enabled; otherwise config params stand."""
    cfg = model_config()["stage2"]["tuning"]
    try:
        import optuna
    except ImportError:
        log.warning("tuning enabled but optuna is not installed; using config params")
        return None

    from sklearn.model_selection import cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
        }
        model, _ = base.stage2_learner()
        model.set_params(**params)
        return -cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=3).mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=False)
    log.info("optuna best params: %s", study.best_params)
    return study.best_params


def main() -> int:
    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
