"""Stage-2 evaluation, including the Stage-1 -> Stage-2 improvement that
justifies collecting documents at all."""

from __future__ import annotations

import pandas as pd

from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger
from .. import base
from ..stage1.predict import Stage1PricingModel
from .predict import Stage2PricingModel

log = get_logger(__name__)


def evaluate(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = loader.load_training_table() if df is None else df
    _, test = loader.train_test_split(df)
    y = test[TARGETS["stage2"]].to_numpy(float)

    s2 = Stage2PricingModel.load()
    rows = [{"model": "stage2", "n": len(test), **base.regression_metrics(y, s2.predict_frame(test).to_numpy())}]

    try:
        s1 = Stage1PricingModel.load()
        s1_pred = s1.predict_frame(test)
        rows.append({"model": "stage1_indicative", "n": len(test),
                     **base.regression_metrics(y, s1_pred["indicative_rate"].to_numpy())})
    except FileNotFoundError:
        log.warning("stage1 artefact missing; skipping the stage comparison")

    result = pd.DataFrame(rows)
    if len(result) == 2:
        lift = result.loc[1, "mae_bps"] - result.loc[0, "mae_bps"]
        log.info("assessment is worth %.1f bps of pricing accuracy (stage1 MAE -> stage2 MAE)", lift)
    log.info("stage2 evaluation\n%s", result.round(2).to_string(index=False))
    return result


if __name__ == "__main__":
    evaluate()
