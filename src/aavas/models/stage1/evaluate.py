"""Stage-1 evaluation: coverage, width and the slices that matter commercially."""

from __future__ import annotations

import pandas as pd

from ...ingestion import loader
from ...ingestion.schemas import TARGETS
from ...logging_utils import get_logger
from .. import base
from .predict import Stage1PricingModel

log = get_logger(__name__)


def evaluate(df: pd.DataFrame | None = None, by: list[str] | None = None) -> pd.DataFrame:
    df = loader.load_training_table() if df is None else df
    _, test = loader.train_test_split(df)
    model = Stage1PricingModel.load()
    pred = model.predict_frame(test)
    y = test[TARGETS["stage1"]].to_numpy(float)

    overall = base.interval_metrics(y, pred["min_rate"].to_numpy(), pred["max_rate"].to_numpy())
    overall["mean_confidence"] = float(pred["confidence"].mean())
    rows = [{"slice": "overall", "n": len(test), **overall}]

    for column in (by or ["segment", "risk_grade", "channel"]):
        if column not in test.columns:
            continue
        for value, idx in test.groupby(column).groups.items():
            sub_pred, sub_y = pred.loc[idx], test.loc[idx, TARGETS["stage1"]].to_numpy(float)
            metrics = base.interval_metrics(sub_y, sub_pred["min_rate"].to_numpy(), sub_pred["max_rate"].to_numpy())
            rows.append({"slice": f"{column}={value}", "n": len(idx), **metrics})

    result = pd.DataFrame(rows)
    log.info("stage1 evaluation\n%s", result.round(2).to_string(index=False))
    return result


if __name__ == "__main__":
    evaluate()
