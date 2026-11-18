#!/usr/bin/env python3
"""Stage-1 quote band, the simple version: one flat dataset, one short training run.

    python scripts/stage1_simple.py                    # build the dataset, train, evaluate
    python scripts/stage1_simple.py --dataset-only     # only write the CSVs
    python scripts/stage1_simple.py --save stage1_simple.joblib

1. Dataset. One row per booked loan in data/raw/home_loan_data.csv, keeping only
   what the branch knows at the first enquiry (config/feature_config.yaml, stage1)
   and the rate the loan was finally sanctioned at - the number the quote band has
   to contain. Oldest 80% by application date to train, newest 20% to test, as the
   main pipeline splits. Written to data/processed/stage1_simple/.
2. Model. Three gradient-boosted quantile regressors - the 10th, 50th and 90th
   percentile of the final rate - with the learner and settings the live Stage-1
   model uses (config/model_config.yaml, stage1).
3. Quote. The 10-90 interval is the band. Policy then holds its width between
   stage1.min_band_width_bps and max_band_width_bps around its midpoint, rounds it
   to round_to_bps and keeps it inside the 13-17% quote envelope.
4. Report. How often the quoted band contained the final rate on the test loans,
   how wide it was, and one worked example.

Reads the same config files as the engine but imports nothing from it, and never
writes to models/: the live champion is untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
TARGET = "sanctioned_rate"
DATE = "application_date"
SEED = 42


def config(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 1. dataset


def build_dataset(raw_path: Path, out_dir: Path, test_share: float):
    stage1 = config("feature_config.yaml")["stage1"]
    numeric, categorical = list(stage1["numeric"]), list(stage1["categorical"])

    raw = pd.read_csv(raw_path, parse_dates=[DATE])
    data = raw[["loan_id", DATE, *numeric, *categorical, TARGET]].sort_values(DATE).reset_index(drop=True)
    cut = int(len(data) * (1 - test_share))
    train, test = data.iloc[:cut], data.iloc[cut:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    (out_dir / "description.md").write_text(_describe(train, test, numeric, categorical), encoding="utf-8")
    return train, test, numeric, categorical


def _describe(train: pd.DataFrame, test: pd.DataFrame, numeric: list, categorical: list) -> str:
    pol = config("pricing_policy.yaml")
    return f"""# Stage-1 quote band - simple training dataset

Task: from what a branch knows at the first enquiry, predict the range the home
loan's final interest rate will fall in. Quantile regression on `{TARGET}`
(10th, 50th, 90th percentile); the 10-90 interval is the quote band.

- Source: data/raw/home_loan_data.csv (synthetic), policy {pol['policy_version']}
- Rates: annual %, inside {pol['reference']['card_rate_floor']:.2f}-{pol['reference']['card_rate_ceiling']:.2f}%
- train.csv: {len(train):,} loans, {train[DATE].min():%Y-%m-%d} to {train[DATE].max():%Y-%m-%d}
- test.csv: {len(test):,} loans, {test[DATE].min():%Y-%m-%d} to {test[DATE].max():%Y-%m-%d} (later dates: an out-of-time test)
- Target: `{TARGET}` - the rate the loan was finally sanctioned at
- Inputs, numeric: {', '.join(numeric)}
- Inputs, categorical: {', '.join(categorical)}
- `bureau_score` is -1 for new-to-credit applicants (no bureau file)
- `loan_id` and `{DATE}` identify and order rows; they are not inputs
- Not included on purpose: the risk grade, verified income and anything else
  learned after documents are collected - unknown at the first quote
"""


# --------------------------------------------------------------------- 2. model


def train_models(train: pd.DataFrame, numeric: list, categorical: list) -> dict:
    cfg = config("model_config.yaml")["stage1"]
    models = {}
    for q in cfg["quantiles"]:
        prep = ColumnTransformer([("num", "passthrough", numeric),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
        model = make_pipeline(prep, GradientBoostingRegressor(loss="quantile", alpha=q, random_state=SEED,
                                                              **cfg["params"]))
        model.fit(train[numeric + categorical], train[TARGET])
        models[q] = model
        print(f"  trained the {q:.0%} quantile model")
    return models


# --------------------------------------------------------------------- 3. quote


def quote(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    pol = config("pricing_policy.yaml")
    s1 = pol["stage1"]
    q_low, q_mid, q_high = sorted(models)
    low, mid, high = (models[q].predict(rows) for q in (q_low, q_mid, q_high))
    low, high = np.minimum(low, high), np.maximum(low, high)        # independent fits can cross
    raw_low, raw_high = low.copy(), high.copy()

    centre = (low + high) / 2
    half = np.clip((high - low) / 2, s1["min_band_width_bps"] / 200, s1["max_band_width_bps"] / 200)
    step = s1["round_to_bps"] / 100
    floor = s1.get("quote_band_floor", pol["reference"]["card_rate_floor"])
    ceiling = s1.get("quote_band_ceiling", pol["reference"]["card_rate_ceiling"])
    low = np.clip(np.round((centre - half) / step) * step, floor, ceiling)
    high = np.clip(np.round((centre + half) / step) * step, floor, ceiling)

    return pd.DataFrame({"raw_min": raw_low, "raw_max": raw_high,
                         "min_rate": low.round(2), "max_rate": high.round(2),
                         "indicative_rate": np.clip(mid, low, high).round(2)}, index=rows.index)


# -------------------------------------------------------------------- 4. report


def report(test: pd.DataFrame, quoted: pd.DataFrame) -> None:
    actual = test[TARGET].to_numpy()
    raw_in = (actual >= quoted["raw_min"]) & (actual <= quoted["raw_max"])
    quoted_in = (actual >= quoted["min_rate"]) & (actual <= quoted["max_rate"])
    cap = config("pricing_policy.yaml")["stage1"]["max_band_width_bps"]
    print(f"\nTest loans: {len(test):,} (applied {test[DATE].min():%Y-%m-%d} to {test[DATE].max():%Y-%m-%d})")
    print(f"  model's 10-90 band   contains the final rate {raw_in.mean():6.1%}   "
          f"mean width {(quoted['raw_max'] - quoted['raw_min']).mean() * 100:5.0f} bps")
    print(f"  quoted band (cap {cap}) contains the final rate {quoted_in.mean():6.1%}   "
          f"mean width {(quoted['max_rate'] - quoted['min_rate']).mean() * 100:5.0f} bps")
    print(f"  indicative rate: median miss {np.median(np.abs(quoted['indicative_rate'] - actual)) * 100:.0f} bps")
    print("  promise: an 80% band should contain the final rate about 80% of the time")

    i = test.index[0]
    row, q = test.loc[i], quoted.loc[i]
    print(f"\nExample - {row['loan_id']}: {row['segment']}, {row['city']}, "
          f"Rs {row['requested_loan_amount']:,.0f} over {row['requested_tenure_years']} years, "
          f"bureau {row['bureau_score']}")
    print(f"  quoted {q['min_rate']:.2f}-{q['max_rate']:.2f}% (indicative {q['indicative_rate']:.2f}%); "
          f"final rate {row[TARGET]:.2f}% - {'inside' if q['min_rate'] <= row[TARGET] <= q['max_rate'] else 'OUTSIDE'} the band")


def main(argv: list[str] | None = None) -> int:
    data_cfg = config("model_config.yaml")["data"]
    parser = argparse.ArgumentParser(description="Stage-1 quote band: simple dataset and training")
    parser.add_argument("--raw", type=Path, default=ROOT / data_cfg["raw_path"])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "stage1_simple")
    parser.add_argument("--dataset-only", action="store_true", help="write the CSVs and stop")
    parser.add_argument("--save", type=Path, help="also save the three models to this .joblib file")
    args = parser.parse_args(argv)

    train, test, numeric, categorical = build_dataset(args.raw, args.out, float(data_cfg["test_size"]))
    print(f"Dataset -> {args.out}  (train {len(train):,} rows, test {len(test):,} rows, "
          f"{len(numeric) + len(categorical)} inputs + target {TARGET})")
    if args.dataset_only:
        return 0

    print("Training:")
    models = train_models(train, numeric, categorical)
    report(test, quote(models, test[numeric + categorical]))

    if args.save:
        joblib.dump({"models": models, "numeric": numeric, "categorical": categorical,
                     "policy_version": config("pricing_policy.yaml")["policy_version"]}, args.save)
        print(f"\nSaved -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
