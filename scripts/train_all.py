#!/usr/bin/env python3
"""Train every model, register it, and print a comparison table.

    python scripts/train_all.py [--regenerate] [--rows 20000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aavas.config import model_config, versions  # noqa: E402
from aavas.governance import model_registry  # noqa: E402
from aavas.ingestion import loader  # noqa: E402
from aavas.logging_utils import get_logger  # noqa: E402
from aavas.models.conversion import train as conversion_train  # noqa: E402
from aavas.models import base as model_base  # noqa: E402
from aavas.models.lgd import train as lgd_train  # noqa: E402
from aavas.models.override import train as override_train  # noqa: E402
from aavas.models.prepayment import train as prepayment_train  # noqa: E402
from aavas.models.risk import train as risk_train  # noqa: E402
from aavas.models.stage1 import train as stage1_train  # noqa: E402
from aavas.models.stage2 import train as stage2_train  # noqa: E402
from aavas.models.uplift import train as uplift_train  # noqa: E402
from aavas.paths import ensure_dirs, resolve  # noqa: E402

log = get_logger("train_all")

# Order. LGD is not an input to the PD fit - RiskModel pairs the two when it is
# *loaded*, not trained - so it could go anywhere. It goes right before risk so a
# severity model that refuses to fit (too few defaults) is reported beside the
# PD it would have completed, and so the risk card registered next is the one
# that will be served with it. Uplift goes last: it consumes nothing trained
# here, and it is the negotiation step that follows the override model.
TRAINERS = [
    ("stage1", stage1_train), ("stage2", stage2_train), ("conversion", conversion_train),
    ("prepayment", prepayment_train), ("lgd", lgd_train), ("risk", risk_train),
    ("override", override_train), ("uplift", uplift_train),
]

HEADLINE = {
    "stage1": ["coverage_pct", "mean_width_bps", "median_mae_bps"],
    "stage2": ["mae_bps", "within_25bps_pct", "r2"],
    "conversion": ["auc", "calibration_gap", "s2d_drop_per_25bps_pts"],
    "prepayment": ["auc", "base_rate"],
    "risk": ["auc", "base_rate"],
    "override": ["auc", "magnitude_mae_bps"],
    "lgd": ["rmse_pts", "policy_rmse_pts", "mae_pts", "policy_mae_pts"],
    "uplift": ["qini", "ate_pts", "naive_difference_pts", "synthetic_truth_rank_corr"],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the full Aavas pricing model suite")
    parser.add_argument("--regenerate", action="store_true", help="regenerate the synthetic dataset first")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--only", nargs="*", help="train only these models")
    args = parser.parse_args(argv)

    ensure_dirs()
    raw_path = resolve(model_config()["data"]["raw_path"])
    if args.regenerate or not raw_path.exists():
        from aavas.synthetic.generator import main as generate

        log.info("generating the synthetic lifecycle dataset")
        generate(["-n", str(args.rows)] if args.rows else [])

    df = loader.build_training_table()
    log.info("training on %s applications (%s)", len(df), versions())

    rows = []
    for name, module in TRAINERS:
        if args.only and name not in args.only:
            continue
        log.info("--- training %s ---", name)
        card = module.train(df=df)
        model_registry.register(card, role="champion")
        headline = ", ".join(
            f"{k}={card.metrics[k]:.4g}" for k in HEADLINE[name] if k in card.metrics)
        rows.append({"model": name, "learner": card.learner,
                     "n_train": card.n_train, "headline_metrics": headline})

    table = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("TRAINED MODELS")
    print("=" * 100)
    for row in rows:
        print(f"  {row['model']:<11} {row['learner']:<34} n={row['n_train']:>6}  {row['headline_metrics']}")
    print("\nStage-1 coverage is the share of files whose final rate fell inside the quoted band.")
    print("Stage-2 mae_bps is the pricing anchor error. conversion s2d_drop_per_25bps_pts is the")
    print("average conversion cost of pricing 25 bps higher - the number the optimizer trades against.")
    print("lgd is judged against the policy table on the same holdout defaults (policy_*): RMSE is")
    print("the comparison that matters for expected loss; MAE favours the table on a bimodal target.")
    print("uplift qini is the IPW-weighted Qini (acceptance pts); ate_pts is the lift of the reference")
    print("discount, naive_difference_pts the unadjusted read it corrects; synthetic_truth_* is")
    print("evaluation against the generator's truth and exists only because the data is synthetic.")
    print(f"\nArtefacts: {model_base.registry_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
