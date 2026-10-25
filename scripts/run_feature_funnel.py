#!/usr/bin/env python3
"""Measure the feature-engineering funnel for each model contract.

    python scripts/run_feature_funnel.py                   # stage2, risk, conversion, stage1
    python scripts/run_feature_funnel.py --models stage2   # one
    python scripts/run_feature_funnel.py --models all      # every contract in feature_config
    python scripts/run_feature_funnel.py --sample 10000    # a larger correlation sample

Writes ``reports/features/funnel_<model>.json`` and prints the funnel and the
"rejected: strong, stable and unexplainable" table as the walkthrough slide reads
them. ``GET /features/funnel`` serves these files while they are fresh.

Nothing here admits a feature. The approved stage counts reason labels people have
written in ``config/feature_config.yaml``; the script reads them and never writes
one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aavas.api.features import feature_approvals  # noqa: E402
from aavas.features import funnel  # noqa: E402
from aavas.ingestion import loader  # noqa: E402
from aavas.logging_utils import get_logger  # noqa: E402

log = get_logger("run_feature_funnel")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", default=",".join(funnel.SLIDE_MODELS),
                        help="comma-separated contract names, or 'all' (default: %(default)s)")
    parser.add_argument("--sample", type=int, default=funnel.DEFAULT_SAMPLE,
                        help="rows sampled for the redundancy correlations (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None,
                        help="sampling seed (default: the seed in model_config.yaml)")
    parser.add_argument("--top", type=int, default=12,
                        help="rows of the rejected table to print (default: %(default)s)")
    parser.add_argument("--no-save", action="store_true", help="print only; write nothing")
    args = parser.parse_args(argv)

    models = funnel.contracts() if args.models == "all" else \
        [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in funnel.contracts()]
    if unknown:
        parser.error(f"no feature contract named {unknown}; expected one of {funnel.contracts()}")

    # One read of the raw table for every model; the loader is not asked to
    # rewrite data/processed, which other processes read.
    frame = loader.build_training_table(save=False)
    for model in models:
        # The same governance evidence the API reads, so a saved report is fresh
        # for the API exactly when nothing has been approved since.
        report = funnel.run(model, frame=frame, sample=args.sample, seed=args.seed,
                            approvals=feature_approvals(model))
        print(funnel.text_summary(report, top=args.top))
        if not args.no_save:
            path = funnel.save(report)
            shown = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
            print(f"  -> {shown}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
