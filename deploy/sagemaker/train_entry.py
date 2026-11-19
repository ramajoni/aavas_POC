"""SageMaker training entry point: train the aavas model suite inside the job.

Runs in the SageMaker scikit-learn image (script mode). The agentic workflow's
SageMaker engine (src/aavas/agentic/engines/sagemaker.py) packs the aavas source,
config and this file into ``sourcedir.tar.gz`` and passes the dataset as the
``train`` channel. The model artefact it leaves in /opt/ml/model holds:

    models/        the trained artefacts and model cards (as scripts/train_all.py writes them)
    code/          inference.py, the aavas source and config: the endpoint serves this
    reference.json predictions on sample rows, recorded here, for the endpoint parity check
    environment.json  Python and library versions the models were trained with
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import runpy
import shutil
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
MODEL = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
TRAIN = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))


def versions() -> dict:
    import joblib
    import numpy
    import pandas
    import scipy
    import sklearn
    import yaml
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__,
            "pyyaml": yaml.__version__}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="stage1,stage2,conversion,prepayment,lgd,risk,override,uplift")
    parser.add_argument("--role", default="challenger")
    parser.add_argument("--reference-rows", type=int, default=20)
    args, _ = parser.parse_known_args()
    models = [m for m in args.models.split(",") if m]

    env = versions()
    print("environment " + json.dumps(env), flush=True)
    sys.path.insert(0, str(CODE / "src"))
    raw = CODE / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("home_loan_data.csv", "sales_notes.csv"):
        if (TRAIN / name).exists():
            shutil.copy2(TRAIN / name, raw / name)
    os.environ.setdefault("AAVAS_STORE_URL", "sqlite:////tmp/aavas_governance.db")   # scratch registry in the job
    os.environ.update({"AAVAS_TRACKING_ENABLED": "false", "AAVAS_MODEL_DIR": str(MODEL / "models")})

    sys.argv = ["train_all.py", "--only", *models, "--role", args.role]
    try:
        runpy.run_path(str(CODE / "scripts" / "train_all.py"), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise

    # The serving code travels with the models: the endpoint runs what trained them.
    serve = MODEL / "code"
    serve.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CODE / "inference.py", serve / "inference.py")
    shutil.copytree(CODE / "src", serve / "src", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(CODE / "config", serve / "config", dirs_exist_ok=True)
    import yaml                                                  # installed for this job, not in the image
    lib = serve / "lib"
    lib.mkdir(exist_ok=True)
    yaml_dir = Path(yaml.__file__).parent
    shutil.copytree(yaml_dir, lib / yaml_dir.name, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for sibling in yaml_dir.parent.glob("_yaml*"):
        (shutil.copytree if sibling.is_dir() else shutil.copy2)(sibling, lib / sibling.name)

    # Reference predictions on sample rows: the endpoint must reproduce these exactly.
    sys.path.insert(0, str(serve))
    import inference
    from aavas.ingestion import loader
    table = loader.load_training_table()
    rows = json.loads(table.tail(args.reference_rows).to_json(orient="records", date_format="iso"))  # NaN -> null
    bundle = inference.model_fn(str(MODEL))
    reference = {"rows": rows, "predictions": {}}
    for name in models:
        try:
            reference["predictions"][name] = inference.predict(bundle, {"model": name, "rows": rows})["predictions"]
        except Exception as exc:                                    # noqa: BLE001 - recorded, not fatal
            reference["predictions"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    (MODEL / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    (MODEL / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    print(f"packed {len(models)} models, serving code and {len(rows)} reference rows into {MODEL}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
