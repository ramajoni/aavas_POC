"""SageMaker inference handler: serves the aavas model bundle trained by train_entry.py.

The training job packs this file, the aavas source and the config into the model
artefact's ``code/`` folder, so the endpoint runs exactly the code the models were
trained with, in the same image. Request (JSON):

    {"model": "stage2", "rows": [{...application fields...}], "rate": 14.5}

``model`` is one of stage1, stage2, conversion, prepayment, risk, lgd, override,
uplift, or ``health``. ``rate`` is used by conversion and prepayment (default: each
row's own sanctioned_rate); ``recommended_rate`` by override; ``discount_bps`` by uplift.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for extra in (HERE / "lib", HERE / "src"):
    if extra.exists() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
os.environ.setdefault("AAVAS_STORE_URL", "sqlite:////tmp/aavas_governance.db")
os.environ.setdefault("AAVAS_TRACKING_ENABLED", "false")


def model_fn(model_dir):
    os.environ["AAVAS_MODEL_DIR"] = str(Path(model_dir) / "models")
    from aavas.governance import model_registry
    return model_registry.load_bundle(refresh=True)


def input_fn(body, content_type="application/json"):
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    return json.loads(body)


def predict_fn(payload, bundle):
    return predict(bundle, payload)


def output_fn(result, accept="application/json"):
    return json.dumps(result), "application/json"


def predict(bundle, payload: dict) -> dict:
    import numpy as np
    import pandas as pd

    name = payload.get("model")
    if name == "health":
        loaded = [n for n in ("stage1", "stage2", "conversion", "prepayment", "risk", "lgd", "override", "uplift")
                  if getattr(bundle, n, None) is not None]
        return {"model": "health", "loaded": loaded, "missing": list(bundle.missing or [])}
    model = getattr(bundle, name or "", None)
    if model is None:
        raise ValueError(f"model '{name}' is not in this bundle")
    rows = pd.DataFrame(payload["rows"])
    if name == "stage1":
        out = model.predict_frame(rows).to_dict(orient="records")
    elif name == "stage2":
        out = list(model.predict_frame(rows))
    elif name in ("conversion", "prepayment"):
        rate = payload.get("rate")
        rate = rows["sanctioned_rate"].to_numpy(float) if rate is None else float(rate)
        out = list(model.predict_at_rate(rows, rate))
    elif name == "risk":
        out = model.assess(rows).to_dict(orient="records")
    elif name == "lgd":
        out = list(model.predict(rows))
    elif name == "override":
        out = model.predict(rows, payload.get("recommended_rate")).to_dict(orient="records")
    elif name == "uplift":
        out = [model.assess(rows.iloc[[i]], float(payload.get("discount_bps", 25)))
               for i in range(len(rows))]
    else:
        raise ValueError(f"no prediction method for '{name}'")
    return {"model": name, "predictions": _jsonable(out, np)}


def _jsonable(value, np):
    if isinstance(value, dict):
        return {str(k): _jsonable(v, np) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, np) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v, np) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
