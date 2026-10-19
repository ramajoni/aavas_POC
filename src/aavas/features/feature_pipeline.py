"""The one place feature contracts are enforced.

Every model declares its features in ``config/feature_config.yaml``. A model
never assembles its own column list, so:

* a Stage-2-only field cannot drift into Stage-1,
* the columns used at training and at scoring time cannot diverge,
* monotone constraints travel with the contract rather than living in code.

Categoricals are ordinal-encoded (all learners here are tree-based, so ordinal
encoding is sufficient and keeps attributions readable per original feature).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import feature_config
# The column primitives moved to the kernel: the pricing optimiser needed them
# and had to import this package to get them. Re-exported here because the
# feature builders use them by these names.
from ..kernel.columns import boolean_column, first_available, numeric_column
from ..logging_utils import get_logger

log = get_logger(__name__)

UNKNOWN = "__unknown__"
MISSING_NUMERIC = np.nan


@dataclass
class FeatureSpec:
    name: str
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    monotone: dict[str, int] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)

    def monotone_vector(self) -> list[int] | None:
        """Constraint vector aligned to ``columns``; None when unconstrained."""
        if not self.monotone:
            return None
        return [int(self.monotone.get(c, 0)) for c in self.columns]


def spec(name: str) -> FeatureSpec:
    cfg = feature_config()
    if name not in cfg:
        raise KeyError(f"no feature contract named {name!r} in feature_config.yaml")
    block = cfg[name]
    return FeatureSpec(
        name=name,
        numeric=list(block.get("numeric", [])),
        categorical=list(block.get("categorical", [])),
        monotone=dict(block.get("monotone", {})),
    )


class FeatureTransformer:
    """Fit on training data, saved next to the model, replayed at scoring time."""

    def __init__(self, spec_: FeatureSpec):
        self.spec = spec_
        self.categories_: dict[str, list] = {}
        self.medians_: dict[str, float] = {}
        self.fitted_ = False

    # ------------------------------------------------------------------ fitting
    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        missing = [c for c in self.spec.columns if c not in df.columns]
        if missing:
            raise KeyError(f"[{self.spec.name}] missing feature columns: {missing}")
        for col in self.spec.categorical:
            values = df[col].astype(str).fillna(UNKNOWN)
            self.categories_[col] = sorted(values.unique().tolist())
        for col in self.spec.numeric:
            self.medians_[col] = float(pd.to_numeric(df[col], errors="coerce").median())
        self.fitted_ = True
        return self

    # ---------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError(f"[{self.spec.name}] transformer used before fit")
        missing = [c for c in self.spec.columns if c not in df.columns]
        if missing:
            raise KeyError(f"[{self.spec.name}] missing feature columns at scoring time: {missing}")

        out = pd.DataFrame(index=df.index)
        for col in self.spec.numeric:
            series = pd.to_numeric(df[col], errors="coerce")
            out[col] = series.fillna(self.medians_.get(col, 0.0)).astype(float)
        for col in self.spec.categorical:
            cats = self.categories_[col]
            lookup = {c: i for i, c in enumerate(cats)}
            out[col] = df[col].astype(str).map(lookup).fillna(-1).astype(float)
        return out[self.spec.columns]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ----------------------------------------------------------------- helpers
    def baseline_row(self) -> pd.Series:
        """Median/modal reference point - the 'average customer' the ablation
        explainer attributes against."""
        values = {c: self.medians_.get(c, 0.0) for c in self.spec.numeric}
        for col in self.spec.categorical:
            cats = self.categories_[col]
            values[col] = float(len(cats) // 2) if cats else -1.0
        return pd.Series(values)[self.spec.columns]

    def decode_category(self, column: str, code: float) -> str:
        cats = self.categories_.get(column, [])
        idx = int(code)
        return cats[idx] if 0 <= idx < len(cats) else UNKNOWN


def reason_label(feature: str) -> dict[str, str]:
    labels = feature_config().get("reason_code_labels", {})
    return labels.get(feature, {"label": feature.replace("_", " ").title(),
                                "up": f"Higher {feature.replace('_', ' ')}",
                                "down": f"Lower {feature.replace('_', ' ')}"})


__all__ = ["FeatureSpec", "FeatureTransformer", "spec", "reason_label",
           "numeric_column", "boolean_column", "first_available"]
