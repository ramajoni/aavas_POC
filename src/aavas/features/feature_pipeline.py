"""The one place feature contracts are enforced.

Every model declares its features in ``config/feature_config.yaml``. A model
never assembles its own column list, so:

* a Stage-2-only field cannot drift into Stage-1,
* the columns used at training and at scoring time cannot diverge,
* monotone constraints travel with the contract rather than living in code.

Categoricals are ordinal-encoded (all learners here are tree-based, so ordinal
encoding is sufficient and keeps attributions readable per original feature).

A numeric value can also be a *code*: ``bureau_score = -1`` means "no bureau
file", not a score 300 points below the worst one. Such sentinels are declared in
``feature_config.yaml`` (``sentinels``), read as missing here, and their meaning
carried by the flag column the contract pairs them with. See ``sentinel_rules``.
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


def sentinel_rules() -> dict[str, dict]:
    """Declared sentinel codes, by column. Empty when none are configured."""
    return dict(feature_config().get("sentinels") or {})


def _is_code(series: pd.Series, rule: dict) -> pd.Series:
    """True where a numeric value is the sentinel code rather than a measurement."""
    values = pd.to_numeric(series, errors="coerce")
    hit = pd.Series(False, index=series.index)
    if "below" in rule:
        hit |= values < float(rule["below"])
    if "values" in rule:
        hit |= values.isin([float(v) for v in rule["values"]])
    return hit.fillna(False).astype(bool)


def is_sentinel(feature: str, value) -> bool:
    """Is this one value a sentinel code for ``feature``?"""
    rule = sentinel_rules().get(feature)
    if not rule or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(_is_code(pd.Series([value]), rule).iloc[0])
    except (TypeError, ValueError):
        return False


def display_value(feature: str, value):
    """A feature value as a person should read it.

    A new-to-credit file shows "No bureau score (new to credit)", never -1: the
    code is an artefact of the bureau feed, and on a reason code or a credit
    review it reads as the worst score on the book.
    """
    if is_sentinel(feature, value):
        return sentinel_rules()[feature].get("display", "Not available")
    return value


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
        # The sentinel rules in force when this transformer was fitted. Captured
        # rather than re-read at scoring time, so an artefact keeps replaying the
        # encoding it was trained on after the config moves on. An artefact
        # pickled before sentinels existed has no such attribute and keeps its
        # original behaviour - the -1 it was fitted on stays a -1.
        self.sentinels_: dict[str, dict] = {}
        self.fitted_ = False

    # ------------------------------------------------------------------ fitting
    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        self.sentinels_ = {col: dict(rule) for col, rule in sentinel_rules().items()
                           if col in self.spec.numeric}
        df = self._with_flags(df)
        missing = [c for c in self.spec.columns if c not in df.columns]
        if missing:
            raise KeyError(f"[{self.spec.name}] missing feature columns: {missing}")
        for col in self.spec.categorical:
            values = df[col].astype(str).fillna(UNKNOWN)
            self.categories_[col] = sorted(values.unique().tolist())
        for col in self.spec.numeric:
            # Over measured values only: a median dragged towards -1 by the
            # new-to-credit files would impute a score nobody has.
            self.medians_[col] = float(self._numeric(df, col).median())
        self.fitted_ = True
        return self

    def _numeric(self, df: pd.DataFrame, col: str) -> pd.Series:
        series = pd.to_numeric(df[col], errors="coerce")
        rule = getattr(self, "sentinels_", {}).get(col)
        return series.mask(_is_code(series, rule)) if rule else series

    def _with_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive a sentinel's flag column when the contract needs it and the frame
        only carries the coded value. The two say the same thing, so a caller that
        sent a score of -1 without the flag is not refused for it."""
        derived = {}
        for col, rule in getattr(self, "sentinels_", {}).items():
            flag = rule.get("flag")
            if flag and flag in self.spec.columns and flag not in df.columns and col in df.columns:
                derived[flag] = _is_code(df[col], rule)
        return df.assign(**derived) if derived else df

    # ---------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError(f"[{self.spec.name}] transformer used before fit")
        df = self._with_flags(df)
        missing = [c for c in self.spec.columns if c not in df.columns]
        if missing:
            raise KeyError(f"[{self.spec.name}] missing feature columns at scoring time: {missing}")

        out = pd.DataFrame(index=df.index)
        for col in self.spec.numeric:
            series = self._numeric(df, col)
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
           "sentinel_rules", "is_sentinel", "display_value",
           "numeric_column", "boolean_column", "first_available"]
