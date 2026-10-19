"""Column coercion. Pure, and deliberately dependency-free.

These lived in ``features.feature_pipeline``, which meant the pricing optimiser
had to import the feature-engineering package to read a number off a frame. The
functions were never about feature engineering; they are about reading a column
that may be missing, mistyped or absent without raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_column(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    """A column as floats, defaulting when it is absent, null or unparseable.

    ``DataFrame.get(col, default)`` returns the bare *scalar* default when the
    column is missing, so every chained pandas call after it raises
    ``AttributeError``. That only ever bites on partially populated
    applications - exactly the ones a live API receives and the training data
    never contains - so it is centralised here rather than rediscovered per call
    site.
    """
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def boolean_column(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(bool(default), index=df.index, dtype=bool)
    # infer_objects first: filling an object column then casting triggers pandas'
    # silent-downcasting deprecation.
    return df[column].fillna(default).infer_objects(copy=False).astype(bool)


def first_available(df: pd.DataFrame, columns: list[str], default: float) -> pd.Series:
    """First of ``columns`` that is present and populated, else ``default``."""
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for column in columns:
        if column in df.columns:
            result = result.fillna(pd.to_numeric(df[column], errors="coerce"))
    return result.fillna(default).astype(float)
