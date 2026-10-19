"""Dataset identity for a trained model.

A training window and a row count are provenance; they are not identity. Two
models fitted on differently filtered extracts of the same window carry identical
cards and are indistinguishable in an audit - which is exactly the question an
auditor asks when two models disagree.

The identity here is a content hash over the frame the model was actually fitted
on: its columns, its shape and its values. It is derived rather than declared, for
the same reason the agent manifest is - a dataset name someone types is a promise
kept on every run but one.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


def content_hash(frame: pd.DataFrame) -> str:
    """A stable hash over the frame's schema and values.

    ``hash_pandas_object`` is stable across processes for a given pandas major
    version, so the column names and dtypes are folded in as well: a frame whose
    values are unchanged but whose schema moved is a different dataset.
    """
    schema = "|".join(f"{c}:{frame[c].dtype}" for c in frame.columns)
    digest = hashlib.sha256(schema.encode())
    digest.update(str(frame.shape).encode())
    if len(frame):
        values = pd.util.hash_pandas_object(frame, index=False).to_numpy()
        digest.update(values.tobytes())
    return digest.hexdigest()


def identity(frame: pd.DataFrame, *, source: str = "", slice_of: str = "",
             date_column: str = "application_date") -> dict[str, Any]:
    """The dataset record that goes onto a model card."""
    window: dict[str, str] = {}
    if date_column in frame.columns and len(frame):
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        if dates.notna().any():
            window = {"from": str(dates.min().date()), "to": str(dates.max().date())}

    digest = content_hash(frame)
    return {
        "dataset_id": f"DS-{digest[:12]}",
        "content_hash": digest,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names_hash": hashlib.sha256("|".join(frame.columns).encode()).hexdigest()[:12],
        "window": window,
        "source": source,
        # What was filtered out of the parent frame to produce this one. The risk
        # model fits on disbursed loans only; a card that did not say so would
        # describe a model of all applicants.
        "slice_of": slice_of,
    }


def same_dataset(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not left or not right:
        return False
    return left.get("content_hash") == right.get("content_hash")
