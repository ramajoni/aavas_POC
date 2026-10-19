"""Group fairness, re-exported from the kernel.

The implementation moved to ``aavas.kernel.fairness``: it is a primitive that the
model trainers and the AutoML search both need, and neither package owns it.
Kept importable from here because the trainers and the model card use these names.
"""

from __future__ import annotations

from ..kernel.fairness import (AGE_BANDS, DEFAULT_GROUP_COLUMN, MIN_GROUP_SIZE, age_band,
                               by_group, evaluate, gaps, groups_for)

__all__ = ["evaluate", "by_group", "gaps", "age_band", "groups_for",
           "AGE_BANDS", "DEFAULT_GROUP_COLUMN", "MIN_GROUP_SIZE"]
