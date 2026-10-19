"""The shared kernel: pure primitives every agent may depend on.

Measuring the dependencies between the four reusable agents turned up exactly one
peer edge - and every cross-package edge in the system was a *single small pure
function* living in whichever package happened to define it first:

    optimization -> features.feature_pipeline.numeric_column
    features     -> learning.drift_detection.psi
    automl       -> models.fairness

None of those functions belongs to the package it sat in. Column coercion is not
a feature-engineering concept; PSI is not a drift-detection concept; group
fairness is not a modelling concept. They are primitives, and putting them here
is what makes "every agent is independent" true rather than aspirational.

The rule for this package: **it imports nothing from aavas except logging.** If
something here needs config, a store, a model or a connector, it is not a
primitive and does not belong. ``tests/test_agent_boundaries.py`` enforces that.
"""

from __future__ import annotations

from .columns import boolean_column, first_available, numeric_column
from .fairness import evaluate as fairness_evaluate
from .statistics import psi

__all__ = ["numeric_column", "boolean_column", "first_available", "psi",
           "fairness_evaluate"]
