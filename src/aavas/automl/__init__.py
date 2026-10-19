"""Agentic AutoML.

    algorithm selection -> HPO -> imbalance -> calibration -> ensemble
        -> fairness -> profit -> champion selection

The catalogue in ``registry`` is complete: every family is registered, including
the ones that cannot run on this data. An algorithm that reports
``no graph data in the schema`` has told you something an omission would not.
"""

from ..kernel import fairness
from . import (calibration, ensembles, hpo, imbalance, locality,
               registry, selection, tracking)
from .registry import ALGORITHMS, Algorithm, WoEScorecard, candidates, inventory, summary
from .search import AutoMLSearch, SearchConfig, SearchResult
from .selection import Candidate, Selection

__all__ = [
    "registry", "hpo", "imbalance", "calibration", "ensembles", "selection", "locality",
    "fairness", "tracking", "AutoMLSearch", "SearchConfig", "SearchResult",
    "Candidate", "Selection", "Algorithm", "ALGORITHMS", "WoEScorecard",
    "candidates", "inventory", "summary",
]
