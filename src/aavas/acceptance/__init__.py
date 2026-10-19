"""Loan Acceptance Confidence.

Four agents, each doing one job:

* ``SourceRegistry``   - configurable data sources; one, several or none
* ``ExtractionAgent``  - free text to typed signals (Claude, with a rule fallback)
* ``EvidenceAgent``    - how much the assessment should be trusted
* ``AcceptanceAgent``  - composes them, and asks the *calibrated conversion model*
                         for the probability

The division of labour is the point: the language model reads, the fitted model
scores. See docs/target-architecture-gap.md section 10.
"""

from .acceptance_agent import AcceptanceAgent
from .evidence_agent import EvidenceAgent
from .extraction_agent import ExtractionAgent, RuleExtractor
from .signals import AcceptanceSignals, Signal
from .sources import SourceDocument, SourceRegistry

__all__ = [
    "AcceptanceAgent", "EvidenceAgent", "ExtractionAgent", "RuleExtractor",
    "AcceptanceSignals", "Signal", "SourceRegistry", "SourceDocument",
]
