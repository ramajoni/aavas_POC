"""Token use of Claude calls, so what each one costs can be seen and summed.

Every call that reaches Claude adds its response's usage here and logs the total.
``cache_read_input_tokens`` is the share served from the prompt cache (billed at
about a twentieth of the input price on Claude Opus 5.5); if it stays at zero
across calls that share a prefix, the cache is not working.
"""

from __future__ import annotations

from typing import Any

FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def tally(response: Any, into: dict[str, int] | None = None) -> dict[str, int]:
    """Add one response's token use to ``into`` (a fresh tally when None)."""
    total = dict.fromkeys(FIELDS, 0) if into is None else into
    usage = getattr(response, "usage", None)
    for name in FIELDS:
        total[name] += int(getattr(usage, name, 0) or 0)
    return total
