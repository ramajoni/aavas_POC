"""A step trace for the agentic pipeline.

The orchestrator has always reported one number - how long the whole decision
took. That is enough to know the API is fast and nothing at all about which
agent decided what. A reviewer asking "why is this file priced at 11.24?" has to
read a nested JSON payload and reconstruct the order of operations from it.

This records the order as it happens: which agent ran, in what position, how
long it took, and one factual line about what it produced. The line is composed
from the agent's own output, never invented - if an agent returns nothing worth
summarising, the step says so.

The trace is additive. It rides along on the decision payload, so it is
persisted with the decision and can be read back later by ``decision_id``;
nothing in the priced path reads it, and turning it off changes no number.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Step:
    """One agent's turn in the pipeline."""

    order: int
    agent: str
    action: str
    ms: float = 0.0
    summary: str = ""
    status: str = "ok"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {"order": self.order, "agent": self.agent, "action": self.action,
               "ms": round(self.ms, 2), "summary": self.summary, "status": self.status}
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class Trace:
    """The steps of one decision, in the order they actually ran."""

    steps: list[Step] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def step(self, agent: str, action: str) -> Iterator[Step]:
        """Time one agent's turn. The body sets ``step.summary``.

        A failure is recorded and re-raised: a trace that quietly swallowed the
        exception would show a pipeline that succeeded when it did not.
        """
        record = Step(order=len(self.steps) + 1, agent=agent, action=action)
        self.steps.append(record)
        began = time.perf_counter()
        try:
            yield record
        except Exception as exc:                                # noqa: BLE001
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.ms = (time.perf_counter() - began) * 1000
            if not record.summary and record.status == "ok":
                record.summary = "no output worth summarising"

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    def as_dict(self) -> dict[str, Any]:
        steps = [s.as_dict() for s in self.steps]
        accounted = sum(s["ms"] for s in steps)
        total = self.elapsed_ms
        return {
            "steps": steps,
            "agents": [s["agent"] for s in steps],
            "total_ms": round(total, 2),
            # What the pipeline spent outside any agent: the payload assembly,
            # the persistence call, the framing. Worth showing rather than
            # letting the step times silently fail to add up.
            "unattributed_ms": round(max(0.0, total - accounted), 2),
            "failed": [s["agent"] for s in steps if s["status"] != "ok"],
        }
