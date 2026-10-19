# Acceptance Scoring Agent

`aavas.acceptance` · version 1.0

Turns a customer's own words into a closed vocabulary of acceptance signals, weighs how far each can be trusted, and scores the likelihood that this customer accepts.


**Not standalone.** It calls aavas.agents.capabilities to check it may reach a connector. That edge is a control, not an accident: lifting this package means taking the permission model with it.

## What it produces

A Loan Acceptance Confidence Score with a verbatim quote behind every signal.

## Public surface

This is the reuse contract. `tests/test_agent_boundaries.py` fails if any of these
stops existing, and if this package imports anything it does not declare.

| Symbol | Module |
|---|---|
| `AcceptanceAgent` | `aavas.acceptance` |
| `ExtractionAgent` | `aavas.acceptance` |
| `EvidenceAgent` | `aavas.acceptance` |
| `AcceptanceSignals` | `aavas.acceptance` |
| `Signal` | `aavas.acceptance` |
| `SourceRegistry` | `aavas.acceptance` |
| `SourceDocument` | `aavas.acceptance` |

## Dependencies

| Depends on | Kind |
|---|---|
| `aavas.kernel` | shared kernel |
| `aavas.config` | shared kernel |
| `aavas.logging_utils` | shared kernel |
| `aavas.connectors` | package |
| `aavas.agents` | package |

The **shared kernel** (`aavas.kernel`) holds the pure primitives - column
coercion, PSI, group fairness. It imports nothing from `aavas` except logging, and
a test enforces that: a primitive that needs config is not a primitive, and every
agent would inherit the dependency.

## Configuration it owns

- `config/acceptance.yaml`

## Entry points

- dormant - not called by the orchestrator

## Lifting it into another codebase

Portable wherever free text carries intent. Two things travel with it: the signal vocabulary, which is the leakage guard, and the permission check at the connector boundary - the dependency on `agents` is that check, and removing it would remove the control.

Take: this directory, `aavas/kernel/`, the config file above, and the tests that
name it. The manifest in `aavas/agents/packages.py` is the checklist.
