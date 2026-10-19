# Agentic Feature Engineering

`aavas.features` · version 1.0

Generates candidate features from a stage's own contract, tests each for signal, redundancy, stability and coverage, and refuses to admit any of them.


**Standalone.** It depends on nothing but the shared kernel and its own config.

## What it produces

A discovery report: candidates with tests and refusals. Never a feature.

## Public surface

This is the reuse contract. `tests/test_agent_boundaries.py` fails if any of these
stops existing, and if this package imports anything it does not declare.

| Symbol | Module |
|---|---|
| `discover` | `aavas.features.discovery` |
| `FeatureDiscoveryAgent` | `aavas.features.discovery` |
| `allowed_columns` | `aavas.features.discovery` |
| `research_needed` | `aavas.features.discovery` |
| `profile` | `aavas.features.profiling` |
| `examine` | `aavas.features.profiling` |
| `build_card` | `aavas.features.cards` |
| `decide` | `aavas.features.cards` |
| `lineage` | `aavas.features.cards` |

## Dependencies

| Depends on | Kind |
|---|---|
| `aavas.kernel` | shared kernel |
| `aavas.config` | shared kernel |
| `aavas.ingestion` | shared kernel |
| `aavas.logging_utils` | shared kernel |

The **shared kernel** (`aavas.kernel`) holds the pure primitives - column
coercion, PSI, group fairness. It imports nothing from `aavas` except logging, and
a test enforces that: a primitive that needs config is not a primitive, and every
agent would inherit the dependency.

## Configuration it owns

- `config/feature_config.yaml`

## Entry points

- GET /features
- POST /features/discover
- GET /features/cards

## Lifting it into another codebase

Takes a DataFrame and a stage name. The stage contract comes from feature_config.yaml; point that at another contract and it works on another dataset unchanged. Nothing in it touches the pricing domain.

Take: this directory, `aavas/kernel/`, the config file above, and the tests that
name it. The manifest in `aavas/agents/packages.py` is the checklist.
