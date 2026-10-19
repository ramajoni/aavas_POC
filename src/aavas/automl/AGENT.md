# Agentic AutoML

`aavas.automl` · version 1.0

Algorithm selection, hyper-parameter search, imbalance handling, calibration, ensembling and champion selection under hard gates.


**Standalone.** It depends on nothing but the shared kernel and its own config.

## What it produces

A leaderboard and a champion recommendation, with the gates each candidate cleared or failed.

## Public surface

This is the reuse contract. `tests/test_agent_boundaries.py` fails if any of these
stops existing, and if this package imports anything it does not declare.

| Symbol | Module |
|---|---|
| `AutoMLSearch` | `aavas.automl` |
| `SearchConfig` | `aavas.automl` |
| `SearchResult` | `aavas.automl` |
| `candidates` | `aavas.automl` |
| `inventory` | `aavas.automl` |
| `Candidate` | `aavas.automl.selection` |
| `Selection` | `aavas.automl.selection` |
| `optimize` | `aavas.automl.hpo` |
| `apply` | `aavas.automl.imbalance` |
| `diagnose` | `aavas.automl.imbalance` |
| `fit` | `aavas.automl.calibration` |
| `expected_calibration_error` | `aavas.automl.calibration` |

## Dependencies

| Depends on | Kind |
|---|---|
| `aavas.kernel` | shared kernel |
| `aavas.config` | shared kernel |
| `aavas.logging_utils` | shared kernel |

The **shared kernel** (`aavas.kernel`) holds the pure primitives - column
coercion, PSI, group fairness. It imports nothing from `aavas` except logging, and
a test enforces that: a primitive that needs config is not a primitive, and every
agent would inherit the dependency.

## Configuration it owns

- `config/automl.yaml`

## Entry points

- make automl
- scripts/run_automl.py
- GET /tracking/runs

## Lifting it into another codebase

The most portable of the four: it takes X, y and a contract, and knows nothing about lending. The fairness gate needs a group column, and reports 'not measurable' rather than passing when there is none.

Take: this directory, `aavas/kernel/`, the config file above, and the tests that
name it. The manifest in `aavas/agents/packages.py` is the checklist.
