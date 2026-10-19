# Price Optimization Agent

`aavas.optimization` · version 1.0

Searches the policy-permissible band for the price with the best risk-adjusted expected value, and reports what it rejected and why.


**Standalone.** It depends on nothing but the shared kernel and its own config.

## What it produces

A recommended rate with its economics, the scenarios rejected, and the binding constraint.

## Public surface

This is the reuse contract. `tests/test_agent_boundaries.py` fails if any of these
stops existing, and if this package imports anything it does not declare.

| Symbol | Module |
|---|---|
| `optimize` | `aavas.optimization.price_optimizer` |
| `evaluate_scenarios` | `aavas.optimization.price_optimizer` |
| `generate` | `aavas.optimization.scenario_generator` |
| `compute` | `aavas.optimization.yield_calculator` |
| `decompose` | `aavas.optimization.yield_calculator` |
| `breakeven_rate` | `aavas.optimization.yield_calculator` |
| `card_rate` | `aavas.optimization.constraints` |
| `permissible_band` | `aavas.optimization.constraints` |
| `approval_level` | `aavas.optimization.constraints` |
| `check` | `aavas.optimization.constraints` |

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

- `config/pricing_policy.yaml`

## Entry points

- POST /pricing/final-assessment

## Lifting it into another codebase

The least portable, and honestly so: it encodes a lending pricing policy - card rates, LTV and FOIR limits, approval authority. The *shape* transfers to any priced product; the policy does not.

Take: this directory, `aavas/kernel/`, the config file above, and the tests that
name it. The manifest in `aavas/agents/packages.py` is the checklist.
