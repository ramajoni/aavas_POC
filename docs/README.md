# Documentation

Technical documentation for the Aavas home-loan pricing, yield-maximization and
sales-override intelligence engine. The root [README](../README.md) makes the business case
and reports results; these documents describe how the system is built and how to run it.

| Document | Read it for |
|---|---|
| **[solution-architecture.md](solution-architecture.md)** | The formal architecture: drivers, principles, the four views, quality attributes with measured evidence, twelve architecture decision records, risks and a dependency-ordered roadmap. Written for architecture review, model risk and credit policy |
| **[feature-intelligence.md](feature-intelligence.md)** | Agentic feature engineering: the profiler, the statistical battery, the four refusals, the veto-ordered decision, the research trigger, and why code generation is not built |
| **[architecture.md](architecture.md)** | Layers, components, the agent pipeline, the decision kernel, model suite, governance, and the reasoning behind each structural choice |
| **[flows.md](flows.md)** | Every path through the system, step by step, with sequence diagrams — including the guard-rail and degradation paths |
| **[api-reference.md](api-reference.md)** | Endpoints, request contracts, response shapes, error semantics, a copy-pasteable demo flow |
| **[data-model.md](data-model.md)** | The lifecycle table, feature contracts, governance-store schema, model artefacts, every configuration knob |
| **[operations.md](operations.md)** | Running, testing, building, containers, the XAI overlay venv, troubleshooting, and what you can change without touching code |

## Where to start

| If you are… | Start here |
|---|---|
| New to the repo | [architecture.md §1–§5](architecture.md#1-what-the-system-does), then [flows.md F2](flows.md#f2--stage-2-final-assessment) |
| Integrating an LOS / CRM | [api-reference.md](api-reference.md), then [flows.md F1–F4](flows.md#f1--stage-1-pre-assessment) |
| Reviewing governance or model risk | [solution-architecture.md](solution-architecture.md) end to end — §10 governance, §14 the approval gap, §17 decision records, §18 risks |
| On an architecture review board | [solution-architecture.md §1–§6](solution-architecture.md#1-executive-summary), then §17–§20 |
| Auditing a specific priced decision | [solution-architecture.md §10.3](solution-architecture.md#103-reconstruction-tested-rather-than-asserted), then `GET /system-of-record/reconstruct/{decision_id}` |
| Changing pricing policy | [operations.md §10](operations.md#10-changing-behaviour-without-changing-code), [data-model.md §5](data-model.md#5-configuration-reference) |
| Running or debugging it | [operations.md](operations.md) |
| Working on the UI | [architecture.md §13](architecture.md#13-frontend-application), [flows.md F9](flows.md#f9--ui-journey), [application/README.md](../application/README.md) |

## The one-paragraph version

An application is priced twice: Stage 1 returns an honest interval before anything is
verified; Stage 2 simulates every policy-permissible price, prices each in rupees of
risk-adjusted value (conversion × lifetime economics − credit loss − funding/servicing −
prepayment impact), and recommends the value-maximising one subject to conversion and
portfolio guard-rails. The engine recommends and explains — it never approves. Every
recommendation names the authority that may sign it, every human override is captured beside
it with a reason, every outcome is captured against the decision, and the learning loop reports
what those overrides actually did once selection bias is accounted for.

> The data is synthetic. Every metric in these documents demonstrates that the pipeline works,
> not that it will perform this way on Aavas data.
