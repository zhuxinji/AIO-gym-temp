# AIO-Gym architecture

This page describes the current source-code layout and the ownership boundaries
between models, environments, controllers, evaluation, and generated artifacts.
For user-facing vocabulary, read [Concepts](concepts.md). For supported API
entry points, read the [Public API guide](public_api.md).

## Layer overview

```text
scenario + task
      |
      v
ProcessModelContract
      |
      v
AIOGymNativeEnv <--- controller
      |
      v
BenchmarkProtocol + BenchmarkCase
      |
      v
evaluation + rollout
      |
      v
aggregation
      |
      v
artifacts + plots + report + checks
```

The ownership rule is:

- models own physical equations, parameter metadata, and model-specific tasks;
- environments own Gymnasium interaction and runtime state;
- controllers own control algorithms, policy adapters, and controller presets;
- evaluation owns objectives, metrics, benchmark cases, suites, and acceptance;
- the output layer owns artifact persistence, plots, reports, and validation.

## Source tree

```text
AIO-Gym/
|-- aiogym/
|   |-- __init__.py                 top-level public API and Gym registration
|   |-- catalog.py                  scenario/task/suite/controller discovery
|   |-- env.py                      AIOGymNativeEnv orchestration and Gym API
|   |-- env_factory.py              high-level make_env factory
|   |
|   |-- _environment/               internal environment runtime helpers
|   |   |-- config.py               defaults and validation
|   |   |-- disturbances.py         events, disturbances, and plant drift
|   |   |-- observations.py         observations, setpoints, supervisory actions
|   |   `-- transitions.py          reward and termination evaluation
|   |
|   |-- _internal/                  shared internal utilities
|   |   |-- config.py
|   |   |-- identifiers.py
|   |   |-- paths.py
|   |   |-- serialization.py
|   |   `-- validation.py
|   |
|   |-- models/                     physical models and model-owned resources
|   |   |-- core.py                 ProcessModelContract
|   |   |-- backends.py             numeric and CasADi operation adapters
|   |   |-- integration.py          fixed-step RK4 Integrator
|   |   |-- registry.py             model registry and construction
|   |   |-- declarative.py          declarative custom-model support
|   |   |-- validation.py           numerical readiness checks
|   |   |-- cards.py                model-card generation
|   |   |-- scenarios/              eight built-in process models
|   |   |-- parameters/             parameter metadata JSON files
|   |   |-- tasks/                  task schema, registry, and built-in task JSON
|   |   `-- examples/               custom-model examples
|   |
|   |-- controllers/                controller implementations and construction
|   |   |-- contracts.py            controller protocol and step context
|   |   |-- adapters.py             Python-policy and SB3 adapters
|   |   |-- configs.py              controller configuration loading
|   |   |-- registry.py             controller registry and factories
|   |   |-- pid.py / mpc.py
|   |   |-- oracle.py / onnx.py
|   |   |-- configs/                built-in controller preset JSON files
|   |   `-- tuning/                 tuning utilities
|   |
|   |-- evaluation/                 benchmark semantics and execution
|   |   |-- metric_catalog.py       metric definitions, direction, and schema
|   |   |-- objective_specs.py      objective and reward-mode resolution
|   |   |-- protocols.py            BenchmarkProtocol
|   |   |-- cases.py                EnvironmentSpec and BenchmarkCase
|   |   |-- evaluator.py            multi-episode evaluation
|   |   |-- rollouts.py             scenario-neutral rollout recording
|   |   |-- aggregation.py          result aggregation and report tables
|   |   |-- metadata.py             diagnostics and reproducibility metadata
|   |   |-- runner.py               one resolved benchmark-case execution path
|   |   |-- benchmark.py            public configuration-driven benchmark API
|   |   |-- task_acceptance.py      task acceptance evaluation
|   |   |-- metrics/                tracking/economic/KPI/safety/robustness metrics
|   |   |-- suite_loading.py        suite loading, inheritance, and presets
|   |   |-- suite_cases.py          resolved suite-case expansion
|   |   |-- suite_results.py        suite summaries and artifact paths
|   |   |-- suites/                 built-in suite JSON and shared presets
|   |   |-- artifact_tables.py      leaderboard and CSV data preparation
|   |   |-- artifact_writers.py     standard artifact file writing
|   |   |-- artifact_plotting.py    plot orchestration
|   |   |-- artifact_checks.py      artifact structure/content checks
|   |   |-- report_rendering.py     Markdown report generation
|   |   `-- plots.py                SVG rendering
|   |
|   |-- cli/                        command-line adapters
|   |-- rl/                         SB3/RLPD training and transition datasets
|   `-- tests/                      contract, regression, physics, and API tests
|
|-- docs/                           user and contributor documentation
|-- runs/                           default benchmark/training artifacts
|-- README.md
|-- pyproject.toml                  packaging, dependencies, and CLI entry points
`-- MANIFEST.in                     source-distribution files
```

## Public APIs and compatibility facades

Implementation modules are deliberately focused, but established imports remain
available through stable facades:

- `aiogym` exposes the primary user API;
- `aiogym.controllers` exposes controller construction and registration;
- `aiogym.evaluation.core` preserves earlier evaluation imports;
- `aiogym.evaluation.protocols` preserves protocol and metric imports;
- `aiogym.evaluation.artifacts` exposes artifact writing and plotting;
- `aiogym.evaluation.reports` exposes report rendering and artifact checks;
- `aiogym.evaluation.task_profiles` preserves the former task-profile path while
  task definitions live under `aiogym.models.tasks`.

Applications should prefer the top-level `aiogym` API or the documented package
facades. The focused implementation modules are useful for extension and
maintenance work, but are not all intended as permanent public APIs.

## Configuration and generated data

Runtime package data is stored alongside its owner:

- model parameter metadata: `aiogym/models/parameters/`;
- model task declarations: `aiogym/models/tasks/builtin/`;
- controller presets: `aiogym/controllers/configs/`;
- benchmark suites and shared presets: `aiogym/evaluation/suites/`.

Generated directories are not source code:

- `runs/` contains benchmark and training artifacts;
- `build/` and `dist/` are Python packaging outputs;
- `output/` is reserved for manually exported reports or documents;
- `tmp/` is reserved for disposable intermediate files.

These generated paths are ignored by Git. Deleting `build/`, `dist/`, or `tmp/`
does not affect source code; deleting `runs/` or `output/` may remove results that
cannot be recovered unless they are regenerated.

## Extension points

- Add a process by implementing `ProcessModelContract` and registering it.
- Add a formal experiment under `models/tasks` when it changes model operating
  conditions, setpoints, events, or constraints.
- Add a controller through the controller registry and declare its action mode.
- Add an objective metric under `evaluation/metrics` and register its direction
  and definition in the metric catalog.
- Add reusable benchmark coverage through a suite JSON declaration rather than
  embedding a matrix in CLI code.

See [Physical-model infrastructure](model_infrastructure.md) for model extension
requirements and the [Public API guide](public_api.md) for executable examples.
