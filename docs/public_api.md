# Public API guide

This page is the user-facing entry point for selecting a process model, creating
an environment, choosing a reproducible task, and running controller benchmarks.
Import these APIs from the top-level `aiogym` package unless noted otherwise.

## Choose the right layer

| Goal | Public entry point | What it selects |
| --- | --- | --- |
| Inspect or simulate process equations | `aiogym.make_model(...)` | Scenario/model only |
| Train or interact through Gymnasium | `aiogym.make_env(...)` | Scenario plus optional task and objective |
| Inspect bundled experiments | `aiogym.list_task_profiles(...)`, `aiogym.load_task_profile(...)` | Task declarations |
| Compare controllers on one experiment | `aiogym.run_benchmark(...)` or `aiogym-single-benchmark` | One scenario/task/objective protocol |
| Run a published matrix of experiments | `aiogym-suite-benchmark` | Multiple resolved benchmark cases |

The concepts are intentionally separate:

- **scenario** selects the process equations, state/action contract, and default
  model parameters;
- **task** selects operating conditions such as initialization, setpoints,
  disturbances, constraints, timing, noise, and optional `model_params`;
- **objective** selects the reward/reporting semantics and primary ranking metric;
- **controller** selects the policy evaluated against that resolved experiment.

## Models

```python
model = aiogym.make_model(scenario="cascade")
```

`scenario` can be a registered scenario name, a model instance, or a zero-argument
model factory. A model is independent of task and objective selection.

```python
import aiogym

model = aiogym.make_model("cascade_recirculating")
equilibrium = model.nominal_steady_state()
```

Use `aiogym.SCENARIOS` to inspect registered scenario names. Custom models can be
managed with `register_model()` and `unregister_model()`.

## Environments

```python
env = aiogym.make_env(
    scenario="cascade",
    objective=None,
    seed=None,
    config=None,
    *,
    protocol=None,
    **overrides,
)
```

For reproducible work, select a named task. If `objective` is omitted, the task's
`default_objective` is used.

```python
import aiogym

env = aiogym.make_env(
    "cascade_recirculating",
    task="temperature-step",
    seed=7,
)
obs, info = env.reset(seed=7)
```

A no-task `make_env()` call preserves the direct interactive environment defaults.
It does not acquire a benchmark task identity or task acceptance criteria.

### What `dynamic` means

`dynamic=False` does **not** turn the physical process into a static model. The
model still advances through `model.dynamics(...)` on every integration step.
The environment's `dynamic` option only controls generic automatically generated
within-episode events. A named task can set it to `false` and still apply its own
deterministic setpoint and disturbance schedule.

Use `model_params={...}` when the numerical plant parameters should differ from
the scenario defaults. Initial state, setpoints, disturbance schedules, and
`control_dt` change the experiment without changing the underlying equations.

## Tasks

```python
tasks = aiogym.list_task_profiles("cascade_recirculating")
task = aiogym.load_task_profile(
    "cascade_recirculating/temperature-step"
)
```

The short form is accepted when the scenario is supplied by the caller:

```python
env = aiogym.make_env(
    "cascade_recirculating",
    task="temperature-step",
)
```

Task files live under `aiogym/evaluation/tasks/<scenario>/`. They are validated,
versioned declarations rather than executable model implementations.

## Objective resolution

One objective is resolved before an environment or benchmark case is created:

```text
explicit API/CLI objective
-> case objective
-> suite objective
-> task.default_objective
```

If all four are absent, benchmark protocol construction raises an error.

Resolved benchmark metadata records `objective_source` as `explicit`,
`case-config`, `suite-config`, or `task-default`.

Changing objective does not change task conditions. `robustness` and `safety`
use the environment's KPI reward implementation but retain distinct benchmark
metrics and result identities.

## One-protocol benchmarks

```python
payload = aiogym.run_benchmark(config, *, objective=None)
```

`config` can be a mapping or a JSON path. One resolved
scenario/task/objective protocol is evaluated with every listed controller and
seed.

The canonical top-level fields are `scenario`, `task`, `objective`,
`controllers`, `controller_configs`, `seeds`, `environment`, `output_dir`, and
`strict`. Environment settings may also be written directly at the top level.
Removed aliases are rejected instead of being interpreted silently.

```python
import aiogym

payload = aiogym.run_benchmark({
    "scenario": "cascade_recirculating",
    "task": "temperature-step",
    "controllers": ["pid", "mpc"],
    "seeds": [9000, 9001, 9002],
    "output_dir": "aiogym/runs/recirculating-temperature-step",
})
```

The equivalent command-line workflow is:

```bash
aiogym-single-benchmark \
  --scenario cascade_recirculating \
  --task temperature-step \
  --controllers pid,mpc \
  --episodes 3
```

Use `aiogym-single-benchmark --help` for all runtime overrides.

## Benchmark suites

A suite is a JSON declaration containing multiple cases. Each case resolves to a
`BenchmarkCase` with an `EnvironmentSpec`, `ObjectiveSpec`, controller
configuration, and seed list.

```bash
aiogym-suite-benchmark \
  --suite cascade-recirculating \
  --episodes 3
```

The suite runner supports per-case tasks, objectives, controller profiles, and
environment settings. It writes a standard artifact directory and keeps rankings
separate for every `(scenario, task, objective)` identity. Use
`aiogym-suite-benchmark --help` to list built-in suites and filtering options.

## Advanced resolved specifications

Most users should prefer `make_env()` and `run_benchmark()`. Framework and
integration code can construct:

- `ObjectiveSpec`: resolved scoring semantics and provenance;
- `EnvironmentSpec`: environment construction independent of ranking;
- `BenchmarkProtocol`: a convenient declaration that resolves both;
- `BenchmarkCase`: the canonical resolved environment/objective/controller/seed evaluation unit.

Benchmark result rows distinguish `execution_status` (whether the run completed)
from `objective_status` (whether task acceptance criteria were met).
