# AIO-Gym

AIO-Gym is a native Python backend for process-control reinforcement learning,
controller benchmarking, offline-data generation, and model-based research. It
provides synchronous, seedable, and vectorizable Gymnasium environments without
requiring a browser or external simulator.

## Features

- Eight process scenarios: `cascade`, `cascade_recirculating`, `quadruple`,
  `cstr`, `hvac`, `extraction`, `heater`, and `crystallization`.
- A generic model contract based on state `x`, action `u`, controlled output
  `y`, setpoint `y_sp`, and declared disturbances.
- PID, linear MPC, nonlinear MPC oracle, Python policy, SB3, and ONNX
  controller support.
- Tracking, economic, KPI, robustness, and safety benchmark protocols.
- Reproducible benchmark suites, reports, plots, leaderboards, and artifact
  validation.
- SB3 and RLPD training workflows with standard benchmark artifacts and ONNX
  export.
- Declarative and Python extension interfaces for custom process models and
  controllers.

## Requirements

- Python 3.10 or newer
- NumPy and Gymnasium
- CasADi for nonlinear MPC
- PyTorch and Stable-Baselines3 for training
- ONNX and ONNX Runtime for policy export and inference

The default installation includes the complete backend dependency stack.

## Installation

Clone the repository and install it from the repository root:

```bash
python -m pip install -e .
```

Install the test dependency as well:

```bash
python -m pip install -e ".[dev]"
```

A non-editable local installation is also supported:

```bash
python -m pip install .
```

## Quick Start

Importing `aiogym` registers all built-in Gymnasium environment IDs:

```python
import gymnasium as gym
import aiogym

env = gym.make("AIOGym/CSTR-v0")
obs, info = env.reset(seed=7)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

The direct factory accepts scenario, task, and benchmark-objective options:

```python
import aiogym

env = aiogym.make_env(
    scenario="quadruple",
    task="minimum-phase-classic",
    objective="tracking",
    seed=7,
    episode_steps=200,
    dynamic=True,
)
```

`dynamic` controls generic automatically generated within-episode events; it
does not switch the process differential equations on or off. Named tasks can
disable generic events and still declare deterministic setpoint or disturbance
schedules.

`scenario` selects the process, `task` selects the operating point and
experiment, and `objective` selects scoring. Reproducible benchmark artifacts
record all three identities separately.

Every scenario uses the same backend contracts:

```text
observation = [x, y_sp, disturbances]
action      = flat u vector in [0, 1]
```

Set `action_mode="setpoint"` to evaluate supervisory policies over the built-in
PID layer when the selected model declares a supervisory layout.

See the [public API guide](docs/public_api.md) for the complete model,
environment, task, objective, controller, and benchmark calling hierarchy.

## Benchmarking

Run the standard controller-comparison suite:

```bash
aiogym-suite-benchmark \
  --suite standard-baselines \
  --episodes 3
```

Run a smaller single-scenario comparison:

```bash
aiogym-single-benchmark \
  --scenario cstr \
  --objective tracking \
  --controllers pid,mpc
```

Compare the minimum- and nonminimum-phase quadruple-tank tasks without mixing
their rankings:

```bash
aiogym-suite-benchmark --suite quadruple-phase-comparison --episodes 1
```

Run the heated-tank cascade as an explicit continuous-production economic task:

```bash
aiogym-single-benchmark \
  --scenario cascade \
  --task continuous-benchmark \
  --objective economic \
  --controllers pid,mpc
```

Direct `cascade` environments retain batch semantics. The
continuous task's `4.0e-4 m3/s` target is an assumed benchmark throughput, not a
validated equipment rating. See [the cascade scenario guide](docs/scenarios/cascade.md).

Run the PDF-derived closed-loop retrofit independently from the historical
open cascade:

```bash
aiogym-single-benchmark \
  --scenario cascade_recirculating \
  --task commissioning \
  --objective tracking \
  --controllers pid,mpc,oracle
```

Run all four formal recirculating tasks under their task-owned default
objectives with PID and MPC:

```bash
aiogym-suite-benchmark \
  --suite cascade-recirculating \
  --episodes 3
```

This scenario has one 2 kW Tank 1 heater, four actuator actions, passive
overflow returns to Tank 3, and no production-economic objective. See the
[recirculating scenario guide](docs/scenarios/cascade_recirculating.md) and
[accuracy/authenticity report](docs/reports/cascade-recirculating-accuracy-authenticity.md).

Run all six formal quadruple-tank tasks with PID, MPC, and NMPC Oracle on every
task:

```bash
aiogym-suite-benchmark --suite quadruple --episodes 1
```

Run only the paper-reference decentralized-PI reproductions, or run the
zero-boundary and deterministic-disturbance subsets:

```bash
aiogym-suite-benchmark --suite quadruple-paper-reference --episodes 1
aiogym-suite-benchmark --suite quadruple-zero-boundary --episodes 1
aiogym-suite-benchmark --suite quadruple-disturbance-rejection --episodes 1
```

Use an ONNX policy:

```bash
aiogym-suite-benchmark \
  --suite economic-supervisory \
  --scenarios cstr \
  --controllers onnx \
  --onnx-path path/to/policy.onnx \
  --episodes 1
```

The same workflow is available through Python:

```python
import aiogym

payload = aiogym.run_benchmark({
    "scenario": "cstr",
    "objective": "tracking",
    "controllers": ["pid", "mpc"],
    "seeds": [7, 8, 9],
    "output_dir": "runs/cstr_tracking",
})
```

## Benchmark Artifacts

Suite and public benchmark runs write a standard artifact directory:

```text
<artifact_dir>/
  benchmark.json
  config/
  metadata/
  summary/
  results/
  rollouts/
  training/
  figures/
```

Generate a Markdown report or validate an artifact directory:

```bash
aiogym-report path/to/artifact_dir
aiogym-artifact-check path/to/artifact_dir
```

Default suite outputs use timestamped directories under `aiogym/runs/`. Pass
`--artifact-dir` when a stable path is required.

## Training

Train an SB3 policy and optionally export ONNX:

```bash
aiogym-train-sb3 \
  --scenario cstr \
  --algo sac \
  --steps 10000 \
  --onnx
```

Run the offline-to-online RLPD workflow:

```bash
aiogym-train-rlpd \
  --scenario cstr \
  --offline-episodes 20 \
  --online-steps 10000
```

Training runs write checkpoints, exports, learning curves, evaluation results,
and standard benchmark artifacts.

## Custom Models and Controllers

Start with the declarative model example when the dynamics can be expressed as
formulas:

```bash
python aiogym/models/examples/declarative_model.py
```

Use the Python model example for lower-level behavior:

```bash
python aiogym/models/examples/custom_model.py
```

Human-readable model documentation is generated under `docs/model_cards/`:

```bash
aiogym-model-cards --format markdown --out-dir docs/model_cards
aiogym-model-cards --check --format markdown --out-dir docs/model_cards
```

The shared physical-model foundation is described in
[`docs/model_infrastructure.md`](docs/model_infrastructure.md). It covers
versioned parameter provenance, scenario task profiles, solver metadata, and
generic numerical readiness checks. Unmigrated profiles retain
`legacy-unverified`; `quadruple` is the first `reference-parameterized` physical
model.

## Package Layout

```text
aiogym/
  _internal/      private shared utilities
  cli/            console entrypoints
  controllers/    controller API, algorithms, configs, and tuning
  evaluation/     protocols, objectives, metrics, reports, and artifacts
  models/         model contracts, registry, scenarios, and model cards
  rl/             transition data, RLPD, and training workflows
  tests/          backend contract and regression tests
  env.py          Gymnasium environment
  env_factory.py  public environment factory
```

## Validation

Run the complete backend test suite from the repository root:

```bash
python -m pytest -q
```

Run a short end-to-end benchmark check:

```bash
aiogym-suite-benchmark \
  --suite standard-baselines \
  --scenarios cstr \
  --objectives tracking \
  --controllers pid,mpc \
  --episodes 1 \
  --episode-steps 2
```

## License

MIT
