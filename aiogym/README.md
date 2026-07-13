# aiogym - native Gymnasium backend for AIO-Gym-temp

A fast, synchronous, **Gymnasium-first** process-control backend in numpy,
built for benchmarking, controller evaluation, and RL **training**.

It is checked against a JavaScript reference implementation: the ODE right-hand
sides, constants, and RK4 integrator are validated by `tests/test_parity.py`
against golden trajectory checkpoints (max delta <= 1e-15, i.e. bit-for-bit).

## Install & verify
```bash
cd aiogym
pip install -e .                                # full local benchmark/oracle/train/export stack
cd ..
node aiogym/tests/generate_golden.mjs --check     # verify golden data from JS source
python aiogym/tests/test_parity.py               # verify native dynamics parity
```

The default install includes CasADi/IPOPT oracle support, SB3/Torch training,
and ONNX export/runtime dependencies. Extras such as `[oracle]`, `[train]`,
`[export]`, and `[all]` are deprecated compatibility aliases scheduled for
removal in `aiogym 0.3`.

## Use
```python
import gymnasium as gym, aiogym                   # registers the ids on import
env = gym.make("AIOGym/CSTR-v0")                  # Cascade / Quadruple / CSTR / HVAC
obs, _ = env.reset(seed=0)
obs, r, term, trunc, info = env.step(env.action_space.sample())
```
Single env runs roughly 2k steps/s (pure python); wrap in `gymnasium.vector` for more.

## User entry points

AIO-Gym-temp exposes two user-facing entry points. They are intentionally separate:

- **Python API**: use `import aiogym` from notebooks, research scripts, or another
  Python project. This is the stable programmatic interface.
- **CLI commands**: use `aiogym-*` commands from a shell or automation job. These
  commands parse arguments, print progress, set exit codes, and write artifacts.

Before installing the package, run CLI modules directly from the repository root:

```bash
python -m aiogym.cli.suite_benchmark --suite standard-baselines --artifact-dir aiogym/runs/bench_suite_standard-baselines_artifacts
python -m aiogym.cli.artifact_tools report aiogym/runs/bench_suite_standard-baselines_artifacts
```

After installing with `pip install -e ./aiogym` from the repository root, the
same workflows are available as shorter console commands:

```bash
aiogym-suite-benchmark --suite standard-baselines --artifact-dir aiogym/runs/bench_suite_standard-baselines_artifacts
aiogym-report aiogym/runs/bench_suite_standard-baselines_artifacts
```

Generated files are written under timestamped directories in `aiogym/runs/` by
default. Pass `--artifact-dir` when you want a stable path for follow-up report
or artifact-check commands.

## Contract
- Generic model contract: `x` is the state vector, `u` is the action vector,
  `y` is the controlled-output vector, and `y_sp` is the matching setpoint vector.
- Generic observations use `[x, y_sp, disturbances]`; legacy built-in scenarios
  still expose the browser-compatible physical observation adapter.
- Direct-actuator mode uses a flat `u` vector in `[0,1]`. Legacy built-ins keep
  physical actuator aliases for PID/MPC compatibility.
- `reward_mode="tracking"` uses PC-Gym-style setpoint tracking:
  `reward = -(normalized squared setpoint error + input move penalty)`.
  `reward_mode="track"` remains accepted as a backward-compatible alias.
  `reward_mode="kpi"` uses the composite tracking + excess-energy + safety score;
  `reward_mode="economic"` uses value minus energy and soft constraint penalties.
  `control_dt`, `episode_steps`, `randomize`, and `randomize_setpoints` are
  constructor args.

## CLI command guide

The common commands are:

```bash
aiogym-suite-benchmark --suite standard-baselines --episodes 3 --artifact-dir aiogym/runs/bench_suite_standard-baselines_artifacts
aiogym-artifact-check aiogym/runs/bench_suite_standard-baselines_artifacts
aiogym-report aiogym/runs/bench_suite_standard-baselines_artifacts
```

Use these when you need a narrower or more advanced workflow:

```bash
aiogym-single-benchmark --scenario cstr --objective tracking --controllers pid,mpc
aiogym-suite-benchmark --suite economic-supervisory --scenarios cstr --controllers onnx --onnx-path frontend/models/rlpd_cstr.onnx --episodes 1
aiogym-train-sb3 --scenario cstr --algo sac --n-envs 8 --vec-env subproc --steps 10000 --onnx
aiogym-model-cards --check
aiogym-model-cards --format markdown --out-dir docs/model_cards
```

The `aiogym-train-rlpd` command is mainly for research training experiments.
RL algorithms and training entrypoints live under `aiogym.rl`; their default
checkpoints, ONNX exports, and reports are written under `aiogym/runs/rl/`.
Controller tuning utilities are kept as provenance tools under
`aiogym.controllers.tuning` and can be run with `python -m` when maintaining
baseline parameters.

## Python API workflow

The backend offline workflow is registry-driven: model coverage comes from
`aiogym.models.SCENARIOS`, not from a separate core/all completion list. The
current built-ins are `cascade`, `quadruple`, `cstr`, `hvac`, `extraction`,
`heater`, and `crystallization`.

```python
import aiogym

env = aiogym.make_env(model="cstr", objective="tracking", seed=7, episode_steps=80)
payload = aiogym.run_benchmark({
    "scenario": "cstr",
    "objective": "tracking",
    "controller": "pid",
    "output_dir": "aiogym/runs/cstr_tracking_pid",
})
figures = aiogym.plot_results("aiogym/runs/cstr_tracking_pid")
```

Formula-only custom scenarios use the same public API surface:

```python
model = aiogym.define_model({...})
aiogym.register_model("my_process", model)
env = aiogym.make_env(model="my_process")
```

Here `define_model(...)` defines the model from a declarative spec,
`register_model(...)` gives it a scenario name, `make_model(...)` instantiates or
validates a model object, and `make_env(...)` creates the Gymnasium environment.
In normal custom-scenario scripts, users usually call `define_model(...)`,
`register_model(...)`, and then `make_env(...)`; `make_model(...)` is the
model-level counterpart to `make_env(...)` for code that needs the process model
instance directly.

Suite artifacts are written into one standard directory:

```text
<artifact_dir>/
  benchmark.json
  config/
  metadata/
  summary/
  results/
  training/
  figures/
```

`metadata/` contains model-card JSON, `summary/` contains rows, CSV, and
leaderboard data, `results/` contains full result/report payloads, and
`training/` contains RL training metadata and learning curves when present.
`figures/` contains reproducible SVG plots, including `learning_curve.svg` for
SB3/RLPD training artifacts.

For human-readable process-model documentation, see `docs/model_cards/`. The
Markdown cards are generated from the same model metadata used in benchmark
artifacts, so updating a model contract and rerunning `aiogym-model-cards
--format markdown --out-dir docs/model_cards` refreshes the docs.

For extension templates, start with
`aiogym/models/examples/declarative_model.py` for formula-only scenarios. Use
`aiogym/models/examples/custom_model.py` only when you need lower-level custom
Python methods. Controller examples live in
`aiogym/controllers/examples/custom_controller.py`.

### Custom objectives and transition data

Custom task rewards can use the same stateless function for real environment
steps and predicted transitions:

```python
def task_reward(state, action, next_state, context):
    error = next_state[0] - context.setpoint[0]
    return -(error * error)

env = aiogym.make_env(
    model="cstr",
    objective="tracking",
    action_mode="actuator",
    custom_stage_reward=task_reward,
)

predicted = env.evaluate_transition(
    state,
    candidate_action,
    predicted_next_state,
    previous_action=previous_candidate_action,
)
```

`evaluate_transition(...)` does not advance the environment or update KPI
accumulators. During a multi-step prediction, pass the previous candidate action
explicitly so input-move penalties follow the candidate trajectory. Stateful
legacy `custom_reward(env, ...)` callbacks remain supported for real steps but
are intentionally rejected by predicted-transition evaluation.

Use the common transition data layer for custom offline or model-based training:

```python
from aiogym.rl import TransitionDataset, collect_transitions

dataset = collect_transitions(env, policy=None, episodes=10, seed=7)
states, actions, next_states = dataset.supervised_arrays("state")
rows = dataset.to_rows()
restored = TransitionDataset.from_rows(rows)
```

Each transition retains policy observations, physical states, rewards,
termination flags, setpoints, disturbances, and episode/step indices. Custom
training code remains outside the environment and decides when to fit or update
its own policy or dynamics model.

For the **offline-to-online** goal (RLPD / Cal-QL): generate a historian dataset
from this env, seed the replay buffer, keep exploring online. RLPD = SAC +
offline data in buffer + critic LayerNorm + critic ensemble + symmetric
sampling (official JAX impl: ikostrikov/rlpd).

SB3 and RLPD training entrypoints keep their legacy checkpoint/report outputs
and also write standard benchmark artifacts. Use `--artifact-dir` to choose the
directory; SB3 can add intermediate evaluation points with
`--learning-curve-every`.
