# aiogym - native Gymnasium backend for AIO-Gym-temp

A fast, synchronous, **Gymnasium-first** process-control backend in numpy,
built for benchmarking, controller evaluation, and RL **training**.

It is checked against a JavaScript reference implementation: the ODE right-hand
sides, constants, and RK4 integrator are validated by `tests/test_parity.py`
against golden trajectory checkpoints (max delta <= 1e-15, i.e. bit-for-bit).

## Install & verify
```bash
pip install -e .                                # core Gymnasium backend
pip install -e ".[oracle]"                      # add NMPC oracle support
pip install -e ".[train]"                       # add SB3/Torch training support
node aiogym/tests/generate_golden.mjs --check     # verify golden data from JS source
python aiogym/tests/test_parity.py               # verify native dynamics parity
```

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
python -m aiogym.cli.suite_benchmark --suite standard-baselines
python -m aiogym.cli.artifact_tools report aiogym/runs/bench_suite_standard-baselines_artifacts
```

After `pip install -e .`, the same workflows are available as shorter console
commands:

```bash
aiogym-suite-benchmark --suite standard-baselines
aiogym-report aiogym/runs/bench_suite_standard-baselines_artifacts
```

Generated files are written under `aiogym/runs/` by default. For example,
`report.md` is created by `aiogym-report`; it is not a source file checked into
the repository.

## Contract
- `obs = [levels(n), temps(n), t_sp(n), h_sp(controlled k), t_cold, t_amb]`
- `action = [pumps..., valves..., heaters...] in [0,1]`  (direct-actuator mode)
- `reward_mode="track"` uses pure setpoint tracking: `reward = -tracking_error`.
  `reward_mode="kpi"` uses the composite tracking + excess-energy + safety score;
  `reward_mode="economic"` uses value minus energy and soft constraint penalties.
  `control_dt`, `episode_steps`, `randomize`, and `randomize_setpoints` are
  constructor args.

## CLI command guide

The common commands are:

```bash
aiogym-suite-benchmark --suite standard-baselines --episodes 3
aiogym-artifact-check aiogym/runs/bench_suite_standard-baselines_artifacts
aiogym-report aiogym/runs/bench_suite_standard-baselines_artifacts
```

Use these when you need a narrower or more advanced workflow:

```bash
aiogym-single-benchmark --scenario cstr --objective tracking --controllers pid,mpc
aiogym-train-sb3 --scenario cstr --algo sac --n-envs 8 --vec-env subproc --steps 10000 --onnx
aiogym-model-cards --check
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

env = aiogym.make_env(model="cstr", protocol="tracking", seed=7, episode_steps=80)
payload = aiogym.run_benchmark({
    "scenario": "cstr",
    "objective": "tracking",
    "controller": "pid",
    "output_dir": "aiogym/runs/cstr_tracking_pid",
})
figures = aiogym.plot_results("aiogym/runs/cstr_tracking_pid")
```

Suite artifacts are written into one standard directory:

```text
<artifact_dir>/
  benchmark.json
  config/
  metadata/
  summary/
  results/
  figures/
```

`metadata/` contains model-card JSON, `summary/` contains rows, CSV, and
leaderboard data, `results/` contains full result/report payloads, and
`figures/` contains reproducible SVG plots.

For the **offline-to-online** goal (RLPD / Cal-QL): generate a historian dataset
from this env, seed the replay buffer, keep exploring online. RLPD = SAC +
offline data in buffer + critic LayerNorm + critic ensemble + symmetric
sampling (official JAX impl: ikostrikov/rlpd).
