<div align="center">

# AIO-Gym-temp

**A cleaned-up Python backend for process-control benchmarking, controller
evaluation, and reinforcement-learning training.**

[**English**](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

MIT License

</div>

---

This checkout is referred to as **AIO-Gym-temp**. It is the reorganized backend
version derived from **AIO-Gym**. In this document, **AIO-Gym** refers to the
older backend snapshot before this reorganization.

This README only describes backend changes: package layout, Python API,
terminal commands, benchmark evaluation, controller organization, model
registry, RL training, artifacts, and generated-output policy.

The detailed backend usage guide lives in [aiogym/README.md](aiogym/README.md).

## Backend Scope

`AIO-Gym-temp` provides a native Python backend under `aiogym/`:

- Native Gymnasium environments.
- Seven registered backend scenarios: `cascade`, `quadruple`, `cstr`, `hvac`,
  `extraction`, `heater`, and `crystallization`.
- Controller evaluation for PID, MPC, oracle, generic policy objects, and SB3
  policies.
- Named benchmark suites with summary tables, reports, plots, and reusable
  artifacts.
- RL training entrypoints for SB3 and RLPD.
- A small public Python API for notebooks, scripts, and external Python users.
- Console commands for terminal and automation workflows.

Install from the repository root:

```bash
pip install -e .
pip install -e ".[oracle]"   # optional: CasADi/IPOPT oracle support
pip install -e ".[train]"    # optional: SB3/Torch training support
pip install -e ".[export]"   # optional: ONNX export support
```

Common commands:

```bash
aiogym-suite-benchmark --suite standard-baselines --episodes 3
aiogym-single-benchmark --scenario cstr --objective tracking --controllers pid,mpc
aiogym-report aiogym/runs/bench_suite_standard-baselines_artifacts
aiogym-artifact-check aiogym/runs/bench_suite_standard-baselines_artifacts
aiogym-train-sb3 --scenario cstr --algo sac --steps 10000 --onnx
```

Python API:

```python
import aiogym

env = aiogym.make_env(model="cstr", protocol="tracking", seed=7)
payload = aiogym.run_benchmark({
    "scenario": "cstr",
    "objective": "tracking",
    "controller": "pid",
})
figures = aiogym.plot_results(payload["run_dir"])
```

## AIO-Gym-temp vs AIO-Gym

**AIO-Gym** means the older backend layout before this reorganization. It was
script-oriented: most backend source files lived directly under `aiogym/`, and
users ran individual Python files.

**AIO-Gym-temp** means the current reorganized backend. It is package-oriented:
source code is split by responsibility, user entrypoints are explicit, and
generated outputs are separated from source files.

| Area | AIO-Gym | AIO-Gym-temp |
|---|---|---|
| Package setup | No `pyproject.toml`; not installable as a standard package from the repo root. | Installable package with `pyproject.toml`, optional dependency groups, package data, and console scripts. |
| User entrypoints | Direct scripts such as `python aiogym/train.py`, `python aiogym/train_rlpd.py`, and `train_all.sh`. | Stable commands such as `aiogym-suite-benchmark`, `aiogym-report`, `aiogym-train-sb3`, and `aiogym-train-rlpd`. |
| Python API | Mostly direct imports from `aiogym.__init__` and internal modules. | Small public API: `aiogym.make_env`, `aiogym.run_benchmark`, and `aiogym.plot_results`. |
| Backend layout | Flat files: `models.py`, `kernel.py`, `baselines.py`, `oracle.py`, `rlpd.py`, `train.py`, `train_sac.py`, `train_rlpd.py`. | Layered packages: `models/`, `controllers/`, `evaluation/`, `rl/`, `cli/`, plus a thin `api.py`. |
| Model coverage | Backend registry covered cascade, quadruple, CSTR, HVAC, and fired heater. | Backend registry covers cascade, quadruple, CSTR, HVAC, extraction, fired heater, and crystallization. |
| Model metadata | Model-card behavior was not a first-class package concern. | Model cards and contract validation live with `aiogym.models`, and exports are checked by tests. |
| Controllers | PID, MPC, evaluation helpers, and baseline concepts were concentrated in `baselines.py` plus `oracle.py`. | Controller interface, registry, configs, adapters, PID, MPC, oracle, and tuning tools live under `aiogym.controllers`. |
| Evaluation | Evaluation was mostly script/helper driven. | `aiogym.evaluation` owns benchmark protocols, rollout collection, metrics, reports, plots, artifact generation, and suites. |
| Benchmark suites | No canonical suite config package. | Named suite JSON files live under `aiogym/evaluation/suites/`, including `standard-baselines`, tracking/economic suites, robustness, RL-direct, and crystallization. |
| Reports and plots | Generated outputs were scattered around runs or produced by scripts. | Artifacts use one standard layout with `benchmark.json`, `config/`, `metadata/`, `summary/`, `results/`, and `figures/`. |
| RL code | `rlpd.py`, `train_rlpd.py`, `train_sac.py`, and `train.py` lived beside core backend modules. | RL algorithms and training flows live under `aiogym.rl`; default outputs go under `aiogym/runs/rl/`. |
| Generated outputs | Historical run JSON files were tracked under `aiogym/runs/`. | `aiogym/runs/` is treated as local output; only `.gitignore` is kept. |
| Parity tooling | JS parity test existed. | Parity testing is retained, and the golden generator lives with backend tests as `aiogym/tests/generate_golden.mjs`. |
| Documentation | Backend usage was spread across script comments, README text, and old helper locations. | Backend documentation is consolidated into README files with explicit package boundaries and commands. |

## Implemented Backend Improvements

### 1. Installable package and canonical commands

`AIO-Gym-temp` is installable from the repository root. User-facing commands are
defined in `pyproject.toml`:

```text
aiogym-single-benchmark
aiogym-suite-benchmark
aiogym-report
aiogym-artifact-check
aiogym-model-cards
aiogym-train-sb3
aiogym-train-rlpd
```

Old compatibility wrappers and broad shell scripts were removed, including the
old `train_all.sh` flow.

### 2. Thin public API

`aiogym/api.py` intentionally exposes only ordinary user-facing functions:

```python
aiogym.make_env(...)
aiogym.run_benchmark(...)
aiogym.plot_results(...)
```

Lower-level artifact writing remains internal to `aiogym.evaluation.artifacts`
instead of being exposed at the top level.

### 3. Clear command-line layer

`aiogym/cli/` contains only terminal entrypoints:

```text
aiogym/cli/
  single_benchmark.py
  suite_benchmark.py
  artifact_tools.py
```

The CLI parses arguments, prints progress, sets process exit behavior, and calls
shared implementation modules. It does not own core evaluation or RL logic.

### 4. Model package instead of one large model file

The old monolithic `models.py` and `kernel.py` were split into:

```text
aiogym/models/
  core.py
  kernel.py
  registry.py
  scenarios/
```

Scenario implementations live in `models/scenarios/`, while registry,
validation, model-card export, and Gym ID helpers live in `models/registry.py`.

### 5. Controller package with registry and tuning provenance

The previous baseline-controller logic was reorganized into:

```text
aiogym/controllers/
  __init__.py
  pid.py
  mpc.py
  oracle.py
  configs/
  tuning/
```

This package owns the formal controller API, controller registry, policy/SB3
adapters, default configs, and reproducible tuning scripts. PID, MPC, and oracle
are now parallel controller implementations rather than being mixed in one
baseline file.

### 6. Evaluation, metrics, reports, plots, and artifacts in one area

Evaluation-related code was consolidated under:

```text
aiogym/evaluation/
  core.py
  metrics/
  reports.py
  plots.py
  artifacts.py
  suites/
```

This boundary is now explicit:

- `core.py` defines benchmark protocols and rollout/evaluation logic.
- `metrics/` contains metric calculators.
- `reports.py` renders benchmark reports.
- `plots.py` produces SVG plots.
- `artifacts.py` writes benchmark artifact directories.
- `suites/` contains named benchmark definitions.

### 7. Standard benchmark artifacts

Benchmark runs now write a reusable artifact directory:

```text
<artifact_dir>/
  benchmark.json
  config/
  metadata/
  summary/
  results/
  figures/
```

This separates source code from generated outputs and gives reports, plots,
model cards, summaries, and full result payloads a predictable home.

### 8. RL code grouped under `aiogym.rl`

RL-specific source is grouped together:

```text
aiogym/rl/
  rlpd.py
  train_rlpd.py
  train_sb3.py
```

Default trained-model outputs are written under:

```text
aiogym/runs/rl/
  rlpd/
  sb3/
```

`aiogym.rl` uses lazy imports so viewing training help does not require optional
Torch/SB3 dependencies.

### 9. Runs cleanup

`aiogym/runs/` is treated as a local output area. Historical tracked run
JSON/SVG files were removed, and the directory keeps only `.gitignore` in source
control.

### 10. Test and parity coverage retained

The cleanup kept the important validation paths:

```bash
python aiogym/tests/test_interface.py
python aiogym/tests/test_parity.py
node aiogym/tests/generate_golden.mjs --check
```

The interface tests cover model contracts, public API entrypoints, controller
evaluation, suite configs, reports/artifacts, setpoint alignment, and oracle
baseline behavior. The parity test checks the native NumPy dynamics against
golden trajectories from the JavaScript reference implementation.

## Recommended Mental Model

Use this structure when deciding where new backend code belongs:

```text
aiogym/
  api.py          # stable Python user entrypoint
  cli/            # terminal entrypoints only
  models/         # process models, kernel, registry, model cards
  controllers/    # controller API, built-in controllers, configs, tuning
  evaluation/     # benchmark protocols, metrics, reports, plots, artifacts
  rl/             # RL algorithms and training workflows
  runs/           # local generated outputs
```

The key design rule is that `api.py` and `cli/` are doors into the system, not
places to hide core logic. Core behavior should live in `models/`,
`controllers/`, `evaluation/`, or `rl/`.
