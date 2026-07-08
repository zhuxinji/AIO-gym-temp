"""Stable Python API for user code.

Use this module from notebooks, scripts, or external Python projects. Command
line argument parsing lives in ``aiogym.cli`` and should depend on this API or
the shared evaluation modules, not the other way around.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .controllers import make_controller as _make_controller
from .env import AIOGymNativeEnv as _AIOGymNativeEnv
from .evaluation import (
    BenchmarkConfig as _BenchmarkConfig,
    BenchmarkProtocol as _BenchmarkProtocol,
    build_evaluation_report as _build_evaluation_report,
    evaluate_controller as _evaluate_controller,
    rollout_controller as _rollout_controller,
)
from .evaluation.artifacts import plot_results as plot_results
from .evaluation.artifacts import write_benchmark_artifacts as _write_benchmark_artifacts


__all__ = ["make_env", "run_benchmark", "plot_results"]
API_BENCHMARK_SCHEMA_VERSION = "aiogym.benchmark_api.v1"
PROTOCOL_KEYS = {
    "objective",
    "env_reward_mode",
    "reward_mode",
    "action_mode",
    "control_dt",
    "episode_steps",
    "dynamic",
    "randomize",
    "randomize_setpoints",
    "randomize_plant",
    "plant_drift",
    "integral_obs",
    "terminate_on_runaway",
    "noise",
    "noise_pct",
}


def make_env(model: str = "cascade", protocol: str | Mapping[str, Any] | _BenchmarkProtocol | None = None,
             seed: int | None = None, config: str | Path | Mapping[str, Any] | None = None,
             **overrides) -> _AIOGymNativeEnv:
    """Create an AIO-Gym environment from direct arguments or a config mapping."""

    data = _load_config(config)
    data.update(overrides)
    if protocol is None and "protocol" in data:
        protocol = data.pop("protocol")
    scenario = data.pop("scenario", data.pop("model", model))
    if protocol is not None or "objective" in data:
        proto = _make_protocol(scenario, protocol=protocol, data=_protocol_data(data))
        env = proto.make_env()
    else:
        env = _AIOGymNativeEnv(scenario, **data)
    if seed is not None:
        env.reset(seed=seed)
    return env


def run_benchmark(config: str | Path | Mapping[str, Any] | _BenchmarkConfig) -> dict:
    """Run a config-driven controller benchmark and write reusable artifacts."""

    cfg = _benchmark_config_dict(config)
    scenario = cfg.get("scenario", cfg.get("model", "cstr"))
    objective = cfg.get("objective", cfg.get("protocol", "tracking"))
    if isinstance(objective, Mapping):
        objective_name = objective.get("objective", "tracking")
    else:
        objective_name = str(objective)
    protocol_data = _protocol_data(cfg)
    protocol_data.update(dict(cfg.get("env", {})))
    protocol_data.update(dict(cfg.get("protocol_kwargs", {})))
    protocol_data["objective"] = objective_name
    protocol = _make_protocol(scenario, protocol=cfg.get("protocol", objective_name), data=protocol_data)
    controller_names = _as_list(cfg.get("controllers", cfg.get("controller", "pid")))
    controller_configs = dict(cfg.get("controller_configs", {}))
    seeds = tuple(int(seed) for seed in cfg.get("seeds", cfg.get("seed_list", [cfg.get("seed", 0)])))
    include_episodes = bool(cfg.get("include_episodes", True))
    save_rollouts = bool(cfg.get("save_rollouts", False))
    rollout_steps = cfg.get("rollout_steps")
    out_dir = Path(cfg.get("output_dir", cfg.get("run_dir", f"aiogym/runs/api_{scenario}_{objective_name}")))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    rows = []
    rollouts = []
    errors = []
    for name in controller_names:
        try:
            controller = _make_controller(
                str(name),
                scenario=scenario,
                config=controller_configs.get(str(name), cfg.get("controller_config", {})),
            )
            result = _evaluate_controller(
                controller,
                protocol.make_env(),
                episodes=len(seeds),
                seed=seeds[0],
                seed_list=seeds,
                protocol=protocol,
                include_episodes=include_episodes,
            )
            results.append(result)
            rows.append(_compact_row(result, scenario, protocol.objective))
            if save_rollouts:
                rollouts.append(_rollout_controller(
                    controller,
                    protocol.make_env(),
                    seed=seeds[0],
                    max_steps=rollout_steps,
                    protocol=protocol,
                ))
        except Exception as ex:
            row = {
                "scenario": scenario,
                "objective": protocol.objective,
                "controller": str(name),
                "status": "failed",
                "message": str(ex),
            }
            rows.append(row)
            errors.append({"controller": str(name), "type": ex.__class__.__name__, "message": str(ex)})

    benchmark_config = _BenchmarkConfig.from_protocol(
        protocol,
        controller=",".join(str(name) for name in controller_names),
        seeds=seeds,
    ).metadata()
    payload = {
        "schema_version": API_BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "public_api_benchmark",
        "run_dir": str(out_dir),
        "config": _jsonable(cfg),
        "benchmark_config": benchmark_config,
        "scenario": scenario,
        "objective": protocol.objective,
        "controllers": [str(name) for name in controller_names],
        "rows": rows,
        "results": results,
        "rollouts": rollouts,
        "report": _build_evaluation_report(results) if results else {},
        "errors": errors,
    }
    benchmark_path = out_dir / "benchmark.json"
    payload["artifacts"] = _write_benchmark_artifacts(out_dir, payload)
    _write_json(benchmark_path, payload)
    return payload


def _make_protocol(scenario: str, protocol: str | Mapping[str, Any] | _BenchmarkProtocol | None,
                   data: Mapping[str, Any]) -> _BenchmarkProtocol:
    if isinstance(protocol, _BenchmarkProtocol):
        return protocol
    cfg = dict(data)
    if isinstance(protocol, Mapping):
        cfg.update(protocol)
        objective = cfg.pop("objective", "tracking")
    elif protocol is None:
        objective = cfg.pop("objective", "tracking")
    else:
        objective = cfg.pop("objective", str(protocol))
    factory = {
        "economic": _BenchmarkProtocol.economic,
        "tracking": _BenchmarkProtocol.tracking,
        "robustness": _BenchmarkProtocol.robustness,
        "safety": _BenchmarkProtocol.safety,
        "kpi": _BenchmarkProtocol.kpi,
    }[objective]
    return factory(scenario, **cfg)


def _benchmark_config_dict(config: str | Path | Mapping[str, Any] | _BenchmarkConfig) -> dict[str, Any]:
    if isinstance(config, _BenchmarkConfig):
        data = config.metadata()
        data["protocol"] = config.protocol
        return data
    return _load_config(config)


def _protocol_data(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: data[key] for key in PROTOCOL_KEYS if key in data}


def _load_config(config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    with path.open() as f:
        return json.load(f)


def _as_list(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _compact_row(result: Mapping[str, Any], scenario: str, objective: str) -> dict[str, Any]:
    metric = result["metric"]
    diagnostics = dict(result.get("controller_diagnostics") or {})
    controller = dict(result.get("controller") or {})
    return {
        "scenario": scenario,
        "name": result["name"],
        "objective": objective,
        "controller": result["name"],
        "control_structure": controller.get("control_structure"),
        "status": "degraded" if result.get("controller_status") == "degraded" else "passed",
        "metric": metric,
        metric: result.get(metric),
        f"{metric}_std": result.get(f"{metric}_std"),
        "kpi": result.get("kpi"),
        "profit": result.get("profit"),
        "return": result.get("return"),
        "track": result.get("track"),
        "constraint": result.get("constraint"),
        "tracking_iae": result.get("tracking_iae"),
        "energy_kwh": result.get("energy_kwh"),
        "constraint_violation_count": result.get("constraint_violation_count"),
        "constraint_violation_severity": result.get("constraint_violation_severity"),
        "safety_margin_min": result.get("safety_margin_min"),
        "runtime_seconds_per_step": result.get("runtime_seconds_per_step"),
        "controller_fallback_count": diagnostics.get("fallback_count", 0),
        "controller_solver_failure_count": diagnostics.get("solver_failure_count", 0),
        "episodes": result.get("episodes"),
        "seed_list": result.get("seed_list"),
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(_jsonable(data), f, indent=2)
        f.write("\n")


def _jsonable(value):
    if hasattr(value, "metadata") and callable(value.metadata):
        return _jsonable(value.metadata())
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
