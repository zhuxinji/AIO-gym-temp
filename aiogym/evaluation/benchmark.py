"""Config-driven public benchmark orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .._config import as_list, load_config, protocol_data
from .._serialization import jsonable, write_json
from .protocols import (
    PUBLIC_BENCHMARK_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkProtocol,
    resolve_protocol,
)
from .core import build_evaluation_report
from .runner import run_evaluation_case


def run_benchmark(config: str | Path | Mapping[str, Any] | BenchmarkConfig) -> dict:
    """Run a config-driven controller benchmark and write reusable artifacts."""

    cfg = _benchmark_config_dict(config)
    scenario = cfg.get("scenario", cfg.get("model", "cstr"))
    config_protocol = config.protocol if isinstance(config, BenchmarkConfig) else None
    objective_spec = config_protocol if config_protocol is not None else cfg.get(
        "objective", cfg.get("protocol", "tracking")
    )
    objective_name = (
        objective_spec.get("objective", "tracking")
        if isinstance(objective_spec, Mapping)
        else str(objective_spec)
    )
    if isinstance(objective_spec, BenchmarkProtocol):
        objective_name = objective_spec.objective
    protocol_options = protocol_data(cfg)
    protocol_options.update(dict(cfg.get("env", {})))
    protocol_options.update(dict(cfg.get("protocol_kwargs", {})))
    protocol_options["objective"] = objective_name
    protocol = resolve_protocol(scenario, objective=objective_spec, data=protocol_options)
    controller_names = as_list(cfg.get("controllers", cfg.get("controller", "pid")))
    if not controller_names:
        raise ValueError("benchmark must include at least one controller")
    controller_configs = dict(cfg.get("controller_configs", {}))
    seeds = tuple(int(seed) for seed in cfg.get("seeds", cfg.get("seed_list", [cfg.get("seed", 0)])))
    if not seeds:
        raise ValueError("benchmark must include at least one seed")
    strict = bool(cfg.get("strict", cfg.get("raise_on_error", False)))
    include_episodes = bool(cfg.get("include_episodes", True))
    save_rollouts = bool(cfg.get("save_rollouts", False))
    rollout_steps = cfg.get("rollout_steps")
    out_dir = Path(cfg.get(
        "output_dir",
        cfg.get("run_dir", f"aiogym/runs/benchmark_{scenario}_{objective_name}"),
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    rows = []
    rollouts = []
    errors = []
    for name in controller_names:
        try:
            case = run_evaluation_case(
                scenario=scenario,
                controller=str(name),
                protocol=protocol,
                seeds=seeds,
                controller_config=controller_configs.get(str(name), cfg.get("controller_config", {})),
                include_episodes=include_episodes,
                save_rollout=save_rollouts,
                rollout_steps=rollout_steps,
            )
            result = case["result"]
            results.append(result)
            rows.append(case["row"])
            if case["rollout"] is not None:
                rollouts.append(case["rollout"])
        except Exception as ex:
            if strict:
                raise RuntimeError(f"controller '{name}' benchmark failed: {ex}") from ex
            rows.append({
                "scenario": scenario,
                "objective": protocol.objective,
                "controller": str(name),
                "status": "failed",
                "message": str(ex),
            })
            errors.append({
                "controller": str(name),
                "type": ex.__class__.__name__,
                "message": str(ex),
            })

    benchmark_config = BenchmarkConfig.from_protocol(
        protocol,
        controller=",".join(str(name) for name in controller_names),
        seeds=seeds,
    ).metadata()
    payload = {
        "schema_version": PUBLIC_BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "public_benchmark",
        "run_dir": str(out_dir),
        "config": jsonable(cfg),
        "benchmark_config": benchmark_config,
        "scenario": scenario,
        "objective": protocol.objective,
        "controllers": [str(name) for name in controller_names],
        "rows": rows,
        "results": results,
        "rollouts": rollouts,
        "report": build_evaluation_report(results) if results else {},
        "errors": errors,
    }
    benchmark_path = out_dir / "benchmark.json"
    from .artifacts import write_benchmark_artifacts

    payload["artifacts"] = write_benchmark_artifacts(out_dir, payload)
    write_json(benchmark_path, payload)
    return payload


def _benchmark_config_dict(
    config: str | Path | Mapping[str, Any] | BenchmarkConfig,
) -> dict[str, Any]:
    if isinstance(config, BenchmarkConfig):
        data = config.metadata()
        data["protocol"] = config.protocol
        return data
    return load_config(config)
