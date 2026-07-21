#!/usr/bin/env python3
"""Run named AIO-Gym benchmark suites."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from aiogym._internal.config import parse_seed_list
from aiogym.evaluation import (
    BenchmarkCase,
    build_evaluation_report,
    primary_metric_for_objective,
    resolve_protocol,
)
from aiogym.evaluation.artifacts import finalize_benchmark_artifacts
from aiogym.evaluation.runner import execute_benchmark_case
from aiogym.models import SCENARIOS


CONFIG_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "suites"
SCENARIO_ALIASES = {
    "ALL_SCENARIOS": tuple(SCENARIOS),
    "CORE_SCENARIOS": ("cascade", "quadruple", "cstr", "hvac"),
    "ECONOMIC_SCENARIOS": ("cascade", "cstr", "hvac", "heater"),
}
SUMMARY_COLUMNS = (
    "suite_case",
    "scenario",
    "task",
    "task_status",
    "task_profile_hash",
    "objective",
    "objective_source",
    "objective_status",
    "action_mode",
    "controller",
    "control_structure",
    "execution_status",
    "metric",
    "metric_mean",
    "metric_std",
    "normalized_score",
    "profit",
    "production",
    "return",
    "track",
    "tracking_cost",
    "tracking_return",
    "tracking_error_cost",
    "tracking_move_cost",
    "tracking_steady_cost",
    "tracking_mse",
    "tracking_iae",
    "energy_kwh",
    "constraint",
    "constraint_violation_count",
    "constraint_violation_severity",
    "safety_margin_min",
    "runtime_seconds_per_step",
    "controller_fallback_count",
    "controller_solver_failure_count",
    "episodes",
    "seed_list",
)


def builtin_suites():
    if not CONFIG_DIR.is_dir():
        return ()
    return tuple(sorted(path.stem for path in CONFIG_DIR.glob("*.json")))


def load_suite(name_or_path: str):
    path = name_or_path
    if os.path.sep not in name_or_path and not name_or_path.endswith(".json"):
        path = str(CONFIG_DIR / f"{name_or_path}.json")
    with open(path) as f:
        suite = json.load(f)
    suite.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    suite["scenarios"] = expand_scenarios(suite.get("scenarios", ()))
    suite["objectives"] = list(suite.get("objectives", ()))
    suite["controllers"] = list(suite.get("controllers", ()))
    if "cases" in suite:
        if not isinstance(suite["cases"], list) or not suite["cases"]:
            raise ValueError("suite cases must be a non-empty list")
        if any(not isinstance(case, dict) for case in suite["cases"]):
            raise TypeError("each suite case must be a mapping")
    suite.setdefault("action_mode", "actuator")
    suite.setdefault("description", "")
    if "cases" not in suite and suite.get("task") is None:
        suite.setdefault("episode_steps", 80)
        suite.setdefault("control_dt", 0.5)
    return suite


def expand_scenarios(value):
    if isinstance(value, str):
        if value in SCENARIO_ALIASES:
            return list(SCENARIO_ALIASES[value])
        return [value]
    out = []
    for item in value:
        if item in SCENARIO_ALIASES:
            out.extend(SCENARIO_ALIASES[item])
        else:
            out.append(item)
    return out


def parse_csv(raw: str | None, default):
    if raw is None:
        return list(default)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("comma-separated options must contain at least one value")
    return values


def build_summary_table(rows: list[dict]):
    table = []
    for row in rows:
        metric = row.get("metric")
        out = {key: row.get(key) for key in SUMMARY_COLUMNS}
        out["metric_mean"] = row.get(metric) if metric else None
        out["metric_std"] = row.get(f"{metric}_std") if metric else None
        table.append(out)
    return table


def artifact_run_id(now: datetime | None = None) -> str:
    stamp = now or datetime.now(timezone.utc)
    return stamp.strftime("%Y%m%dT%H%M%S%fZ")


def artifact_dir_for(suite_name: str, artifact_dir: str | None = None, run_id: str | None = None):
    if artifact_dir:
        return artifact_dir
    safe_suite = re.sub(r"[^A-Za-z0-9_.-]+", "-", suite_name).strip("-") or "suite"
    return f"aiogym/runs/bench_suite_{safe_suite}_{run_id or artifact_run_id()}_artifacts"


def effective_suite_config(suite: dict, cases: list[dict], episode_steps: int | None,
                           control_dt: float | None):
    config = dict(suite)
    config["scenarios"] = list(dict.fromkeys(case["scenario"] for case in cases))
    config["objectives"] = list(dict.fromkeys(case["objective"] for case in cases))
    config["controllers"] = list(dict.fromkeys(case["controller"] for case in cases))
    config["tasks"] = list(dict.fromkeys(case.get("task", "default") for case in cases))
    config["cases"] = [
        {
            "name": case.get(
                "name",
                f"{case['objective']}:{case['scenario']}:{case.get('task', 'default')}:{case['controller']}",
            ),
            "scenario": case["scenario"],
            "task": case.get("task", "default"),
            "objective": case["objective"],
            "objective_source": case.get("objective_source"),
            "controller": case["controller"],
            "episode_steps": case["protocol"].episode_steps if case.get("protocol") else episode_steps,
            "control_dt": case["protocol"].control_dt if case.get("protocol") else control_dt,
        }
        for case in cases
    ]
    resolved_steps = list(dict.fromkeys(case["protocol"].episode_steps for case in cases if case.get("protocol")))
    resolved_dt = list(dict.fromkeys(case["protocol"].control_dt for case in cases if case.get("protocol")))
    config["episode_steps"] = episode_steps if episode_steps is not None else (resolved_steps[0] if len(resolved_steps) == 1 else None)
    config["control_dt"] = control_dt if control_dt is not None else (resolved_dt[0] if len(resolved_dt) == 1 else None)
    return config


def build_cases(args):
    suite = load_suite(args.suite)
    explicit_cases = bool(suite.get("cases"))
    scenario_filter = (
        set(parse_csv(args.scenarios, suite["scenarios"]))
        if explicit_cases and args.scenarios is not None else None
    )
    explicit_objectives = (
        parse_csv(args.objectives, ()) if args.objectives is not None else None
    )
    controller_filter = (
        set(parse_csv(args.controllers, suite["controllers"]))
        if explicit_cases and args.controllers is not None else None
    )
    seeds = parse_seed_list(args.seed_list, args.seed, args.episodes)
    cases = []
    declarations = suite.get("cases") or [{
        "scenarios": parse_csv(args.scenarios, suite["scenarios"]),
        "controllers": parse_csv(args.controllers, suite["controllers"]),
    }]
    for declaration in declarations:
        scenarios = expand_scenarios(declaration.get("scenarios", declaration.get("scenario", suite["scenarios"])))
        if explicit_objectives is not None:
            objective_candidates = [("explicit", value) for value in explicit_objectives]
        elif "objectives" in declaration:
            objective_candidates = [
                ("case-config", value) for value in declaration["objectives"]
            ]
        elif "objective" in declaration:
            objective_candidates = [("case-config", declaration["objective"])]
        elif suite["objectives"]:
            objective_candidates = [
                ("suite-config", value) for value in suite["objectives"]
            ]
        else:
            objective_candidates = [("task-default", None)]
        controllers = list(declaration.get("controllers", [declaration["controller"]] if "controller" in declaration else suite["controllers"]))
        for scenario in scenarios:
            if scenario_filter is not None and scenario not in scenario_filter:
                continue
            for objective_source, objective_value in objective_candidates:
                action_mode = declaration.get("action_mode", suite["action_mode"])
                protocol_config = {"action_mode": action_mode}
                declaration_environment = dict(declaration.get("environment", {}))
                suite_environment = dict(suite.get("environment", {}))
                task = declaration.get("task", suite.get("task"))
                if task is not None:
                    protocol_config["task"] = task
                for key in (
                    "dynamic", "randomize", "randomize_setpoints", "randomize_plant",
                    "plant_drift", "integral_obs", "terminate_on_runaway", "noise",
                    "noise_pct", "tracking_q_y", "tracking_r_move", "tracking_r_steady", "model_params",
                ):
                    if key in declaration:
                        protocol_config[key] = declaration[key]
                    elif key in declaration_environment:
                        protocol_config[key] = declaration_environment[key]
                    elif key in suite:
                        protocol_config[key] = suite[key]
                    elif key in suite_environment:
                        protocol_config[key] = suite_environment[key]
                if args.episode_steps is not None:
                    protocol_config["episode_steps"] = int(args.episode_steps)
                elif "episode_steps" in declaration:
                    protocol_config["episode_steps"] = int(declaration["episode_steps"])
                elif "episode_steps" in suite:
                    protocol_config["episode_steps"] = int(suite["episode_steps"])
                if args.control_dt is not None:
                    protocol_config["control_dt"] = float(args.control_dt)
                elif "control_dt" in declaration:
                    protocol_config["control_dt"] = float(declaration["control_dt"])
                elif "control_dt" in suite:
                    protocol_config["control_dt"] = float(suite["control_dt"])
                protocol = resolve_protocol(
                    scenario,
                    objective_value if objective_source == "explicit" else None,
                    protocol_config,
                    case_objective=(
                        objective_value if objective_source == "case-config" else None
                    ),
                    suite_objective=(
                        objective_value if objective_source == "suite-config" else None
                    ),
                )
                task_meta = protocol.metadata()["task_identity"]
                objective = protocol.objective
                for controller in controllers:
                    if controller_filter is not None and controller not in controller_filter:
                        continue
                    controller_config = controller_config_for(args, controller, action_mode, objective)
                    controller_config = _merge_config(controller_config, suite.get("controller_configs", {}).get(controller, {}))
                    controller_config = _merge_config(controller_config, declaration.get("controller_configs", {}).get(controller, {}))
                    case_name = f"{objective}:{scenario}:{task_meta['name']}:{controller}"
                    case_spec = BenchmarkCase.from_protocol(
                        protocol,
                        controller=controller,
                        seeds=seeds,
                        controller_config=controller_config,
                        case_id=case_name,
                    )
                    cases.append({
                        "name": case_name,
                        "scenario": scenario,
                        "task": task_meta["name"],
                        "task_status": task_meta["status"],
                        "task_profile_hash": task_meta["profile_hash"],
                        "objective": objective,
                        "objective_source": protocol.objective_source,
                        "controller": controller,
                        "action_mode": action_mode,
                        "controller_config": controller_config,
                        "protocol": protocol,
                        "case_spec": case_spec,
                        "seeds": seeds,
                    })
    if not cases:
        raise ValueError("suite filters selected no benchmark cases")
    return suite, cases


def _merge_config(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in dict(override or {}).items():
        if key == "parameters" and isinstance(value, dict):
            out[key] = {**dict(out.get(key) or {}), **value}
        else:
            out[key] = value
    return out


def controller_config_for(args, controller: str, action_mode: str, objective: str | None = None):
    if controller == "sb3":
        if not args.sb3_path:
            raise SystemExit("controller 'sb3' requires --sb3-path")
        return {
            "path": args.sb3_path,
            "algo": args.sb3_algo,
            "action_mode": action_mode,
        }
    if controller == "onnx":
        if not args.onnx_path:
            raise SystemExit("controller 'onnx' requires --onnx-path")
        return {"path": args.onnx_path, "action_mode": action_mode}
    if controller == "oracle" and objective == "tracking":
        return {"profile": "tracking", "mode": "tracking"}
    return {}


def run_case(suite_case: dict, include_tracebacks: bool):
    return execute_benchmark_case(
        suite_case["case_spec"],
        include_episodes=True,
        save_rollout=suite_case["objective"] == "tracking",
        suite_case=suite_case["name"],
        include_tracebacks=include_tracebacks,
    )


def print_case(row: dict):
    status = row["execution_status"].upper()
    prefix = (
        f"{status:7s} {row['objective']:9s} {row['scenario']:10s} "
        f"{row.get('task', 'default'):28s} {row['controller']:14s}"
    )
    if row["execution_status"] not in ("passed", "degraded"):
        print(f"{prefix} {row.get('message', '')}")
        return
    metric = row["metric"]
    value = row.get(metric, 0.0)
    std = row.get(f"{metric}_std", 0.0)
    value_text = _format_metric_value(value)
    std_text = _format_metric_value(std)
    step_ms = float(row.get("runtime_seconds_per_step") or 0.0) * 1000.0
    print(
        f"{prefix} {metric}={value_text:>9s} +/- {std_text} "
        f"score={row['normalized_score']:8.2f} profit={row['profit']:9.2f} "
        f"track={row['track']:8.2f} safety={row.get('constraint_violation_count', 0.0):6.1f} "
        f"fallback={row.get('controller_fallback_count', 0):3} "
        f"step={step_ms:7.2f}ms"
    )


def _format_metric_value(value) -> str:
    number = float(value or 0.0)
    if number != 0.0 and abs(number) < 0.01:
        return f"{number:.3e}"
    return f"{number:.2f}"


def main():
    ap = argparse.ArgumentParser(
        description="Run a benchmark suite and write a standard artifact directory."
    )
    ap.add_argument("--suite", default="core", help=f"built-in suite or JSON path; built-ins: {', '.join(builtin_suites())}")
    ap.add_argument("--scenarios", default=None, help="comma-separated override")
    ap.add_argument("--objectives", default=None, help="comma-separated override")
    ap.add_argument("--controllers", default=None, help="comma-separated override")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--episode-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--seed-list", default=None, help="comma-separated fixed seeds; overrides --seed/--episodes")
    ap.add_argument("--control-dt", type=float, default=None)
    ap.add_argument("--sb3-path", default=None)
    ap.add_argument("--sb3-algo", default="sac", choices=["sac", "ppo", "td3"])
    ap.add_argument("--onnx-path", default=None)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--fail-on-degraded", action="store_true",
                    help="exit non-zero when any controller reports fallback/degraded diagnostics")
    ap.add_argument("--artifact-dir", default=None,
                    help="standard artifact directory; defaults to a timestamped aiogym/runs/bench_suite_<suite>_<time>_artifacts")
    ap.add_argument("--tracebacks", action="store_true")
    args = ap.parse_args()

    suite, cases = build_cases(args)
    episode_steps = int(args.episode_steps) if args.episode_steps is not None else suite.get("episode_steps")
    control_dt = float(args.control_dt) if args.control_dt is not None else suite.get("control_dt")
    started = perf_counter()
    rows = []
    results = []
    configs = []
    rollouts = []
    errors = []

    for suite_case in cases:
        artifact = run_case(suite_case, include_tracebacks=args.tracebacks)
        rows.append(artifact["row"])
        print_case(artifact["row"])
        if artifact["status"] in ("passed", "degraded"):
            results.append(artifact["result"])
            configs.append(artifact["config"])
            if artifact.get("rollout") is not None:
                rollouts.append(artifact["rollout"])
            if args.fail_fast and args.fail_on_degraded and artifact["status"] == "degraded":
                break
        else:
            errors.append({
                "suite_case": suite_case["name"],
                "scenario": suite_case["scenario"],
                "task": suite_case["task"],
                "objective": suite_case["objective"],
                "controller": suite_case["controller"],
                "status": artifact["status"],
                "error": artifact["error"],
            })
            if args.fail_fast and artifact["status"] == "failed":
                break

    counts = {
        "passed": sum(1 for row in rows if row["execution_status"] == "passed"),
        "degraded": sum(1 for row in rows if row["execution_status"] == "degraded"),
        "skipped": sum(1 for row in rows if row["execution_status"] == "skipped"),
        "failed": sum(1 for row in rows if row["execution_status"] == "failed"),
        "total": len(rows),
    }
    summary_table = build_summary_table(rows)
    suite_config = effective_suite_config(suite, cases, episode_steps, control_dt)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "benchmark_suite",
        "suite": suite["name"],
        "description": suite["description"],
        "suite_config": suite_config,
        "suite_source_config": suite,
        "defaults": {
            "episodes": args.episodes,
            "episode_steps": episode_steps,
            "seed": args.seed,
            "seed_list": parse_seed_list(args.seed_list, args.seed, args.episodes),
            "control_dt": control_dt,
            "fail_on_degraded": bool(args.fail_on_degraded),
        },
        "counts": counts,
        "runtime_seconds": float(perf_counter() - started),
        "rows": rows,
        "summary_table": summary_table,
        "degraded_cases": [row for row in rows if row["execution_status"] == "degraded"],
        "configs": configs,
        "results": results,
        "rollouts": rollouts,
        "report": build_evaluation_report(results) if results else {},
        "errors": errors,
    }
    artifact_dir = artifact_dir_for(suite["name"], args.artifact_dir)
    payload["artifact_dir"] = artifact_dir
    finalize_benchmark_artifacts(artifact_dir, payload, create_plots=True)

    print(
        f"saved artifacts {artifact_dir} "
        f"passed={counts['passed']} degraded={counts['degraded']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )
    if counts["failed"]:
        raise SystemExit(1)
    if args.fail_on_degraded and counts["degraded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
