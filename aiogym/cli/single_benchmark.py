#!/usr/bin/env python3
"""Run a single controller benchmark from command-line arguments."""
from __future__ import annotations

import argparse
import re
from dataclasses import replace
from datetime import datetime, timezone

from aiogym._internal.config import parse_seed_list
from aiogym._internal.paths import run_path
from aiogym.catalog import list_scenarios
from aiogym.evaluation import (
    BenchmarkProtocol,
    PUBLIC_BENCHMARK_SCHEMA_VERSION,
    build_evaluation_report,
    finalize_benchmark_artifacts,
    metric_direction,
    primary_metric_for_objective,
    resolve_protocol,
)
from aiogym.evaluation.execution import run_evaluation_case


def parse_controllers(raw: str):
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--controllers must include at least one controller")
    return names


def parse_setpoint_vector(raw: str, option: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",")]
    except ValueError as ex:
        raise ValueError(f"{option} must be a comma-separated numeric vector") from ex
    if not values or any(part.strip() == "" for part in raw.split(",")):
        raise ValueError(f"{option} must be a comma-separated numeric vector")
    return values


def parse_setpoint_step(raw: str) -> dict:
    step_text, separator, values_text = raw.partition(":")
    if not separator:
        raise ValueError("--setpoint-step must use STEP:VALUE1,VALUE2 syntax")
    try:
        step = int(step_text)
    except ValueError as ex:
        raise ValueError("--setpoint-step STEP must be a non-negative integer") from ex
    if step < 0:
        raise ValueError("--setpoint-step STEP must be a non-negative integer")
    return {
        "at_step": step,
        "values": parse_setpoint_vector(values_text, "--setpoint-step"),
    }


def controller_specs(args, baseline_protocol: BenchmarkProtocol):
    specs = []
    for name in parse_controllers(args.controllers):
        if name in {"sb3", "onnx"}:
            path = args.sb3_path if name == "sb3" else args.onnx_path
            action_mode = args.sb3_action_mode if name == "sb3" else args.onnx_action_mode
            if not path:
                raise ValueError(f"controller '{name}' requires --{name}-path")
            policy_protocol = replace(baseline_protocol, action_mode=action_mode)
            config = {"path": path, "action_mode": action_mode}
            if name == "sb3":
                config["algo"] = args.sb3_algo
            specs.append({
                "name": name,
                "protocol": policy_protocol,
                "seed_list": parse_seed_list(args.seed_list, args.seed, args.episodes),
                "config": config,
            })
            continue
        if name == "oracle":
            mode = (
                "tracking"
                if baseline_protocol.objective == "tracking"
                else baseline_protocol.env_reward_mode
            )
            config = {"mode": mode}
            oracle_profile = args.oracle_profile or args.controller_profile
            if oracle_profile:
                config["profile"] = oracle_profile
            specs.append({
                "name": name,
                "protocol": baseline_protocol,
                "seed_list": parse_seed_list(args.seed_list, args.seed, args.oracle_episodes),
                "config": config,
            })
            continue
        profile = args.controller_profile
        config = {"profile": profile} if profile else {}
        if name == "mpc" and getattr(args, "mpc_horizon", None) is not None:
            config["P"] = args.mpc_horizon
        specs.append({
            "name": name,
            "protocol": baseline_protocol,
            "seed_list": parse_seed_list(args.seed_list, args.seed, args.episodes),
            "config": config,
        })
    return specs


def artifact_run_id(now: datetime | None = None) -> str:
    stamp = now or datetime.now(timezone.utc)
    return stamp.strftime("%Y%m%dT%H%M%S%fZ")


def artifact_dir_for(args, protocol: BenchmarkProtocol, run_id: str | None = None) -> str:
    if args.artifact_dir:
        return args.artifact_dir
    task = protocol.metadata()["task_identity"]["name"]
    identity = f"{protocol.scenario}_{task}_{protocol.objective}"
    safe_identity = re.sub(r"[^A-Za-z0-9_.-]+", "-", identity).strip("-")
    return str(run_path(f"benchmark_{safe_identity}_{run_id or artifact_run_id()}_artifacts"))


def standard_artifact_payload(payload: dict, args, protocol: BenchmarkProtocol,
                              artifact_dir: str) -> dict:
    """Add the canonical directory-artifact envelope without changing results."""

    task_identity = protocol.metadata()["task_identity"]
    config = {
        "scenario": protocol.scenario,
        "task": args.task,
        "objective": args.objective,
        "controllers": parse_controllers(args.controllers),
        "controller_profile": args.controller_profile,
        "oracle_profile": args.oracle_profile,
        "mpc_horizon": args.mpc_horizon,
        "episodes": args.episodes,
        "oracle_episodes": args.oracle_episodes,
        "episode_steps": args.episode_steps,
        "initial_setpoint": args.initial_setpoint,
        "setpoint_schedule": args.setpoint_step,
        "clear_setpoint_schedule": args.clear_setpoint_schedule,
        "seed": args.seed,
        "seed_list": args.seed_list,
        "control_dt": args.control_dt,
        "tracking_q_y": args.tracking_q_y,
        "tracking_r_move": args.tracking_r_move,
        "disturbance_obs": args.disturbance_obs,
        "previous_action_obs": args.previous_action_obs,
        "normalize_observations": args.normalize_observations,
        "tracking_error_obs": args.tracking_error_obs,
        "sb3_path": args.sb3_path,
        "sb3_algo": args.sb3_algo,
        "sb3_action_mode": args.sb3_action_mode,
        "onnx_path": args.onnx_path,
        "onnx_action_mode": args.onnx_action_mode,
        "save_rollouts": args.save_rollouts,
        "plot": args.plot,
        "rollout_steps": args.rollout_steps,
        "output_dir": artifact_dir,
    }
    benchmark_config = {
        "schema_version": PUBLIC_BENCHMARK_SCHEMA_VERSION,
        "scenario": protocol.scenario,
        "task": task_identity["name"],
        "task_status": task_identity["status"],
        "task_profile_hash": task_identity["profile_hash"],
        "objective": protocol.objective,
        "objective_source": protocol.objective_source,
        "controllers": list(payload["controllers"]),
        "protocol": protocol.metadata(),
        "cases": list(payload["configs"]),
    }
    return {
        **payload,
        "schema_version": PUBLIC_BENCHMARK_SCHEMA_VERSION,
        "benchmark": "public_benchmark",
        "run_dir": artifact_dir,
        "artifact_dir": artifact_dir,
        "task_status": task_identity["status"],
        "task_profile_hash": task_identity["profile_hash"],
        "config": config,
        "benchmark_config": benchmark_config,
    }


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Run one scenario/objective/controller benchmark from the command line."
    )
    ap.add_argument("--scenario", default="cstr", choices=list_scenarios())
    ap.add_argument(
        "--objective",
        default=None,
        choices=["economic", "tracking", "robustness", "safety", "kpi"],
        help="explicit evaluation objective; otherwise use task.default_objective",
    )
    ap.add_argument("--task", default=None, help="named scenario task profile")
    ap.add_argument("--controllers", default="pid,mpc")
    ap.add_argument("--controller-profile", default=None, help="controller tuning profile for PID/MPC baselines")
    ap.add_argument("--oracle-profile", default=None, help="Oracle-specific tuning profile")
    ap.add_argument("--mpc-horizon", type=int, default=None, help="override MPC prediction horizon P")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--oracle-episodes", type=int, default=1)
    ap.add_argument("--episode-steps", type=int, default=None)
    ap.add_argument(
        "--initial-setpoint",
        default=None,
        help="override the initial controlled-output setpoint as VALUE1,VALUE2",
    )
    ap.add_argument(
        "--setpoint-step",
        action="append",
        default=None,
        metavar="STEP:VALUE1,VALUE2",
        help="replace the task schedule with an absolute setpoint event; repeatable",
    )
    ap.add_argument(
        "--clear-setpoint-schedule",
        action="store_true",
        help="remove the task's default setpoint events",
    )
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--seed-list", default=None, help="comma-separated fixed seeds; overrides --seed/--episodes")
    ap.add_argument("--control-dt", type=float, default=None)
    ap.add_argument(
        "--tracking-q-y",
        type=float,
        default=None,
        help="override scalar tracking weight Q; task-owned when omitted",
    )
    ap.add_argument(
        "--tracking-r-move",
        type=float,
        default=None,
        help="override move weight R; task-owned when omitted",
    )
    ap.add_argument(
        "--disturbance-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include current disturbances in observations; task-owned when omitted",
    )
    ap.add_argument(
        "--previous-action-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include the previous applied action; task-owned when omitted",
    )
    ap.add_argument(
        "--normalize-observations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="scale observations with fixed physical bounds; task-owned when omitted",
    )
    ap.add_argument(
        "--tracking-error-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replace setpoints with normalized tracking errors; task-owned when omitted",
    )
    ap.add_argument("--sb3-path", default=None)
    ap.add_argument("--sb3-algo", default="sac", choices=["sac", "ppo", "td3"])
    ap.add_argument("--sb3-action-mode", default="setpoint", choices=["actuator", "setpoint"])
    ap.add_argument("--onnx-path", default=None)
    ap.add_argument("--onnx-action-mode", default="setpoint", choices=["actuator", "setpoint"])
    ap.add_argument(
        "--artifact-dir",
        default=None,
        help="standard artifact directory; defaults to a timestamped directory under runs/",
    )
    ap.add_argument("--save-rollouts", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--rollout-steps", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        initial_setpoint = (
            parse_setpoint_vector(args.initial_setpoint, "--initial-setpoint")
            if args.initial_setpoint is not None
            else None
        )
        setpoint_schedule = (
            [parse_setpoint_step(raw) for raw in args.setpoint_step]
            if args.setpoint_step is not None
            else ([] if args.clear_setpoint_schedule else None)
        )
    except ValueError as ex:
        ap.error(str(ex))

    timing = {}
    if args.episode_steps is not None:
        timing["episode_steps"] = args.episode_steps
    elif args.task is None:
        timing["episode_steps"] = 80
    if args.control_dt is not None:
        timing["control_dt"] = args.control_dt
    elif args.task is None:
        timing["control_dt"] = 0.5
    baseline_protocol = resolve_protocol(
        args.scenario,
        args.objective,
        {
            "action_mode": "actuator",
            "tracking_q_y": args.tracking_q_y,
            "tracking_r_move": args.tracking_r_move,
            "disturbance_obs": args.disturbance_obs,
            "previous_action_obs": args.previous_action_obs,
            "normalize_observations": args.normalize_observations,
            "tracking_error_obs": args.tracking_error_obs,
            "initial_setpoint": initial_setpoint,
            "setpoint_schedule": setpoint_schedule,
            **timing,
            **({"task": args.task} if args.task else {}),
        },
    )
    specs = controller_specs(args, baseline_protocol)
    case_artifacts = [
        run_evaluation_case(
            scenario=args.scenario,
            controller=spec["name"],
            protocol=spec["protocol"],
            seeds=spec["seed_list"],
            controller_config=spec.get("config") or {},
            include_episodes=True,
            save_rollout=args.save_rollouts or args.plot,
            rollout_steps=args.rollout_steps,
        )
        for spec in specs
    ]
    results = [case["result"] for case in case_artifacts]
    rows = [case["row"] for case in case_artifacts]
    configs = [case["config"] for case in case_artifacts]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "controller_benchmark",
        "scenario": baseline_protocol.scenario,
        "task": baseline_protocol.metadata()["task_identity"]["name"],
        "objective": baseline_protocol.objective,
        "objective_source": baseline_protocol.objective_source,
        "controllers": [spec["name"] for spec in specs],
        "metric": primary_metric_for_objective(baseline_protocol.objective),
        "metric_direction": metric_direction(primary_metric_for_objective(baseline_protocol.objective)),
        "protocol": baseline_protocol.metadata(),
        "configs": configs,
        "rows": rows,
        "results": results,
        "report": build_evaluation_report(results),
    }

    rollouts = []
    if args.save_rollouts or args.plot:
        rollouts = [case["rollout"] for case in case_artifacts]
        payload["rollouts"] = rollouts

    artifact_dir = artifact_dir_for(args, baseline_protocol)
    payload = standard_artifact_payload(payload, args, baseline_protocol, artifact_dir)
    finalize_benchmark_artifacts(
        artifact_dir,
        payload,
        create_plots=True,
        markdown_report=True,
    )
    print(f"saved artifacts {artifact_dir}")
    for row in payload["rows"]:
        metric = row["metric"]
        print(
            f"{row['controller']:18s} {row['control_structure']:18s} "
            f"{metric}={row[metric]:9.2f} +/- {row[f'{metric}_std']:.2f} "
            f"score={row['normalized_score']:8.2f} profit={row['profit']:8.2f} "
            f"track={row['track']:8.2f} constraint={row['constraint']:8.2f} "
            f"safety={row.get('constraint_violation_count', 0):6.1f} "
            f"status={row.get('controller_status', 'ok'):8s} "
            f"fallback={row.get('controller_fallback_count', 0):3} "
            f"runtime={row.get('runtime_seconds', 0.0):7.3f}s "
            f"step={row.get('runtime_seconds_per_step', 0.0) * 1000.0:7.2f}ms"
        )


if __name__ == "__main__":
    main()
