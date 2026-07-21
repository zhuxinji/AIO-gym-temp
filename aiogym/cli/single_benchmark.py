#!/usr/bin/env python3
"""Run a single controller benchmark from command-line arguments."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from datetime import datetime, timezone

from aiogym._internal.config import parse_seed_list
from aiogym._internal.serialization import write_json
from aiogym.evaluation import (
    BenchmarkProtocol,
    build_evaluation_report,
    metric_direction,
    primary_metric_for_objective,
    resolve_protocol,
    plot_rollouts,
    plot_summary,
)
from aiogym.evaluation.runner import run_evaluation_case
from aiogym.models import SCENARIOS


def parse_controllers(raw: str):
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--controllers must include at least one controller")
    return names


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
            specs.append({
                "name": name,
                "protocol": baseline_protocol,
                "seed_list": parse_seed_list(args.seed_list, args.seed, args.oracle_episodes),
                "config": {"mode": mode},
            })
            continue
        specs.append({
            "name": name,
            "protocol": baseline_protocol,
            "seed_list": parse_seed_list(args.seed_list, args.seed, args.episodes),
            "config": {"profile": args.controller_profile} if args.controller_profile else {},
        })
    return specs


def figure_paths(out_path: str, scenario: str):
    base, _ = os.path.splitext(out_path)
    return {
        "summary": f"{base}_summary.svg",
        "rollout": f"{base}_{scenario}_rollout.svg",
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run one scenario/objective/controller benchmark from the command line."
    )
    ap.add_argument("--scenario", default="cstr", choices=SCENARIOS)
    ap.add_argument(
        "--objective",
        default=None,
        choices=["economic", "tracking", "robustness", "safety", "kpi"],
        help="explicit evaluation objective; otherwise use task.default_objective",
    )
    ap.add_argument("--task", default=None, help="named scenario task profile")
    ap.add_argument("--controllers", default="pid,mpc")
    ap.add_argument("--controller-profile", default=None, help="controller tuning profile for PID/MPC baselines")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--oracle-episodes", type=int, default=1)
    ap.add_argument("--episode-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--seed-list", default=None, help="comma-separated fixed seeds; overrides --seed/--episodes")
    ap.add_argument("--control-dt", type=float, default=None)
    ap.add_argument("--sb3-path", default=None)
    ap.add_argument("--sb3-algo", default="sac", choices=["sac", "ppo", "td3"])
    ap.add_argument("--sb3-action-mode", default="setpoint", choices=["actuator", "setpoint"])
    ap.add_argument("--onnx-path", default=None)
    ap.add_argument("--onnx-action-mode", default="setpoint", choices=["actuator", "setpoint"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-rollouts", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--rollout-steps", type=int, default=None)
    args = ap.parse_args()

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
            **timing,
            **({"task": args.task} if args.task else {}),
        },
    )
    out_path = args.out or f"aiogym/runs/bench_{args.scenario}_controllers.json"
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
        "scenario": args.scenario,
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

    if args.plot:
        figures = figure_paths(out_path, args.scenario)
        plot_summary(rows, figures["summary"], args.scenario)
        plot_rollouts(rollouts, figures["rollout"], args.scenario)
        payload["figures"] = figures

    write_json(out_path, payload)

    print(f"saved {out_path}")
    if args.plot:
        for kind, path in payload["figures"].items():
            print(f"saved {kind} figure {path}")
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
