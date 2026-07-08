#!/usr/bin/env python3
"""Run a single controller benchmark from command-line arguments."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from aiogym.controllers import make_controller
from aiogym.evaluation import (
    BenchmarkConfig,
    BenchmarkProtocol,
    build_evaluation_report,
    evaluate_controller,
    metric_direction,
    primary_metric_for_objective,
    rollout_controller,
    plot_rollouts,
    plot_summary,
)
from aiogym.models import SCENARIOS


def protocol_factory(objective: str):
    return {
        "economic": BenchmarkProtocol.economic,
        "tracking": BenchmarkProtocol.tracking,
        "robustness": BenchmarkProtocol.robustness,
        "safety": BenchmarkProtocol.safety,
        "kpi": BenchmarkProtocol.kpi,
    }[objective]


def parse_controllers(raw: str):
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--controllers must include at least one controller")
    return names


def parse_seed_list(raw: str | None, seed: int, episodes: int):
    if not raw:
        return [seed + i for i in range(episodes)]
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seed-list must contain at least one integer seed")
    return seeds


def compact_row(result: dict):
    controller = result["controller"]
    metric = result["metric"]
    diagnostics = dict(result.get("controller_diagnostics") or {})
    return {
        "name": result["name"],
        "control_structure": controller.get("control_structure"),
        "controller_status": result.get("controller_status", "ok"),
        "controller_solve_count": diagnostics.get("solve_count", 0),
        "controller_solver_success_count": diagnostics.get("solver_success_count", 0),
        "controller_solver_failure_count": diagnostics.get("solver_failure_count", 0),
        "controller_fallback_count": diagnostics.get("fallback_count", 0),
        "controller_last_solver_error": diagnostics.get("last_solver_error"),
        "metric": metric,
        metric: result[metric],
        f"{metric}_std": result[f"{metric}_std"],
        "kpi": result["kpi"],
        "normalized_score": result["normalized_score"],
        "profit": result["profit"],
        "return": result["return"],
        "track": result["track"],
        "constraint": result["constraint"],
        "tracking_iae": result.get("tracking_iae"),
        "energy_kwh": result.get("energy_kwh"),
        "runtime_seconds": result.get("runtime_seconds"),
        "runtime_seconds_per_step": result.get("runtime_seconds_per_step"),
        "runtime_total_seconds": result.get("runtime_total_seconds"),
        "constraint_violation_count": result.get("constraint_violation_count"),
        "constraint_violation_severity": result.get("constraint_violation_severity"),
        "safety_margin_min": result.get("safety_margin_min"),
        "episodes": result["episodes"],
        "seed": result["seed"],
        "seed_list": result["seed_list"],
    }


def controller_specs(args, baseline_protocol: BenchmarkProtocol):
    specs = []
    for name in parse_controllers(args.controllers):
        if name == "sb3":
            if not args.sb3_path:
                raise ValueError("controller 'sb3' requires --sb3-path")
            sb3_protocol = protocol_factory(args.objective)(
                args.scenario,
                action_mode=args.sb3_action_mode,
                episode_steps=args.episode_steps,
                control_dt=args.control_dt,
            )
            specs.append({
                "name": "sb3",
                "protocol": sb3_protocol,
                "seed_list": parse_seed_list(args.seed_list, args.seed, args.episodes),
                "config": {
                    "path": args.sb3_path,
                    "algo": args.sb3_algo,
                    "action_mode": args.sb3_action_mode,
                },
            })
            continue
        if name in ("oracle", "nmpc"):
            mode = "track" if args.objective == "tracking" else baseline_protocol.env_reward_mode
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
            "config": {},
        })
    return specs


def run_spec(spec: dict, scenario: str):
    controller = make_controller(spec["name"], scenario=scenario, config=spec.get("config") or {})
    seeds = spec["seed_list"]
    return evaluate_controller(
        controller,
        spec["protocol"].make_env(),
        episodes=len(seeds),
        seed=seeds[0],
        seed_list=seeds,
        protocol=spec["protocol"],
        include_episodes=True,
    )


def rollout_spec(spec: dict, scenario: str, max_steps: int | None = None):
    controller = make_controller(spec["name"], scenario=scenario, config=spec.get("config") or {})
    protocol = spec["protocol"]
    return rollout_controller(
        controller,
        protocol.make_env(),
        seed=spec["seed_list"][0],
        max_steps=max_steps,
        protocol=protocol,
    )


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
    ap.add_argument("--objective", default="economic", choices=["economic", "tracking", "robustness", "safety", "kpi"])
    ap.add_argument("--controllers", default="pid,mpc")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--oracle-episodes", type=int, default=1)
    ap.add_argument("--episode-steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--seed-list", default=None, help="comma-separated fixed seeds; overrides --seed/--episodes")
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--sb3-path", default=None)
    ap.add_argument("--sb3-algo", default="sac", choices=["sac", "ppo", "td3"])
    ap.add_argument("--sb3-action-mode", default="setpoint", choices=["actuator", "setpoint"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-rollouts", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--rollout-steps", type=int, default=None)
    args = ap.parse_args()

    make_protocol = protocol_factory(args.objective)
    baseline_protocol = make_protocol(
        args.scenario,
        action_mode="actuator",
        episode_steps=args.episode_steps,
        control_dt=args.control_dt,
    )
    out_path = args.out or f"aiogym/runs/bench_{args.scenario}_controllers.json"
    specs = controller_specs(args, baseline_protocol)
    results = [run_spec(spec, args.scenario) for spec in specs]
    rows = [compact_row(r) for r in results]
    configs = [
        BenchmarkConfig.from_protocol(
            spec["protocol"],
            controller=spec["name"],
            seeds=spec["seed_list"],
            controller_config=spec.get("config") or {},
        ).metadata()
        for spec in specs
    ]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "controller_benchmark",
        "scenario": args.scenario,
        "objective": args.objective,
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
        rollouts = [rollout_spec(spec, args.scenario, max_steps=args.rollout_steps) for spec in specs]
        payload["rollouts"] = rollouts

    if args.plot:
        figures = figure_paths(out_path, args.scenario)
        plot_summary(rows, figures["summary"], args.scenario)
        plot_rollouts(rollouts, figures["rollout"], args.scenario)
        payload["figures"] = figures

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"saved {out_path}")
    if args.plot:
        for kind, path in payload["figures"].items():
            print(f"saved {kind} figure {path}")
    for row in payload["rows"]:
        metric = row["metric"]
        print(
            f"{row['name']:18s} {row['control_structure']:18s} "
            f"{metric}={row[metric]:9.2f} +/- {row[f'{metric}_std']:.2f} "
            f"kpi={row['kpi']:8.2f} profit={row['profit']:8.2f} "
            f"track={row['track']:8.2f} constraint={row['constraint']:8.2f} "
            f"safety={row.get('constraint_violation_count', 0):6.1f} "
            f"status={row.get('controller_status', 'ok'):8s} "
            f"fallback={row.get('controller_fallback_count', 0):3} "
            f"runtime={row.get('runtime_seconds', 0.0):7.3f}s "
            f"step={row.get('runtime_seconds_per_step', 0.0) * 1000.0:7.2f}ms"
        )


if __name__ == "__main__":
    main()
