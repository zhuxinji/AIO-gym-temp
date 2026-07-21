#!/usr/bin/env python3
"""Grid-search task-specific quadruple-tank NMPC Oracle parameters."""
from __future__ import annotations

import argparse
import itertools
import json
from time import perf_counter

from aiogym.evaluation import resolve_protocol
from aiogym.evaluation.runner import run_evaluation_case


TASK_PROFILES = {
    "minimum-phase-classic": "quadruple-minimum-phase",
    "nonminimum-phase-classic": "quadruple-nonminimum-phase",
    "pminus-reference-step": "quadruple-minimum-phase",
    "pplus-reference-step": "quadruple-nonminimum-phase",
    "zero-boundary-stress": "quadruple-zero-boundary",
}


def _csv(raw: str, cast):
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_PROFILES))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--horizons", default="1", help="comma-separated integers")
    parser.add_argument("--solve-every", default="1", help="comma-separated integers")
    parser.add_argument("--terminal-weights", default="0", help="comma-separated floats")
    parser.add_argument("--move-weights", default="0", help="comma-separated floats")
    parser.add_argument("--transcriptions", default="multiple_shooting")
    parser.add_argument("--preview-setpoints", action="store_true")
    parser.add_argument("--ipopt-max-iter", type=int, default=200)
    parser.add_argument("--ipopt-tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--episode-steps", type=int, default=None)
    args = parser.parse_args()

    protocol = resolve_protocol(
        "quadruple",
        "tracking",
        {
            "task": args.task,
            "action_mode": "actuator",
            **({"episode_steps": args.episode_steps} if args.episode_steps else {}),
        },
    )
    profile = args.profile or TASK_PROFILES[args.task]
    grid = itertools.product(
        _csv(args.horizons, int),
        _csv(args.solve_every, int),
        _csv(args.terminal_weights, float),
        _csv(args.move_weights, float),
        _csv(args.transcriptions, str),
    )
    rows = []
    for horizon, solve_every, terminal, move, transcription in grid:
        parameters = {
            "horizon": horizon,
            "solve_every": solve_every,
            "terminal_weight": terminal,
            "r_move": move,
            "transcription": transcription,
            "ipopt_max_iter": args.ipopt_max_iter,
            "ipopt_tol": args.ipopt_tol,
            "warm_start": True,
            "preview_setpoints": args.preview_setpoints,
        }
        started = perf_counter()
        case = run_evaluation_case(
            scenario="quadruple",
            controller="oracle",
            protocol=protocol,
            seeds=[args.seed],
            controller_config={"profile": profile, "mode": "tracking", "parameters": parameters},
            include_episodes=False,
        )
        row = case["row"]
        elapsed = perf_counter() - started
        result = {
            **parameters,
            "task": args.task,
            "tracking_error_cost": row.get("tracking_error_cost"),
            "tracking_cost": row.get("tracking_cost"),
            "constraint_violation_count": row.get("constraint_violation_count"),
            "controller_fallback_count": row.get("controller_fallback_count"),
            "controller_solver_failure_count": row.get("controller_solver_failure_count"),
            "runtime_seconds_per_step": row.get("runtime_seconds_per_step"),
            "wall_seconds": elapsed,
        }
        rows.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    feasible = [
        row for row in rows
        if not row["constraint_violation_count"]
        and not row["controller_fallback_count"]
        and not row["controller_solver_failure_count"]
    ]
    ranked = feasible or rows
    best = min(ranked, key=lambda row: float(row["tracking_error_cost"]))
    print(json.dumps({"best": best, "candidate_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
