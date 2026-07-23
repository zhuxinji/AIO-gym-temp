#!/usr/bin/env python3
"""Run named AIO-Gym benchmark suites."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from aiogym._internal.config import parse_seed_list
from aiogym.catalog import list_suites
from aiogym.evaluation import build_evaluation_report
from aiogym.evaluation.artifact import finalize_benchmark_artifacts
from aiogym.evaluation.execution import execute_benchmark_case
from aiogym.evaluation import suite as suite_pipeline


CONFIG_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "suites"
PRESET_DIR = CONFIG_DIR / "presets"


def _sync_suite_paths():
    suite_pipeline.CONFIG_DIR = CONFIG_DIR
    suite_pipeline.PRESET_DIR = PRESET_DIR


def load_suite(name_or_path: str):
    _sync_suite_paths()
    return suite_pipeline.load_suite(name_or_path)


def build_cases(args):
    return suite_pipeline.build_cases(args, load_suite_fn=load_suite)


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


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Run a benchmark suite and write a standard artifact directory."
    )
    ap.add_argument("--suite", default="core", help=f"built-in suite or JSON path; built-ins: {', '.join(list_suites())}")
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
                    help="standard artifact directory; defaults to a timestamped runs/bench_suite_<suite>_<time>_artifacts")
    ap.add_argument("--tracebacks", action="store_true")
    args = ap.parse_args(argv)

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
    summary_table = suite_pipeline.build_summary_table(rows)
    suite_config = suite_pipeline.effective_suite_config(
        suite, cases, episode_steps, control_dt
    )
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
    artifact_dir = suite_pipeline.artifact_dir_for(suite["name"], args.artifact_dir)
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
