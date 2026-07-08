#!/usr/bin/env python3
"""Run named AIO-Gym controller benchmark suites."""
from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from aiogym.controllers import make_controller
from aiogym.evaluation import (
    BenchmarkConfig,
    BenchmarkProtocol,
    build_evaluation_report,
    evaluate_controller,
    primary_metric_for_objective,
)
from aiogym.models import SCENARIOS


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "benchmarks"
SCENARIO_ALIASES = {
    "ALL_SCENARIOS": tuple(SCENARIOS),
    "CORE_SCENARIOS": ("cascade", "quadruple", "cstr", "hvac"),
}
SUMMARY_COLUMNS = (
    "suite_case",
    "scenario",
    "objective",
    "action_mode",
    "controller",
    "control_structure",
    "status",
    "metric",
    "metric_mean",
    "metric_std",
    "kpi",
    "profit",
    "return",
    "track",
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


def protocol_factory(objective: str):
    return {
        "economic": BenchmarkProtocol.economic,
        "tracking": BenchmarkProtocol.tracking,
        "robustness": BenchmarkProtocol.robustness,
        "safety": BenchmarkProtocol.safety,
        "kpi": BenchmarkProtocol.kpi,
    }[objective]


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
    suite.setdefault("action_mode", "actuator")
    suite.setdefault("description", "")
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


def seed_list(raw: str | None, seed: int, episodes: int):
    if not raw:
        return [seed + i for i in range(episodes)]
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seed-list must contain at least one integer seed")
    return seeds


def compact_row(result: dict, suite_case: dict):
    controller = result["controller"]
    metric = result["metric"]
    diagnostics = dict(result.get("controller_diagnostics") or {})
    controller_status = result.get("controller_status", "ok")
    status = "degraded" if controller_status == "degraded" else "passed"
    row = {
        "suite_case": suite_case["name"],
        "scenario": suite_case["scenario"],
        "objective": suite_case["objective"],
        "controller": result["name"],
        "control_structure": controller.get("control_structure"),
        "action_mode": suite_case["action_mode"],
        "status": status,
        "controller_status": controller_status,
        "controller_solve_count": diagnostics.get("solve_count", 0),
        "controller_solver_success_count": diagnostics.get("solver_success_count", 0),
        "controller_solver_failure_count": diagnostics.get("solver_failure_count", 0),
        "controller_fallback_count": diagnostics.get("fallback_count", 0),
        "controller_last_solver_error": diagnostics.get("last_solver_error"),
        "metric": metric,
        metric: result[metric],
        f"{metric}_std": result[f"{metric}_std"],
        "kpi": result["kpi"],
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
    return row


def skipped_row(suite_case: dict, status: str, message: str):
    return {
        "suite_case": suite_case["name"],
        "scenario": suite_case["scenario"],
        "objective": suite_case["objective"],
        "controller": suite_case["controller"],
        "control_structure": None,
        "action_mode": suite_case["action_mode"],
        "status": status,
        "metric": primary_metric_for_objective(suite_case["objective"]),
        "message": message,
        "episodes": 0,
        "seed": suite_case["seeds"][0] if suite_case["seeds"] else None,
        "seed_list": suite_case["seeds"],
    }


def build_summary_table(rows: list[dict]):
    table = []
    for row in rows:
        metric = row.get("metric")
        out = {key: row.get(key) for key in SUMMARY_COLUMNS}
        out["metric_mean"] = row.get(metric) if metric else None
        out["metric_std"] = row.get(f"{metric}_std") if metric else None
        table.append(out)
    return table


def summary_paths(out_path: str, summary_out: str | None = None):
    if summary_out:
        base, ext = os.path.splitext(summary_out)
        if ext.lower() in (".csv", ".md", ".json"):
            return {ext.lower().lstrip("."): summary_out}
        return {"csv": f"{summary_out}.csv", "md": f"{summary_out}.md"}
    base, _ = os.path.splitext(out_path)
    return {"csv": f"{base}_summary.csv", "md": f"{base}_summary.md"}


def write_summary_files(table: list[dict], paths: dict[str, str]):
    written = {}
    if "csv" in paths:
        _write_summary_csv(table, paths["csv"])
        written["csv"] = paths["csv"]
    if "md" in paths:
        _write_summary_markdown(table, paths["md"])
        written["markdown"] = paths["md"]
    if "json" in paths:
        _write_json(paths["json"], table)
        written["json"] = paths["json"]
    return written


def _write_summary_csv(table: list[dict], path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        for row in table:
            writer.writerow({key: _cell_value(row.get(key)) for key in SUMMARY_COLUMNS})


def _write_summary_markdown(table: list[dict], path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    columns = list(SUMMARY_COLUMNS)
    with open(path, "w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in table:
            f.write("| " + " | ".join(_escape_markdown(_cell_value(row.get(key))) for key in columns) + " |\n")


def _write_json(path: str, data):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _cell_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _escape_markdown(value: str):
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def build_cases(args):
    suite = load_suite(args.suite)
    scenarios = parse_csv(args.scenarios, suite["scenarios"])
    objectives = parse_csv(args.objectives, suite["objectives"])
    controllers = parse_csv(args.controllers, suite["controllers"])
    seeds = seed_list(args.seed_list, args.seed, args.episodes)
    episode_steps = int(args.episode_steps if args.episode_steps is not None else suite.get("episode_steps", 80))
    control_dt = float(args.control_dt if args.control_dt is not None else suite.get("control_dt", 0.5))
    cases = []
    for objective in objectives:
        for scenario in scenarios:
            protocol = protocol_factory(objective)(
                scenario,
                action_mode=suite["action_mode"],
                episode_steps=episode_steps,
                control_dt=control_dt,
            )
            for controller in controllers:
                controller_config = controller_config_for(args, controller, suite["action_mode"])
                cases.append({
                    "name": f"{objective}:{scenario}:{controller}",
                    "scenario": scenario,
                    "objective": objective,
                    "controller": controller,
                    "action_mode": suite["action_mode"],
                    "controller_config": controller_config,
                    "protocol": protocol,
                    "seeds": seeds,
                })
    return suite, cases


def controller_config_for(args, controller: str, action_mode: str):
    if controller != "sb3":
        return {}
    if not args.sb3_path:
        raise SystemExit("controller 'sb3' requires --sb3-path")
    return {
        "path": args.sb3_path,
        "algo": args.sb3_algo,
        "action_mode": action_mode,
    }


def run_case(suite_case: dict, include_tracebacks: bool):
    started = perf_counter()
    try:
        controller = make_controller(
            suite_case["controller"],
            scenario=suite_case["scenario"],
            config=suite_case.get("controller_config") or {},
        )
        result = evaluate_controller(
            controller,
            suite_case["protocol"].make_env(),
            episodes=len(suite_case["seeds"]),
            seed=suite_case["seeds"][0],
            seed_list=suite_case["seeds"],
            protocol=suite_case["protocol"],
            include_episodes=True,
        )
    except (KeyError, ValueError) as ex:
        row = skipped_row(suite_case, "skipped", str(ex))
        return {"status": "skipped", "row": row, "error": _error_payload(ex, include_tracebacks)}
    except Exception as ex:
        row = skipped_row(suite_case, "failed", str(ex))
        return {"status": "failed", "row": row, "error": _error_payload(ex, include_tracebacks)}

    row = compact_row(result, suite_case)
    row["suite_runtime_seconds"] = float(perf_counter() - started)
    config = BenchmarkConfig.from_protocol(
        suite_case["protocol"],
        controller=suite_case["controller"],
        seeds=suite_case["seeds"],
        controller_config=suite_case.get("controller_config") or {},
    ).metadata()
    return {"status": row["status"], "row": row, "result": result, "config": config}


def _error_payload(ex: Exception, include_tracebacks: bool):
    data = {"type": ex.__class__.__name__, "message": str(ex)}
    if include_tracebacks:
        data["traceback"] = traceback.format_exc()
    return data


def print_case(row: dict):
    status = row["status"].upper()
    prefix = f"{status:7s} {row['objective']:9s} {row['scenario']:10s} {row['controller']:14s}"
    if row["status"] not in ("passed", "degraded"):
        print(f"{prefix} {row.get('message', '')}")
        return
    metric = row["metric"]
    value = row.get(metric, 0.0)
    std = row.get(f"{metric}_std", 0.0)
    step_ms = float(row.get("runtime_seconds_per_step") or 0.0) * 1000.0
    print(
        f"{prefix} {metric}={value:9.2f} +/- {std:.2f} "
        f"kpi={row['kpi']:8.2f} profit={row['profit']:9.2f} "
        f"track={row['track']:8.2f} safety={row.get('constraint_violation_count', 0.0):6.1f} "
        f"fallback={row.get('controller_fallback_count', 0):3} "
        f"step={step_ms:7.2f}ms"
    )


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--fail-on-degraded", action="store_true",
                    help="exit non-zero when any controller reports fallback/degraded diagnostics")
    ap.add_argument("--summary-out", default=None,
                    help="summary table path or basename; defaults to <out>_summary.csv and <out>_summary.md")
    ap.add_argument("--tracebacks", action="store_true")
    args = ap.parse_args()

    suite, cases = build_cases(args)
    episode_steps = int(args.episode_steps if args.episode_steps is not None else suite.get("episode_steps", 80))
    control_dt = float(args.control_dt if args.control_dt is not None else suite.get("control_dt", 0.5))
    out_path = args.out or f"aiogym/runs/bench_suite_{args.suite}.json"
    started = perf_counter()
    rows = []
    results = []
    configs = []
    errors = []

    for suite_case in cases:
        artifact = run_case(suite_case, include_tracebacks=args.tracebacks)
        rows.append(artifact["row"])
        print_case(artifact["row"])
        if artifact["status"] in ("passed", "degraded"):
            results.append(artifact["result"])
            configs.append(artifact["config"])
            if args.fail_fast and args.fail_on_degraded and artifact["status"] == "degraded":
                break
        else:
            errors.append({
                "suite_case": suite_case["name"],
                "scenario": suite_case["scenario"],
                "objective": suite_case["objective"],
                "controller": suite_case["controller"],
                "status": artifact["status"],
                "error": artifact["error"],
            })
            if args.fail_fast and artifact["status"] == "failed":
                break

    counts = {
        "passed": sum(1 for row in rows if row["status"] == "passed"),
        "degraded": sum(1 for row in rows if row["status"] == "degraded"),
        "skipped": sum(1 for row in rows if row["status"] == "skipped"),
        "failed": sum(1 for row in rows if row["status"] == "failed"),
        "total": len(rows),
    }
    summary_table = build_summary_table(rows)
    summary_files = write_summary_files(summary_table, summary_paths(out_path, args.summary_out))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "benchmark_suite",
        "suite": suite["name"],
        "description": suite["description"],
        "suite_config": suite,
        "defaults": {
            "episodes": args.episodes,
            "episode_steps": episode_steps,
            "seed": args.seed,
            "seed_list": seed_list(args.seed_list, args.seed, args.episodes),
            "control_dt": control_dt,
            "fail_on_degraded": bool(args.fail_on_degraded),
        },
        "counts": counts,
        "runtime_seconds": float(perf_counter() - started),
        "rows": rows,
        "summary_table": summary_table,
        "summary_files": summary_files,
        "degraded_cases": [row for row in rows if row["status"] == "degraded"],
        "configs": configs,
        "results": results,
        "report": build_evaluation_report(results) if results else {},
        "errors": errors,
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"saved {out_path} "
        f"passed={counts['passed']} degraded={counts['degraded']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )
    for kind, path in summary_files.items():
        print(f"saved {kind} summary {path}")
    if counts["failed"]:
        raise SystemExit(1)
    if args.fail_on_degraded and counts["degraded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
