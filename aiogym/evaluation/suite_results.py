"""Benchmark-suite summary and artifact-path preparation."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .._internal.paths import run_path


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
    return str(run_path(f"bench_suite_{safe_suite}_{run_id or artifact_run_id()}_artifacts"))


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
