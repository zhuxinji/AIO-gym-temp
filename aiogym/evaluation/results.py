"""Evaluation schemas, compact rows, aggregation, and task acceptance."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .metric_catalog import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_DEFINITIONS,
    METRIC_DIRECTIONS,
    PROTOCOL_METRICS,
    ROLLOUT_SCHEMA,
)
from .metrics.robustness import robustness_extrema
from ..models.tasks import validate_task_profile


def result_schema():
    return {
        "version": EVALUATION_SCHEMA_VERSION,
        "rollout": dict(ROLLOUT_SCHEMA),
        "episode_metrics": dict(METRIC_DEFINITIONS),
        "tables": {
            "tracking": list(PROTOCOL_METRICS["tracking"]),
            "economic": list(PROTOCOL_METRICS["economic"]),
            "safety": list(PROTOCOL_METRICS["safety"]),
            "robustness": list(PROTOCOL_METRICS["robustness"]),
        },
    }

def build_evaluation_report(results: Sequence[Mapping[str, Any]]):
    """Return roadmap-style tracking/economic/safety/robustness tables."""

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "tracking": [_table_row(result, PROTOCOL_METRICS["tracking"]) for result in results],
        "economic": [_table_row(result, PROTOCOL_METRICS["economic"]) for result in results],
        "safety": [_table_row(result, PROTOCOL_METRICS["safety"]) for result in results],
        "robustness": [_robustness_row(result) for result in results],
        "metric_definitions": dict(METRIC_DEFINITIONS),
    }

def _aggregate_metric_keys(per_episode):
    keys = set()
    for row in per_episode:
        for key, value in row.items():
            if key in ("episode", "seed", "steps"):
                continue
            if isinstance(value, (int, float, np.number)):
                keys.add(key)
    return sorted(keys)

def _table_row(result: Mapping[str, Any], keys: Sequence[str]):
    row = {
        "controller": result.get("controller_name"),
        "objective": result.get("objective"),
        "task": result.get("task", "default"),
        "task_status": result.get("task_status", "implicit-default"),
        "task_profile_hash": result.get("task_profile_hash"),
        "control_structure": dict(result.get("controller", {})).get("control_structure"),
        "controller_status": result.get("controller_status", "ok"),
        "controller_diagnostics": result.get("controller_diagnostics", {}),
        "episodes": result.get("episodes"),
        "seed_list": result.get("seed_list", []),
    }
    for key in keys:
        if key in result:
            row[key] = result[key]
        std_key = f"{key}_std"
        if std_key in result:
            row[std_key] = result[std_key]
    return row
def _robustness_row(result: Mapping[str, Any]):
    row = _table_row(result, PROTOCOL_METRICS["robustness"])
    row.update(robustness_extrema(
        result.get("episode_metrics", []),
        PROTOCOL_METRICS["robustness"],
        METRIC_DIRECTIONS,
    ))
    return row


def compact_result_row(
    result: Mapping[str, Any],
    *,
    scenario: str | None = None,
    objective: str | None = None,
    action_mode: str | None = None,
    task: str | None = None,
    task_status: str | None = None,
    task_profile_hash: str | None = None,
    suite_case: str | None = None,
    controller: str | None = None,
) -> dict[str, Any]:
    """Normalize one evaluation result into the stable summary-row superset."""

    controller_meta = dict(result.get("controller") or {})
    diagnostics = dict(result.get("controller_diagnostics") or {})
    metric = str(result.get("metric") or "")
    name = str(
        result.get("controller_name")
        or controller_meta.get("name")
        or controller
        or ""
    )
    controller_status = str(result.get("controller_status", "ok"))
    execution_status = str(result.get(
        "execution_status",
        "degraded" if controller_status == "degraded" else "passed",
    ))
    row = {
        "suite_case": suite_case,
        "scenario": (
            scenario
            or result.get("scenario")
            or dict(result.get("model") or {}).get("scenario")
        ),
        "task": task or result.get("task") or "default",
        "task_status": task_status or result.get("task_status") or "implicit-default",
        "task_profile_hash": task_profile_hash or result.get("task_profile_hash"),
        "objective": objective or result.get("objective"),
        "objective_source": result.get("objective_source"),
        "objective_status": result.get("objective_status", "not-defined"),
        "action_mode": action_mode or controller_meta.get("action_mode"),
        "controller": name,
        "control_structure": controller_meta.get("control_structure"),
        "execution_status": execution_status,
        "controller_status": controller_status,
        "controller_solve_count": diagnostics.get("solve_count", 0),
        "controller_solver_success_count": diagnostics.get("solver_success_count", 0),
        "controller_solver_failure_count": diagnostics.get("solver_failure_count", 0),
        "controller_fallback_count": diagnostics.get("fallback_count", 0),
        "controller_last_solver_error": diagnostics.get("last_solver_error"),
        "metric": metric,
        "episodes": result.get("episodes"),
        "seed": result.get("seed"),
        "seed_list": result.get("seed_list", []),
    }
    if metric:
        row[metric] = result.get(metric)
        row[f"{metric}_std"] = result.get(f"{metric}_std")
    for key in (
        "normalized_score", "profit", "production", "return", "track",
        "constraint", "tracking_cost", "tracking_return", "tracking_error_cost",
        "tracking_move_cost", "tracking_mse",
        "tracking_iae", "energy_kwh", "runtime_seconds",
        "runtime_seconds_per_step", "runtime_total_seconds",
        "constraint_violation_count", "constraint_violation_severity",
        "safety_margin_min",
    ):
        row[key] = result.get(key)
    return row


def evaluate_task_acceptance(
    profile: Mapping[str, Any] | None,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate optional metric thresholds separately from execution status."""

    if profile is None:
        return {"status": "not-defined", "checks": []}
    validate_task_profile(profile)
    acceptance = profile.get("acceptance", {})
    thresholds = acceptance.get("metrics") if isinstance(acceptance, Mapping) else None
    if not thresholds:
        return {"status": "not-defined", "checks": []}
    checks = []
    met = True
    for metric, bounds in thresholds.items():
        value = result.get(metric)
        check = {"metric": metric, "value": value, **dict(bounds)}
        check_met = isinstance(value, (int, float))
        if check_met and "min" in bounds:
            check_met = float(value) >= float(bounds["min"])
        if check_met and "max" in bounds:
            check_met = float(value) <= float(bounds["max"])
        check["met"] = bool(check_met)
        checks.append(check)
        met = met and bool(check_met)
    return {"status": "met" if met else "not-met", "checks": checks}
