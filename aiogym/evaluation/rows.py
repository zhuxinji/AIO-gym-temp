"""Canonical compact rows shared by API, CLI, suite, and training artifacts."""
from __future__ import annotations

from typing import Any, Mapping


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
    name = str(result.get("controller_name") or controller_meta.get("name") or controller or "")
    controller_status = str(result.get("controller_status", "ok"))
    execution_status = str(
        result.get(
            "execution_status",
            "degraded" if controller_status == "degraded" else "passed",
        )
    )
    row = {
        "suite_case": suite_case,
        "scenario": scenario or result.get("scenario") or dict(result.get("model") or {}).get("scenario"),
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
        "tracking_move_cost", "tracking_steady_cost", "tracking_mse", "tracking_iae", "energy_kwh",
        "runtime_seconds", "runtime_seconds_per_step", "runtime_total_seconds",
        "constraint_violation_count", "constraint_violation_severity", "safety_margin_min",
    ):
        row[key] = result.get(key)
    return row
