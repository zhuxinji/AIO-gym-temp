"""Artifact row grouping, leaderboard, and CSV preparation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def _artifact_scenarios(payload: Mapping[str, Any]) -> list[str]:
    if payload.get("scenario"):
        return [str(payload["scenario"])]
    suite_config = payload.get("suite_config") or {}
    if suite_config.get("scenarios"):
        return list(dict.fromkeys(str(scenario) for scenario in suite_config["scenarios"]))
    rows = payload.get("rows") or []
    return list(dict.fromkeys(str(row["scenario"]) for row in rows if row.get("scenario")))


def _leaderboard(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("execution_status") == "failed":
            continue
        metric = row.get("metric")
        value = row.get(metric) if metric else None
        item = {
            "rank": 0,
            "controller": row.get("controller"),
            "scenario": row.get("scenario"),
            "task": row.get("task", "default"),
            "task_status": row.get("task_status", "implicit-default"),
            "task_profile_hash": row.get("task_profile_hash"),
            "objective": row.get("objective"),
            "objective_source": row.get("objective_source"),
            "objective_status": row.get("objective_status", "not-defined"),
            "execution_status": row.get("execution_status"),
            "metric": metric,
            "metric_value": value,
            "normalized_score": row.get("normalized_score"),
            "profit": row.get("profit"),
            "tracking_cost": row.get("tracking_cost"),
            "tracking_return": row.get("tracking_return"),
            "tracking_error_cost": row.get("tracking_error_cost"),
            "tracking_move_cost": row.get("tracking_move_cost"),
            "tracking_mse": row.get("tracking_mse"),
            "tracking_iae": row.get("tracking_iae"),
            "constraint_violation_count": row.get("constraint_violation_count"),
            "constraint_violation_severity": row.get("constraint_violation_severity"),
        }
        key = (
            str(item.get("scenario") or "benchmark"),
            str(item.get("task") or "default"),
            str(item.get("objective") or "benchmark"),
        )
        groups.setdefault(key, []).append(item)
    out = []
    for group in groups.values():
        group.sort(key=lambda item: (
            item["execution_status"] not in {"passed", "degraded"},
            _sort_value(item["metric"], item["metric_value"]),
        ))
        for i, item in enumerate(group, 1):
            item["rank"] = i
            out.append(item)
    return out

def _sort_value(metric: str | None, value):
    if value is None:
        return float("inf")
    if metric in {
        "tracking_cost", "tracking_error_cost", "tracking_move_cost",
        "tracking_mse", "tracking_iae", "tracking_ise", "tracking_itae", "tracking_overshoot",
        "tracking_settling_time", "constraint_violation_count",
        "constraint_violation_severity", "action_violation_count",
        "action_violation_severity", "runaway_count", "runaway_duration",
    }:
        return float(value)
    return -float(value)


def _rows_by_objective(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        objective = str(row.get("objective") or "benchmark")
        groups.setdefault(objective, []).append(row)
    return groups


def _benchmark_case_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario") or "benchmark"),
        str(row.get("task") or "default"),
    )


def _benchmark_case_label(row: Mapping[str, Any]) -> str:
    scenario, task = _benchmark_case_key(row)
    return scenario if task == "default" else f"{scenario} / {task}"


def _rows_by_benchmark_case(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = _benchmark_case_label(row)
        groups.setdefault(label, []).append(row)
    return groups


def _tracking_rollout_groups(rollouts: Sequence[Mapping[str, Any]]):
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for rollout in rollouts:
        if rollout.get("objective") != "tracking" or not rollout.get("scenario"):
            continue
        key = (
            str(rollout["scenario"]),
            str(rollout.get("task") or "default"),
        )
        groups.setdefault(key, []).append(rollout)
    return groups


def _tracking_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tracking_rows = [
        row for row in rows
        if row.get("objective") == "tracking" and row.get("execution_status") in {"passed", "degraded"}
    ]
    if not tracking_rows:
        return []
    benchmark_cases = list(dict.fromkeys(_benchmark_case_key(row) for row in tracking_rows))
    controllers = list(dict.fromkeys(str(row.get("controller") or "controller") for row in tracking_rows))
    out = []
    for scenario, task in benchmark_cases:
        scenario_rows = [row for row in tracking_rows if _benchmark_case_key(row) == (scenario, task)]
        values = {
            str(row.get("controller") or "controller"): _float_or_none(row.get("tracking_cost"))
            for row in scenario_rows
        }
        runtime_seconds = {
            str(row.get("controller") or "controller"): _runtime_total_seconds(row)
            for row in scenario_rows
        }
        ranked = [(controller, value) for controller, value in values.items() if value is not None]
        if not ranked:
            continue
        best_controller, best_value = min(ranked, key=lambda item: item[1])
        row = {
            "scenario": scenario,
            "task": task,
            "best_controller": best_controller,
            "best_tracking_cost": best_value,
            "best_runtime_total_seconds": runtime_seconds.get(best_controller),
        }
        oracle_value = values.get("NMPC-oracle")
        row["oracle_gap_vs_best"] = None if oracle_value is None else oracle_value - best_value
        for controller in controllers:
            row[f"{controller}_tracking_cost"] = values.get(controller)
            row[f"{controller}_runtime_total_seconds"] = runtime_seconds.get(controller)
        out.append(row)
    return out


def _write_tracking_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base = ["scenario", "task", "best_controller", "best_tracking_cost", "best_runtime_total_seconds", "oracle_gap_vs_best"]
    controllers = []
    for row in rows:
        for key in row:
            if key.endswith("_tracking_cost") and key not in {"best_tracking_cost"}:
                controllers.append(key[: -len("_tracking_cost")])
    controllers = list(dict.fromkeys(controllers))
    columns = base + [column for controller in controllers for column in (f"{controller}_tracking_cost", f"{controller}_runtime_total_seconds")]
    _write_summary_csv(path, rows, columns)


def _runtime_total_seconds(row: Mapping[str, Any]):
    return _float_or_none(row.get("runtime_total_seconds"))


def _float_or_none(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "objective"


def _write_summary_index_csv(path: Path, groups: Mapping[str, Sequence[Mapping[str, Any]]],
                             summary_csvs: Mapping[str, str], leaderboards: Mapping[str, str]) -> None:
    columns = ["objective", "rows", "metrics", "summary_csv", "leaderboard"]
    with path.open("w") as f:
        f.write(",".join(columns) + "\n")
        for objective, rows in groups.items():
            metrics = list(dict.fromkeys(str(row.get("metric", "")) for row in rows if row.get("metric")))
            row = {
                "objective": objective,
                "rows": len(rows),
                "metrics": metrics,
                "summary_csv": summary_csvs[objective],
                "leaderboard": leaderboards[objective],
            }
            f.write(",".join(_csv_cell(row.get(column)) for column in columns) + "\n")


FULL_SUMMARY_COLUMNS = [
    "suite_case", "scenario", "task", "task_status", "task_profile_hash",
    "objective", "objective_source", "objective_status", "action_mode", "controller",
    "control_structure", "execution_status", "metric", "normalized_score", "profit", "production",
    "return", "track", "tracking_cost", "tracking_return", "tracking_error_cost",
    "tracking_move_cost", "tracking_mse", "tracking_iae", "energy_kwh", "constraint",
    "constraint_violation_count", "constraint_violation_severity",
    "safety_margin_min",
    "runtime_seconds_per_step", "episodes", "seed_list",
]


OBJECTIVE_SUMMARY_COLUMNS = {
    "tracking": [
        "suite_case", "scenario", "task", "task_status", "task_profile_hash",
        "objective_source", "objective_status", "controller", "control_structure",
        "execution_status",
        "metric", "tracking_cost", "tracking_return", "tracking_error_cost",
        "tracking_move_cost", "tracking_mse", "tracking_iae", "track", "normalized_score", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity",
        "runtime_seconds_per_step", "episodes", "seed_list",
    ],
    "economic": [
        "suite_case", "scenario", "task", "task_status", "task_profile_hash",
        "objective_source", "objective_status", "controller", "control_structure",
        "execution_status",
        "metric", "profit", "production", "energy_kwh", "normalized_score",
        "constraint", "constraint_violation_count", "constraint_violation_severity",
        "safety_margin_min", "runtime_seconds_per_step", "episodes", "seed_list",
    ],
    "safety": [
        "suite_case", "scenario", "task", "task_status", "task_profile_hash",
        "objective_source", "objective_status", "controller", "control_structure",
        "execution_status",
        "metric", "constraint_violation_count", "constraint_violation_duration",
        "constraint_violation_severity", "action_violation_count",
        "action_violation_severity", "runaway_count", "safety_margin_min",
        "runtime_seconds_per_step", "episodes", "seed_list",
    ],
}


def _summary_columns_for_objective(objective: str) -> list[str]:
    return list(OBJECTIVE_SUMMARY_COLUMNS.get(objective, FULL_SUMMARY_COLUMNS))


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]],
                       columns: Sequence[str] = FULL_SUMMARY_COLUMNS) -> None:
    with path.open("w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(_csv_cell(row.get(column)) for column in columns) + "\n")


def _write_learning_curve_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    preferred = [
        "step", "timesteps", "phase", "metric", "metric_value", "normalized_score", "profit",
        "return", "track", "tracking_cost", "tracking_return", "tracking_error_cost",
        "tracking_move_cost", "tracking_mse", "tracking_iae", "constraint_violation_count",
        "constraint_violation_severity", "runtime_total_seconds",
    ]
    keys = list(dict.fromkeys(
        [key for key in preferred if any(key in row for row in rows)]
        + sorted({key for row in rows for key in row}.difference(preferred))
    ))
    with path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(_csv_cell(row.get(column)) for column in keys) + "\n")


def _csv_cell(value) -> str:
    if value is None:
        return ""
    text = json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict, tuple)) else str(value)
    if any(ch in text for ch in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def _plot_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("profit", "normalized_score", "track", "constraint"):
        out.setdefault(key, 0.0)
    return out
