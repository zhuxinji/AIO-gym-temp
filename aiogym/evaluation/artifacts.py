"""Standard benchmark artifact directory helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from aiogym._internal.serialization import jsonable as _jsonable, write_json as _write_json
from aiogym.evaluation.plots import (
    plot_constraint_timeline,
    plot_grouped_leaderboard,
    plot_learning_curve,
    plot_leaderboard,
    plot_rollouts,
    plot_summary,
    plot_tracking_comparison_table,
)
from aiogym.models import collect_model_cards


def write_benchmark_artifacts(out_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Write the standard artifact directory for an API or suite benchmark payload."""

    return _write_benchmark_artifacts(Path(out_dir), payload)


def plot_results(run_dir: str | Path) -> dict[str, str]:
    """Generate summary and rollout SVGs from a benchmark artifact directory."""

    run_path = Path(run_dir)
    benchmark_path = run_path / "benchmark.json"
    with benchmark_path.open() as f:
        payload = json.load(f)
    figures_dir = run_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _clear_comparison_figures(figures_dir)
    title = payload.get("scenario", payload.get("suite", "benchmark"))
    rows = [
        _plot_row(row)
        for row in payload.get("rows", [])
        if row.get("status") in {"passed", "degraded"}
    ]
    figures = {}
    artifact_figures = {}
    if rows:
        tracking_comparison_rows = _tracking_comparison_rows(rows)
        if tracking_comparison_rows:
            tracking_comparison_path = figures_dir / "tracking_comparison.svg"
            plot_tracking_comparison_table(tracking_comparison_rows, str(tracking_comparison_path), title)
            figures["tracking_comparison"] = str(tracking_comparison_path)
            artifact_figures["tracking_comparison_figure"] = str(tracking_comparison_path)
        groups = _rows_by_objective(rows)
        scenario_count = len({str(row.get("scenario") or "benchmark") for row in rows})
        if len(groups) == 1 and scenario_count <= 1:
            objective, objective_rows = next(iter(groups.items()))
            summary_path = figures_dir / "summary.svg"
            plot_summary(objective_rows, str(summary_path), f"{title} {objective}")
            figures["summary"] = str(summary_path)
            artifact_figures["summary_figure"] = str(summary_path)
            leaderboard_path = figures_dir / "leaderboard.svg"
            plot_leaderboard(_leaderboard(objective_rows), str(leaderboard_path), objective)
            figures["leaderboard"] = str(leaderboard_path)
            artifact_figures["leaderboard_figure"] = str(leaderboard_path)
        else:
            summary_figures = {}
            leaderboard_sections = []
            for objective, objective_rows in groups.items():
                summary_figures[objective] = {}
                for scenario, scenario_rows in _rows_by_scenario(objective_rows).items():
                    slug = f"{_slug(objective)}_{_slug(scenario)}"
                    summary_path = figures_dir / f"summary_{slug}.svg"
                    plot_summary(scenario_rows, str(summary_path), f"{title} {objective} {scenario}")
                    summary_figures[objective][scenario] = str(summary_path)
                    leaderboard_sections.append({
                        "title": f"{objective} / {scenario}",
                        "metric": scenario_rows[0].get("metric") if scenario_rows else None,
                        "board": _leaderboard(scenario_rows),
                    })
            leaderboard_path = figures_dir / "leaderboard.svg"
            plot_grouped_leaderboard(leaderboard_sections, str(leaderboard_path), title)
            figures["summary_by_scenario"] = summary_figures
            figures["leaderboard_by_scenario"] = str(leaderboard_path)
            artifact_figures["summary_figures"] = summary_figures
            artifact_figures["leaderboard_figure"] = str(leaderboard_path)
    rollouts = payload.get("rollouts") or []
    if rollouts:
        rollout_path = figures_dir / f"{title}_rollout.svg"
        plot_rollouts(rollouts, str(rollout_path), title)
        figures["rollout"] = str(rollout_path)
        artifact_figures["rollout_figure"] = str(rollout_path)
        constraint_path = figures_dir / "constraint_timeline.svg"
        plot_constraint_timeline(rollouts, str(constraint_path), title)
        figures["constraint_timeline"] = str(constraint_path)
        artifact_figures["constraint_timeline_figure"] = str(constraint_path)
    learning_curve = payload.get("learning_curve") or []
    if learning_curve:
        curve_path = figures_dir / "learning_curve.svg"
        plot_learning_curve(learning_curve, str(curve_path), title)
        figures["learning_curve"] = str(curve_path)
        artifact_figures["learning_curve_figure"] = str(curve_path)
    payload.setdefault("artifacts", {})
    for key in ("summary_figure", "leaderboard_figure", "summary_figures", "leaderboard_figures"):
        payload["artifacts"].pop(key, None)
    payload["artifacts"].update(artifact_figures)
    _write_json(benchmark_path, payload)
    return figures


def _write_benchmark_artifacts(out_dir: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    artifacts = {"benchmark": str(out_dir / "benchmark.json")}
    dirs = {
        "config": out_dir / "config",
        "metadata": out_dir / "metadata",
        "summary": out_dir / "summary",
        "results": out_dir / "results",
        "rollouts": out_dir / "rollouts",
        "figures": out_dir / "figures",
        "training": out_dir / "training",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    paths = {
        "input_config": dirs["config"] / "config.json",
        "benchmark_config": dirs["config"] / "benchmark_config.json",
        "rows": dirs["summary"] / "rows.json",
        "summary_csv": dirs["summary"] / "summary.csv",
        "leaderboard": dirs["summary"] / "leaderboard.json",
        "results": dirs["results"] / "results.json",
        "report": dirs["results"] / "report.json",
    }
    _write_json(paths["input_config"], payload.get("config", payload.get("suite_config", {})))
    _write_json(paths["benchmark_config"], payload.get("benchmark_config", payload.get("configs", {})))
    model_card_artifacts = _write_model_card_artifacts(dirs["metadata"], payload)
    _write_json(paths["rows"], payload.get("rows", []))
    rows = list(payload.get("rows", []))
    objective_groups = _rows_by_objective(rows)
    tracking_comparison_rows = _tracking_comparison_rows(rows)
    if tracking_comparison_rows:
        tracking_comparison_path = dirs["summary"] / "tracking_comparison.csv"
        _write_tracking_comparison_csv(tracking_comparison_path, tracking_comparison_rows)
        artifacts["tracking_comparison"] = str(tracking_comparison_path)
    if len(objective_groups) <= 1:
        _write_summary_csv(paths["summary_csv"], rows)
        _write_json(paths["leaderboard"], _leaderboard(rows))
    else:
        summary_csvs = {}
        leaderboards = {}
        objective_dir = dirs["summary"] / "objectives"
        objective_dir.mkdir(parents=True, exist_ok=True)
        for objective, objective_rows in objective_groups.items():
            slug = _slug(objective)
            summary_path = objective_dir / f"{slug}.csv"
            leaderboard_path = objective_dir / f"{slug}_leaderboard.json"
            _write_summary_csv(summary_path, objective_rows, _summary_columns_for_objective(objective))
            _write_json(leaderboard_path, _leaderboard(objective_rows))
            summary_csvs[objective] = str(summary_path)
            leaderboards[objective] = str(leaderboard_path)
        _write_summary_index_csv(paths["summary_csv"], objective_groups, summary_csvs, leaderboards)
        _write_json(paths["leaderboard"], {
            objective: _leaderboard(objective_rows)
            for objective, objective_rows in objective_groups.items()
        })
        all_summary_path = dirs["summary"] / "all_summary.csv"
        all_leaderboard_path = dirs["summary"] / "all_leaderboard.json"
        _write_summary_csv(all_summary_path, rows)
        _write_json(all_leaderboard_path, _leaderboard(rows))
        artifacts["summary_csvs"] = summary_csvs
        artifacts["leaderboards"] = leaderboards
        artifacts["all_summary_csv"] = str(all_summary_path)
        artifacts["all_leaderboard"] = str(all_leaderboard_path)
    _write_json(paths["results"], payload.get("results", []))
    _write_json(paths["report"], payload.get("report", {}))
    for key, path in paths.items():
        artifacts[key] = str(path)
    artifacts.update(model_card_artifacts)

    if payload.get("training"):
        training_path = dirs["training"] / "training.json"
        _write_json(training_path, payload.get("training", {}))
        artifacts["training"] = str(training_path)
    if payload.get("learning_curve"):
        curve_json = dirs["training"] / "learning_curve.json"
        curve_csv = dirs["training"] / "learning_curve.csv"
        _write_json(curve_json, payload.get("learning_curve", []))
        _write_learning_curve_csv(curve_csv, payload.get("learning_curve", []))
        artifacts["learning_curve"] = str(curve_json)
        artifacts["learning_curve_csv"] = str(curve_csv)

    if payload.get("rollouts"):
        rollout_path = dirs["rollouts"] / "rollouts.json"
        _write_json(rollout_path, payload.get("rollouts", []))
        artifacts["rollouts"] = str(rollout_path)
    return artifacts


def _clear_comparison_figures(figures_dir: Path) -> None:
    for pattern in ("summary*.svg", "leaderboard*.svg", "tracking_comparison.svg"):
        for path in figures_dir.glob(pattern):
            path.unlink()


def _write_model_card_artifacts(metadata_dir: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    scenarios = _artifact_scenarios(payload)
    if not scenarios:
        return {}
    cards = collect_model_cards(scenarios)
    artifacts = {}
    if len(cards) == 1:
        cards_dir = metadata_dir / "model_cards"
        _clear_json_files(cards_dir)
        scenario, card = next(iter(cards.items()))
        path = metadata_dir / "model_card.json"
        _write_json(path, card)
        artifacts["model_card"] = str(path)
        return artifacts
    single_path = metadata_dir / "model_card.json"
    if single_path.exists():
        single_path.unlink()
    cards_dir = metadata_dir / "model_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    _clear_json_files(cards_dir)
    artifact_root = metadata_dir.parent
    manifest = {"scenarios": list(cards), "cards": {}}
    for scenario, card in cards.items():
        path = cards_dir / f"{scenario}.json"
        _write_json(path, card)
        manifest["cards"][scenario] = str(path.relative_to(artifact_root))
    manifest_path = cards_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    artifacts["model_cards_dir"] = str(cards_dir)
    artifacts["model_cards_manifest"] = str(manifest_path)
    return artifacts


def _clear_json_files(path: Path) -> None:
    if not path.exists():
        return
    for child in path.glob("*.json"):
        child.unlink()


def _artifact_scenarios(payload: Mapping[str, Any]) -> list[str]:
    if payload.get("scenario"):
        return [str(payload["scenario"])]
    suite_config = payload.get("suite_config") or {}
    if suite_config.get("scenarios"):
        return list(dict.fromkeys(str(scenario) for scenario in suite_config["scenarios"]))
    rows = payload.get("rows") or []
    return list(dict.fromkeys(str(row["scenario"]) for row in rows if row.get("scenario")))


def _leaderboard(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("status") == "failed":
            continue
        metric = row.get("metric")
        value = row.get(metric) if metric else None
        out.append({
            "rank": 0,
            "controller": row.get("controller") or row.get("name"),
            "scenario": row.get("scenario"),
            "objective": row.get("objective"),
            "status": row.get("status"),
            "metric": metric,
            "metric_value": value,
            "kpi": row.get("kpi"),
            "profit": row.get("profit"),
            "tracking_cost": row.get("tracking_cost"),
            "tracking_return": row.get("tracking_return"),
            "tracking_error_cost": row.get("tracking_error_cost"),
            "tracking_move_cost": row.get("tracking_move_cost"),
            "tracking_mse": row.get("tracking_mse"),
            "tracking_iae": row.get("tracking_iae"),
            "constraint_violation_count": row.get("constraint_violation_count"),
            "constraint_violation_severity": row.get("constraint_violation_severity"),
        })
    out.sort(key=lambda row: (
        row["status"] not in {"passed", "degraded"},
        _sort_value(row["metric"], row["metric_value"]),
    ))
    for i, row in enumerate(out, 1):
        row["rank"] = i
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


def _rows_by_scenario(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        scenario = str(row.get("scenario") or "benchmark")
        groups.setdefault(scenario, []).append(row)
    return groups


def _tracking_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tracking_rows = [
        row for row in rows
        if row.get("objective") == "tracking" and row.get("status") in {"passed", "degraded"}
    ]
    if not tracking_rows:
        return []
    scenarios = list(dict.fromkeys(str(row.get("scenario") or "benchmark") for row in tracking_rows))
    controllers = list(dict.fromkeys(str(row.get("controller") or "controller") for row in tracking_rows))
    out = []
    for scenario in scenarios:
        scenario_rows = [row for row in tracking_rows if str(row.get("scenario") or "benchmark") == scenario]
        values = {
            str(row.get("controller") or "controller"): _float_or_none(row.get("tracking_cost"))
            for row in scenario_rows
        }
        runtime_ms = {
            str(row.get("controller") or "controller"): _runtime_step_ms(row)
            for row in scenario_rows
        }
        ranked = [(controller, value) for controller, value in values.items() if value is not None]
        if not ranked:
            continue
        best_controller, best_value = min(ranked, key=lambda item: item[1])
        row = {
            "scenario": scenario,
            "best_controller": best_controller,
            "best_tracking_cost": best_value,
            "best_step_ms": runtime_ms.get(best_controller),
        }
        oracle_value = values.get("NMPC-oracle")
        row["oracle_gap_vs_best"] = None if oracle_value is None else oracle_value - best_value
        for controller in controllers:
            row[f"{controller}_tracking_cost"] = values.get(controller)
            row[f"{controller}_step_ms"] = runtime_ms.get(controller)
        out.append(row)
    return out


def _write_tracking_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base = ["scenario", "best_controller", "best_tracking_cost", "best_step_ms", "oracle_gap_vs_best"]
    controllers = []
    for row in rows:
        for key in row:
            if key.endswith("_tracking_cost") and key not in {"best_tracking_cost"}:
                controllers.append(key[: -len("_tracking_cost")])
    controllers = list(dict.fromkeys(controllers))
    columns = base + [column for controller in controllers for column in (f"{controller}_tracking_cost", f"{controller}_step_ms")]
    _write_summary_csv(path, rows, columns)


def _runtime_step_ms(row: Mapping[str, Any]):
    value = _float_or_none(row.get("runtime_seconds_per_step"))
    return None if value is None else value * 1000.0


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
    "suite_case", "scenario", "objective", "action_mode", "controller",
    "control_structure", "status", "metric", "kpi", "profit", "production",
    "return", "track", "tracking_cost", "tracking_return", "tracking_error_cost",
    "tracking_move_cost", "tracking_mse", "tracking_iae", "energy_kwh", "constraint",
    "constraint_violation_count", "constraint_violation_severity",
    "safety_margin_min", "runtime_seconds_per_step", "episodes", "seed_list",
]


OBJECTIVE_SUMMARY_COLUMNS = {
    "tracking": [
        "suite_case", "scenario", "controller", "control_structure", "status",
        "metric", "tracking_cost", "tracking_return", "tracking_error_cost",
        "tracking_move_cost", "tracking_mse", "tracking_iae", "track", "kpi", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity",
        "runtime_seconds_per_step", "episodes", "seed_list",
    ],
    "economic": [
        "suite_case", "scenario", "controller", "control_structure", "status",
        "metric", "profit", "production", "energy_kwh", "kpi",
        "constraint", "constraint_violation_count", "constraint_violation_severity",
        "safety_margin_min", "runtime_seconds_per_step", "episodes", "seed_list",
    ],
    "safety": [
        "suite_case", "scenario", "controller", "control_structure", "status",
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
        "step", "timesteps", "phase", "metric", "metric_value", "kpi", "profit",
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
    out.setdefault("name", out.get("controller", out.get("suite_case", "case")))
    for key in ("profit", "kpi", "track", "constraint"):
        out.setdefault(key, 0.0)
    return out
