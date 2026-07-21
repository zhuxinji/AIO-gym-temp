"""Standard benchmark artifact directory helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from aiogym._internal.serialization import write_json as _write_json
from aiogym.evaluation.plots import (
    plot_constraint_timeline,
    plot_grouped_leaderboard,
    plot_learning_curve,
    plot_leaderboard,
    plot_rollouts,
    plot_summary,
    plot_tracking_control,
    plot_tracking_comparison_table,
)
from aiogym.models import collect_model_cards


def write_benchmark_artifacts(out_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Write the standard artifact directory for an API or suite benchmark payload."""

    return _write_benchmark_artifacts(Path(out_dir), payload)


def finalize_benchmark_artifacts(
    out_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    create_plots: bool = False,
    markdown_report: bool = False,
) -> dict[str, Any]:
    """Write one canonical benchmark payload and its optional derived outputs."""

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["artifacts"] = write_benchmark_artifacts(root, data)
    benchmark_path = root / "benchmark.json"
    _write_json(benchmark_path, data)
    if create_plots:
        plot_results(root)
        with benchmark_path.open() as stream:
            data = json.load(stream)
    if markdown_report:
        from .reports import render_benchmark_report

        report_path = root / "report.md"
        render_benchmark_report(root, out_path=report_path)
        data.setdefault("artifacts", {})
        data["artifacts"]["markdown_report"] = str(report_path)
        _write_json(benchmark_path, data)
    return data


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
        if row.get("execution_status") in {"passed", "degraded"}
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
        benchmark_case_count = len({_benchmark_case_key(row) for row in rows})
        if len(groups) == 1 and benchmark_case_count <= 1:
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
                for benchmark_case, scenario_rows in _rows_by_benchmark_case(objective_rows).items():
                    slug = f"{_slug(objective)}_{_slug(benchmark_case)}"
                    summary_path = figures_dir / f"summary_{slug}.svg"
                    plot_summary(scenario_rows, str(summary_path), f"{title} {objective} {benchmark_case}")
                    summary_figures[objective][benchmark_case] = str(summary_path)
                    case_title = benchmark_case
                    repeated_scenario = f"{title} / "
                    if case_title.startswith(repeated_scenario):
                        case_title = case_title[len(repeated_scenario):]
                    leaderboard_sections.append({
                        "title": f"{case_title} / {objective}",
                        "metric": scenario_rows[0].get("metric") if scenario_rows else None,
                        "board": _leaderboard(scenario_rows),
                    })
            leaderboard_path = figures_dir / "leaderboard.svg"
            plot_grouped_leaderboard(leaderboard_sections, str(leaderboard_path), title)
            figures["summary_by_scenario"] = summary_figures
            figures["leaderboard_by_scenario"] = str(leaderboard_path)
            artifact_figures["summary_figures"] = summary_figures
            artifact_figures["leaderboard_figure"] = str(leaderboard_path)
    rollouts = list(payload.get("rollouts") or [])
    tracking_rollout_groups = _tracking_rollout_groups(rollouts)
    if tracking_rollout_groups:
        control_figures = {}
        for (scenario, task), case_rollouts in tracking_rollout_groups.items():
            label = scenario if task == "default" else f"{scenario} / {task}"
            control_path = figures_dir / f"tracking_control_{_slug(label)}.svg"
            plot_tracking_control(case_rollouts, str(control_path), scenario, task)
            control_figures[label] = str(control_path)
        figures["tracking_control_by_scenario"] = control_figures
        artifact_figures["tracking_control_figures"] = control_figures
    elif rollouts:
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
    for key in (
        "summary_figure", "leaderboard_figure", "summary_figures", "leaderboard_figures",
        "tracking_control_figures",
    ):
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
    for pattern in (
        "summary*.svg", "leaderboard*.svg", "tracking_comparison.svg",
        "tracking_control_*.svg",
    ):
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
            "tracking_steady_cost": row.get("tracking_steady_cost"),
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
        "tracking_cost", "tracking_error_cost", "tracking_move_cost", "tracking_steady_cost",
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
            str(row.get("controller") or "controller"): _float_or_none(row.get("tracking_error_cost"))
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
            "best_tracking_error_cost": best_value,
            "best_runtime_total_seconds": runtime_seconds.get(best_controller),
        }
        oracle_value = values.get("NMPC-oracle")
        row["oracle_gap_vs_best"] = None if oracle_value is None else oracle_value - best_value
        for controller in controllers:
            row[f"{controller}_tracking_error_cost"] = values.get(controller)
            row[f"{controller}_runtime_total_seconds"] = runtime_seconds.get(controller)
        out.append(row)
    return out


def _write_tracking_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base = ["scenario", "task", "best_controller", "best_tracking_error_cost", "best_runtime_total_seconds", "oracle_gap_vs_best"]
    controllers = []
    for row in rows:
        for key in row:
            if key.endswith("_tracking_error_cost") and key not in {"best_tracking_error_cost"}:
                controllers.append(key[: -len("_tracking_error_cost")])
    controllers = list(dict.fromkeys(controllers))
    columns = base + [column for controller in controllers for column in (f"{controller}_tracking_error_cost", f"{controller}_runtime_total_seconds")]
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
    "tracking_move_cost", "tracking_steady_cost", "tracking_mse", "tracking_iae", "energy_kwh", "constraint",
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
        "tracking_move_cost", "tracking_steady_cost", "tracking_mse", "tracking_iae", "track", "normalized_score", "energy_kwh",
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
        "tracking_move_cost", "tracking_steady_cost", "tracking_mse", "tracking_iae", "constraint_violation_count",
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
