"""Standard benchmark artifact directory helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from aiogym.evaluation.plots import (
    plot_constraint_timeline,
    plot_learning_curve,
    plot_leaderboard,
    plot_rollouts,
    plot_summary,
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
    title = payload.get("scenario", payload.get("suite", "benchmark"))
    rows = [_plot_row(row) for row in payload.get("rows", []) if row.get("status") != "failed"]
    figures = {}
    artifact_figures = {}
    if rows:
        summary_path = figures_dir / "summary.svg"
        plot_summary(rows, str(summary_path), title)
        figures["summary"] = str(summary_path)
        artifact_figures["summary_figure"] = str(summary_path)
        leaderboard_path = figures_dir / "leaderboard.svg"
        plot_leaderboard(_leaderboard(rows), str(leaderboard_path), payload.get("objective", "benchmark"))
        figures["leaderboard"] = str(leaderboard_path)
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
    _write_summary_csv(paths["summary_csv"], payload.get("rows", []))
    _write_json(paths["leaderboard"], _leaderboard(payload.get("rows", [])))
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
    manifest = {"scenarios": list(cards), "cards": {}}
    for scenario, card in cards.items():
        path = cards_dir / f"{scenario}.json"
        _write_json(path, card)
        manifest["cards"][scenario] = str(path)
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
        "tracking_iae", "tracking_ise", "tracking_itae", "tracking_overshoot",
        "tracking_settling_time", "constraint_violation_count",
        "constraint_violation_severity", "action_violation_count",
        "action_violation_severity", "runaway_count", "runaway_duration",
    }:
        return float(value)
    return -float(value)


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "suite_case", "scenario", "objective", "action_mode", "controller",
        "control_structure", "status", "metric", "kpi", "profit", "return", "track", "tracking_iae",
        "energy_kwh", "constraint", "constraint_violation_count",
        "constraint_violation_severity", "safety_margin_min",
        "runtime_seconds_per_step", "episodes", "seed_list",
    ]
    with path.open("w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(_csv_cell(row.get(column)) for column in columns) + "\n")


def _write_learning_curve_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    preferred = [
        "step", "timesteps", "phase", "metric", "metric_value", "kpi", "profit",
        "return", "track", "tracking_iae", "constraint_violation_count",
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


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(_jsonable(data), f, indent=2)
        f.write("\n")


def _jsonable(value):
    if hasattr(value, "metadata") and callable(value.metadata):
        return _jsonable(value.metadata())
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
