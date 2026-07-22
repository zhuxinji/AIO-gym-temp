"""Standard benchmark artifact file writers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .._internal.identifiers import canonicalize_artifact_ids
from .._internal.serialization import write_json as _write_json
from ..models import collect_model_cards
from .artifact_tables import (
    _artifact_scenarios,
    _leaderboard,
    _rows_by_objective,
    _slug,
    _summary_columns_for_objective,
    _tracking_comparison_rows,
    _write_learning_curve_csv,
    _write_summary_csv,
    _write_summary_index_csv,
    _write_tracking_comparison_csv,
)


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
        _write_json(path, canonicalize_artifact_ids(card))
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
        _write_json(path, canonicalize_artifact_ids(card))
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
