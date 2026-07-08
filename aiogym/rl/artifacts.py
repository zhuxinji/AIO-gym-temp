"""Helpers for publishing RL training runs as standard benchmark artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from aiogym.evaluation import build_evaluation_report, plot_results, render_benchmark_report
from aiogym.evaluation.artifacts import write_benchmark_artifacts


RL_ARTIFACT_SCHEMA_VERSION = "aiogym.rl_training_artifact.v1"


def result_row(result: Mapping[str, Any], scenario: str, action_mode: str,
               controller: str | None = None, suite_case: str | None = None) -> dict[str, Any]:
    """Return a summary row compatible with benchmark-suite artifacts."""

    metric = str(result.get("metric") or "")
    row = {
        "suite_case": suite_case or f"{result.get('objective', '')}:{scenario}:{controller or result.get('name', '')}",
        "scenario": scenario,
        "objective": result.get("objective"),
        "action_mode": action_mode,
        "controller": controller or result.get("name"),
        "control_structure": dict(result.get("controller", {})).get("control_structure"),
        "status": "degraded" if result.get("controller_status") == "degraded" else "passed",
        "metric": metric,
        "episodes": result.get("episodes"),
        "seed_list": result.get("seed_list", []),
        "runtime_seconds_per_step": result.get("runtime_seconds_per_step"),
    }
    for key in (
        metric,
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
        "normalized_score",
    ):
        if key and key in result:
            row[key] = result[key]
    return row


def learning_curve_point(step: int, result: Mapping[str, Any], phase: str = "eval") -> dict[str, Any]:
    """Condense an evaluation result into one training-history row."""

    metric = str(result.get("metric") or "")
    row = {
        "step": int(step),
        "phase": phase,
        "metric": metric,
        "metric_value": result.get(metric) if metric else None,
        "metric_direction": result.get("metric_direction"),
        "episodes": result.get("episodes"),
        "runtime_total_seconds": result.get("runtime_total_seconds"),
    }
    for key in (
        "kpi",
        "profit",
        "return",
        "track",
        "tracking_iae",
        "constraint_violation_count",
        "constraint_violation_severity",
        "safety_margin_min",
    ):
        if key in result:
            row[key] = result[key]
    return _jsonable(row)


def rl_payload(kind: str, scenario: str, objective: str, action_mode: str,
               training: Mapping[str, Any], protocol: Mapping[str, Any],
               results: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
               learning_curve: Sequence[Mapping[str, Any]] | None = None,
               rollouts: Sequence[Mapping[str, Any]] | None = None,
               extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a standard benchmark payload for an RL training run."""

    payload = {
        "schema_version": RL_ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "rl_training",
        "kind": kind,
        "scenario": scenario,
        "objective": objective,
        "action_mode": action_mode,
        "config": dict(training),
        "benchmark_config": dict(protocol),
        "training": dict(training),
        "evaluation_protocol": dict(protocol),
        "rows": list(rows),
        "results": list(results),
        "report": build_evaluation_report(results),
        "learning_curve": list(learning_curve or []),
        "rollouts": list(rollouts or []),
    }
    if extra:
        payload.update(dict(extra))
    return _jsonable(payload)


def write_rl_artifacts(artifact_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write benchmark.json, standard children, figures, and Markdown report."""

    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(_jsonable(payload))
    payload["artifact_dir"] = str(root)
    payload["artifacts"] = write_benchmark_artifacts(root, payload)
    benchmark_path = root / "benchmark.json"
    _write_json(benchmark_path, payload)
    plot_results(root)
    with benchmark_path.open() as f:
        payload = json.load(f)
    report_path = root / "report.md"
    render_benchmark_report(root, out_path=report_path)
    payload.setdefault("artifacts", {})
    payload["artifacts"]["markdown_report"] = str(report_path)
    _write_json(benchmark_path, payload)
    return payload


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(_jsonable(data), f, indent=2)
        f.write("\n")


def _jsonable(value):
    if hasattr(value, "metadata") and callable(value.metadata):
        return _jsonable(value.metadata())
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
