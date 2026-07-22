"""Stable facade for benchmark artifact writing and derived outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .._internal.identifiers import canonicalize_artifact_ids
from .._internal.serialization import write_json as _write_json
from .artifact_plotting import _clear_comparison_figures, plot_results
from .artifact_tables import (
    FULL_SUMMARY_COLUMNS,
    OBJECTIVE_SUMMARY_COLUMNS,
    _artifact_scenarios,
    _benchmark_case_key,
    _benchmark_case_label,
    _csv_cell,
    _float_or_none,
    _leaderboard,
    _plot_row,
    _rows_by_benchmark_case,
    _rows_by_objective,
    _runtime_total_seconds,
    _slug,
    _sort_value,
    _summary_columns_for_objective,
    _tracking_comparison_rows,
    _tracking_rollout_groups,
    _write_learning_curve_csv,
    _write_summary_csv,
    _write_summary_index_csv,
    _write_tracking_comparison_csv,
)
from .artifact_writers import (
    _clear_json_files,
    _write_benchmark_artifacts,
    _write_model_card_artifacts,
)


def write_benchmark_artifacts(out_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Write the standard artifact directory for an API or suite benchmark payload."""

    return _write_benchmark_artifacts(
        Path(out_dir), canonicalize_artifact_ids(dict(payload))
    )


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
    data = canonicalize_artifact_ids(dict(payload))
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
