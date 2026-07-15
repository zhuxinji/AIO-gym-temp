"""Evaluation report generation for benchmark artifact directories."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_SCHEMA_VERSION = "aiogym.offline_report.v1"
ARTIFACT_CHECK_SCHEMA_VERSION = "aiogym.artifact_check.v1"


def render_benchmark_report(artifact_dir: str | Path, out_path: str | Path | None = None) -> str:
    """Render a Markdown report from a standard benchmark artifact directory."""

    root = Path(artifact_dir)
    benchmark = _read_json(root / "benchmark.json")
    artifacts = dict(benchmark.get("artifacts") or {})
    leaderboard = _read_json(_artifact_path(root, artifacts, "leaderboard", "summary/leaderboard.json"), default=[])
    objective_report = _read_json(_artifact_path(root, artifacts, "report", "results/report.json"), default={})
    summary_path = _artifact_path(root, artifacts, "all_summary_csv", "summary/all_summary.csv")
    if not summary_path.exists():
        summary_path = _artifact_path(root, artifacts, "summary_csv", "summary/summary.csv")
    summary_rows = _read_csv(summary_path)
    tracking_comparison = _read_csv(_artifact_path(root, artifacts, "tracking_comparison", "summary/tracking_comparison.csv"))
    model_manifest = _read_model_manifest(root, artifacts)

    title = benchmark.get("suite") or benchmark.get("scenario") or benchmark.get("benchmark", "benchmark")
    lines = [
        f"# AIO-Gym Benchmark Report: {title}",
        "",
        f"Report schema: `{REPORT_SCHEMA_VERSION}`",
        f"Artifact directory: `{root}`",
        "",
        "## Run Summary",
        "",
    ]
    lines.extend(_markdown_table(["Field", "Value"], _summary_rows(benchmark, summary_rows, model_manifest)))
    lines.extend(["", "## Scenario Coverage", ""])
    lines.extend(_scenario_section(benchmark, model_manifest))
    if tracking_comparison:
        lines.extend(["", "## Tracking Comparison", ""])
        lines.extend(_tracking_comparison_section(tracking_comparison))
    lines.extend(["", "## Leaderboard", ""])
    lines.extend(_leaderboard_sections(leaderboard))
    lines.extend(["", "## Objective Report", ""])
    lines.extend(_objective_section(objective_report))
    lines.extend(["", "## Stable Inputs", ""])
    lines.extend(_artifact_section(root, benchmark, artifacts))
    lines.append("")
    text = "\n".join(lines)
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


def _summary_rows(benchmark: Mapping[str, Any], summary_rows: Sequence[Mapping[str, str]],
                  model_manifest: Mapping[str, Any] | None) -> list[list[str]]:
    suite_config = benchmark.get("suite_config") or {}
    counts = benchmark.get("counts") or {}
    scenarios = _scenario_names(benchmark, model_manifest)
    objectives = suite_config.get("objectives") or sorted({row.get("objective", "") for row in summary_rows if row.get("objective")})
    controllers = suite_config.get("controllers") or sorted({row.get("controller", "") for row in summary_rows if row.get("controller")})
    rows = [
        ["Benchmark", benchmark.get("benchmark", "")],
        ["Created At", benchmark.get("created_at", "")],
        ["Suite", benchmark.get("suite", benchmark.get("scenario", ""))],
        ["Scenarios", ", ".join(str(item) for item in scenarios)],
        ["Objectives", ", ".join(str(item) for item in objectives)],
        ["Controllers", ", ".join(str(item) for item in controllers)],
        ["Rows", str(len(benchmark.get("rows") or []) or len(summary_rows))],
    ]
    if benchmark.get("learning_curve"):
        rows.append(["Learning Curve Rows", str(len(benchmark.get("learning_curve") or []))])
    if counts:
        rows.append(["Counts", ", ".join(f"{key}={value}" for key, value in counts.items())])
    defaults = benchmark.get("defaults") or {}
    if defaults:
        rows.append(["Seeds", _fmt(defaults.get("seed_list", defaults.get("seed", "")))])
        rows.append(["Episode Steps", _fmt(defaults.get("episode_steps", ""))])
    return rows


def _scenario_section(benchmark: Mapping[str, Any], model_manifest: Mapping[str, Any] | None) -> list[str]:
    scenarios = _scenario_names(benchmark, model_manifest)
    if not scenarios:
        return ["No scenario metadata was found."]
    rows = []
    cards = dict((model_manifest or {}).get("cards") or {})
    for scenario in scenarios:
        rows.append([scenario, _rel(cards.get(scenario, "")) if scenario in cards else "metadata/model_card.json"])
    return _markdown_table(["Scenario", "Model Card"], rows)


def _leaderboard_sections(leaderboard) -> list[str]:
    if isinstance(leaderboard, Mapping):
        lines = []
        for objective, rows in leaderboard.items():
            lines.extend([f"### {objective}", ""])
            lines.extend(_leaderboard_section(rows if isinstance(rows, list) else []))
            lines.append("")
        return lines[:-1] if lines else ["No leaderboard rows were found."]
    return _leaderboard_section(leaderboard)


def _tracking_comparison_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = ["scenario", "task", "best_controller", "best_tracking_cost", "best_step_ms", "oracle_gap_vs_best"]
    controller_columns = [
        key for key in rows[0].keys()
        if key.endswith("_tracking_cost") or key.endswith("_step_ms")
    ] if rows else []
    columns = preferred + [key for key in controller_columns if key not in preferred]
    table_rows = []
    for row in rows:
        table_rows.append([
            _fmt(row.get(column)) if not column.endswith(("_tracking_cost", "_step_ms")) and column != "oracle_gap_vs_best"
            else _fmt_number(row.get(column))
            for column in columns
        ])
    return _markdown_table(columns, table_rows)


def _tracking_benchmark_case_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count distinct scenario/task cases represented by successful tracking rows."""
    return len({
        (
            str(row.get("scenario") or "benchmark"),
            str(row.get("task") or "default"),
        )
        for row in rows
        if row.get("objective") == "tracking" and row.get("status") != "failed"
    })


def _leaderboard_section(leaderboard: Sequence[Mapping[str, Any]]) -> list[str]:
    if not leaderboard:
        return ["No leaderboard rows were found."]
    rows = []
    for row in leaderboard:
        rows.append([
            _fmt(row.get("rank")),
            _fmt(row.get("scenario")),
            _fmt(row.get("task", "default")),
            _fmt(row.get("objective")),
            _fmt(row.get("controller")),
            _fmt(row.get("status")),
            _fmt(row.get("metric")),
            _fmt_number(row.get("metric_value")),
            _fmt_number(row.get("kpi")),
            _fmt_number(row.get("profit")),
            _fmt_number(row.get("constraint_violation_count")),
        ])
    return _markdown_table(
        ["Rank", "Scenario", "Task", "Objective", "Controller", "Status", "Metric", "Value", "KPI", "Profit", "Violations"],
        rows,
    )


def _objective_section(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return ["No objective-grouped report was found."]
    rows = []
    for objective, entries in report.items():
        if not isinstance(entries, list):
            continue
        metrics = _objective_metrics(entries)
        rows.append([str(objective), str(len(entries)), ", ".join(metrics)])
    return _markdown_table(["Objective", "Entries", "Primary Metrics"], rows) if rows else ["No objective entries were found."]


def _objective_metrics(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "tracking_cost",
        "tracking_iae",
        "profit",
        "constraint_violation_count",
        "return",
        "normalized_score",
        "energy_kwh",
    ]
    keys = set()
    for entry in entries:
        if entry.get("metric"):
            keys.add(str(entry["metric"]))
        for key in preferred:
            if key in entry:
                keys.add(key)
    return [key for key in preferred if key in keys] + sorted(keys.difference(preferred))


def _artifact_section(root: Path, benchmark: Mapping[str, Any], artifacts: Mapping[str, str]) -> list[str]:
    keys = [
        ("benchmark", "benchmark.json"),
        ("summary_csv", "summary/summary.csv"),
        ("tracking_comparison", "summary/tracking_comparison.csv"),
        ("tracking_comparison_figure", "figures/tracking_comparison.svg"),
        ("rollouts", "rollouts/rollouts.json"),
        ("leaderboard", "summary/leaderboard.json"),
        ("all_summary_csv", "summary/all_summary.csv"),
        ("all_leaderboard", "summary/all_leaderboard.json"),
        ("report", "results/report.json"),
        ("model_cards_manifest", "metadata/model_cards/manifest.json"),
        ("model_card", "metadata/model_card.json"),
        ("training", "training/training.json"),
        ("learning_curve", "training/learning_curve.json"),
        ("learning_curve_csv", "training/learning_curve.csv"),
        ("summary_figure", "figures/summary.svg"),
        ("leaderboard_figure", "figures/leaderboard.svg"),
        ("learning_curve_figure", "figures/learning_curve.svg"),
    ]
    rows = []
    for key, default in keys:
        path = _artifact_path(root, artifacts, key, default)
        if path.exists():
            rows.append([key, _rel(path)])
    rows.extend(_artifact_mapping_rows(root, artifacts, "summary_csvs"))
    rows.extend(_artifact_mapping_rows(root, artifacts, "leaderboards"))
    rows.extend(_artifact_mapping_rows(root, artifacts, "summary_figures"))
    rows.extend(_artifact_mapping_rows(root, artifacts, "tracking_control_figures"))
    rows.extend(_artifact_mapping_rows(root, artifacts, "leaderboard_figures"))
    if not rows and benchmark.get("artifacts"):
        rows = [[key, _rel(value)] for key, value in sorted(artifacts.items())]
    return _markdown_table(["Input", "Path"], rows)


def _artifact_mapping_rows(root: Path, artifacts: Mapping[str, Any], key: str) -> list[list[str]]:
    value = artifacts.get(key)
    if not isinstance(value, Mapping):
        return []
    rows = []
    for name, raw in _flatten_artifact_mapping(value):
        path = _resolve_artifact_path(root, raw, "")
        if path.exists():
            rows.append([f"{key}:{name}", _rel(path)])
    return rows


def _flatten_artifact_mapping(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows = []
    for name, raw in sorted(value.items()):
        label = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(raw, Mapping):
            rows.extend(_flatten_artifact_mapping(raw, label))
        else:
            rows.append((label, raw))
    return rows


def _scenario_names(benchmark: Mapping[str, Any], model_manifest: Mapping[str, Any] | None) -> list[str]:
    if model_manifest and model_manifest.get("scenarios"):
        return [str(item) for item in model_manifest["scenarios"]]
    if benchmark.get("scenario"):
        return [str(benchmark["scenario"])]
    suite_config = benchmark.get("suite_config") or {}
    if suite_config.get("scenarios"):
        return [str(item) for item in suite_config["scenarios"]]
    return list(dict.fromkeys(str(row["scenario"]) for row in benchmark.get("rows", []) if row.get("scenario")))


def _read_model_manifest(root: Path, artifacts: Mapping[str, str]) -> Mapping[str, Any] | None:
    path = _artifact_path(root, artifacts, "model_cards_manifest", "metadata/model_cards/manifest.json")
    if path.exists():
        return _read_json(path)
    return None


def _artifact_path(root: Path, artifacts: Mapping[str, str], key: str, default: str) -> Path:
    raw = artifacts.get(key)
    if raw:
        return _resolve_artifact_path(root, raw, default)
    return root / default


def _read_json(path: Path, default=None):
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open() as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_escape_cell(cell) for cell in row) + " |")
    return out


def _escape_cell(value) -> str:
    return _fmt(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _fmt_number(value) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.4g}"


def _rel(path) -> str:
    if not path:
        return ""
    raw = str(path)
    marker = "/metadata/"
    if marker in raw:
        return "metadata/" + raw.split(marker, 1)[1]
    marker = "/summary/"
    if marker in raw:
        return "summary/" + raw.split(marker, 1)[1]
    marker = "/results/"
    if marker in raw:
        return "results/" + raw.split(marker, 1)[1]
    marker = "/training/"
    if marker in raw:
        return "training/" + raw.split(marker, 1)[1]
    marker = "/figures/"
    if marker in raw:
        return "figures/" + raw.split(marker, 1)[1]
    return raw


def check_benchmark_artifacts(artifact_dir: str | Path) -> dict[str, Any]:
    """Check a standard benchmark artifact directory and return structured status."""

    root = Path(artifact_dir)
    checks = []
    benchmark_path = root / "benchmark.json"
    _add_exists(checks, "benchmark", benchmark_path, required=True)
    if not benchmark_path.exists():
        return _check_result(root, checks)

    try:
        benchmark = _read_json(benchmark_path)
        checks.append(_check("benchmark_json", True, str(benchmark_path), "benchmark.json is valid JSON"))
    except Exception as ex:
        checks.append(_check("benchmark_json", False, str(benchmark_path), str(ex)))
        return _check_result(root, checks)

    artifacts = dict(benchmark.get("artifacts") or {})
    rows = list(benchmark.get("rows") or [])
    expected_scenarios = _artifact_scenario_names(benchmark)
    expected_rows = len(rows)
    paths = {
        "rows": _check_artifact_path(root, artifacts, "rows", "summary/rows.json"),
        "summary_csv": _check_artifact_path(root, artifacts, "summary_csv", "summary/summary.csv"),
        "tracking_comparison": _check_artifact_path(root, artifacts, "tracking_comparison", "summary/tracking_comparison.csv"),
        "tracking_comparison_figure": _check_artifact_path(root, artifacts, "tracking_comparison_figure", "figures/tracking_comparison.svg"),
        "rollouts": _check_artifact_path(root, artifacts, "rollouts", "rollouts/rollouts.json"),
        "leaderboard": _check_artifact_path(root, artifacts, "leaderboard", "summary/leaderboard.json"),
        "all_summary_csv": _check_artifact_path(root, artifacts, "all_summary_csv", "summary/all_summary.csv"),
        "all_leaderboard": _check_artifact_path(root, artifacts, "all_leaderboard", "summary/all_leaderboard.json"),
        "results": _check_artifact_path(root, artifacts, "results", "results/results.json"),
        "report": _check_artifact_path(root, artifacts, "report", "results/report.json"),
        "input_config": _check_artifact_path(root, artifacts, "input_config", "config/config.json"),
        "benchmark_config": _check_artifact_path(root, artifacts, "benchmark_config", "config/benchmark_config.json"),
        "summary_figure": _check_artifact_path(root, artifacts, "summary_figure", "figures/summary.svg"),
        "leaderboard_figure": _check_artifact_path(root, artifacts, "leaderboard_figure", "figures/leaderboard.svg"),
        "training": _check_artifact_path(root, artifacts, "training", "training/training.json"),
        "learning_curve": _check_artifact_path(root, artifacts, "learning_curve", "training/learning_curve.json"),
        "learning_curve_csv": _check_artifact_path(root, artifacts, "learning_curve_csv", "training/learning_curve.csv"),
        "learning_curve_figure": _check_artifact_path(root, artifacts, "learning_curve_figure", "figures/learning_curve.svg"),
    }
    for key in ("rows", "summary_csv", "leaderboard", "results", "report", "input_config", "benchmark_config"):
        _add_exists(checks, key, paths[key], required=True)
    tracking_rows_present = any(row.get("objective") == "tracking" for row in rows)
    if tracking_rows_present:
        _add_exists(checks, "tracking_comparison", paths["tracking_comparison"], required=True)
        _add_exists(checks, "tracking_comparison_figure", paths["tracking_comparison_figure"], required=True)
    rollouts = list(benchmark.get("rollouts") or [])
    tracking_control_figures = (
        artifacts.get("tracking_control_figures")
        if isinstance(artifacts.get("tracking_control_figures"), Mapping)
        else {}
    )
    if rollouts:
        _add_exists(checks, "rollouts", paths["rollouts"], required=True)
        for benchmark_case, raw in sorted(tracking_control_figures.items()):
            path = _resolve_artifact_path(root, raw, "missing")
            _add_exists(checks, f"tracking_control_figure:{benchmark_case}", path, required=True)
        expected_control_cases = len({
            (
                str(rollout.get("scenario")),
                str(rollout.get("task") or "default"),
            )
            for rollout in rollouts
            if rollout.get("objective") == "tracking" and rollout.get("scenario")
        })
        checks.append(_check(
            "tracking_control_figure_count",
            len(tracking_control_figures) == expected_control_cases,
            str(root / "figures"),
            f"expected {expected_control_cases}, found {len(tracking_control_figures)}",
        ))
    summary_csvs = artifacts.get("summary_csvs") if isinstance(artifacts.get("summary_csvs"), Mapping) else {}
    leaderboards = artifacts.get("leaderboards") if isinstance(artifacts.get("leaderboards"), Mapping) else {}
    summary_figures = artifacts.get("summary_figures") if isinstance(artifacts.get("summary_figures"), Mapping) else {}
    leaderboard_figures = artifacts.get("leaderboard_figures") if isinstance(artifacts.get("leaderboard_figures"), Mapping) else {}
    has_objective_outputs = bool(summary_csvs or leaderboards)
    if has_objective_outputs:
        _add_exists(checks, "all_summary_csv", paths["all_summary_csv"], required=True)
        _add_exists(checks, "all_leaderboard", paths["all_leaderboard"], required=True)
    for key in ("summary_figure", "leaderboard_figure"):
        _add_exists(checks, key, paths[key], required=bool(rows) and not summary_figures and not leaderboard_figures)
    if benchmark.get("training"):
        _add_exists(checks, "training", paths["training"], required=True)
    learning_curve = list(benchmark.get("learning_curve") or [])
    if learning_curve:
        for key in ("learning_curve", "learning_curve_csv", "learning_curve_figure"):
            _add_exists(checks, key, paths[key], required=True)

    row_data = _safe_json_list(checks, "rows_json", paths["rows"])
    leaderboard = _safe_json(checks, "leaderboard_json", paths["leaderboard"])
    summary_rows = _safe_csv_rows(checks, "summary_csv_rows", paths["summary_csv"])
    tracking_comparison_rows = _safe_csv_rows(checks, "tracking_comparison_rows", paths["tracking_comparison"]) if tracking_rows_present else None
    all_summary_rows = _safe_csv_rows(checks, "all_summary_csv_rows", paths["all_summary_csv"]) if has_objective_outputs else None
    all_leaderboard = _safe_json_list(checks, "all_leaderboard_json", paths["all_leaderboard"]) if has_objective_outputs else None
    curve_rows = _safe_json_list(checks, "learning_curve_json", paths["learning_curve"]) if learning_curve else None
    curve_csv_rows = _safe_csv_rows(checks, "learning_curve_csv_rows", paths["learning_curve_csv"]) if learning_curve else None
    rollout_rows = _safe_json_list(checks, "rollouts_json", paths["rollouts"]) if rollouts else None
    if row_data is not None:
        _add_count_check(checks, "rows_json_count", len(row_data), expected_rows, paths["rows"])
    if summary_rows is not None:
        expected_summary_rows = len(_rows_by_objective(rows)) if has_objective_outputs else expected_rows
        _add_count_check(checks, "summary_csv_count", len(summary_rows), expected_summary_rows, paths["summary_csv"])
    if tracking_comparison_rows is not None:
        expected_tracking_cases = _tracking_benchmark_case_count(rows)
        _add_count_check(checks, "tracking_comparison_count", len(tracking_comparison_rows), expected_tracking_cases, paths["tracking_comparison"])
    if all_summary_rows is not None:
        _add_count_check(checks, "all_summary_csv_count", len(all_summary_rows), expected_rows, paths["all_summary_csv"])
    if leaderboard is not None:
        active_rows = sum(1 for row in rows if row.get("status") != "failed")
        leaderboard_count = _leaderboard_count(leaderboard)
        _add_count_check(checks, "leaderboard_count", leaderboard_count, active_rows, paths["leaderboard"])
    if all_leaderboard is not None:
        active_rows = sum(1 for row in rows if row.get("status") != "failed")
        _add_count_check(checks, "all_leaderboard_count", len(all_leaderboard), active_rows, paths["all_leaderboard"])
    _check_objective_artifacts(root, checks, rows, summary_csvs, "summary_csv")
    _check_objective_artifacts(root, checks, rows, leaderboards, "leaderboard")
    _check_objective_artifacts(root, checks, rows, summary_figures, "summary_figure", count_rows=False)
    _check_objective_artifacts(root, checks, rows, leaderboard_figures, "leaderboard_figure", count_rows=False)
    if curve_rows is not None:
        _add_count_check(checks, "learning_curve_json_count", len(curve_rows), len(learning_curve), paths["learning_curve"])
    if curve_csv_rows is not None:
        _add_count_check(checks, "learning_curve_csv_count", len(curve_csv_rows), len(learning_curve), paths["learning_curve_csv"])
    if rollout_rows is not None:
        _add_count_check(checks, "rollouts_json_count", len(rollout_rows), len(rollouts), paths["rollouts"])

    _check_model_cards(root, artifacts, expected_scenarios, checks)
    return _check_result(root, checks)


def _check_model_cards(root: Path, artifacts: Mapping[str, str], expected_scenarios: list[str],
                       checks: list[dict[str, Any]]) -> None:
    manifest_path = _check_artifact_path(root, artifacts, "model_cards_manifest", "metadata/model_cards/manifest.json")
    single_path = _check_artifact_path(root, artifacts, "model_card", "metadata/model_card.json")
    cards_dir = root / "metadata" / "model_cards"
    if len(expected_scenarios) <= 1:
        _add_exists(checks, "model_card", single_path, required=True)
        if cards_dir.exists():
            stale = sorted(path.name for path in cards_dir.glob("*.json"))
            checks.append(_check("stale_model_cards", not stale, str(cards_dir), _stale_message(stale)))
        return

    _add_exists(checks, "model_cards_manifest", manifest_path, required=True)
    if not manifest_path.exists():
        return
    try:
        manifest = _read_json(manifest_path)
    except Exception as ex:
        checks.append(_check("model_cards_manifest_json", False, str(manifest_path), str(ex)))
        return
    manifest_scenarios = [str(item) for item in manifest.get("scenarios", [])]
    checks.append(_check(
        "model_cards_manifest_scenarios",
        manifest_scenarios == expected_scenarios,
        str(manifest_path),
        f"expected {expected_scenarios}, found {manifest_scenarios}",
    ))
    cards = dict(manifest.get("cards") or {})
    for scenario in expected_scenarios:
        path = _resolve_artifact_path(root, cards.get(scenario), f"metadata/model_cards/{scenario}.json")
        _add_exists(checks, f"model_card:{scenario}", path, required=True)
    if cards_dir.exists():
        expected_files = {f"{scenario}.json" for scenario in expected_scenarios} | {"manifest.json"}
        stale = sorted(path.name for path in cards_dir.glob("*.json") if path.name not in expected_files)
        checks.append(_check("stale_model_cards", not stale, str(cards_dir), _stale_message(stale)))


def _safe_json_list(checks: list[dict[str, Any]], name: str, path: Path) -> list[Any] | None:
    if not path.exists():
        return None
    try:
        data = _read_json(path)
    except Exception as ex:
        checks.append(_check(name, False, str(path), str(ex)))
        return None
    ok = isinstance(data, list)
    checks.append(_check(name, ok, str(path), "valid JSON list" if ok else "expected JSON list"))
    return data if ok else None


def _safe_json(checks: list[dict[str, Any]], name: str, path: Path):
    if not path.exists():
        return None
    try:
        data = _read_json(path)
    except Exception as ex:
        checks.append(_check(name, False, str(path), str(ex)))
        return None
    ok = isinstance(data, (list, dict))
    checks.append(_check(name, ok, str(path), "valid JSON" if ok else "expected JSON list or object"))
    return data if ok else None


def _safe_csv_rows(checks: list[dict[str, Any]], name: str, path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as ex:
        checks.append(_check(name, False, str(path), str(ex)))
        return None
    checks.append(_check(name, True, str(path), "valid CSV"))
    return rows


def _add_count_check(checks: list[dict[str, Any]], name: str, actual: int, expected: int, path: Path) -> None:
    checks.append(_check(name, actual == expected, str(path), f"expected {expected}, found {actual}"))


def _add_exists(checks: list[dict[str, Any]], name: str, path: Path, required: bool) -> None:
    exists = path.exists()
    checks.append(_check(name, exists or not required, str(path), "exists" if exists else "missing"))


def _rows_by_objective(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("objective") or "benchmark"), []).append(row)
    return groups


def _leaderboard_count(data) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, Mapping):
        return sum(len(rows) for rows in data.values() if isinstance(rows, list))
    return 0


def _check_objective_artifacts(root: Path, checks: list[dict[str, Any]], rows: Sequence[Mapping[str, Any]],
                               paths_by_objective, name: str, count_rows: bool = True) -> None:
    if not isinstance(paths_by_objective, Mapping) or not paths_by_objective:
        return
    groups = _rows_by_objective(rows)
    for objective, objective_rows in groups.items():
        raw = paths_by_objective.get(objective)
        if isinstance(raw, Mapping):
            for scenario, scenario_raw in sorted(raw.items()):
                path = _resolve_artifact_path(root, scenario_raw, "missing")
                _add_exists(checks, f"{name}:{objective}:{scenario}", path, required=True)
            continue
        path = _resolve_artifact_path(root, raw, "missing")
        check_name = f"{name}:{objective}"
        _add_exists(checks, check_name, path, required=True)
        if not count_rows or not path.exists():
            continue
        if name == "summary_csv":
            csv_rows = _safe_csv_rows(checks, f"{check_name}_rows", path)
            if csv_rows is not None:
                _add_count_check(checks, f"{check_name}_count", len(csv_rows), len(objective_rows), path)
        elif name == "leaderboard":
            data = _safe_json_list(checks, f"{check_name}_json", path)
            if data is not None:
                active_rows = sum(1 for row in objective_rows if row.get("status") != "failed")
                _add_count_check(checks, f"{check_name}_count", len(data), active_rows, path)


def _check(name: str, ok: bool, path: str, message: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "path": path, "message": message}


def _check_result(root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row for row in checks if not row["ok"]]
    return {
        "schema_version": ARTIFACT_CHECK_SCHEMA_VERSION,
        "artifact_dir": str(root),
        "ok": not failed,
        "checks": checks,
        "failed": failed,
    }


def _artifact_scenario_names(benchmark: Mapping[str, Any]) -> list[str]:
    if benchmark.get("scenario"):
        return [str(benchmark["scenario"])]
    suite_config = benchmark.get("suite_config") or {}
    if suite_config.get("scenarios"):
        return [str(item) for item in suite_config["scenarios"]]
    return list(dict.fromkeys(str(row["scenario"]) for row in benchmark.get("rows", []) if row.get("scenario")))


def _check_artifact_path(root: Path, artifacts: Mapping[str, str], key: str, default: str) -> Path:
    raw = artifacts.get(key)
    return _resolve_artifact_path(root, raw, default)


def _resolve_artifact_path(root: Path, raw, default: str) -> Path:
    if raw:
        path = Path(raw)
        if path.is_absolute() or path.exists():
            return path
        return root / path
    return root / default


def _stale_message(stale: list[str]) -> str:
    if not stale:
        return "no stale model-card JSON files"
    return "stale files: " + ", ".join(stale)
