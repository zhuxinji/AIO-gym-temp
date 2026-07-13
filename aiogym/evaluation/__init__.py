"""Evaluation protocols, metrics, and artifact reports."""

from .objectives import StageRewardContext, StageRewardResult, stage_reward

from .protocols import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_DEFINITIONS,
    METRIC_DIRECTIONS,
    PRIMARY_METRICS,
    PROTOCOL_METRICS,
    PUBLIC_BENCHMARK_SCHEMA_VERSION,
    ROLLOUT_SCHEMA,
    BenchmarkConfig,
    BenchmarkProtocol,
    metric_definitions,
    metric_direction,
    metric_for_reward_mode,
    primary_metric_for_objective,
    resolve_protocol,
)
from .core import (
    _tracking_step_metrics,
    build_evaluation_report,
    evaluate_controller,
    result_schema,
    rollout_controller,
)
from .benchmark import run_benchmark
from .reports import (
    ARTIFACT_CHECK_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    check_benchmark_artifacts,
    render_benchmark_report,
)
from .artifacts import plot_results, write_benchmark_artifacts
from .plots import plot_constraint_timeline, plot_leaderboard, plot_learning_curve, plot_rollouts, plot_summary

__all__ = [
    "StageRewardContext",
    "StageRewardResult",
    "stage_reward",
    "EVALUATION_SCHEMA_VERSION",
    "ROLLOUT_SCHEMA",
    "METRIC_DEFINITIONS",
    "PROTOCOL_METRICS",
    "PRIMARY_METRICS",
    "METRIC_DIRECTIONS",
    "PUBLIC_BENCHMARK_SCHEMA_VERSION",
    "BenchmarkConfig",
    "BenchmarkProtocol",
    "metric_for_reward_mode",
    "primary_metric_for_objective",
    "metric_direction",
    "metric_definitions",
    "resolve_protocol",
    "evaluate_controller",
    "rollout_controller",
    "run_benchmark",
    "result_schema",
    "build_evaluation_report",
    "_tracking_step_metrics",
    "REPORT_SCHEMA_VERSION",
    "ARTIFACT_CHECK_SCHEMA_VERSION",
    "render_benchmark_report",
    "check_benchmark_artifacts",
    "write_benchmark_artifacts",
    "plot_results",
    "plot_summary",
    "plot_rollouts",
    "plot_leaderboard",
    "plot_constraint_timeline",
    "plot_learning_curve",
]
