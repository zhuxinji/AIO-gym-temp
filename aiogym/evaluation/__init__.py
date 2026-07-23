"""Evaluation protocols, metrics, and artifact reports."""

from .objectives import StageRewardContext, StageRewardResult, stage_reward
from .results import evaluate_task_acceptance

from .metric_catalog import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_DEFINITIONS,
    METRIC_DIRECTIONS,
    PRIMARY_METRICS,
    PROTOCOL_METRICS,
    PUBLIC_BENCHMARK_SCHEMA_VERSION,
    ROLLOUT_SCHEMA,
    metric_definitions,
    metric_direction,
    primary_metric_for_objective,
)
from .cases import BenchmarkCase, EnvironmentSpec
from .objective_specs import (
    ObjectiveSpec,
    metric_for_reward_mode,
    reward_mode_for_objective,
    objective_spec,
    resolve_objective,
)
from .protocols import (
    BenchmarkProtocol,
    resolve_protocol,
)
from .results import build_evaluation_report, result_schema
from .execution import evaluate_controller
from .metrics.tracking import tracking_step_metrics as _tracking_step_metrics
from .execution import rollout_controller, run_benchmark
from .artifact import (
    ARTIFACT_CHECK_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    check_benchmark_artifacts,
    finalize_benchmark_artifacts,
    plot_constraint_timeline,
    plot_leaderboard,
    plot_learning_curve,
    plot_results,
    plot_rollouts,
    plot_summary,
    render_benchmark_report,
    write_benchmark_artifacts,
)

__all__ = [
    "StageRewardContext",
    "StageRewardResult",
    "stage_reward",
    "evaluate_task_acceptance",
    "EVALUATION_SCHEMA_VERSION",
    "ROLLOUT_SCHEMA",
    "METRIC_DEFINITIONS",
    "PROTOCOL_METRICS",
    "PRIMARY_METRICS",
    "METRIC_DIRECTIONS",
    "PUBLIC_BENCHMARK_SCHEMA_VERSION",
    "BenchmarkCase",
    "BenchmarkProtocol",
    "EnvironmentSpec",
    "ObjectiveSpec",
    "metric_for_reward_mode",
    "reward_mode_for_objective",
    "objective_spec",
    "primary_metric_for_objective",
    "metric_direction",
    "metric_definitions",
    "resolve_objective",
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
    "finalize_benchmark_artifacts",
    "plot_results",
    "plot_summary",
    "plot_rollouts",
    "plot_leaderboard",
    "plot_constraint_timeline",
    "plot_learning_curve",
]
