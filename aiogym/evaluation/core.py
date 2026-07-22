"""Compatibility facade for evaluation execution and aggregation.

Implementations live in focused modules; existing deep imports remain valid.
"""

from .aggregation import (
    _aggregate_metric_keys,
    _robustness_row,
    _table_row,
    build_evaluation_report,
    result_schema,
)
from .evaluator import evaluate_controller
from .metadata import (
    _aggregate_controller_diagnostics,
    _controller_diagnostics,
    _controller_diagnostic_totals,
    _env_disturbances,
    _env_metadata,
    _env_objective,
    _git_commit,
    _model_metadata,
    _reproducibility_metadata,
)
from .metrics.tracking import tracking_step_metrics as _tracking_step_metrics
from .rollouts import rollout_controller

__all__ = [
    "evaluate_controller",
    "rollout_controller",
    "result_schema",
    "build_evaluation_report",
    "_tracking_step_metrics",
]
