"""Compatibility imports for task APIs moved to :mod:`aiogym.models.tasks`.

New code should import model-bound task declarations from ``aiogym.models.tasks``.
Acceptance evaluation remains owned by ``aiogym.evaluation``.
"""

from ..models.tasks import (
    ENVIRONMENT_BOOLEAN_FIELDS,
    TASK_ENVIRONMENT_FIELDS,
    TASK_OPERATION_FIELDS,
    TASK_PROFILE_SCHEMA_VERSION,
    configure_model_for_task,
    list_task_profiles,
    load_task_profile,
    resolve_environment_options,
    task_environment,
    task_identity,
    task_operation,
    validate_task_profile,
)
from .task_acceptance import evaluate_task_acceptance

__all__ = [
    "ENVIRONMENT_BOOLEAN_FIELDS",
    "TASK_ENVIRONMENT_FIELDS",
    "TASK_OPERATION_FIELDS",
    "TASK_PROFILE_SCHEMA_VERSION",
    "configure_model_for_task",
    "evaluate_task_acceptance",
    "list_task_profiles",
    "load_task_profile",
    "resolve_environment_options",
    "task_environment",
    "task_identity",
    "task_operation",
    "validate_task_profile",
]
