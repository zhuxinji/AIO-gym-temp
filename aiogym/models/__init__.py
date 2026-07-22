"""Process-model package for AIO-Gym.

The public API intentionally matches the old aiogym.models module.
"""

from .core import (
    CP,
    G,
    RHO,
    RHO_CP,
    ProcessModelContract,
)
from .integration import Integrator
from .declarative import DeclarativeProcessModel, define_model
from .scenarios import (
    CascadeModel,
    RecirculatingCascadeModel,
    QuadrupleModel,
    CSTRModel,
    HVACModel,
    ExtractionModel,
    FiredHeaterModel,
    CrystallizationModel,
)
from .registry import (
    MODELS,
    BUILTIN_MODELS,
    SCENARIOS,
    apply_model_params,
    builtin_gym_ids,
    gym_id_name,
    make_model,
    register_model,
    unregister_model,
    validate_model_contract,
)
from .cards import (
    MODEL_CARD_SCHEMA_VERSION,
    collect_model_cards,
    export_model_card_markdown,
    export_model_cards,
    iter_model_cards,
    render_model_card_markdown,
    validate_model_card,
)
from .parameter_profiles import (
    PARAMETER_PROFILE_SCHEMA_VERSION,
    list_parameter_profiles,
    load_parameter_profile,
    validate_parameter_profile,
)
from .validation import validate_model_readiness
from .tasks import (
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

__all__ = [
    "CP",
    "G",
    "RHO",
    "RHO_CP",
    "ProcessModelContract",
    "DeclarativeProcessModel",
    "define_model",
    "Integrator",
    "CascadeModel",
    "RecirculatingCascadeModel",
    "QuadrupleModel",
    "CSTRModel",
    "HVACModel",
    "ExtractionModel",
    "FiredHeaterModel",
    "CrystallizationModel",
    "MODELS",
    "BUILTIN_MODELS",
    "MODEL_CARD_SCHEMA_VERSION",
    "PARAMETER_PROFILE_SCHEMA_VERSION",
    "TASK_PROFILE_SCHEMA_VERSION",
    "SCENARIOS",
    "apply_model_params",
    "builtin_gym_ids",
    "collect_model_cards",
    "export_model_card_markdown",
    "export_model_cards",
    "gym_id_name",
    "iter_model_cards",
    "make_model",
    "list_parameter_profiles",
    "load_parameter_profile",
    "list_task_profiles",
    "load_task_profile",
    "render_model_card_markdown",
    "register_model",
    "unregister_model",
    "validate_model_card",
    "validate_model_contract",
    "validate_model_readiness",
    "validate_parameter_profile",
    "configure_model_for_task",
    "resolve_environment_options",
    "task_environment",
    "task_identity",
    "task_operation",
    "validate_task_profile",
]
