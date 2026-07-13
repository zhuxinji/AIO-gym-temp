"""Process-model package for AIO-Gym.

The public API intentionally matches the old aiogym.models module.
"""

from .core import (
    CP,
    G,
    RHO,
    RHO_CP,
    Integrator,
    ProcessModelContract,
    obs_vector,
)
from .declarative import DeclarativeProcessModel, define_model
from .scenarios import (
    CascadeModel,
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
    "QuadrupleModel",
    "CSTRModel",
    "HVACModel",
    "ExtractionModel",
    "FiredHeaterModel",
    "CrystallizationModel",
    "MODELS",
    "BUILTIN_MODELS",
    "MODEL_CARD_SCHEMA_VERSION",
    "SCENARIOS",
    "apply_model_params",
    "builtin_gym_ids",
    "collect_model_cards",
    "export_model_card_markdown",
    "export_model_cards",
    "gym_id_name",
    "iter_model_cards",
    "make_model",
    "obs_vector",
    "render_model_card_markdown",
    "register_model",
    "unregister_model",
    "validate_model_card",
    "validate_model_contract",
]
