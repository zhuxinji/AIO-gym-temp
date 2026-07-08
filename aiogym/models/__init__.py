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
from .kernel import Integrator
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
    MODEL_CARD_SCHEMA_VERSION,
    SCENARIOS,
    builtin_gym_ids,
    collect_model_cards,
    export_model_cards,
    gym_id_name,
    iter_model_cards,
    make_model,
    obs_vector,
    register_model,
    unregister_model,
    validate_model_card,
    validate_model_contract,
)

__all__ = [
    "CP",
    "G",
    "RHO",
    "RHO_CP",
    "ProcessModelContract",
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
    "builtin_gym_ids",
    "collect_model_cards",
    "export_model_cards",
    "gym_id_name",
    "iter_model_cards",
    "make_model",
    "obs_vector",
    "register_model",
    "unregister_model",
    "validate_model_card",
    "validate_model_contract",
]
