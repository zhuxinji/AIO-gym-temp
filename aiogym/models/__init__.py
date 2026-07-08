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
    builtin_gym_ids,
    gym_id_name,
    make_model,
    obs_vector,
    register_model,
    unregister_model,
    validate_model_contract,
)

__all__ = [
    "CP",
    "G",
    "RHO",
    "RHO_CP",
    "ProcessModelContract",
    "CascadeModel",
    "QuadrupleModel",
    "CSTRModel",
    "HVACModel",
    "ExtractionModel",
    "FiredHeaterModel",
    "CrystallizationModel",
    "MODELS",
    "BUILTIN_MODELS",
    "SCENARIOS",
    "builtin_gym_ids",
    "gym_id_name",
    "make_model",
    "obs_vector",
    "register_model",
    "unregister_model",
    "validate_model_contract",
]
