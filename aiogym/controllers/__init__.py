"""Unified controller public API.

The implementation is split by responsibility; this module remains the stable
compatibility facade for existing imports.
"""

from .adapters import PolicyController, SB3PolicyController, as_controller
from .configs import (
    _CONFIG_META_KEYS,
    _controller_params,
    _merged_controller_config,
    load_controller_config,
)
from .contracts import (
    CONTROLLER_API_VERSION,
    Controller,
    ControllerContext,
    _call_compatible,
    build_context,
    controller_metadata,
    make_meas,
    validate_action,
)
from .registry import (
    BUILTIN_CONTROLLERS,
    ControllerFactory,
    _REGISTRY,
    make_controller,
    register_controller,
    registered_controllers,
    unregister_controller,
)

_metadata = controller_metadata

__all__ = [
    "CONTROLLER_API_VERSION",
    "Controller",
    "ControllerContext",
    "PolicyController",
    "SB3PolicyController",
    "BUILTIN_CONTROLLERS",
    "ControllerFactory",
    "as_controller",
    "build_context",
    "load_controller_config",
    "make_controller",
    "make_meas",
    "register_controller",
    "registered_controllers",
    "unregister_controller",
    "validate_action",
]
