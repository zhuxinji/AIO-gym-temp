"""Unified controller public API."""

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
    "unregister_controller",
    "validate_action",
]
