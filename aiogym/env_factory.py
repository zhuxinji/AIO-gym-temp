"""Public environment construction from direct arguments or config mappings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ._config import (
    PROTOCOL_KEYS,
    load_config,
    protocol_data,
    protocol_env_overrides,
)
from .env import AIOGymNativeEnv
from .evaluation.protocols import resolve_protocol


def make_env(model: str = "cascade", objective: str | Mapping[str, Any] | None = None,
             seed: int | None = None, config: str | Path | Mapping[str, Any] | None = None,
             *, protocol: str | Mapping[str, Any] | None = None,
             **overrides) -> AIOGymNativeEnv:
    """Create an AIO-Gym environment from direct arguments or a config mapping."""

    data = load_config(config)
    nested_env = data.pop("env", {})
    if not isinstance(nested_env, Mapping):
        raise TypeError("config['env'] must be a mapping of environment options")
    data.update(nested_env)
    data.update(overrides)
    config_objective = data.pop("objective", None)
    if objective is None:
        objective = config_objective
    config_protocol = data.pop("protocol", None)
    if protocol is None:
        protocol = config_protocol
    scenario = data.pop("scenario", data.pop("model", model))
    objective_spec = objective if objective is not None else protocol
    if objective_spec is not None:
        if isinstance(objective_spec, Mapping):
            objective_data = dict(objective_spec)
            objective_spec = objective_data.pop("objective", "tracking")
            data.update(objective_data)
        protocol_options = protocol_data(data)
        benchmark_protocol = resolve_protocol(
            scenario,
            objective=objective_spec,
            data=protocol_options,
        )
        env_kwargs = benchmark_protocol.env_kwargs()
        env_kwargs.update(protocol_env_overrides(protocol_options))
        env_kwargs.update({key: value for key, value in data.items() if key not in PROTOCOL_KEYS})
        env = AIOGymNativeEnv(scenario, **env_kwargs)
    else:
        env = AIOGymNativeEnv(scenario, **data)
    if seed is not None:
        env.reset(seed=seed)
    return env
