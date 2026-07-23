"""Public environment construction from direct arguments or config mappings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ._internal.config import (
    PROTOCOL_KEYS,
    load_config,
    protocol_data,
    protocol_env_overrides,
)
from .env import AIOGymNativeEnv
from .evaluation.protocols import BenchmarkProtocol, resolve_protocol


def make_env(scenario: Any = "cascade", objective: str | Mapping[str, Any] | None = None,
             seed: int | None = None, config: str | Path | Mapping[str, Any] | None = None,
             *, protocol: BenchmarkProtocol | Mapping[str, Any] | None = None,
             **overrides) -> AIOGymNativeEnv:
    """Create an AIO-Gym environment from direct arguments or a config mapping."""

    data = load_config(config)
    removed = sorted(set(data) & {"model", "env"})
    if removed:
        raise ValueError(
            f"unsupported environment config field(s): {', '.join(removed)}; "
            "use 'scenario' and 'environment'"
        )
    nested_environment = data.pop("environment", {})
    if not isinstance(nested_environment, Mapping):
        raise TypeError("config['environment'] must be a mapping of environment options")
    data.update(nested_environment)
    data.update(overrides)
    removed_options = sorted(set(data) & {"dynamic", "reward_mode", "env_reward_mode"})
    if removed_options:
        raise ValueError(
            f"unsupported environment option(s): {', '.join(removed_options)}"
        )
    config_objective = data.pop("objective", None)
    config_protocol = data.pop("protocol", None)
    if protocol is None:
        protocol = config_protocol
    scenario = data.pop("scenario", scenario)
    protocol_options = protocol_data(data)
    configured_objective = config_objective
    if isinstance(protocol, Mapping):
        protocol_options.update(dict(protocol))
        configured_objective = protocol_options.pop(
            "objective", configured_objective
        )
    elif protocol is not None and not isinstance(protocol, BenchmarkProtocol):
        raise TypeError("protocol must be a BenchmarkProtocol or mapping")

    task_has_default = False
    if data.get("task") is not None:
        from .models.tasks import load_task_profile

        task_profile = load_task_profile(data["task"], scenario=scenario)
        task_has_default = task_profile.get("default_objective") is not None

    use_protocol = (
        objective is not None
        or configured_objective is not None
        or task_has_default
        or isinstance(protocol, BenchmarkProtocol)
    )
    if use_protocol:
        protocol_options = protocol_data({**data, **protocol_options})
        if configured_objective is not None:
            protocol_options["objective"] = configured_objective
        benchmark_protocol = resolve_protocol(
            scenario,
            objective=protocol if isinstance(protocol, BenchmarkProtocol) else objective,
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
