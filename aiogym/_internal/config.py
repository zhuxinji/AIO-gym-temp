"""Shared config parsing helpers for public environment and benchmark APIs."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_KEYS = frozenset({
    "objective",
    "env_reward_mode",
    "reward_mode",
    "action_mode",
    "control_dt",
    "episode_steps",
    "task",
    "auto_events",
    "dynamic",
    "randomize",
    "randomize_setpoints",
    "randomize_plant",
    "plant_drift",
    "integral_obs",
    "terminate_on_runaway",
    "tracking_q_y",
    "tracking_r_move",
    "tracking_r_steady",
    "noise",
    "noise_pct",
    "model_params",
})


def resolve_auto_events(
    auto_events: bool | None = None,
    dynamic: bool | None = None,
    *,
    default: bool | None = None,
    warn_legacy: bool = False,
) -> bool | None:
    """Resolve the public automatic-event flag and its deprecated alias."""

    for name, value in (("auto_events", auto_events), ("dynamic", dynamic)):
        if value is not None and not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
    if auto_events is not None and dynamic is not None and auto_events != dynamic:
        raise ValueError(
            f"auto_events {auto_events!r} conflicts with deprecated dynamic {dynamic!r}"
        )
    if warn_legacy and dynamic is not None:
        warnings.warn(
            "dynamic is deprecated for automatic event generation; use auto_events instead",
            FutureWarning,
            stacklevel=2,
        )
    if auto_events is not None:
        return auto_events
    if dynamic is not None:
        return dynamic
    return default


def canonicalize_auto_events(
    data: Mapping[str, Any],
    *,
    warn_legacy: bool = False,
) -> dict[str, Any]:
    """Return a mapping using only the canonical ``auto_events`` field."""

    result = dict(data)
    if "auto_events" not in result and "dynamic" not in result:
        return result
    result["auto_events"] = resolve_auto_events(
        result.get("auto_events"),
        result.get("dynamic"),
        warn_legacy=warn_legacy,
    )
    result.pop("dynamic", None)
    return result


def load_config(config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    with Path(config).open() as stream:
        return json.load(stream)


def as_list(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def parse_seed_list(
    raw: str | None,
    seed: int,
    episodes: int,
    *,
    option: str = "--seed-list",
) -> list[int]:
    """Parse explicit seeds or generate one deterministic seed per episode."""

    if raw:
        seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
        if not seeds:
            raise ValueError(f"{option} must contain at least one integer seed")
        return seeds
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    return [seed + i for i in range(episodes)]


def protocol_data(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: data[key] for key in PROTOCOL_KEYS if key in data}


def protocol_env_overrides(data: Mapping[str, Any]) -> dict[str, Any]:
    overrides = {
        key: value
        for key, value in data.items()
        if key not in {"objective", "reward_mode", "env_reward_mode"}
    }
    return canonicalize_auto_events(overrides)
