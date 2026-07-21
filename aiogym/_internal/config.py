"""Shared config parsing helpers for public environment and benchmark APIs."""
from __future__ import annotations

import json
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
    overrides = {key: value for key, value in data.items() if key != "objective"}
    if "env_reward_mode" in overrides:
        overrides["reward_mode"] = overrides.pop("env_reward_mode")
    return overrides
