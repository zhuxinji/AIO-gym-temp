"""Objective selection and environment reward-mode semantics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import warnings

from .._internal.serialization import jsonable as _jsonable
from .metric_catalog import (
    PRIMARY_METRICS,
    PROTOCOL_METRICS,
    metric_direction,
    primary_metric_for_objective,
)


OBJECTIVE_REWARD_MODES = {
    "tracking": "tracking",
    "economic": "economic",
    "kpi": "kpi",
    "robustness": "kpi",
    "safety": "kpi",
}
REWARD_MODES = frozenset(OBJECTIVE_REWARD_MODES.values())


def reward_mode_for_objective(objective: str) -> str:
    if objective not in OBJECTIVE_REWARD_MODES:
        raise ValueError(f"objective must be one of: {', '.join(PRIMARY_METRICS)}")
    return OBJECTIVE_REWARD_MODES[objective]


def objective_for_reward_mode(reward_mode: str) -> str:
    if reward_mode not in REWARD_MODES:
        raise ValueError("reward_mode must be one of: economic, kpi, tracking")
    return reward_mode


def resolve_objective_reward_mode(
    objective: str | None = None,
    reward_mode: str | None = None,
    *,
    default_objective: str | None = "kpi",
    warn_legacy: bool = False,
) -> tuple[str, str]:
    if reward_mode is not None and reward_mode not in REWARD_MODES:
        raise ValueError("reward_mode must be one of: economic, kpi, tracking")
    selected = objective
    if selected is None and reward_mode is not None:
        selected = objective_for_reward_mode(reward_mode)
    if selected is None:
        selected = default_objective
    if selected is None:
        raise ValueError("no objective was resolved")
    resolved_reward = reward_mode_for_objective(selected)
    if reward_mode is not None and reward_mode != resolved_reward:
        raise ValueError(
            f"objective {selected!r} resolves to reward mode {resolved_reward!r}, "
            f"which conflicts with reward_mode {reward_mode!r}"
        )
    if warn_legacy and reward_mode is not None:
        warnings.warn(
            "reward_mode is deprecated on high-level APIs and CLIs; use objective instead",
            FutureWarning,
            stacklevel=2,
        )
    return selected, resolved_reward


def metric_for_reward_mode(reward_mode: str) -> str:
    if reward_mode == "economic":
        return "profit"
    if reward_mode == "tracking":
        return "return"
    return "kpi"


@dataclass(frozen=True)
class ObjectiveSpec:
    """Resolved evaluation semantics and the source that selected them."""

    name: str
    source: str
    reward_mode: str
    primary_metric: str
    direction: str
    metrics: tuple[str, ...]
    reward_options: Mapping[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "reward_mode": self.reward_mode,
            "resolved_reward_mode": self.reward_mode,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "metrics": list(self.metrics),
            "reward_options": _jsonable(dict(self.reward_options)),
        }


def objective_spec(
    name: str,
    *,
    source: str = "explicit",
    reward_options: Mapping[str, Any] | None = None,
) -> ObjectiveSpec:
    if name not in PRIMARY_METRICS:
        raise ValueError(f"objective must be one of: {', '.join(PRIMARY_METRICS)}")
    primary_metric = primary_metric_for_objective(name)
    return ObjectiveSpec(
        name=name,
        source=str(source),
        reward_mode=reward_mode_for_objective(name),
        primary_metric=primary_metric,
        direction=metric_direction(primary_metric),
        metrics=tuple(PROTOCOL_METRICS.get(name, ())),
        reward_options=dict(reward_options or {}),
    )


def resolve_objective(
    *,
    explicit: str | None = None,
    case_config: str | None = None,
    suite_config: str | None = None,
    task_profile: Mapping[str, Any] | None = None,
) -> ObjectiveSpec:
    candidates = (
        ("explicit", explicit),
        ("case-config", case_config),
        ("suite-config", suite_config),
        ("task-default", (task_profile or {}).get("default_objective")),
    )
    for source, value in candidates:
        if value is not None:
            return objective_spec(str(value), source=source)
    raise ValueError(
        "no objective was resolved; specify an objective explicitly or use a "
        "task with default_objective"
    )
