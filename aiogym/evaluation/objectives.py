"""Pure task-objective evaluation shared by environments and planners."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .metrics.kpi import KPIStep, kpi_step
from .metrics.tracking import normalized_tracking_error_sum, normalized_tracking_errors


@dataclass(frozen=True)
class StageRewardResult:
    """Reward and diagnostics for one state transition."""

    reward: float
    terminated: bool
    info: Mapping[str, Any]
    kpi: KPIStep


@dataclass(frozen=True)
class StageRewardContext:
    """Read-only transition context supplied to a custom stage reward."""

    model: Any
    setpoint: tuple[float, ...]
    disturbance: Mapping[str, Any]
    previous_action: Any
    reward_mode: str
    reward_scale: float
    base_reward: float
    terminated: bool
    info: Mapping[str, Any]


def stage_reward(
    model,
    state: Sequence[float],
    action,
    next_state: Sequence[float],
    *,
    setpoint: Sequence[float],
    disturbance: Mapping[str, Any],
    previous_action,
    reward_mode: str,
    reward_scale: float,
    tracking_q_y: Sequence[float],
    tracking_r_move: float,
    terminate_on_runaway: bool,
    economic_config: Mapping[str, Any] | None = None,
    reward_override: Callable[[Sequence[float], Any, Sequence[float], StageRewardContext], float] | None = None,
) -> StageRewardResult:
    """Evaluate a real or predicted transition without changing environment state."""

    if reward_mode not in {"economic", "kpi", "tracking"}:
        raise ValueError("reward_mode must be one of: economic, kpi, tracking")

    # ``state`` is part of the public transition contract even though the current
    # built-in objectives depend only on the resulting state and action move.
    x = model.state_vector(state)
    x_next = model.state_vector(next_state)
    action = _model_action(model, action)
    previous_action = _model_action(model, previous_action)
    env = dict(disturbance)
    y_sp = [float(value) for value in setpoint]
    out = model.outputs(x_next)
    levels = list(out["levels"])
    temps = list(out["temps"])
    y = list(out["y"])

    track = normalized_tracking_error_sum(model, y, y_sp)
    tracking_error_cost, tracking_move_cost = _tracking_cost_terms(
        model,
        y,
        y_sp,
        action,
        previous_action,
        tracking_q_y,
        tracking_r_move,
    )
    tracking_cost = tracking_error_cost + tracking_move_cost

    cons_info = dict(model.common_constraint_info(levels, temps))
    runaway = bool(model.runaway_state(levels, temps))
    cons_info.update(model.process_constraint_info(x_next, levels, temps, env))
    process_extra = _process_info(model, x_next, levels, temps, env, action)
    constraint = _constraint_penalty(model, cons_info)

    action_energy_kw = float(model.action_energy_kw(action, x_next, env))
    ideal_energy_kw = float(model.ideal_energy_kw(x_next, y_sp, env, action))
    kpi = kpi_step(
        model,
        y,
        y_sp,
        action_energy_kw,
        ideal_energy_kw,
        runaway,
        energy_scored=bool(getattr(model, "energy_scored", True)),
    )

    prod = 0.0
    profit = 0.0
    if reward_mode == "economic":
        profit, prod = _economic_profit(
            model,
            x_next,
            action,
            levels,
            temps,
            runaway,
            env,
            economic_config,
        )
        reward = profit * float(reward_scale)
    elif reward_mode == "kpi":
        reward = -kpi.penalty * float(reward_scale)
    else:
        reward = -tracking_cost

    info = {
        "track": track,
        "constraint": constraint,
        "prod": prod,
        "profit": profit,
        "tracking_cost": tracking_cost,
        "tracking_return": -tracking_cost,
        "tracking_error_cost": tracking_error_cost,
        "tracking_move_cost": tracking_move_cost,
        "energy_kw": action_energy_kw,
        "runaway": runaway,
        "cons_info": cons_info,
        "cons_violated": any(value > 0 for value in cons_info.values()),
        "levels": levels,
        "temps": temps,
        "y": y,
        "y_sp": y_sp,
    }
    if process_extra:
        info.update(process_extra)

    terminated = bool(terminate_on_runaway and runaway)
    if reward_override is not None:
        context = StageRewardContext(
            model=model,
            setpoint=tuple(y_sp),
            disturbance=copy.deepcopy(env),
            previous_action=copy.deepcopy(previous_action),
            reward_mode=reward_mode,
            reward_scale=float(reward_scale),
            base_reward=float(reward),
            terminated=terminated,
            info=copy.deepcopy(info),
        )
        reward = float(reward_override(
            tuple(x),
            copy.deepcopy(action),
            tuple(x_next),
            context,
        ))
        if not np.isfinite(reward):
            raise ValueError("custom_stage_reward must return a finite scalar")
    if terminated:
        reward -= 50.0
    return StageRewardResult(float(reward), terminated, info, kpi)


def _model_action(model, action):
    return model.action_vector(action)


def _tracking_cost_terms(model, y, y_sp, action, previous_action, q_y, r_move):
    errors = normalized_tracking_errors(model, y, y_sp)
    error_cost = 0.0
    for i, error in enumerate(errors):
        weight = float(q_y[i]) if i < len(q_y) else 1.0
        error_cost += weight * error * error

    u = np.asarray(model.action_vector(action), dtype=np.float64)
    u_previous = np.asarray(model.action_vector(previous_action), dtype=np.float64)
    move_cost = 0.0
    if u.shape == u_previous.shape:
        move_cost = float(r_move) * float(np.sum((u - u_previous) ** 2))
    return float(error_cost), float(move_cost)


def _process_info(model, state, levels, temps, disturbance, action):
    process_info = model.process_info
    args = (state, levels, temps, disturbance)
    try:
        inspect.signature(process_info).bind(*args, action)
    except (TypeError, ValueError):
        return process_info(*args)
    return process_info(*args, action)


def _constraint_penalty(model, cons_info):
    height_max = list(model.height_max)
    level_scale = 0.1 * max(max(height_max), 1e-9) if height_max else 0.1
    scales = {"temp_high": 10.0, "temp_trip": 10.0}
    scales.update(model.constraint_penalty_scales())
    total = 0.0
    for key, value in cons_info.items():
        violation = max(0.0, float(value))
        scale = level_scale if key.startswith("level_") else scales.get(key, 1.0)
        total += violation / max(scale, 1e-9)
    return float(total)


def _economic_profit(model, state, action, levels, temps, runaway, disturbance, config):
    cfg = dict(config or model.economic_config)
    value = 0.0
    production = 0.0
    if cfg["value"] == "production" and callable(getattr(model, "production", None)):
        production = float(model.production(state, action, disturbance))
        value = production

    energy_kw = float(model.economic_energy_kw(action, state, disturbance))
    violation = 0.0
    for i, (lower, upper) in enumerate(cfg["temp_band"]):
        if lower is not None and temps[i] < lower:
            violation += (lower - temps[i]) / 10.0
        if upper is not None and temps[i] > upper:
            violation += (temps[i] - upper) / 10.0

    level_scale = float(cfg.get("level_scale", 0.1))
    y = list(model.controlled_output(state))
    for i, (lower, upper) in enumerate(cfg["level_band"]):
        controlled_value = y[i] if i < len(y) else 0.0
        if lower is not None and controlled_value < lower:
            violation += (lower - controlled_value) / level_scale
        if upper is not None and controlled_value > upper:
            violation += (controlled_value - upper) / level_scale

    profit = cfg["w_value"] * value - cfg["w_energy"] * energy_kw - cfg["w_viol"] * violation
    if runaway:
        profit -= 50.0
    return float(profit), float(production)
