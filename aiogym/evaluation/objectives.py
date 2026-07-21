"""Pure task-objective evaluation shared by environments and planners."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .metrics.kpi import KPIStep, kpi_step
from .metrics.tracking import normalized_tracking_error_sum, raw_tracking_errors


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
    tracking_r_steady: float = 1.0,
    dt: float = 1.0,
    economic_config: Mapping[str, Any] | None = None,
    reward_override: Callable[[Sequence[float], Any, Sequence[float], StageRewardContext], float] | None = None,
) -> StageRewardResult:
    """Evaluate a real or predicted transition without changing environment state."""

    if reward_mode not in {"economic", "kpi", "tracking"}:
        raise ValueError("reward_mode must be one of: economic, kpi, tracking")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    # ``state`` is part of the public transition contract even though the current
    # built-in objectives depend only on the resulting state and action terms.
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
    tracking_error_cost, tracking_move_cost, tracking_steady_cost, tracking_steady_action = _tracking_cost_terms(
        model,
        y,
        y_sp,
        action,
        previous_action,
        tracking_q_y,
        tracking_r_move,
        tracking_r_steady,
    )
    tracking_cost = tracking_error_cost + tracking_move_cost + tracking_steady_cost

    cons_info = dict(model.common_constraint_info(levels, temps))
    cons_info.update(model.process_constraint_info(x_next, levels, temps, env))
    hard_termination_resolver = getattr(model, "hard_termination_reasons", None)
    hard_termination_reasons = (
        tuple(
            str(reason)
            for reason in hard_termination_resolver(x_next, levels, temps, env)
        )
        if callable(hard_termination_resolver)
        else ()
    )
    runaway = bool(model.runaway_state(levels, temps)) or bool(hard_termination_reasons)
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
        profit_rate, production_rate = _economic_profit(
            model,
            x_next,
            action,
            levels,
            temps,
            runaway,
            env,
            action_energy_kw,
            economic_config,
        )
        profit = profit_rate * dt
        prod = production_rate * dt
        reward = profit * float(reward_scale)
    elif reward_mode == "kpi":
        reward = -kpi.penalty * float(reward_scale)
    else:
        reward = -tracking_cost

    info = {
        "track": track,
        "constraint": constraint,
        "production": prod,
        "profit": profit,
        "tracking_cost": tracking_cost,
        "tracking_return": -tracking_cost,
        "tracking_error_cost": tracking_error_cost,
        "tracking_move_cost": tracking_move_cost,
        "tracking_steady_cost": tracking_steady_cost,
        "tracking_steady_action": tracking_steady_action,
        "energy_kw": action_energy_kw,
        "runaway": runaway,
        "cons_info": cons_info,
        "cons_violated": any(value > 0 for value in cons_info.values()),
        "levels": levels,
        "temps": temps,
        "y": y,
        "y_sp": y_sp,
        "safety_events": list(hard_termination_reasons),
    }
    if process_extra:
        info.update(process_extra)

    terminated = bool(hard_termination_reasons) or bool(terminate_on_runaway and runaway)
    if terminated:
        info["termination_reason"] = (
            hard_termination_reasons[0] if hard_termination_reasons else "runaway"
        )
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


def _tracking_cost_terms(model, y, y_sp, action, previous_action, q_y, r_move, r_steady):
    errors = raw_tracking_errors(y, y_sp)
    error_cost = 0.0
    for i, error in enumerate(errors):
        weight = float(q_y[i]) if i < len(q_y) else 1.0
        error_cost += weight * error * error

    u = np.asarray(model.physical_action_vector(action), dtype=np.float64)
    u_previous = np.asarray(model.physical_action_vector(previous_action), dtype=np.float64)
    move_cost = 0.0
    if u.shape == u_previous.shape:
        move_cost = float(r_move) * float(np.sum((u - u_previous) ** 2))

    steady_cost = 0.0
    steady_action = None
    steady_resolver = getattr(model, "tracking_steady_state_action", None)
    if callable(steady_resolver):
        resolved = steady_resolver(y_sp)
        if resolved is not None:
            u_steady = np.asarray(model.physical_action_vector(resolved), dtype=np.float64)
            if u_steady.shape != u.shape or not np.all(np.isfinite(u_steady)):
                raise ValueError("tracking steady-state action must match the finite action vector")
            squared_deviation = float(np.sum((u - u_steady) ** 2))
            # Suppress roundoff such as 0.3 versus 0.30000000000000004 so an
            # exact nominal equilibrium reports a true zero auxiliary cost.
            steady_cost = 0.0 if squared_deviation <= 1e-24 else float(r_steady) * squared_deviation
            steady_action = [float(value) for value in u_steady]
    return float(error_cost), float(move_cost), float(steady_cost), steady_action


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


def _economic_profit(
    model, state, action, levels, temps, runaway, disturbance, energy_kw, config
):
    cfg = dict(config or model.economic_config)
    value = 0.0
    production = 0.0
    if cfg["value"] == "production" and callable(getattr(model, "production", None)):
        production = float(model.production(state, action, disturbance))
        value = production

    shortfall = 0.0
    shortfall_resolver = getattr(model, "product_flow_shortfall", None)
    if callable(shortfall_resolver):
        shortfall = float(shortfall_resolver(production))

    energy_kw = float(energy_kw)
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

    profit = (
        cfg["w_value"] * value
        - cfg["w_energy"] * energy_kw
        - cfg["w_viol"] * violation
        - float(cfg.get("w_product_shortfall", 0.0)) * shortfall
    )
    if runaway:
        profit -= 50.0
    return float(profit), float(production)
