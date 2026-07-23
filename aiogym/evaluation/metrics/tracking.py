"""Setpoint-tracking metrics for evaluation."""
from __future__ import annotations


def tracking_step_metrics(info, setpoint, time_sec: float, dt: float, env):
    y = list(info["y"])
    y_sp = list(setpoint.get("y_sp") or env.model.default_setpoint_vector())
    errors = raw_tracking_errors(y, y_sp)
    overshoot = max(errors, default=0.0)
    abs_errors = [abs(err) for err in errors]
    error_cost = sum(err * err for err in errors)
    normalized_error_cost = sum(
        err * err for err in normalized_tracking_errors(env.model, y, y_sp)
    )
    move_cost = float(info.get("tracking_move_cost", 0.0) or 0.0)
    cost = float(info.get("tracking_cost", normalized_error_cost + move_cost) or 0.0)
    iae = sum(abs_errors) * dt
    ise = error_cost * dt
    mse = error_cost / max(len(errors), 1)
    itae = time_sec * sum(abs_errors) * dt
    settled = True
    normalized_tol = 0.02
    for err in errors:
        if abs(err) > normalized_tol:
            settled = False
    return {
        "tracking_cost": float(cost),
        "tracking_return": float(-cost),
        "tracking_error_cost": float(info.get("tracking_error_cost", normalized_error_cost) or 0.0),
        "tracking_move_cost": float(move_cost),
        "tracking_iae": float(iae),
        "tracking_mse": float(mse * dt),
        "tracking_ise": float(ise),
        "tracking_itae": float(itae),
        "tracking_overshoot": float(max(0.0, overshoot)),
        "tracking_settled": settled,
    }


def normalized_tracking_error_sum(model, y, y_sp) -> float:
    return float(sum(abs(err) for err in normalized_tracking_errors(model, y, y_sp)))


def raw_tracking_errors(y, y_sp):
    """Return signed controlled-output errors without range normalization."""

    return [float(value) - float(setpoint) for value, setpoint in zip(y, y_sp)]


def normalized_tracking_errors(model, y, y_sp):
    errors = []
    scales = list(model.controlled_output_scales())
    for i, (value, setpoint) in enumerate(zip(y, y_sp)):
        scale = scales[i] if i < len(scales) else 1.0
        errors.append((float(value) - float(setpoint)) / max(float(scale), 1e-12))
    return errors
