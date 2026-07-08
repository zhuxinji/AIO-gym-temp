"""Setpoint-tracking metrics for evaluation."""
from __future__ import annotations


def tracking_step_metrics(info, setpoint, time_sec: float, dt: float, env):
    errors = []
    overshoot = 0.0
    temps = list(info.get("temps", []))
    levels = list(info.get("levels", []))
    t_sp = list(setpoint.get("t_sp", []))
    h_sp = list(setpoint.get("h_sp", []))
    level_scale = float(getattr(env.model, "kpi_level_scale", 1.0))
    for i, temp in enumerate(temps):
        if i < len(t_sp):
            err = float(temp) - float(t_sp[i])
            errors.append(err)
            overshoot = max(overshoot, err)
    for i in getattr(env.model, "controlled_levels", lambda: [])():
        if i < len(levels) and i < len(h_sp):
            err = (float(levels[i]) - float(h_sp[i])) / level_scale
            errors.append(err)
            overshoot = max(overshoot, err)
    abs_errors = [abs(err) for err in errors]
    iae = sum(abs_errors) * dt
    ise = sum(err * err for err in errors) * dt
    itae = time_sec * sum(abs_errors) * dt
    settled = True
    temp_tol = 1.0
    level_tol = 0.02
    for i, temp in enumerate(temps):
        if i < len(t_sp) and abs(float(temp) - float(t_sp[i])) > temp_tol:
            settled = False
    for i in getattr(env.model, "controlled_levels", lambda: [])():
        if i < len(levels) and i < len(h_sp) and abs((float(levels[i]) - float(h_sp[i])) / level_scale) > level_tol:
            settled = False
    return {
        "tracking_iae": float(iae),
        "tracking_ise": float(ise),
        "tracking_itae": float(itae),
        "tracking_overshoot": float(max(0.0, overshoot)),
        "tracking_settled": settled,
    }
