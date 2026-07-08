"""Safety and constraint-violation metrics for evaluation."""
from __future__ import annotations

import numpy as np


def action_bound_metrics(action, env):
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    arr = np.asarray(action, dtype=np.float64)
    below = np.maximum(low - arr, 0.0)
    above = np.maximum(arr - high, 0.0)
    severity = float(np.sum(below + above))
    return {
        "violated": severity > 0.0,
        "severity": severity,
    }


def safety_step_metrics(info, bound_metrics, dt: float):
    cons = dict(info.get("cons_info", {}))
    cons_severity = float(sum(max(0.0, float(v)) for v in cons.values()))
    cons_violated = bool(info.get("cons_violated", False)) or cons_severity > 0.0
    action_violated = bool(bound_metrics.get("violated", False))
    runaway = bool(info.get("runaway", False))
    worst = max(cons_severity, float(bound_metrics.get("severity", 0.0)), 1.0 if runaway else 0.0)
    return {
        "constraint_violation_count": 1.0 if cons_violated else 0.0,
        "constraint_violation_duration": dt if cons_violated else 0.0,
        "constraint_violation_severity": cons_severity,
        "action_violation_count": 1.0 if action_violated else 0.0,
        "action_violation_duration": dt if action_violated else 0.0,
        "action_violation_severity": float(bound_metrics.get("severity", 0.0)),
        "runaway_count": 1.0 if runaway else 0.0,
        "runaway_duration": dt if runaway else 0.0,
        "safety_margin_min": -float(worst) if worst > 0.0 else 0.0,
    }
