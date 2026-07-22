"""Versioned evaluation schemas and metric catalog."""
from __future__ import annotations


EVALUATION_SCHEMA_VERSION = "aiogym.evaluation.v3"
PUBLIC_BENCHMARK_SCHEMA_VERSION = "aiogym.public_benchmark.v2"

ROLLOUT_SCHEMA = {
    "step": "integer control-step index",
    "time": "seconds since episode start",
    "obs": "observation before control action",
    "state": "physical state before integration",
    "action": "controller output passed to env.step",
    "setpoint": "active setpoint context exposed to the controller",
    "disturbance": "disturbance values applied by the process model",
    "reward": "training reward returned by the environment",
    "profit": "time-integrated economic contribution for the transition",
    "constraint": "normalized process constraint penalty for the step",
    "info": "environment-specific diagnostic fields",
}

METRIC_DEFINITIONS = {
    "return": "sum of environment reward over the rollout; reward is the training signal",
    "profit": "time-integrated economic profit over the rollout",
    "normalized_score": "0-100 KPI score from KPIScorer; score is for reporting, not raw economics",
    "production": "time-integrated process production over the rollout",
    "energy_kwh": "total action energy over the rollout",
    "runtime_seconds": "wall-clock seconds spent evaluating one episode",
    "runtime_total_seconds": "total wall-clock seconds spent evaluating all episodes",
    "runtime_seconds_per_step": "wall-clock seconds per environment control step",
    "tracking_cost": "cumulative tracking cost: raw squared setpoint error plus physical-input move and nominal steady-input deviation penalties",
    "tracking_return": "negative cumulative tracking_cost, matching the tracking reward returned by the environment",
    "tracking_error_cost": "cumulative raw squared setpoint-tracking error over all control steps and tracked outputs",
    "tracking_move_cost": "cumulative squared control move in physical actuator units",
    "tracking_steady_cost": "cumulative squared deviation from the model-resolved nominal steady input in physical actuator units; zero when no resolver is available",
    "tracking_mse": "mean squared raw tracking error over time and tracked outputs",
    "tracking_iae": "integral absolute raw tracking error",
    "tracking_ise": "integral squared raw tracking error",
    "tracking_itae": "time-weighted integral absolute raw tracking error",
    "tracking_overshoot": "largest positive raw excursion above the active setpoint",
    "tracking_settling_time": "last time at which any tracked variable exceeded tolerance",
    "constraint": "sum of normalized soft constraint penalty reported by the environment",
    "constraint_violation_count": "number of steps with any process constraint violation",
    "constraint_violation_duration": "seconds spent with any process constraint violation",
    "constraint_violation_severity": "sum of positive per-constraint violation magnitudes",
    "action_violation_count": "number of controller outputs outside action bounds before env clipping",
    "action_violation_duration": "seconds with any action-bound violation",
    "action_violation_severity": "sum of action-bound excess before env clipping",
    "runaway_count": "number of runaway steps reported by the environment",
    "runaway_duration": "seconds spent in runaway state",
    "safety_margin_min": "minimum negative violation margin; 0 means no violation was observed",
    "controller_solve_count": "number of optimization or policy solve calls reported by the controller",
    "controller_solver_success_count": "number of successful controller solve calls",
    "controller_solver_failure_count": "number of failed controller solve calls",
    "controller_fallback_count": "number of times the controller fell back to a previous or safe action",
    "controller_degraded_count": "number of episodes with any controller-side degradation",
}

PROTOCOL_METRICS = {
    "tracking": (
        "tracking_error_cost", "tracking_cost", "tracking_return", "tracking_move_cost",
        "tracking_steady_cost", "tracking_mse", "tracking_iae", "tracking_ise",
        "tracking_itae", "tracking_overshoot", "tracking_settling_time",
    ),
    "economic": (
        "profit", "normalized_score", "production", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity", "safety_margin_min",
        "controller_solver_failure_count", "controller_fallback_count",
        "runtime_seconds", "runtime_seconds_per_step",
    ),
    "robustness": (
        "return", "profit", "normalized_score", "tracking_cost", "tracking_mse",
        "tracking_iae", "energy_kwh", "constraint_violation_count",
        "constraint_violation_severity", "controller_solver_failure_count",
        "controller_fallback_count", "runtime_seconds", "runtime_seconds_per_step",
    ),
    "safety": (
        "constraint_violation_count", "constraint_violation_duration",
        "constraint_violation_severity", "action_violation_count",
        "action_violation_duration", "action_violation_severity", "runaway_count",
        "runaway_duration", "safety_margin_min", "controller_solver_failure_count",
        "controller_fallback_count", "runtime_seconds", "runtime_seconds_per_step",
    ),
    "kpi": (
        "normalized_score", "tracking_cost", "tracking_mse", "tracking_iae", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity",
        "controller_solver_failure_count", "controller_fallback_count",
        "runtime_seconds", "runtime_seconds_per_step",
    ),
}

PRIMARY_METRICS = {
    "tracking": "tracking_error_cost",
    "economic": "profit",
    "kpi": "normalized_score",
    "robustness": "normalized_score",
    "safety": "constraint_violation_count",
}

METRIC_DIRECTIONS = {
    "tracking_cost": "minimize", "tracking_error_cost": "minimize",
    "tracking_move_cost": "minimize", "tracking_steady_cost": "minimize",
    "tracking_return": "maximize", "tracking_mse": "minimize",
    "tracking_iae": "minimize", "tracking_ise": "minimize",
    "tracking_itae": "minimize", "tracking_overshoot": "minimize",
    "tracking_settling_time": "minimize", "constraint_violation_count": "minimize",
    "constraint_violation_duration": "minimize", "constraint_violation_severity": "minimize",
    "action_violation_count": "minimize", "action_violation_duration": "minimize",
    "action_violation_severity": "minimize", "runaway_count": "minimize",
    "runaway_duration": "minimize", "energy_kwh": "minimize",
    "runtime_seconds": "minimize", "runtime_total_seconds": "minimize",
    "runtime_seconds_per_step": "minimize", "controller_solver_failure_count": "minimize",
    "controller_fallback_count": "minimize", "controller_degraded_count": "minimize",
    "safety_margin_min": "maximize", "profit": "maximize",
    "normalized_score": "maximize", "return": "maximize", "production": "maximize",
}


def primary_metric_for_objective(objective: str) -> str:
    return PRIMARY_METRICS.get(objective, "return")


def metric_direction(metric: str) -> str:
    return METRIC_DIRECTIONS.get(metric, "maximize")


def metric_definitions(objective: str | None = None):
    if objective is None:
        return dict(METRIC_DEFINITIONS)
    return {
        key: METRIC_DEFINITIONS[key]
        for key in PROTOCOL_METRICS.get(objective, ())
        if key in METRIC_DEFINITIONS
    }
