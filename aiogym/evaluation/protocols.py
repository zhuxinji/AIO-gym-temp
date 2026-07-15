"""Benchmark protocol declarations and metric catalog."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Integral
from typing import Any, Mapping, Sequence


from .._internal.serialization import jsonable as _jsonable

EVALUATION_SCHEMA_VERSION = "aiogym.evaluation.v2"
PUBLIC_BENCHMARK_SCHEMA_VERSION = "aiogym.public_benchmark.v1"


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
    "tracking_cost": "cumulative PC-Gym-style tracking cost: normalized squared setpoint error plus input move penalty",
    "tracking_return": "negative cumulative tracking_cost, matching the tracking reward returned by the environment",
    "tracking_error_cost": "cumulative normalized squared setpoint-tracking error term",
    "tracking_move_cost": "cumulative normalized control move penalty term",
    "tracking_mse": "mean squared normalized tracking error over time and tracked outputs",
    "tracking_iae": "integral absolute error normalized by each tracked variable range",
    "tracking_ise": "integral squared normalized tracking error",
    "tracking_itae": "time-weighted integral absolute normalized tracking error",
    "tracking_overshoot": "largest positive normalized excursion above the active setpoint",
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
        "tracking_cost", "tracking_return", "tracking_error_cost", "tracking_move_cost",
        "tracking_mse", "tracking_iae", "tracking_ise", "tracking_itae", "tracking_overshoot",
        "tracking_settling_time",
    ),
    "economic": (
        "profit", "normalized_score", "production", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity", "safety_margin_min",
        "controller_solver_failure_count", "controller_fallback_count",
        "runtime_seconds", "runtime_seconds_per_step",
    ),
    "robustness": (
        "return", "profit", "normalized_score", "tracking_cost", "tracking_mse", "tracking_iae", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity",
        "controller_solver_failure_count", "controller_fallback_count",
        "runtime_seconds", "runtime_seconds_per_step",
    ),
    "safety": (
        "constraint_violation_count", "constraint_violation_duration",
        "constraint_violation_severity", "action_violation_count",
        "action_violation_duration", "action_violation_severity",
        "runaway_count", "runaway_duration", "safety_margin_min",
        "controller_solver_failure_count", "controller_fallback_count",
        "runtime_seconds", "runtime_seconds_per_step",
    ),
    "kpi": (
        "normalized_score", "tracking_cost", "tracking_mse", "tracking_iae", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity",
        "controller_solver_failure_count", "controller_fallback_count",
        "runtime_seconds", "runtime_seconds_per_step",
    ),
}


PRIMARY_METRICS = {
    "tracking": "tracking_cost",
    "economic": "profit",
    "kpi": "normalized_score",
    "robustness": "normalized_score",
    "safety": "constraint_violation_count",
}


METRIC_DIRECTIONS = {
    "tracking_cost": "minimize",
    "tracking_error_cost": "minimize",
    "tracking_move_cost": "minimize",
    "tracking_return": "maximize",
    "tracking_mse": "minimize",
    "tracking_iae": "minimize",
    "tracking_ise": "minimize",
    "tracking_itae": "minimize",
    "tracking_overshoot": "minimize",
    "tracking_settling_time": "minimize",
    "constraint_violation_count": "minimize",
    "constraint_violation_duration": "minimize",
    "constraint_violation_severity": "minimize",
    "action_violation_count": "minimize",
    "action_violation_duration": "minimize",
    "action_violation_severity": "minimize",
    "runaway_count": "minimize",
    "runaway_duration": "minimize",
    "energy_kwh": "minimize",
    "runtime_seconds": "minimize",
    "runtime_total_seconds": "minimize",
    "runtime_seconds_per_step": "minimize",
    "controller_solver_failure_count": "minimize",
    "controller_fallback_count": "minimize",
    "controller_degraded_count": "minimize",
    "safety_margin_min": "maximize",
    "profit": "maximize",
    "normalized_score": "maximize",
    "kpi": "maximize",
    "return": "maximize",
    "production": "maximize",
}


def metric_for_reward_mode(reward_mode: str) -> str:
    """Training/checkpoint metric implied by an environment reward mode.

    Benchmark reports should use ``primary_metric_for_objective`` instead.
    """

    if reward_mode == "economic":
        return "profit"
    if reward_mode == "tracking":
        return "return"
    return "kpi"


def primary_metric_for_objective(objective: str) -> str:
    return PRIMARY_METRICS.get(objective, "return")


def metric_direction(metric: str) -> str:
    return METRIC_DIRECTIONS.get(metric, "maximize")


def metric_definitions(objective: str | None = None):
    if objective is None:
        return dict(METRIC_DEFINITIONS)
    return {key: METRIC_DEFINITIONS[key] for key in PROTOCOL_METRICS.get(objective, ()) if key in METRIC_DEFINITIONS}


def _empty_episode_totals(ep: int, seed: int):
    return {
        "episode": int(ep),
        "seed": int(seed),
        "return": 0.0,
        "track": 0.0,
        "constraint": 0.0,
        "profit": 0.0,
        "prod": 0.0,
        "production": 0.0,
        "energy_kwh": 0.0,
        "runtime_seconds": 0.0,
        "runtime_seconds_per_step": 0.0,
        "tracking_cost": 0.0,
        "tracking_return": 0.0,
        "tracking_error_cost": 0.0,
        "tracking_move_cost": 0.0,
        "tracking_mse": 0.0,
        "tracking_iae": 0.0,
        "tracking_ise": 0.0,
        "tracking_itae": 0.0,
        "tracking_overshoot": 0.0,
        "tracking_settling_time": 0.0,
        "constraint_violation_count": 0.0,
        "constraint_violation_duration": 0.0,
        "constraint_violation_severity": 0.0,
        "action_violation_count": 0.0,
        "action_violation_duration": 0.0,
        "action_violation_severity": 0.0,
        "runaway_count": 0.0,
        "runaway_duration": 0.0,
        "safety_margin_min": 0.0,
        "controller_solve_count": 0.0,
        "controller_solver_success_count": 0.0,
        "controller_solver_failure_count": 0.0,
        "controller_fallback_count": 0.0,
        "controller_degraded_count": 0.0,
    }


def _protocol_kwargs(defaults: Mapping[str, Any], overrides: Mapping[str, Any]):
    data = dict(defaults)
    data.update(overrides)
    return data


# Compatibility conditions for protocols without an explicit task. New task
# profiles own these environment settings; objectives own scoring and reports.
_IMPLICIT_TASK_DEFAULTS = {
    "economic": {
        "action_mode": "actuator", "dynamic": True, "randomize": True,
        "randomize_setpoints": False, "randomize_plant": True, "plant_drift": True,
        "integral_obs": False, "terminate_on_runaway": False, "noise": False,
    },
    "tracking": {
        "action_mode": "actuator", "dynamic": False, "randomize": False,
        "randomize_setpoints": False, "randomize_plant": False, "plant_drift": False,
        "integral_obs": False, "terminate_on_runaway": False, "noise": False,
    },
    "kpi": {
        "action_mode": "actuator", "dynamic": True, "randomize": True,
        "randomize_setpoints": True, "randomize_plant": True, "plant_drift": True,
        "integral_obs": False, "terminate_on_runaway": False, "noise": False,
    },
    "robustness": {
        "action_mode": "actuator", "dynamic": True, "randomize": True,
        "randomize_setpoints": True, "randomize_plant": True, "plant_drift": True,
        "integral_obs": False, "terminate_on_runaway": False, "noise": True,
    },
    "safety": {
        "action_mode": "actuator", "dynamic": True, "randomize": True,
        "randomize_setpoints": True, "randomize_plant": True, "plant_drift": True,
        "integral_obs": False, "terminate_on_runaway": False, "noise": False,
    },
}


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Reproducible benchmark environment configuration.

    Use ``economic`` for supervisory/RTO benchmarks and ``tracking`` for
    setpoint-following benchmarks.  Baseline PID/MPC/oracle controllers usually
    run with ``action_mode="actuator"``; learned supervisory policies usually run
    with ``action_mode="setpoint"`` on the same protocol. ``env_reward_mode`` is
    passed to the Gym environment; benchmark metrics are chosen from ``objective``.
    """

    scenario: str = "cstr"
    objective: str = "economic"
    env_reward_mode: str = "economic"
    action_mode: str | None = None
    control_dt: float | None = None
    episode_steps: int | None = None
    task: Any = None
    dynamic: bool | None = None
    randomize: bool | None = None
    randomize_setpoints: bool | None = None
    randomize_plant: bool | None = None
    plant_drift: bool | None = None
    integral_obs: bool | None = None
    terminate_on_runaway: bool | None = None
    tracking_q_y: Any = 1.0
    tracking_r_move: float = 0.05
    noise: bool | None = None
    noise_pct: float | None = None
    model_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.scenario, str) or not self.scenario:
            raise ValueError("scenario must be a non-empty string")
        if self.objective not in PRIMARY_METRICS:
            raise ValueError(
                f"objective must be one of: {', '.join(PRIMARY_METRICS)}"
            )
        if self.env_reward_mode not in {"economic", "kpi", "tracking"}:
            raise ValueError("env_reward_mode must be one of: economic, kpi, tracking")
        task = None
        task_defaults = {}
        if self.task is not None:
            from .task_profiles import load_task_profile, task_environment

            task = load_task_profile(self.task, scenario=self.scenario)
            task_defaults = task_environment(task)
            supported = task.get("supported_objectives")
            if supported is not None and self.objective not in supported:
                raise ValueError(
                    f"task {task['name']!r} does not support objective {self.objective!r}; "
                    f"supported: {', '.join(supported)}"
                )
        implicit = _IMPLICIT_TASK_DEFAULTS[self.objective]
        resolved_conditions = {}
        for name in (
            "action_mode", "dynamic", "randomize", "randomize_setpoints",
            "randomize_plant", "plant_drift", "integral_obs",
            "terminate_on_runaway", "noise",
        ):
            explicit = getattr(self, name)
            resolved_conditions[name] = (
                explicit if explicit is not None else task_defaults.get(name, implicit[name])
            )
        if resolved_conditions["action_mode"] not in {"actuator", "setpoint"}:
            raise ValueError("action_mode must be one of: actuator, setpoint")
        control_dt = float(
            self.control_dt if self.control_dt is not None else task_defaults.get("control_dt", 0.5)
        )
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        episode_steps = (
            self.episode_steps
            if self.episode_steps is not None
            else task_defaults.get("episode_steps", 400)
        )
        if (
            isinstance(episode_steps, bool)
            or not isinstance(episode_steps, Integral)
            or int(episode_steps) <= 0
        ):
            raise ValueError("episode_steps must be a positive integer")
        noise_pct = float(
            self.noise_pct if self.noise_pct is not None else task_defaults.get("noise_pct", 0.01)
        )
        if not math.isfinite(noise_pct) or noise_pct < 0:
            raise ValueError("noise_pct must be finite and non-negative")
        tracking_r_move = float(self.tracking_r_move)
        if not math.isfinite(tracking_r_move) or tracking_r_move < 0:
            raise ValueError("tracking_r_move must be finite and non-negative")
        if not isinstance(self.model_params, Mapping):
            raise TypeError("model_params must be a mapping")
        model_params = dict((task or {}).get("model_params", {}))
        model_params.update(dict(self.model_params))
        object.__setattr__(self, "control_dt", control_dt)
        object.__setattr__(self, "episode_steps", int(episode_steps))
        object.__setattr__(self, "task", task)
        for name, value in resolved_conditions.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "noise_pct", noise_pct)
        object.__setattr__(self, "tracking_r_move", tracking_r_move)
        object.__setattr__(self, "model_params", model_params)

    @classmethod
    def economic(cls, scenario: str, **kw):
        defaults = dict(objective="economic", env_reward_mode="economic")
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def tracking(cls, scenario: str, **kw):
        defaults = dict(objective="tracking", env_reward_mode="tracking")
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def kpi(cls, scenario: str, **kw):
        defaults = dict(objective="kpi", env_reward_mode="kpi")
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def robustness(cls, scenario: str, **kw):
        defaults = dict(objective="robustness", env_reward_mode="kpi")
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def safety(cls, scenario: str, **kw):
        defaults = dict(objective="safety", env_reward_mode="kpi")
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    def env_kwargs(self, action_mode: str | None = None):
        data = asdict(self)
        data.pop("scenario")
        data.pop("objective")
        data["reward_mode"] = data.pop("env_reward_mode")
        if action_mode is not None:
            data["action_mode"] = action_mode
        return data

    def make_env(self, action_mode: str | None = None):
        from ..env import AIOGymNativeEnv

        return AIOGymNativeEnv(self.scenario, **self.env_kwargs(action_mode=action_mode))

    def metadata(self):
        from .task_profiles import task_identity

        primary_metric = primary_metric_for_objective(self.objective)
        data = asdict(self)
        data["task_identity"] = task_identity(self.task)
        data["metrics"] = list(PROTOCOL_METRICS.get(self.objective, ()))
        data["primary_metric"] = primary_metric
        data["primary_metric_direction"] = metric_direction(primary_metric)
        data["metric_definitions"] = metric_definitions(self.objective)
        return data


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fully reproducible benchmark declaration.

    ``BenchmarkProtocol`` describes the environment. ``BenchmarkConfig`` binds
    that protocol to a controller, seed list, disturbance policy, and metrics.
    """

    protocol: BenchmarkProtocol = field(default_factory=BenchmarkProtocol)
    controller: str = "pid"
    seeds: tuple[int, ...] = (0,)
    controller_config: Mapping[str, Any] = field(default_factory=dict)
    disturbance: str = "task_and_model_schema"
    metrics: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.controller, str) or not self.controller.strip():
            raise ValueError("controller must be a non-empty string")
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds:
            raise ValueError("seeds must contain at least one seed")
        object.__setattr__(self, "seeds", seeds)

    @classmethod
    def from_protocol(cls, protocol: BenchmarkProtocol, controller: str = "pid",
                      seeds: Sequence[int] | None = None, **kw):
        return cls(protocol=protocol, controller=controller,
                   seeds=tuple((0,) if seeds is None else seeds), **kw)

    def metadata(self):
        metrics = self.metrics or PROTOCOL_METRICS.get(self.protocol.objective, ())
        primary_metric = primary_metric_for_objective(self.protocol.objective)
        task_meta = self.protocol.metadata()["task_identity"]
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "objective": self.protocol.objective,
            "scenario": self.protocol.scenario,
            "task": task_meta["name"],
            "task_status": task_meta["status"],
            "task_profile_hash": task_meta["profile_hash"],
            "controller": self.controller,
            "controller_config": _jsonable(dict(self.controller_config)),
            "seed_list": list(self.seeds),
            "disturbance": self.disturbance,
            "metrics": list(metrics),
            "primary_metric": primary_metric,
            "primary_metric_direction": metric_direction(primary_metric),
            "episode_steps": self.protocol.episode_steps,
            "protocol": self.protocol.metadata(),
            "metric_definitions": {key: METRIC_DEFINITIONS[key] for key in metrics if key in METRIC_DEFINITIONS},
        }


def resolve_protocol(
    scenario: str,
    objective,
    data: Mapping[str, Any],
) -> BenchmarkProtocol:
    """Resolve a protocol object from an objective name or config mapping."""

    if isinstance(objective, BenchmarkProtocol):
        return objective
    cfg = dict(data)
    if isinstance(objective, Mapping):
        cfg.update(objective)
        objective_name = cfg.pop("objective", "tracking")
    elif objective is None:
        objective_name = cfg.pop("objective", "tracking")
    else:
        objective_name = cfg.pop("objective", str(objective))
    factories = {
        "economic": BenchmarkProtocol.economic,
        "tracking": BenchmarkProtocol.tracking,
        "robustness": BenchmarkProtocol.robustness,
        "safety": BenchmarkProtocol.safety,
        "kpi": BenchmarkProtocol.kpi,
    }
    if objective_name not in factories:
        raise ValueError(f"objective must be one of: {', '.join(factories)}")
    factory = factories[objective_name]
    return factory(scenario, **cfg)
