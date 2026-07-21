"""Benchmark protocol declarations and metric catalog."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


from .._internal.serialization import jsonable as _jsonable

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
        "tracking_steady_cost",
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
    "tracking": "tracking_error_cost",
    "economic": "profit",
    "kpi": "normalized_score",
    "robustness": "normalized_score",
    "safety": "constraint_violation_count",
}


OBJECTIVE_REWARD_MODES = {
    "tracking": "tracking",
    "economic": "economic",
    "kpi": "kpi",
    "robustness": "kpi",
    "safety": "kpi",
}


METRIC_DIRECTIONS = {
    "tracking_cost": "minimize",
    "tracking_error_cost": "minimize",
    "tracking_move_cost": "minimize",
    "tracking_steady_cost": "minimize",
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
    """Build validated objective semantics without constructing an environment."""

    if name not in PRIMARY_METRICS:
        raise ValueError(f"objective must be one of: {', '.join(PRIMARY_METRICS)}")
    primary_metric = primary_metric_for_objective(name)
    return ObjectiveSpec(
        name=name,
        source=str(source),
        reward_mode=OBJECTIVE_REWARD_MODES[name],
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
    """Resolve one objective using a single, user-visible precedence order."""

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


def _empty_episode_totals(ep: int, seed: int):
    return {
        "episode": int(ep),
        "seed": int(seed),
        "return": 0.0,
        "track": 0.0,
        "constraint": 0.0,
        "profit": 0.0,
        "production": 0.0,
        "energy_kwh": 0.0,
        "runtime_seconds": 0.0,
        "runtime_seconds_per_step": 0.0,
        "tracking_cost": 0.0,
        "tracking_return": 0.0,
        "tracking_error_cost": 0.0,
        "tracking_move_cost": 0.0,
        "tracking_steady_cost": 0.0,
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


# Objective selection must not silently change the experiment. Task profiles or
# explicit environment overrides own these neutral protocol defaults.
_DEFAULT_ENVIRONMENT_CONDITIONS = {
    "action_mode": "actuator",
    "dynamic": False,
    "randomize": False,
    "randomize_setpoints": False,
    "randomize_plant": False,
    "plant_drift": False,
    "integral_obs": False,
    "terminate_on_runaway": False,
    "noise": False,
    "noise_pct": 0.01,
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
    objective_source: str = "explicit"
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
    tracking_r_move: float = 1.0
    tracking_r_steady: float = 1.0
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
        from ..models import make_model

        model_supported = tuple(
            getattr(make_model(self.scenario), "supported_objectives", tuple(PRIMARY_METRICS))
        )
        if self.objective not in model_supported:
            raise ValueError(
                f"scenario {self.scenario!r} does not support objective {self.objective!r}; "
                f"supported: {', '.join(model_supported)}"
            )
        if self.env_reward_mode not in {"economic", "kpi", "tracking"}:
            raise ValueError("env_reward_mode must be one of: economic, kpi, tracking")
        expected_reward_mode = OBJECTIVE_REWARD_MODES[self.objective]
        if self.env_reward_mode != expected_reward_mode:
            raise ValueError(
                f"objective {self.objective!r} requires env_reward_mode "
                f"{expected_reward_mode!r}, got {self.env_reward_mode!r}"
            )
        from .task_profiles import resolve_environment_options

        task, environment_options = resolve_environment_options(
            scenario=self.scenario,
            task=self.task,
            explicit={
                "control_dt": self.control_dt,
                "episode_steps": self.episode_steps,
                "action_mode": self.action_mode,
                "dynamic": self.dynamic,
                "randomize": self.randomize,
                "randomize_setpoints": self.randomize_setpoints,
                "randomize_plant": self.randomize_plant,
                "plant_drift": self.plant_drift,
                "integral_obs": self.integral_obs,
                "terminate_on_runaway": self.terminate_on_runaway,
                "noise": self.noise,
                "noise_pct": self.noise_pct,
                "model_params": self.model_params,
            },
            defaults=_DEFAULT_ENVIRONMENT_CONDITIONS,
            default_control_dt=0.5,
            default_episode_steps=400,
        )
        if task is not None:
            supported = task.get("supported_objectives")
            if supported is not None and self.objective not in supported:
                raise ValueError(
                    f"task {task['name']!r} does not support objective {self.objective!r}; "
                    f"supported: {', '.join(supported)}"
                )
        tracking_r_move = float(self.tracking_r_move)
        if not math.isfinite(tracking_r_move) or tracking_r_move < 0:
            raise ValueError("tracking_r_move must be finite and non-negative")
        tracking_r_steady = float(self.tracking_r_steady)
        if not math.isfinite(tracking_r_steady) or tracking_r_steady < 0:
            raise ValueError("tracking_r_steady must be finite and non-negative")
        object.__setattr__(self, "task", task)
        for name, value in environment_options.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "tracking_r_move", tracking_r_move)
        object.__setattr__(self, "tracking_r_steady", tracking_r_steady)

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
        data.pop("objective_source")
        data["reward_mode"] = data.pop("env_reward_mode")
        if action_mode is not None:
            data["action_mode"] = action_mode
        return data

    def make_env(self, action_mode: str | None = None):
        from ..env import AIOGymNativeEnv

        return AIOGymNativeEnv(self.scenario, **self.env_kwargs(action_mode=action_mode))

    def resolved_objective(self) -> ObjectiveSpec:
        return objective_spec(
            self.objective,
            source=self.objective_source,
            reward_options={
                "tracking_q_y": self.tracking_q_y,
                "tracking_r_move": self.tracking_r_move,
                "tracking_r_steady": self.tracking_r_steady,
            },
        )

    def environment_spec(self) -> "EnvironmentSpec":
        return EnvironmentSpec.from_protocol(self)

    def metadata(self):
        from .task_profiles import task_identity

        primary_metric = primary_metric_for_objective(self.objective)
        data = asdict(self)
        data["task_identity"] = task_identity(self.task)
        data["objective_spec"] = self.resolved_objective().metadata()
        data["metrics"] = list(PROTOCOL_METRICS.get(self.objective, ()))
        data["primary_metric"] = primary_metric
        data["primary_metric_direction"] = metric_direction(primary_metric)
        data["metric_definitions"] = metric_definitions(self.objective)
        return data


@dataclass(frozen=True)
class EnvironmentSpec:
    """Task-owned environment construction data, independent of ranking."""

    scenario: str
    task: Any = None
    action_mode: str = "actuator"
    control_dt: float = 0.5
    episode_steps: int = 400
    dynamic: bool = False
    randomize: bool = False
    randomize_setpoints: bool = False
    randomize_plant: bool = False
    plant_drift: bool = False
    integral_obs: bool = False
    terminate_on_runaway: bool = False
    noise: bool = False
    noise_pct: float = 0.01
    model_params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_protocol(cls, protocol: BenchmarkProtocol) -> "EnvironmentSpec":
        return cls(
            scenario=protocol.scenario,
            task=protocol.task,
            action_mode=str(protocol.action_mode),
            control_dt=float(protocol.control_dt),
            episode_steps=int(protocol.episode_steps),
            dynamic=bool(protocol.dynamic),
            randomize=bool(protocol.randomize),
            randomize_setpoints=bool(protocol.randomize_setpoints),
            randomize_plant=bool(protocol.randomize_plant),
            plant_drift=bool(protocol.plant_drift),
            integral_obs=bool(protocol.integral_obs),
            terminate_on_runaway=bool(protocol.terminate_on_runaway),
            noise=bool(protocol.noise),
            noise_pct=float(protocol.noise_pct),
            model_params=dict(protocol.model_params),
        )

    def env_kwargs(self, objective: ObjectiveSpec) -> dict[str, Any]:
        data = asdict(self)
        data.pop("scenario")
        data["reward_mode"] = objective.reward_mode
        data.update(dict(objective.reward_options))
        return data

    def make_env(self, objective: ObjectiveSpec):
        from ..env import AIOGymNativeEnv

        return AIOGymNativeEnv(
            self.scenario,
            **self.env_kwargs(objective),
        )


@dataclass(frozen=True)
class BenchmarkCase:
    """One resolved task/objective/controller/seed evaluation unit."""

    case_id: str
    environment: EnvironmentSpec
    objective: ObjectiveSpec
    controller: str
    seeds: tuple[int, ...]
    controller_config: Mapping[str, Any] = field(default_factory=dict)
    protocol: BenchmarkProtocol | None = field(default=None, repr=False)

    def __post_init__(self):
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds:
            raise ValueError("benchmark case seeds must not be empty")
        if not isinstance(self.controller, str) or not self.controller.strip():
            raise ValueError("benchmark case controller must be a non-empty string")
        object.__setattr__(self, "seeds", seeds)

    @classmethod
    def from_protocol(
        cls,
        protocol: BenchmarkProtocol,
        *,
        controller: str,
        seeds: Sequence[int],
        controller_config: Mapping[str, Any] | None = None,
        case_id: str | None = None,
    ) -> "BenchmarkCase":
        from .task_profiles import task_identity

        task_name = task_identity(protocol.task)["name"]
        return cls(
            case_id=case_id
            or f"{protocol.objective}:{protocol.scenario}:{task_name}:{controller}",
            environment=protocol.environment_spec(),
            objective=protocol.resolved_objective(),
            controller=controller,
            seeds=tuple(seeds),
            controller_config=dict(controller_config or {}),
            protocol=protocol,
        )

    def metadata(self) -> dict[str, Any]:
        from .task_profiles import task_identity

        task_meta = task_identity(self.environment.task)
        metrics = PROTOCOL_METRICS.get(self.objective.name, ())
        primary_metric = self.objective.primary_metric
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "case_id": self.case_id,
            "scenario": self.environment.scenario,
            "task": task_meta["name"],
            "task_status": task_meta["status"],
            "task_profile_hash": task_meta["profile_hash"],
            "objective": self.objective.name,
            "objective_source": self.objective.source,
            "objective_spec": self.objective.metadata(),
            "controller": self.controller,
            "controller_config": _jsonable(dict(self.controller_config)),
            "seed_list": list(self.seeds),
            "environment": _jsonable(asdict(self.environment)),
            "metrics": list(metrics),
            "primary_metric": primary_metric,
            "primary_metric_direction": metric_direction(primary_metric),
            "episode_steps": self.environment.episode_steps,
            "protocol": self.protocol.metadata() if self.protocol is not None else None,
            "metric_definitions": {key: METRIC_DEFINITIONS[key] for key in metrics if key in METRIC_DEFINITIONS},
        }


def resolve_protocol(
    scenario: str,
    objective=None,
    data: Mapping[str, Any] | None = None,
    *,
    case_objective: str | None = None,
    suite_objective: str | None = None,
) -> BenchmarkProtocol:
    """Resolve a protocol after loading task-owned objective defaults."""

    if isinstance(objective, BenchmarkProtocol):
        return objective
    cfg = dict(data or {})
    explicit = None
    configured = cfg.pop("objective", None)
    if isinstance(objective, Mapping):
        cfg.update(objective)
        configured = cfg.pop("objective", configured)
    elif objective is not None:
        explicit = str(objective)

    if "reward_mode" in cfg:
        reward_mode = cfg.pop("reward_mode")
        cfg.setdefault("env_reward_mode", reward_mode)

    task = None
    if cfg.get("task") is not None:
        from .task_profiles import load_task_profile

        task = load_task_profile(cfg["task"], scenario=scenario)
        cfg["task"] = task
    resolved = resolve_objective(
        explicit=explicit,
        case_config=case_objective if case_objective is not None else configured,
        suite_config=suite_objective,
        task_profile=task,
    )
    factories = {
        "economic": BenchmarkProtocol.economic,
        "tracking": BenchmarkProtocol.tracking,
        "robustness": BenchmarkProtocol.robustness,
        "safety": BenchmarkProtocol.safety,
        "kpi": BenchmarkProtocol.kpi,
    }
    factory = factories[resolved.name]
    cfg["objective_source"] = resolved.source
    return factory(scenario, **cfg)
