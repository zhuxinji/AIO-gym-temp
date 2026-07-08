"""Controller evaluation protocols for AIO-Gym.

This module is the common path for fixed actuator controllers (PID/MPC/oracle)
and learned policies (SB3/RLPD style).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence
import numpy as np

from ..controllers import as_controller, build_context, validate_action
from .metrics.economic import economic_step_metrics
from .metrics.robustness import robustness_extrema
from .metrics.safety import action_bound_metrics as _action_bound_metrics
from .metrics.safety import safety_step_metrics as _safety_step_metrics
from .metrics.tracking import tracking_step_metrics as _tracking_step_metrics


EVALUATION_SCHEMA_VERSION = "aiogym.evaluation.v2"


ROLLOUT_SCHEMA = {
    "step": "integer control-step index",
    "time": "seconds since episode start",
    "obs": "observation before control action",
    "state": "physical state before integration",
    "action": "controller output passed to env.step",
    "setpoint": "active setpoint context exposed to the controller",
    "disturbance": "disturbance values applied by the process model",
    "reward": "training reward returned by the environment",
    "profit": "raw economic profit for the step",
    "constraint": "normalized process constraint penalty for the step",
    "info": "environment-specific diagnostic fields",
}


METRIC_DEFINITIONS = {
    "return": "sum of environment reward over the rollout; reward is the training signal",
    "profit": "sum of raw economic profit reported by the environment",
    "normalized_score": "0-100 KPI score from KPIScorer; score is for reporting, not raw economics",
    "production": "sum of process production reported by the environment",
    "energy_kwh": "total heat plus pump energy over the rollout",
    "runtime_seconds": "wall-clock seconds spent evaluating one episode",
    "runtime_total_seconds": "total wall-clock seconds spent evaluating all episodes",
    "runtime_seconds_per_step": "wall-clock seconds per environment control step",
    "tracking_iae": "integral absolute error for temperatures and controlled levels",
    "tracking_ise": "integral squared error for temperatures and controlled levels",
    "tracking_itae": "time-weighted integral absolute error",
    "tracking_overshoot": "largest positive excursion above the active setpoint",
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
        "tracking_iae", "tracking_ise", "tracking_itae", "tracking_overshoot",
        "tracking_settling_time",
    ),
    "economic": (
        "profit", "normalized_score", "production", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity", "safety_margin_min",
        "controller_solver_failure_count", "controller_fallback_count",
        "runtime_seconds", "runtime_seconds_per_step",
    ),
    "robustness": (
        "return", "profit", "normalized_score", "tracking_iae", "energy_kwh",
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
        "normalized_score", "tracking_iae", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity",
        "controller_solver_failure_count", "controller_fallback_count",
        "runtime_seconds", "runtime_seconds_per_step",
    ),
}


PRIMARY_METRICS = {
    "tracking": "tracking_iae",
    "economic": "profit",
    "kpi": "normalized_score",
    "robustness": "normalized_score",
    "safety": "constraint_violation_count",
}


METRIC_DIRECTIONS = {
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
    if reward_mode == "track":
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
        "heat_kwh": 0.0,
        "pump_kwh": 0.0,
        "runtime_seconds": 0.0,
        "runtime_seconds_per_step": 0.0,
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
    legacy_reward_mode = data.pop("reward_mode", None)
    if legacy_reward_mode is not None and "env_reward_mode" not in overrides:
        data["env_reward_mode"] = legacy_reward_mode
    return data


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
    action_mode: str = "actuator"
    control_dt: float = 0.5
    episode_steps: int = 400
    dynamic: bool = True
    randomize: bool = True
    randomize_setpoints: bool = False
    randomize_plant: bool = True
    plant_drift: bool = True
    integral_obs: bool = False
    terminate_on_runaway: bool = False
    noise: bool = False
    noise_pct: float = 0.01

    @classmethod
    def economic(cls, scenario: str, **kw):
        defaults = dict(objective="economic", env_reward_mode="economic",
                        randomize_setpoints=False, randomize_plant=True, plant_drift=True)
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def tracking(cls, scenario: str, **kw):
        defaults = dict(objective="tracking", env_reward_mode="track",
                        dynamic=False, randomize=False, randomize_setpoints=False,
                        randomize_plant=False, plant_drift=False)
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def kpi(cls, scenario: str, **kw):
        defaults = dict(objective="kpi", env_reward_mode="kpi",
                        randomize_setpoints=True, randomize_plant=True, plant_drift=True)
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def robustness(cls, scenario: str, **kw):
        defaults = dict(objective="robustness", env_reward_mode="kpi",
                        dynamic=True, randomize=True, randomize_setpoints=True,
                        randomize_plant=True, plant_drift=True, noise=True)
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def safety(cls, scenario: str, **kw):
        defaults = dict(objective="safety", env_reward_mode="kpi",
                        dynamic=True, randomize=True, randomize_setpoints=True,
                        randomize_plant=True, plant_drift=True,
                        terminate_on_runaway=False)
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @property
    def reward_mode(self):
        """Backward-compatible alias for the environment reward mode."""

        return self.env_reward_mode

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
        primary_metric = primary_metric_for_objective(self.objective)
        data = asdict(self)
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
    disturbance: str = "model_schema_dynamic"
    metrics: tuple[str, ...] = ()

    @classmethod
    def from_protocol(cls, protocol: BenchmarkProtocol, controller: str = "pid",
                      seeds: Sequence[int] | None = None, **kw):
        return cls(protocol=protocol, controller=controller,
                   seeds=tuple(seeds or (0,)), **kw)

    def metadata(self):
        metrics = self.metrics or PROTOCOL_METRICS.get(self.protocol.objective, ())
        primary_metric = primary_metric_for_objective(self.protocol.objective)
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "objective": self.protocol.objective,
            "scenario": self.protocol.scenario,
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


def evaluate_controller(agent, env, episodes: int = 20, seed: int = 0,
                        include_episodes: bool = False, protocol: BenchmarkProtocol | None = None,
                        seed_list: Sequence[int] | None = None):
    """Evaluate any supported controller/policy on an AIOGymNativeEnv.

    Returns aggregate metrics plus the protocol/controller metadata needed to
    reproduce the benchmark.
    """

    controller = as_controller(agent, action_mode=getattr(env, "action_mode", "actuator"))
    seeds = list(seed_list) if seed_list is not None else [seed + ep for ep in range(episodes)]
    per_episode = []
    episode_schedules = []
    eval_start = perf_counter()
    for ep, ep_seed in enumerate(seeds):
        episode_start = perf_counter()
        obs, reset_info = env.reset(seed=ep_seed)
        controller.reset(seed=ep_seed)
        totals = _empty_episode_totals(ep, ep_seed)
        episode_schedules.append(_jsonable(getattr(env, "_dist_events", [])))
        done = False
        steps = 0
        info = reset_info or {}
        last_unsettled_time = 0.0
        while not done:
            context = build_context(env, info)
            action = validate_action(controller.act(obs, context), env, controller.name)
            bound_metrics = _action_bound_metrics(action, env)
            obs, reward, term, trunc, info = env.step(action)
            time_sec = steps * float(env.control_dt)
            active_setpoint = {"h_sp": list(getattr(env, "h_sp", [])), "t_sp": list(getattr(env, "t_sp", []))}
            tracking = _tracking_step_metrics(info, active_setpoint, time_sec, float(env.control_dt), env)
            safety = _safety_step_metrics(info, bound_metrics, float(env.control_dt))
            totals["return"] += float(reward)
            totals["track"] += float(info.get("track", 0.0))
            totals["constraint"] += float(info.get("constraint", 0.0))
            economic = economic_step_metrics(info, float(env.control_dt))
            for key, value in economic.items():
                totals[key] += value
            for key in ("tracking_iae", "tracking_ise", "tracking_itae"):
                totals[key] += tracking[key]
            totals["tracking_overshoot"] = max(totals["tracking_overshoot"], tracking["tracking_overshoot"])
            if not tracking["tracking_settled"]:
                last_unsettled_time = time_sec
            for key, value in safety.items():
                if key == "safety_margin_min":
                    totals[key] = min(totals[key], value)
                else:
                    totals[key] += value
            done = bool(term or trunc)
            steps += 1
        rep = env.scorer.report()
        runtime_seconds = perf_counter() - episode_start
        totals["kpi"] = float(rep["score"])
        totals["normalized_score"] = float(rep["score"])
        totals["steps"] = steps
        totals["runtime_seconds"] = float(runtime_seconds)
        totals["runtime_seconds_per_step"] = float(runtime_seconds / steps) if steps else 0.0
        totals["tracking_settling_time"] = float(last_unsettled_time)
        controller_diag = _controller_diagnostics(controller)
        totals["controller_diagnostics"] = controller_diag
        totals.update(_controller_diagnostic_totals(controller_diag))
        per_episode.append(totals)
    runtime_total_seconds = perf_counter() - eval_start

    def mean(key):
        return float(np.mean([row[key] for row in per_episode]))

    def std(key):
        return float(np.std([row[key] for row in per_episode]))

    objective = protocol.objective if protocol is not None else _env_objective(env)
    primary_metric = primary_metric_for_objective(objective)
    aggregate_keys = _aggregate_metric_keys(per_episode)
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "name": controller.name,
        "metric": primary_metric,
        "metric_direction": metric_direction(primary_metric),
        "objective": objective,
        "episodes": len(seeds),
        "seed": int(seeds[0]) if seeds else int(seed),
        "seed_list": [int(s) for s in seeds],
        "kpi": mean("kpi"),
        "kpi_std": std("kpi"),
        "normalized_score": mean("normalized_score"),
        "normalized_score_std": std("normalized_score"),
        "profit": mean("profit"),
        "profit_std": std("profit"),
        "return": mean("return"),
        "return_std": std("return"),
        "track": mean("track"),
        "track_std": std("track"),
        "constraint": mean("constraint"),
        "constraint_std": std("constraint"),
        "prod": mean("prod"),
        "prod_std": std("prod"),
        "production": mean("production"),
        "production_std": std("production"),
        "runtime_total_seconds": float(runtime_total_seconds),
        "protocol": protocol.metadata() if protocol is not None else _env_metadata(env),
        "controller": controller.metadata(),
        "model": _model_metadata(env),
        "disturbance": {
            "schedule_source": "model_schema_dynamic",
            "episode_schedules": episode_schedules,
        },
        "controller_diagnostics": _aggregate_controller_diagnostics(per_episode),
        "metric_definitions": metric_definitions(objective),
        "result_schema": result_schema(),
        "reproducibility": _reproducibility_metadata(env, seeds, protocol),
    }
    result["controller_status"] = "degraded" if result["controller_diagnostics"].get("degraded") else "ok"
    for key in aggregate_keys:
        result.setdefault(key, mean(key))
        result.setdefault(f"{key}_std", std(key))
    if include_episodes:
        result["episode_metrics"] = per_episode
    return result


def rollout_controller(agent, env, seed: int = 0, max_steps: int | None = None,
                       protocol: BenchmarkProtocol | None = None):
    """Run one episode and return a generic per-step rollout artifact.

    The recorder is scenario-neutral. Common fields are always present, and
    scenario-specific data from env ``info`` is preserved under each step.
    """

    controller = as_controller(agent, action_mode=getattr(env, "action_mode", "actuator"))
    obs, reset_info = env.reset(seed=seed)
    controller.reset(seed=seed)
    rows = []
    info = reset_info or {}
    done = False
    step = 0
    limit = max_steps if max_steps is not None else getattr(env, "episode_steps", None)
    while not done and (limit is None or step < limit):
        context = build_context(env, info)
        action = validate_action(controller.act(obs, context), env, controller.name)
        state = list(getattr(env.integ, "x", []))
        obs_next, reward, term, trunc, info_next = env.step(action)
        rows.append(_jsonable({
            "step": step,
            "time": step * float(env.control_dt),
            "obs": obs,
            "state": state,
            "action": action,
            "setpoint": context.setpoint,
            "measurement": context.measurement,
            "disturbance": _env_disturbances(env),
            "reward": reward,
            "profit": info_next.get("profit", 0.0),
            "constraint": info_next.get("constraint", 0.0),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "next_obs": obs_next,
            "next_state": list(getattr(env.integ, "x", [])),
            "info": info_next,
        }))
        obs = obs_next
        info = info_next
        done = bool(term or trunc)
        step += 1

    return {
        "name": controller.name,
        "seed": int(seed),
        "steps": len(rows),
        "protocol": protocol.metadata() if protocol is not None else _env_metadata(env),
        "controller": controller.metadata(),
        "scorer": _jsonable(env.scorer.report()),
        "rollout_schema": result_schema()["rollout"],
        "rollout": rows,
    }


def result_schema():
    return {
        "version": EVALUATION_SCHEMA_VERSION,
        "rollout": dict(ROLLOUT_SCHEMA),
        "episode_metrics": dict(METRIC_DEFINITIONS),
        "tables": {
            "tracking": list(PROTOCOL_METRICS["tracking"]),
            "economic": list(PROTOCOL_METRICS["economic"]),
            "safety": list(PROTOCOL_METRICS["safety"]),
            "robustness": list(PROTOCOL_METRICS["robustness"]),
        },
    }


def build_evaluation_report(results: Sequence[Mapping[str, Any]]):
    """Return roadmap-style tracking/economic/safety/robustness tables."""

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "tracking": [_table_row(result, PROTOCOL_METRICS["tracking"]) for result in results],
        "economic": [_table_row(result, PROTOCOL_METRICS["economic"]) for result in results],
        "safety": [_table_row(result, PROTOCOL_METRICS["safety"]) for result in results],
        "robustness": [_robustness_row(result) for result in results],
        "metric_definitions": dict(METRIC_DEFINITIONS),
    }


def _env_metadata(env):
    keys = (
        "scenario", "reward_mode", "action_mode", "control_dt", "episode_steps",
        "dynamic", "randomize", "randomize_setpoints", "randomize_plant",
        "plant_drift", "integral_obs", "terminate_on_runaway",
        "noise", "noise_pct",
    )
    return {key: getattr(env, key) for key in keys if hasattr(env, key)}


def _env_objective(env):
    reward_mode = getattr(env, "reward_mode", "")
    if reward_mode == "economic":
        return "economic"
    if reward_mode == "track":
        return "tracking"
    return "kpi"


def _aggregate_metric_keys(per_episode):
    keys = set()
    for row in per_episode:
        for key, value in row.items():
            if key in ("episode", "seed", "steps"):
                continue
            if isinstance(value, (int, float, np.number)):
                keys.add(key)
    return sorted(keys)


def _controller_diagnostics(controller):
    targets = [controller, getattr(controller, "agent", None), getattr(controller, "policy", None)]
    for target in targets:
        if target is None or not hasattr(target, "diagnostics"):
            continue
        try:
            return _jsonable(target.diagnostics())
        except TypeError:
            continue
    return {}


def _controller_diagnostic_totals(diagnostics):
    data = dict(diagnostics or {})
    degraded = bool(
        data.get("degraded")
        or float(data.get("solver_failure_count", 0.0) or 0.0) > 0.0
        or float(data.get("fallback_count", 0.0) or 0.0) > 0.0
    )
    return {
        "controller_solve_count": float(data.get("solve_count", 0.0) or 0.0),
        "controller_solver_success_count": float(data.get("solver_success_count", 0.0) or 0.0),
        "controller_solver_failure_count": float(data.get("solver_failure_count", 0.0) or 0.0),
        "controller_fallback_count": float(data.get("fallback_count", 0.0) or 0.0),
        "controller_degraded_count": 1.0 if degraded else 0.0,
    }


def _aggregate_controller_diagnostics(per_episode):
    rows = [dict(row.get("controller_diagnostics") or {}) for row in per_episode]
    totals = {
        "solve_count": 0.0,
        "solver_success_count": 0.0,
        "solver_failure_count": 0.0,
        "fallback_count": 0.0,
    }
    degraded = False
    last_error = None
    for row in rows:
        for key in totals:
            value = row.get(key, 0.0)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.number)):
                totals[key] += float(value)
        row_degraded = bool(
            row.get("degraded")
            or float(row.get("solver_failure_count", 0.0) or 0.0) > 0.0
            or float(row.get("fallback_count", 0.0) or 0.0) > 0.0
        )
        degraded = degraded or row_degraded
        if row.get("last_solver_error"):
            last_error = row.get("last_solver_error")
    out = {
        "degraded": degraded,
        "last_solver_error": last_error,
        "per_episode": rows,
    }
    out.update({key: int(value) if value.is_integer() else value for key, value in totals.items()})
    return _jsonable(out)


def _model_metadata(env):
    model = getattr(env, "model", None)
    if model is None or not hasattr(model, "metadata"):
        return {}
    return _jsonable(model.metadata())


def _env_disturbances(env):
    if hasattr(env, "_env"):
        return _jsonable(env._env())
    return {}


def _reproducibility_metadata(env, seeds, protocol):
    return {
        "git_commit": _git_commit(),
        "seed_list": [int(seed) for seed in seeds],
        "model_version": getattr(getattr(env, "model", None), "scenario", None),
        "episode_length": int(getattr(env, "episode_steps", 0)),
        "disturbance_schedule": "model_schema_dynamic",
        "metric_definition_version": EVALUATION_SCHEMA_VERSION,
        "protocol": protocol.metadata() if protocol is not None else _env_metadata(env),
    }


def _git_commit():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = out.stdout.strip()
    return commit or None


def _table_row(result: Mapping[str, Any], keys: Sequence[str]):
    row = {
        "name": result.get("name"),
        "objective": result.get("objective"),
        "control_structure": dict(result.get("controller", {})).get("control_structure"),
        "controller_status": result.get("controller_status", "ok"),
        "controller_diagnostics": result.get("controller_diagnostics", {}),
        "episodes": result.get("episodes"),
        "seed_list": result.get("seed_list", []),
    }
    for key in keys:
        if key in result:
            row[key] = result[key]
        std_key = f"{key}_std"
        if std_key in result:
            row[std_key] = result[std_key]
    return row


def _robustness_row(result: Mapping[str, Any]):
    row = _table_row(result, PROTOCOL_METRICS["robustness"])
    row.update(robustness_extrema(result.get("episode_metrics", []), PROTOCOL_METRICS["robustness"]))
    return row


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
