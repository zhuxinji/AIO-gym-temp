"""Controller evaluation protocols for AIO-Gym.

This module is the common path for fixed actuator controllers (PID/MPC/oracle)
and learned policies (SB3/RLPD style).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import subprocess
from typing import Any, Mapping, Sequence
import numpy as np

from .controllers import as_controller, build_context, validate_action


EVALUATION_SCHEMA_VERSION = "aiogym.evaluation.v2"


METRIC_DEFINITIONS = {
    "return": "sum of environment reward over the rollout; reward is the training signal",
    "profit": "sum of raw economic profit reported by the environment",
    "normalized_score": "0-100 KPI score from KPIScorer; score is for reporting, not raw economics",
    "production": "sum of process production reported by the environment",
    "energy_kwh": "total heat plus pump energy over the rollout",
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
}


PROTOCOL_METRICS = {
    "tracking": (
        "tracking_iae", "tracking_ise", "tracking_itae", "tracking_overshoot",
        "tracking_settling_time", "energy_kwh", "constraint_violation_count",
        "constraint_violation_severity",
    ),
    "economic": (
        "profit", "normalized_score", "production", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity", "safety_margin_min",
    ),
    "robustness": (
        "return", "profit", "normalized_score", "tracking_iae", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity",
    ),
    "safety": (
        "constraint_violation_count", "constraint_violation_duration",
        "constraint_violation_severity", "action_violation_count",
        "action_violation_duration", "action_violation_severity",
        "runaway_count", "runaway_duration", "safety_margin_min",
    ),
    "kpi": (
        "normalized_score", "tracking_iae", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity",
    ),
}


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


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Reproducible benchmark environment configuration.

    Use ``economic`` for supervisory/RTO benchmarks and ``tracking`` for
    setpoint-following benchmarks.  Baseline PID/MPC/oracle controllers usually
    run with ``action_mode="actuator"``; learned supervisory policies usually run
    with ``action_mode="setpoint"`` on the same protocol.
    """

    scenario: str = "cstr"
    objective: str = "economic"
    reward_mode: str = "economic"
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
        defaults = dict(objective="economic", reward_mode="economic",
                        randomize_setpoints=False, randomize_plant=True, plant_drift=True)
        defaults.update(kw)
        return cls(scenario=scenario, **defaults)

    @classmethod
    def tracking(cls, scenario: str, **kw):
        defaults = dict(objective="tracking", reward_mode="track",
                        dynamic=False, randomize=False, randomize_setpoints=False,
                        randomize_plant=False, plant_drift=False)
        defaults.update(kw)
        return cls(scenario=scenario, **defaults)

    @classmethod
    def kpi(cls, scenario: str, **kw):
        defaults = dict(objective="kpi", reward_mode="kpi",
                        randomize_setpoints=True, randomize_plant=True, plant_drift=True)
        defaults.update(kw)
        return cls(scenario=scenario, **defaults)

    @classmethod
    def robustness(cls, scenario: str, **kw):
        defaults = dict(objective="robustness", reward_mode="kpi",
                        dynamic=True, randomize=True, randomize_setpoints=True,
                        randomize_plant=True, plant_drift=True, noise=True)
        defaults.update(kw)
        return cls(scenario=scenario, **defaults)

    @classmethod
    def safety(cls, scenario: str, **kw):
        defaults = dict(objective="safety", reward_mode="kpi",
                        dynamic=True, randomize=True, randomize_setpoints=True,
                        randomize_plant=True, plant_drift=True,
                        terminate_on_runaway=False)
        defaults.update(kw)
        return cls(scenario=scenario, **defaults)

    def env_kwargs(self, action_mode: str | None = None):
        data = asdict(self)
        data.pop("scenario")
        data.pop("objective")
        if action_mode is not None:
            data["action_mode"] = action_mode
        return data

    def make_env(self, action_mode: str | None = None):
        from .env import AIOGymNativeEnv

        return AIOGymNativeEnv(self.scenario, **self.env_kwargs(action_mode=action_mode))

    def metadata(self):
        data = asdict(self)
        data["metrics"] = list(PROTOCOL_METRICS.get(self.objective, ()))
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
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "objective": self.protocol.objective,
            "scenario": self.protocol.scenario,
            "controller": self.controller,
            "controller_config": _jsonable(dict(self.controller_config)),
            "seed_list": list(self.seeds),
            "disturbance": self.disturbance,
            "metrics": list(metrics),
            "episode_steps": self.protocol.episode_steps,
            "protocol": self.protocol.metadata(),
            "metric_definitions": {key: METRIC_DEFINITIONS[key] for key in metrics if key in METRIC_DEFINITIONS},
        }


def metric_for_reward_mode(reward_mode: str) -> str:
    return "profit" if reward_mode == "economic" else "kpi"


def metric_definitions(objective: str | None = None):
    if objective is None:
        return dict(METRIC_DEFINITIONS)
    return {key: METRIC_DEFINITIONS[key] for key in PROTOCOL_METRICS.get(objective, ()) if key in METRIC_DEFINITIONS}


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
    for ep, ep_seed in enumerate(seeds):
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
            tracking = _tracking_step_metrics(info, context.setpoint, time_sec, float(env.control_dt), env)
            safety = _safety_step_metrics(info, bound_metrics, float(env.control_dt))
            totals["return"] += float(reward)
            totals["track"] += float(info.get("track", 0.0))
            totals["constraint"] += float(info.get("constraint", 0.0))
            totals["profit"] += float(info.get("profit", 0.0))
            totals["prod"] += float(info.get("prod", 0.0))
            totals["production"] += float(info.get("prod", 0.0))
            totals["energy_kwh"] += float(info.get("energy_kw", 0.0)) * float(env.control_dt) / 3600.0
            totals["heat_kwh"] += float(info.get("heat_kw", 0.0)) * float(env.control_dt) / 3600.0
            totals["pump_kwh"] += float(info.get("pump_kw", 0.0)) * float(env.control_dt) / 3600.0
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
        totals["kpi"] = float(rep["score"])
        totals["normalized_score"] = float(rep["score"])
        totals["steps"] = steps
        totals["tracking_settling_time"] = float(last_unsettled_time)
        per_episode.append(totals)

    def mean(key):
        return float(np.mean([row[key] for row in per_episode]))

    def std(key):
        return float(np.std([row[key] for row in per_episode]))

    reward_mode = getattr(env, "reward_mode", "")
    objective = protocol.objective if protocol is not None else _env_objective(env)
    aggregate_keys = _aggregate_metric_keys(per_episode)
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "name": controller.name,
        "metric": metric_for_reward_mode(reward_mode),
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
        "protocol": protocol.metadata() if protocol is not None else _env_metadata(env),
        "controller": controller.metadata(),
        "model": _model_metadata(env),
        "disturbance": {
            "schedule_source": "model_schema_dynamic",
            "episode_schedules": episode_schedules,
        },
        "metric_definitions": metric_definitions(objective),
        "result_schema": result_schema(),
        "reproducibility": _reproducibility_metadata(env, seeds, protocol),
    }
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
    }


def _tracking_step_metrics(info, setpoint, time_sec: float, dt: float, env):
    errors = []
    overshoot = 0.0
    temps = list(info.get("temps", []))
    levels = list(info.get("levels", []))
    t_sp = list(setpoint.get("t_sp", []))
    h_sp = list(setpoint.get("h_sp", []))
    for i, temp in enumerate(temps):
        if i < len(t_sp):
            err = float(temp) - float(t_sp[i])
            errors.append(err)
            overshoot = max(overshoot, err)
    for i in getattr(env.model, "controlled_levels", lambda: [])():
        if i < len(levels) and i < len(h_sp):
            err = float(levels[i]) - float(h_sp[i])
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
        if i < len(levels) and i < len(h_sp) and abs(float(levels[i]) - float(h_sp[i])) > level_tol:
            settled = False
    return {
        "tracking_iae": float(iae),
        "tracking_ise": float(ise),
        "tracking_itae": float(itae),
        "tracking_overshoot": float(max(0.0, overshoot)),
        "tracking_settled": settled,
    }


def _action_bound_metrics(action, env):
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


def _safety_step_metrics(info, bound_metrics, dt: float):
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


def _aggregate_metric_keys(per_episode):
    keys = set()
    for row in per_episode:
        for key, value in row.items():
            if key in ("episode", "seed", "steps"):
                continue
            if isinstance(value, (int, float, np.number)):
                keys.add(key)
    return sorted(keys)


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
    episode_metrics = result.get("episode_metrics", [])
    for key in PROTOCOL_METRICS["robustness"]:
        vals = [float(ep[key]) for ep in episode_metrics if key in ep]
        if vals:
            row[f"{key}_best"] = max(vals)
            row[f"{key}_worst"] = min(vals)
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
