"""Controller evaluation protocols for AIO-Gym.

This module is the common path for fixed actuator controllers (PID/MPC/oracle)
and learned policies (SB3/RLPD style).
"""
from __future__ import annotations

import os
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence
import numpy as np


from .._internal.serialization import jsonable as _jsonable
from ..controllers import as_controller, build_context, validate_action
from .metrics.economic import economic_step_metrics
from .metrics.robustness import robustness_extrema
from .metrics.safety import action_bound_metrics as _action_bound_metrics
from .metrics.safety import safety_step_metrics as _safety_step_metrics
from .metrics.tracking import tracking_step_metrics as _tracking_step_metrics
from .protocols import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_DEFINITIONS,
    METRIC_DIRECTIONS,
    PROTOCOL_METRICS,
    ROLLOUT_SCHEMA,
    BenchmarkProtocol,
    ObjectiveSpec,
    _empty_episode_totals,
    metric_definitions,
    metric_direction,
    objective_spec,
    primary_metric_for_objective,
)

def evaluate_controller(agent, env, episodes: int = 20, seed: int = 0,
                        include_episodes: bool = False, protocol: BenchmarkProtocol | None = None,
                        seed_list: Sequence[int] | None = None,
                        objective_specification: ObjectiveSpec | str | None = None):
    """Evaluate any supported controller/policy on an AIOGymNativeEnv.

    Returns aggregate metrics plus the protocol/controller metadata needed to
    reproduce the benchmark.
    """

    if seed_list is None:
        if episodes <= 0:
            raise ValueError("episodes must be positive when seed_list is not provided")
        seeds = [int(seed) + ep for ep in range(episodes)]
    else:
        seeds = [int(value) for value in seed_list]
        if not seeds:
            raise ValueError("seed_list must contain at least one seed")
    controller = as_controller(agent, action_mode=getattr(env, "action_mode", "actuator"))
    if protocol is not None:
        resolved_objective = protocol.resolved_objective()
        if (
            objective_specification is not None
            and getattr(objective_specification, "name", objective_specification)
            != resolved_objective.name
        ):
            raise ValueError("objective specification does not match protocol objective")
    elif isinstance(objective_specification, ObjectiveSpec):
        resolved_objective = objective_specification
    elif objective_specification is not None:
        resolved_objective = objective_spec(
            str(objective_specification), source="explicit"
        )
    else:
        resolved_objective = objective_spec(
            _env_objective(env), source="environment-reward-mode"
        )
    objective = resolved_objective.name
    per_episode = []
    episode_schedules = []
    eval_start = perf_counter()
    for ep, ep_seed in enumerate(seeds):
        episode_start = perf_counter()
        obs, reset_info = env.reset(seed=ep_seed)
        controller.reset(seed=ep_seed)
        totals = _empty_episode_totals(ep, ep_seed)
        episode_schedules.append(_jsonable({
            "task": getattr(env, "_task_disturbance_events", {}),
            "dynamic": getattr(env, "_dist_events", []),
        }))
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
            # ``env.y_sp`` may already contain the reference staged for the next
            # control step. Metrics for this completed transition must use the
            # reference recorded by its stage objective.
            active_setpoint = {
                "y_sp": list(info.get("y_sp", getattr(env, "y_sp", [])))
            }
            tracking = _tracking_step_metrics(info, active_setpoint, time_sec, float(env.control_dt), env)
            safety = _safety_step_metrics(info, bound_metrics, float(env.control_dt))
            totals["return"] += float(reward)
            totals["track"] += float(info.get("track", 0.0))
            totals["constraint"] += float(info.get("constraint", 0.0))
            economic = economic_step_metrics(info, float(env.control_dt))
            for key, value in economic.items():
                totals[key] += value
            for key in (
                "tracking_cost", "tracking_return", "tracking_error_cost", "tracking_move_cost",
                "tracking_steady_cost",
                "tracking_mse", "tracking_iae", "tracking_ise", "tracking_itae",
            ):
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
        totals["normalized_score"] = float(rep["score"])
        totals["steps"] = steps
        totals["runtime_seconds"] = float(runtime_seconds)
        totals["runtime_seconds_per_step"] = float(runtime_seconds / steps) if steps else 0.0
        horizon_seconds = steps * float(env.control_dt)
        totals["tracking_mse"] = float(totals["tracking_mse"] / horizon_seconds) if horizon_seconds > 0 else 0.0
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

    if protocol is not None:
        task_meta = protocol.metadata()["task_identity"]
    else:
        from .task_profiles import task_identity

        task_meta = task_identity(getattr(env, "task_profile", None))
    primary_metric = primary_metric_for_objective(objective)
    aggregate_keys = _aggregate_metric_keys(per_episode)
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "controller_name": controller.name,
        "metric": primary_metric,
        "metric_direction": metric_direction(primary_metric),
        "objective": objective,
        "objective_source": resolved_objective.source,
        "objective_spec": resolved_objective.metadata(),
        "task": task_meta["name"],
        "task_status": task_meta["status"],
        "task_profile_hash": task_meta["profile_hash"],
        "episodes": len(seeds),
        "seed": int(seeds[0]) if seeds else int(seed),
        "seed_list": [int(s) for s in seeds],
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
        "production": mean("production"),
        "production_std": std("production"),
        "runtime_total_seconds": float(runtime_total_seconds),
        "protocol": (
            protocol.metadata()
            if protocol is not None
            else {
                "environment": _env_metadata(env),
                "objective_spec": resolved_objective.metadata(),
            }
        ),
        "controller": controller.metadata(),
        "model": _model_metadata(env),
        "disturbance": {
            "schedule_source": "task_and_model_schema",
            "episode_schedules": episode_schedules,
        },
        "controller_diagnostics": _aggregate_controller_diagnostics(per_episode),
        "metric_definitions": metric_definitions(objective),
        "result_schema": result_schema(),
        "reproducibility": _reproducibility_metadata(
            env, seeds, protocol, resolved_objective
        ),
    }
    result["controller_status"] = "degraded" if result["controller_diagnostics"].get("degraded") else "ok"
    for key in aggregate_keys:
        result.setdefault(key, mean(key))
        result.setdefault(f"{key}_std", std(key))
    result["execution_status"] = (
        "degraded" if result["controller_status"] == "degraded" else "passed"
    )
    from .task_profiles import evaluate_task_acceptance

    acceptance = evaluate_task_acceptance(
        protocol.task if protocol is not None else getattr(env, "task_profile", None),
        result,
    )
    result["objective_status"] = acceptance["status"]
    result["objective_acceptance"] = acceptance
    if include_episodes:
        result["episode_metrics"] = per_episode
    return result


def rollout_controller(agent, env, seed: int = 0, max_steps: int | None = None,
                       protocol: BenchmarkProtocol | None = None,
                       objective_specification: ObjectiveSpec | None = None):
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
        "controller_name": controller.name,
        "seed": int(seed),
        "steps": len(rows),
        "objective": (
            protocol.objective
            if protocol is not None
            else getattr(objective_specification, "name", _env_objective(env))
        ),
        "objective_source": (
            protocol.objective_source
            if protocol is not None
            else getattr(objective_specification, "source", "environment-reward-mode")
        ),
        "protocol": (
            protocol.metadata()
            if protocol is not None
            else {
                "environment": _env_metadata(env),
                "objective_spec": (
                    objective_specification.metadata()
                    if objective_specification is not None
                    else objective_spec(_env_objective(env), source="environment-reward-mode").metadata()
                ),
            }
        ),
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
    if reward_mode == "tracking":
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
        return _jsonable(target.diagnostics())
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


def _reproducibility_metadata(env, seeds, protocol, resolved_objective=None):
    return {
        "git_commit": _git_commit(),
        "seed_list": [int(seed) for seed in seeds],
        "model_version": getattr(getattr(env, "model", None), "scenario", None),
        "episode_length": int(getattr(env, "episode_steps", 0)),
        "disturbance_schedule": "task_and_model_schema",
        "metric_definition_version": EVALUATION_SCHEMA_VERSION,
        "protocol": (
            protocol.metadata()
            if protocol is not None
            else {
                "environment": _env_metadata(env),
                "objective_spec": (
                    resolved_objective.metadata()
                    if resolved_objective is not None
                    else None
                ),
            }
        ),
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
        "controller": result.get("controller_name"),
        "objective": result.get("objective"),
        "task": result.get("task", "default"),
        "task_status": result.get("task_status", "implicit-default"),
        "task_profile_hash": result.get("task_profile_hash"),
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
    row.update(robustness_extrema(
        result.get("episode_metrics", []),
        PROTOCOL_METRICS["robustness"],
        METRIC_DIRECTIONS,
    ))
    return row
