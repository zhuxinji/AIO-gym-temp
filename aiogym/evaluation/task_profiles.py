"""Versioned task declarations shared by environments and benchmark runners."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Integral
from collections.abc import Mapping
from pathlib import Path
from typing import Any


TASK_PROFILE_SCHEMA_VERSION = "aiogym.task_profile.v1"
_TASK_DIR = Path(__file__).with_name("tasks")
TASK_ENVIRONMENT_FIELDS = frozenset({
    "control_dt",
    "episode_steps",
    "action_mode",
    "dynamic",
    "randomize",
    "randomize_setpoints",
    "randomize_plant",
    "plant_drift",
    "integral_obs",
    "terminate_on_runaway",
    "noise",
    "noise_pct",
})
TASK_OPERATION_FIELDS = frozenset({"mode", "product_flow_sp", "min_product_flow"})
ENVIRONMENT_BOOLEAN_FIELDS = (
    "dynamic",
    "randomize",
    "randomize_setpoints",
    "randomize_plant",
    "plant_drift",
    "integral_obs",
    "terminate_on_runaway",
    "noise",
)


def list_task_profiles(scenario: str | None = None) -> tuple[str, ...]:
    """List bundled task identifiers as ``scenario/name`` strings."""

    paths = (_TASK_DIR / scenario).glob("*.json") if scenario else _TASK_DIR.glob("*/*.json")
    return tuple(sorted(f"{path.parent.name}/{path.stem}" for path in paths))


def load_task_profile(
    source: str | Path | Mapping[str, Any],
    *,
    scenario: str | None = None,
) -> dict[str, Any]:
    """Load and validate a bundled, file-based, or in-memory task profile."""

    if isinstance(source, Mapping):
        data = copy.deepcopy(dict(source))
    else:
        path = Path(source)
        if isinstance(source, str) and not path.exists():
            parts = source.split("/", 1)
            if len(parts) == 2:
                task_scenario, task_name = parts
                path = _TASK_DIR / task_scenario / f"{task_name}.json"
            elif scenario:
                path = _TASK_DIR / scenario / f"{source}.json"
        if not path.is_file():
            raise FileNotFoundError(f"task profile not found: {source}")
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    validate_task_profile(data, expected_scenario=scenario)
    return data


def validate_task_profile(
    profile: Mapping[str, Any],
    *,
    expected_scenario: str | None = None,
) -> None:
    """Validate task identity, timing, and extension sections."""

    if not isinstance(profile, Mapping):
        raise TypeError("task profile must be a mapping")
    required = ("schema_version", "name", "scenario", "status", "environment")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"task profile is missing required fields: {', '.join(missing)}")
    if profile["schema_version"] != TASK_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"unsupported task profile schema: {profile['schema_version']!r}")
    for key in ("name", "scenario", "status"):
        if not isinstance(profile[key], str) or not profile[key]:
            raise ValueError(f"task profile {key} must be a non-empty string")
    if expected_scenario is not None and profile["scenario"] != expected_scenario:
        raise ValueError(f"expected task for {expected_scenario!r}, got {profile['scenario']!r}")
    environment = profile["environment"]
    if not isinstance(environment, Mapping):
        raise TypeError("task profile environment must be a mapping")
    unknown = set(environment) - TASK_ENVIRONMENT_FIELDS
    if unknown:
        raise ValueError(f"unknown task environment fields: {', '.join(sorted(unknown))}")
    if "control_dt" in environment:
        value = float(environment["control_dt"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("task control_dt must be finite and positive")
    if "episode_steps" in environment:
        value = environment["episode_steps"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("task episode_steps must be a positive integer")
    if "action_mode" in environment and environment["action_mode"] not in {"actuator", "setpoint"}:
        raise ValueError("task action_mode must be one of: actuator, setpoint")
    for key in (
        "dynamic", "randomize", "randomize_setpoints", "randomize_plant",
        "plant_drift", "integral_obs", "terminate_on_runaway", "noise",
    ):
        if key in environment and not isinstance(environment[key], bool):
            raise TypeError(f"task {key} must be a boolean")
    if "noise_pct" in environment:
        value = float(environment["noise_pct"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("task noise_pct must be finite and non-negative")
    if "model_params" in profile and not isinstance(profile["model_params"], Mapping):
        raise TypeError("task profile model_params must be a mapping")
    _validate_operation(profile.get("operation"))
    for section in ("initialization", "setpoints", "disturbances", "constraints", "acceptance"):
        if section in profile and not isinstance(profile[section], (Mapping, list)):
            raise TypeError(f"task profile {section} must be a mapping or list")
    acceptance = profile.get("acceptance", {})
    if isinstance(acceptance, Mapping) and "metrics" in acceptance:
        _validate_acceptance_metrics(acceptance["metrics"])
    initialization = profile.get("initialization", {})
    if isinstance(initialization, Mapping) and "state" in initialization:
        _finite_numeric_vector("task initialization state", initialization["state"])
    setpoints = profile.get("setpoints", {})
    if isinstance(setpoints, Mapping):
        if "initial" in setpoints:
            _finite_numeric_vector("task initial setpoint", setpoints["initial"])
        schedule = setpoints.get("schedule", [])
        if not isinstance(schedule, list):
            raise TypeError("task setpoint schedule must be a list")
        for event in schedule:
            if not isinstance(event, Mapping):
                raise TypeError("each task setpoint event must be a mapping")
            at_step = event.get("at_step")
            if isinstance(at_step, bool) or not isinstance(at_step, int) or at_step < 0:
                raise ValueError("task setpoint event at_step must be a non-negative integer")
            _finite_numeric_vector("task scheduled setpoint", event.get("values"))
    disturbances = profile.get("disturbances", [])
    if not isinstance(disturbances, list):
        raise TypeError("task disturbances must be a list")
    for event in disturbances:
        if not isinstance(event, Mapping):
            raise TypeError("each task disturbance event must be a mapping")
        at_step = event.get("at_step")
        if isinstance(at_step, bool) or not isinstance(at_step, int) or at_step < 0:
            raise ValueError("task disturbance event at_step must be a non-negative integer")
        name = event.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError("task disturbance event name must be a non-empty string")
        _finite_numeric_value("task disturbance value", event.get("value"))
    if "references" in profile and not isinstance(profile["references"], list):
        raise TypeError("task profile references must be a list")
    supported = profile.get("supported_objectives")
    if supported is not None:
        if not isinstance(supported, list) or not supported:
            raise TypeError("task supported_objectives must be a non-empty list")
        if any(not isinstance(value, str) or not value for value in supported):
            raise TypeError("task supported_objectives must contain non-empty strings")
    default_objective = profile.get("default_objective")
    if default_objective is not None:
        if not isinstance(default_objective, str) or not default_objective:
            raise TypeError("task default_objective must be a non-empty string")
        if supported is not None and default_objective not in supported:
            raise ValueError("task default_objective must be listed in supported_objectives")


def task_environment(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the executable environment conditions owned by a task."""

    validate_task_profile(profile)
    return copy.deepcopy(dict(profile["environment"]))


def resolve_environment_options(
    *,
    scenario: str,
    task: str | Path | Mapping[str, Any] | None,
    explicit: Mapping[str, Any],
    defaults: Mapping[str, Any],
    default_control_dt: float,
    default_episode_steps: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Resolve and validate task-owned environment settings in one place."""

    profile = (
        load_task_profile(task, scenario=scenario)
        if task is not None
        else None
    )
    task_defaults = task_environment(profile) if profile is not None else {}
    provided = dict(explicit)

    resolved = {}
    for name in ("action_mode", *ENVIRONMENT_BOOLEAN_FIELDS):
        value = provided.get(name)
        resolved[name] = (
            value
            if value is not None
            else task_defaults.get(name, defaults[name])
        )

    control_dt = provided.get("control_dt")
    if control_dt is None:
        control_dt = task_defaults.get("control_dt", default_control_dt)
    if isinstance(control_dt, bool):
        raise TypeError("control_dt must be numeric")
    control_dt = float(control_dt)
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")

    episode_steps = provided.get("episode_steps")
    if episode_steps is None:
        episode_steps = task_defaults.get("episode_steps", default_episode_steps)
    if (
        isinstance(episode_steps, bool)
        or not isinstance(episode_steps, Integral)
        or int(episode_steps) <= 0
    ):
        raise ValueError("episode_steps must be a positive integer")

    noise_pct = provided.get("noise_pct")
    if noise_pct is None:
        noise_pct = task_defaults.get("noise_pct", defaults["noise_pct"])
    if isinstance(noise_pct, bool):
        raise TypeError("noise_pct must be numeric")
    noise_pct = float(noise_pct)
    if not math.isfinite(noise_pct) or noise_pct < 0:
        raise ValueError("noise_pct must be finite and non-negative")

    if resolved["action_mode"] not in {"actuator", "setpoint"}:
        raise ValueError("action_mode must be one of: actuator, setpoint")
    for name in ENVIRONMENT_BOOLEAN_FIELDS:
        resolved[name] = bool(resolved[name])

    explicit_model_params = provided.get("model_params")
    if explicit_model_params is not None and not isinstance(explicit_model_params, Mapping):
        raise TypeError("model_params must be a mapping")
    model_params = dict((profile or {}).get("model_params", {}))
    model_params.update(dict(explicit_model_params or {}))

    resolved.update({
        "control_dt": control_dt,
        "episode_steps": int(episode_steps),
        "noise_pct": noise_pct,
        "model_params": model_params,
    })
    return profile, resolved


def task_operation(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a normalized batch/continuous operation declaration, if present."""

    validate_task_profile(profile)
    operation = profile.get("operation")
    if operation is None:
        return None
    mode = str(operation["mode"])
    product_flow_sp = float(operation.get("product_flow_sp", 0.0))
    min_product_flow = float(
        operation.get(
            "min_product_flow",
            product_flow_sp if mode == "continuous" else 0.0,
        )
    )
    return {
        "mode": mode,
        "product_flow_sp": product_flow_sp,
        "min_product_flow": min_product_flow,
    }


def configure_model_for_task(model, profile: Mapping[str, Any] | None):
    """Apply task context that affects model economics without changing ``p``."""

    if profile is None:
        return model
    operation = task_operation(profile)
    if operation is None:
        return model
    configure = getattr(model, "configure_operation", None)
    if not callable(configure):
        raise ValueError(
            f"task {profile['name']!r} declares operation settings, but model "
            f"{model.scenario!r} does not support them"
        )
    configure(operation)
    return model


def task_identity(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the stable task identity recorded in benchmark rows and artifacts."""

    if profile is None:
        return {
            "name": "default",
            "status": "implicit-default",
            "schema_version": TASK_PROFILE_SCHEMA_VERSION,
            "profile_hash": None,
        }
    validate_task_profile(profile)
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "name": str(profile["name"]),
        "status": str(profile["status"]),
        "schema_version": str(profile["schema_version"]),
        "profile_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def evaluate_task_acceptance(
    profile: Mapping[str, Any] | None,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate optional metric thresholds without conflating them with execution."""

    if profile is None:
        return {"status": "not-defined", "checks": []}
    validate_task_profile(profile)
    acceptance = profile.get("acceptance", {})
    thresholds = acceptance.get("metrics") if isinstance(acceptance, Mapping) else None
    if not thresholds:
        return {"status": "not-defined", "checks": []}
    checks = []
    met = True
    for metric, bounds in thresholds.items():
        value = result.get(metric)
        check = {"metric": metric, "value": value, **dict(bounds)}
        check_met = isinstance(value, (int, float))
        if check_met and "min" in bounds:
            check_met = float(value) >= float(bounds["min"])
        if check_met and "max" in bounds:
            check_met = float(value) <= float(bounds["max"])
        check["met"] = bool(check_met)
        checks.append(check)
        met = met and bool(check_met)
    return {"status": "met" if met else "not-met", "checks": checks}


def _finite_numeric_vector(name: str, values) -> None:
    if not isinstance(values, list) or not values:
        raise TypeError(f"{name} must be a non-empty list")
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError(f"{name} values must be finite")


def _finite_numeric_value(name: str, value) -> None:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise TypeError(f"{name} must not be empty")
    try:
        numbers = [float(item) for item in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric or a numeric list") from exc
    if not all(math.isfinite(item) for item in numbers):
        raise ValueError(f"{name} must be finite")


def _validate_operation(operation) -> None:
    if operation is None:
        return
    if not isinstance(operation, Mapping):
        raise TypeError("task profile operation must be a mapping")
    unknown = set(operation) - TASK_OPERATION_FIELDS
    if unknown:
        raise ValueError(f"unknown task operation fields: {', '.join(sorted(unknown))}")
    mode = operation.get("mode")
    if mode not in {"batch", "continuous"}:
        raise ValueError("task operation mode must be one of: batch, continuous")
    if mode == "continuous" and "product_flow_sp" not in operation:
        raise ValueError("continuous task operation requires product_flow_sp")
    product_flow_sp = _nonnegative_operation_value(
        "task operation product_flow_sp", operation.get("product_flow_sp", 0.0)
    )
    if mode == "continuous" and product_flow_sp <= 0.0:
        raise ValueError("continuous task product_flow_sp must be positive")
    if mode == "batch" and product_flow_sp != 0.0:
        raise ValueError("batch task product_flow_sp must be zero")
    min_product_flow = _nonnegative_operation_value(
        "task operation min_product_flow",
        operation.get("min_product_flow", product_flow_sp if mode == "continuous" else 0.0),
    )
    if mode == "batch" and min_product_flow != 0.0:
        raise ValueError("batch task min_product_flow must be zero")
    if min_product_flow > product_flow_sp:
        raise ValueError("task operation min_product_flow must not exceed product_flow_sp")


def _nonnegative_operation_value(name: str, value) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _validate_acceptance_metrics(metrics) -> None:
    if not isinstance(metrics, Mapping) or not metrics:
        raise TypeError("task acceptance metrics must be a non-empty mapping")
    for metric, bounds in metrics.items():
        if not isinstance(metric, str) or not metric:
            raise TypeError("task acceptance metric names must be non-empty strings")
        if not isinstance(bounds, Mapping) or not bounds:
            raise TypeError("task acceptance metric bounds must be a non-empty mapping")
        unknown = set(bounds) - {"min", "max"}
        if unknown:
            raise ValueError(
                f"unknown task acceptance bounds for {metric}: {', '.join(sorted(unknown))}"
            )
        for key, value in bounds.items():
            _finite_numeric_value(f"task acceptance {metric} {key}", value)
        if "min" in bounds and "max" in bounds:
            if float(bounds["min"]) > float(bounds["max"]):
                raise ValueError(
                    f"task acceptance minimum exceeds maximum for {metric}"
                )
