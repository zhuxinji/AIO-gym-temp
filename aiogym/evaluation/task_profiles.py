"""Versioned task declarations shared by environments and benchmark runners."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


TASK_PROFILE_SCHEMA_VERSION = "aiogym.task_profile.v1"
_TASK_DIR = Path(__file__).with_name("tasks")
_TASK_ALIASES = {
    ("quadruple", "minimum-phase-tracking"): "minimum-phase-classic",
}
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
                task_name = _TASK_ALIASES.get((task_scenario, task_name), task_name)
                path = _TASK_DIR / task_scenario / f"{task_name}.json"
            elif scenario:
                task_name = _TASK_ALIASES.get((scenario, source), source)
                path = _TASK_DIR / scenario / f"{task_name}.json"
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
    for section in ("initialization", "setpoints", "disturbances", "constraints", "acceptance"):
        if section in profile and not isinstance(profile[section], (Mapping, list)):
            raise TypeError(f"task profile {section} must be a mapping or list")
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
