"""Evaluation of acceptance thresholds declared by model-bound tasks."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..models.tasks import validate_task_profile


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


__all__ = ["evaluate_task_acceptance"]
