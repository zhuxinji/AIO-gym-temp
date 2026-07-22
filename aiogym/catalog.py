"""Public discovery helpers for registered and bundled AIO-Gym resources."""
from __future__ import annotations

from pathlib import Path

from ._internal.identifiers import canonical_scenario_ids
from .controllers import registered_controllers
from .models.tasks import list_task_profiles
from .models.registry import MODELS


_SUITE_DIR = Path(__file__).with_name("evaluation") / "suites"


def list_scenarios() -> tuple[str, ...]:
    """Return canonical IDs for all currently registered process scenarios."""

    return canonical_scenario_ids(tuple(MODELS))


def list_tasks(scenario: str | None = None) -> tuple[str, ...]:
    """Return canonical ``scenario/name`` IDs for bundled task specs."""

    return list_task_profiles(scenario)


def list_suites() -> tuple[str, ...]:
    """Return canonical IDs for bundled benchmark suites."""

    if not _SUITE_DIR.is_dir():
        return ()
    return tuple(sorted(path.stem for path in _SUITE_DIR.glob("*.json")))


def list_controllers() -> tuple[str, ...]:
    """Return canonical IDs for all currently registered controllers."""

    return registered_controllers()


__all__ = ["list_controllers", "list_scenarios", "list_suites", "list_tasks"]
