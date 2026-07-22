"""Canonical public resource IDs and compatibility aliases."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# Aliases are deliberately explicit. Custom IDs are never normalized by a
# blanket underscore/hyphen replacement.
_SCENARIO_CANONICAL_BY_INTERNAL = {
    "cascade_recirculating": "cascade-recirculating",
}
_SCENARIO_INTERNAL_BY_ALIAS = {
    canonical: internal
    for internal, canonical in _SCENARIO_CANONICAL_BY_INTERNAL.items()
}


def canonical_scenario_id(value: str) -> str:
    """Return the public canonical ID for a scenario registration key."""

    return _SCENARIO_CANONICAL_BY_INTERNAL.get(value, value)


def internal_scenario_id(value: str) -> str:
    """Resolve a public built-in alias to its internal registration key."""

    return _SCENARIO_INTERNAL_BY_ALIAS.get(value, value)


def canonical_scenario_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(canonical_scenario_id(value) for value in values))


def scenario_aliases(value: str) -> tuple[str, ...]:
    """Return accepted non-canonical aliases for one scenario ID."""

    internal = internal_scenario_id(value)
    canonical = canonical_scenario_id(internal)
    return (internal,) if internal != canonical else ()


def scenario_catalog_text(values: Sequence[str]) -> str:
    """Render canonical IDs and their accepted aliases for error messages."""

    rows = []
    for internal in sorted(values, key=canonical_scenario_id):
        canonical = canonical_scenario_id(internal)
        aliases = scenario_aliases(internal)
        suffix = f" (alias: {', '.join(aliases)})" if aliases else ""
        rows.append(f"{canonical}{suffix}")
    return ", ".join(rows)


def canonical_task_id(value: str) -> str:
    """Canonicalize the scenario prefix of a ``scenario/task`` ID."""

    parts = value.split("/", 1)
    if len(parts) != 2:
        return value
    return f"{canonical_scenario_id(internal_scenario_id(parts[0]))}/{parts[1]}"


def internal_task_id(value: str) -> str:
    """Resolve the scenario prefix of a ``scenario/task`` ID for storage."""

    parts = value.split("/", 1)
    if len(parts) != 2:
        return value
    return f"{internal_scenario_id(parts[0])}/{parts[1]}"


def resolve_suite_id(value: str, canonical_ids: Sequence[str]) -> str:
    """Resolve snake-case aliases for known built-in suite IDs only."""

    canonical = set(canonical_ids)
    if value in canonical:
        return value
    aliases = {
        item.replace("-", "_"): item
        for item in canonical
        if "-" in item and item.replace("-", "_") not in canonical
    }
    return aliases.get(value, value)


def suite_aliases(value: str, canonical_ids: Sequence[str]) -> tuple[str, ...]:
    alias = value.replace("-", "_")
    return (alias,) if alias != value and alias not in set(canonical_ids) else ()


def suite_catalog_text(canonical_ids: Sequence[str]) -> str:
    rows = []
    for canonical in sorted(canonical_ids):
        aliases = suite_aliases(canonical, canonical_ids)
        suffix = f" (alias: {', '.join(aliases)})" if aliases else ""
        rows.append(f"{canonical}{suffix}")
    return ", ".join(rows)


def canonicalize_artifact_ids(value: Any, *, field: str | None = None) -> Any:
    """Canonicalize scenario-bearing fields in newly written artifacts."""

    if isinstance(value, Mapping):
        return {
            key: canonicalize_artifact_ids(item, field=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [canonicalize_artifact_ids(item, field=field) for item in value]
    if isinstance(value, tuple):
        return [canonicalize_artifact_ids(item, field=field) for item in value]
    if isinstance(value, str):
        if field == "scenario":
            return canonical_scenario_id(internal_scenario_id(value))
        if field == "scenarios":
            return canonical_scenario_id(internal_scenario_id(value))
        if field == "task" and "/" in value:
            return canonical_task_id(value)
    return value


__all__ = [
    "canonical_scenario_id",
    "canonical_scenario_ids",
    "canonical_task_id",
    "canonicalize_artifact_ids",
    "internal_scenario_id",
    "internal_task_id",
    "resolve_suite_id",
    "scenario_aliases",
    "scenario_catalog_text",
    "suite_aliases",
    "suite_catalog_text",
]
