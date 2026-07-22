"""Benchmark-suite declaration loading and reuse resolution."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .._internal.config import canonicalize_auto_events
from .._internal.identifiers import (
    canonical_scenario_id,
    internal_scenario_id,
    resolve_suite_id,
    suite_catalog_text,
)
from ..catalog import list_suites
from ..models import SCENARIOS


CONFIG_DIR = Path(__file__).resolve().parent / "suites"
PRESET_DIR = CONFIG_DIR / "presets"
SCENARIO_ALIASES = {
    "ALL_SCENARIOS": tuple(canonical_scenario_id(scenario) for scenario in SCENARIOS),
    "CORE_SCENARIOS": ("cascade", "quadruple", "cstr", "hvac"),
    "ECONOMIC_SCENARIOS": ("cascade", "cstr", "hvac", "heater"),
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge suite declarations, replacing lists and recursively merging mappings."""

    result = dict(base)
    for key, value in dict(override).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _reference_list(value, field: str) -> list[str]:
    refs = [value] if isinstance(value, str) else value
    if not isinstance(refs, list) or not refs:
        raise TypeError(f"suite {field} must be a non-empty string or list of strings")
    if any(not isinstance(ref, str) or not ref for ref in refs):
        raise TypeError(f"suite {field} must contain non-empty strings")
    return refs


def _load_reuse_registry() -> tuple[dict, dict]:
    presets = {}
    cases = {}
    for path in sorted(PRESET_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise TypeError(f"suite preset file must contain a mapping: {path}")
        unknown = set(data) - {"presets", "cases"}
        if unknown:
            raise ValueError(
                f"unknown suite preset sections in {path.name}: "
                f"{', '.join(sorted(unknown))}"
            )
        for section, target in (("presets", presets), ("cases", cases)):
            entries = data.get(section, {})
            if not isinstance(entries, dict):
                raise TypeError(f"suite preset {section} must be a mapping: {path}")
            for name, declaration in entries.items():
                if not isinstance(name, str) or not name or not isinstance(declaration, dict):
                    raise TypeError(
                        f"suite preset {section} entries must map non-empty IDs to mappings"
                    )
                if name in target:
                    raise ValueError(f"duplicate suite {section[:-1]} ID: {name!r}")
                target[name] = declaration
    return presets, cases


def _unknown_reference(kind: str, name: str, registry: dict) -> ValueError:
    available = ", ".join(sorted(registry)) if registry else "none"
    return ValueError(
        f"unknown suite {kind} ID {name!r}; available {kind} IDs: {available}"
    )


def _apply_preset(
    declaration: dict,
    presets: dict,
    *,
    stack: tuple[str, ...] = (),
) -> dict:
    current = dict(declaration)
    raw_refs = current.pop("preset", None)
    if raw_refs is None:
        return current
    resolved = {}
    for name in _reference_list(raw_refs, "preset"):
        if name in stack:
            cycle = " -> ".join((*stack, name))
            raise ValueError(f"cyclic suite preset reference: {cycle}")
        if name not in presets:
            raise _unknown_reference("preset", name, presets)
        fragment = _apply_preset(
            presets[name], presets, stack=(*stack, name)
        )
        resolved = _deep_merge(resolved, fragment)
    return _deep_merge(resolved, current)


def _resolve_case_reference(
    declaration: dict,
    presets: dict,
    case_refs: dict,
    *,
    stack: tuple[str, ...] = (),
) -> dict:
    current = dict(declaration)
    reference = current.pop("case_ref", None)
    current = _apply_preset(current, presets)
    if isinstance(current.get("environment"), dict):
        current["environment"] = _apply_preset(
            current["environment"], presets
        )
    if reference is not None:
        if not isinstance(reference, str) or not reference:
            raise TypeError("suite case_ref must be a non-empty string")
        if reference in stack:
            cycle = " -> ".join((*stack, reference))
            raise ValueError(f"cyclic suite case reference: {cycle}")
        if reference not in case_refs:
            raise _unknown_reference("case", reference, case_refs)
        base = _resolve_case_reference(
            case_refs[reference],
            presets,
            case_refs,
            stack=(*stack, reference),
        )
        current = _deep_merge(base, current)
    return current


def _resolve_suite_path(reference: str, *, relative_to: Path | None = None) -> Path:
    path = Path(reference)
    if os.path.sep not in reference and not path.suffix:
        available_suites = list_suites()
        suite_id = resolve_suite_id(reference, available_suites)
        path = CONFIG_DIR / f"{suite_id}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"unknown suite ID {reference!r}; available suite IDs: "
                f"{suite_catalog_text(available_suites)}"
            )
    elif not path.is_absolute() and relative_to is not None:
        path = relative_to.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"suite declaration not found: {reference}")
    return path.resolve()


def _load_suite_declaration(path: Path, *, stack: tuple[Path, ...] = ()) -> dict:
    resolved_path = path.resolve()
    if resolved_path in stack:
        cycle = " -> ".join(item.name for item in (*stack, resolved_path))
        raise ValueError(f"cyclic suite extends reference: {cycle}")
    with resolved_path.open(encoding="utf-8") as stream:
        declaration = json.load(stream)
    if not isinstance(declaration, dict):
        raise TypeError(f"suite declaration must be a mapping: {resolved_path}")
    parents = declaration.pop("extends", None)
    merged = {}
    if parents is not None:
        for reference in _reference_list(parents, "extends"):
            parent_path = _resolve_suite_path(reference, relative_to=resolved_path)
            parent = _load_suite_declaration(
                parent_path, stack=(*stack, resolved_path)
            )
            merged = _deep_merge(merged, parent)
    return _deep_merge(merged, declaration)


def load_suite(name_or_path: str):
    builtin_id = os.path.sep not in name_or_path and not name_or_path.endswith(".json")
    path = _resolve_suite_path(name_or_path)
    suite = _load_suite_declaration(path)
    presets, case_refs = _load_reuse_registry()
    suite = _apply_preset(suite, presets)
    if isinstance(suite.get("environment"), dict):
        suite["environment"] = _apply_preset(suite["environment"], presets)
    if isinstance(suite.get("cases"), list):
        suite["cases"] = [
            _resolve_case_reference(case, presets, case_refs)
            if isinstance(case, dict) else case
            for case in suite["cases"]
        ]
    suite = canonicalize_auto_events(suite, warn_legacy=True)
    if isinstance(suite.get("environment"), dict):
        suite["environment"] = canonicalize_auto_events(
            suite["environment"], warn_legacy=True
        )
    if isinstance(suite.get("cases"), list):
        normalized_cases = []
        for case in suite["cases"]:
            if not isinstance(case, dict):
                normalized_cases.append(case)
                continue
            normalized = canonicalize_auto_events(case, warn_legacy=True)
            if isinstance(normalized.get("environment"), dict):
                normalized["environment"] = canonicalize_auto_events(
                    normalized["environment"], warn_legacy=True
                )
            normalized_cases.append(normalized)
        suite["cases"] = normalized_cases
    suite.setdefault("name", path.stem)
    if builtin_id:
        suite["name"] = resolve_suite_id(suite["name"], list_suites())
    suite["scenarios"] = expand_scenarios(suite.get("scenarios", ()))
    suite["objectives"] = list(suite.get("objectives", ()))
    suite["controllers"] = list(suite.get("controllers", ()))
    if "cases" in suite:
        if not isinstance(suite["cases"], list) or not suite["cases"]:
            raise ValueError("suite cases must be a non-empty list")
        if any(not isinstance(case, dict) for case in suite["cases"]):
            raise TypeError("each suite case must be a mapping")
    suite.setdefault("action_mode", "actuator")
    suite.setdefault("description", "")
    if "cases" not in suite and suite.get("task") is None:
        suite.setdefault("episode_steps", 80)
        suite.setdefault("control_dt", 0.5)
    return suite


def expand_scenarios(value):
    if isinstance(value, str):
        if value in SCENARIO_ALIASES:
            return list(SCENARIO_ALIASES[value])
        return [canonical_scenario_id(internal_scenario_id(value))]
    out = []
    for item in value:
        if item in SCENARIO_ALIASES:
            out.extend(SCENARIO_ALIASES[item])
        else:
            out.append(canonical_scenario_id(internal_scenario_id(item)))
    return out
