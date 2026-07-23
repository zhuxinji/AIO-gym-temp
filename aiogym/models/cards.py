"""Structured model metadata validation and JSON export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .._internal.identifiers import internal_scenario_id
from .._internal.serialization import jsonable as _jsonable
from .._internal.identifiers import canonical_scenario_ids
from .registry import MODELS, make_model

MODEL_CARD_SCHEMA_VERSION = "aiogym.model_card.v2"

REQUIRED_MODEL_CARD_FIELDS = (
    "schema_version",
    "scenario",
    "name",
    "summary",
    "states",
    "actions",
    "state_vector",
    "action_vector",
    "dynamics_disturbances",
    "parameters",
    "physical_metadata",
    "solver",
    "disturbances",
    "disturbance_defaults",
    "constraints",
    "plant_regime",
    "economic_config",
    "supervisory_layout",
    "dt_micro",
    "energy_scored",
)


def iter_model_cards(scenarios: Iterable[str] | None = None):
    """Yield ``(scenario, model_card)`` for the requested registered models."""

    for scenario in scenarios or canonical_scenario_ids(tuple(MODELS)):
        card = _jsonable(make_model(scenario).model_card())
        card["schema_version"] = MODEL_CARD_SCHEMA_VERSION
        validate_model_card(card, expected_scenario=scenario)
        yield scenario, card


def collect_model_cards(scenarios: Iterable[str] | None = None) -> dict[str, dict]:
    """Return validated model cards keyed by scenario name."""

    return {scenario: card for scenario, card in iter_model_cards(scenarios)}


def validate_model_card(card: Mapping, expected_scenario: str | None = None) -> None:
    """Validate the stable model metadata surface used by artifacts and docs."""

    missing = [field for field in REQUIRED_MODEL_CARD_FIELDS if field not in card]
    if missing:
        raise ValueError(f"model card is missing required fields: {', '.join(missing)}")
    if card["schema_version"] != MODEL_CARD_SCHEMA_VERSION:
        raise ValueError(f"unsupported model card schema: {card['schema_version']!r}")
    if (
        expected_scenario is not None
        and internal_scenario_id(card["scenario"])
        != internal_scenario_id(expected_scenario)
    ):
        raise ValueError(f"expected scenario {expected_scenario!r}, got {card['scenario']!r}")
    if not isinstance(card["states"], list) or not card["states"]:
        raise ValueError(f"{card['scenario']} model card must include at least one state")
    if not isinstance(card["actions"], list):
        raise ValueError(f"{card['scenario']} model card actions must be a list")
    for row in card["states"]:
        _require_model_card_fields(card["scenario"], "state", row, ("name", "unit", "bounds"))
        _require_model_card_bounds(card["scenario"], "state", row)
    for row in card["actions"]:
        _require_model_card_fields(card["scenario"], "action", row, ("name", "kind", "index", "unit", "bounds"))
        _require_model_card_bounds(card["scenario"], "action", row)
    if "controlled_outputs" in card:
        if not isinstance(card["controlled_outputs"], list):
            raise ValueError(f"{card['scenario']} controlled_outputs must be a list")
        for row in card["controlled_outputs"]:
            _require_model_card_fields(card["scenario"], "controlled output", row, ("name", "unit", "bounds"))
        if "controlled_output_vector" in card and len(card["controlled_outputs"]) != int(card["controlled_output_vector"]["length"]):
            raise ValueError(f"{card['scenario']} controlled output count does not match controlled_output_vector length")
    if "setpoints" in card:
        if not isinstance(card["setpoints"], list):
            raise ValueError(f"{card['scenario']} setpoints must be a list")
        for row in card["setpoints"]:
            _require_model_card_fields(card["scenario"], "setpoint", row, ("name", "unit", "bounds"))
        if "setpoint_vector" in card and len(card["setpoints"]) != int(card["setpoint_vector"]["length"]):
            raise ValueError(f"{card['scenario']} setpoint count does not match setpoint_vector length")
    if len(card["states"]) != int(card["state_vector"]["length"]):
        raise ValueError(f"{card['scenario']} state count does not match state_vector length")
    if len(card["actions"]) != int(card["action_vector"]["length"]):
        raise ValueError(f"{card['scenario']} action count does not match action_vector length")
    if not isinstance(card["parameters"], dict) or not card["parameters"]:
        raise ValueError(f"{card['scenario']} model card must include parameters")
    for name, row in card["parameters"].items():
        _require_model_card_fields(card["scenario"], f"parameter {name}", row, ("value", "unit", "bounds"))
    if not isinstance(card["physical_metadata"], dict):
        raise ValueError(f"{card['scenario']} physical_metadata must be a mapping")
    _require_model_card_fields(
        card["scenario"], "physical metadata", card["physical_metadata"],
        ("parameter_status", "fidelity", "time_unit", "references", "solver"),
    )
    if not isinstance(card["solver"], dict):
        raise ValueError(f"{card['scenario']} solver must be a mapping")
    _require_model_card_fields(card["scenario"], "solver", card["solver"], ("method", "max_step"))
    if not isinstance(card["disturbances"], list):
        raise ValueError(f"{card['scenario']} disturbances must be a list")
    if not isinstance(card["constraints"], list) or not card["constraints"]:
        raise ValueError(f"{card['scenario']} model card must include constraints")
    if not isinstance(card["plant_regime"], dict) or not card["plant_regime"]:
        raise ValueError(f"{card['scenario']} model card must include plant_regime")
    if not isinstance(card["economic_config"], dict) or not card["economic_config"]:
        raise ValueError(f"{card['scenario']} model card must include economic_config")


def export_model_cards(out_dir: str | Path, scenarios: Iterable[str] | None = None,
                       write_manifest: bool = True) -> dict:
    """Write one JSON model card per scenario and an optional manifest."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cards = collect_model_cards(scenarios)
    written = {}
    for scenario, card in cards.items():
        path = out / f"{scenario}.json"
        _write_model_card_json(path, card)
        written[scenario] = str(path)
    manifest = {
        "schema_version": MODEL_CARD_SCHEMA_VERSION,
        "scenarios": list(cards),
        "cards": written,
    }
    if write_manifest:
        manifest_path = out / "manifest.json"
        _write_model_card_json(manifest_path, manifest)
        manifest["manifest"] = str(manifest_path)
    return manifest


def _require_model_card_fields(scenario: str, kind: str, row: Mapping, fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{scenario} {kind} is missing fields: {', '.join(missing)}")


def _require_model_card_bounds(scenario: str, kind: str, row: Mapping) -> None:
    bounds = row.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"{scenario} {kind} {row.get('name', '')!r} must expose [low, high] bounds")


def _write_model_card_json(path: Path, data: Mapping) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
