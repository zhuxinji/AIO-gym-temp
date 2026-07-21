"""Model-card schema validation, export, and Markdown rendering."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .._internal.serialization import jsonable as _jsonable
from .registry import SCENARIOS, make_model

MODEL_CARD_SCHEMA_VERSION = "aiogym.model_card.v2"

HUMAN_MODEL_CARD_NOTES = {
    "cascade": {
        "dynamics": "Cascaded material balances move liquid through three heated tanks; energy balances mix inlet, interstage, ambient-loss, and heater terms.",
        "assumptions": [
            "Constant-area, perfectly mixed tanks use constant liquid density and heat capacity.",
            "Each outlet is a one-way free drain with no downstream backpressure or reverse flow.",
            "Pipes have no volume, transport delay, pressure dynamics, or heat loss.",
            "Tank walls, heaters, pumps, valves, and sensors have no independent dynamics; boiling and other phase changes are excluded.",
            "Negative levels and overflow are hard termination events; levels are not silently clamped.",
            "Low liquid level or a 92 degC tank temperature disables that tank's heater before the 120 degC hard termination boundary.",
            "Direct operation is batch; continuous throughput economics belongs to the continuous-benchmark task.",
            "Heat-transfer and flow coefficients are benchmark parameters, not plant-identification claims.",
        ],
        "best_use": "multi-loop level and temperature tracking, actuator allocation, safety-boundary tests, and task-declared continuous-production benchmarks",
    },
    "cascade_recirculating": {
        "dynamics": "P101 removes liquid and enthalpy from Tank 3 and returns it to Tank 1; V12 and V23 close the gravity-fed loop, while only Tank 1 receives heater power.",
        "assumptions": [
            "The normal process is a closed liquid inventory with no product or makeup flow.",
            "The provisional geometry interprets the PDF ranges as 16 L Tank 1, 16 L Tank 2, and 60 L Tank 3 inventory at the 0.40 m nominal level; these are range-derived design values, not as-built measurements.",
            "The PDF specifies one 2 kW Tank 1 heater and gives design ranges for tank height, effective volume, and pump power; valve capacity, flow calibration, heat loss, and interlock setpoints remain unverified.",
            "Pipes, sensors, pump, valves, and heater have no independent dynamics in the foundational model.",
            "Tank 1 and Tank 2 passive standpipes return mass and enthalpy to Tank 3 using provisional overflow levels and coefficients.",
            "P101 low level and H1 L2/L3/L4 protection disable equipment without projecting the physical state; only hard physical boundaries terminate simulation.",
            "Tracking, KPI, robustness, and safety are supported; production economics is intentionally unsupported because the closed loop has no declared product stream or production-value basis.",
        ],
        "best_use": "closed-loop three-tank control development and staged software validation against the retrofit design intent",
    },
    "quadruple": {
        "dynamics": "Johansson's four nonlinear liquid-level balances route two voltage-driven pump streams through fixed three-way valve splits; upper tanks drain into the two lower tanks.",
        "assumptions": [
            "Liquid is incompressible and each outlet follows Torricelli flow.",
            "Tank cross-sections, outlet areas, pump gains, and valve split fractions are constant within one run.",
            "Pipe, sensor, and pump dynamics are neglected as in the reference nonlinear physical model.",
        ],
        "best_use": "classic two-input/two-output MIMO level control, interaction analysis, and minimum/nonminimum-phase studies",
    },
    "cstr": {
        "dynamics": "A two-state exothermic reactor tracks concentration and temperature with feed dilution, Arrhenius reaction rate, heat release, and cooling action.",
        "assumptions": [
            "Physical constants are simplified and partly lumped for a stable benchmark operating range.",
            "Cooling is represented as a normalized actuator rather than detailed jacket hydraulics.",
            "Economic value is production-oriented and should be compared only within the declared objective.",
        ],
        "best_use": "economic control, nonlinear tracking, and NMPC/oracle comparisons",
    },
    "hvac": {
        "dynamics": "Two coupled thermal-zone balances combine outdoor exchange, inter-zone coupling, internal heat load, and bidirectional HVAC actuation.",
        "assumptions": [
            "Thermal capacitance, envelope conductance, and actuator efficiency are lumped.",
            "Outdoor temperature and heat-load disturbances are benchmark schedules.",
            "Comfort and operating bounds are safety/evaluation constraints, not a building-code model.",
        ],
        "best_use": "comfort tracking, energy-aware control, and disturbance-adaptation tests",
    },
    "extraction": {
        "dynamics": "A five-stage counter-current extraction column updates liquid and gas concentrations using simplified stage-transfer balances.",
        "assumptions": [
            "The column follows the PC-Gym-style benchmark abstraction rather than a detailed tray model.",
            "Concentrations are bounded to a normalized operating interval.",
            "Flow actions and mass-transfer coefficients are scaled benchmark parameters.",
        ],
        "best_use": "multi-stage concentration tracking and PC-Gym-style controller comparisons",
    },
    "heater": {
        "dynamics": "A fired-heater surrogate couples firebox temperature, outlet temperature, and flue oxygen through fuel, air, heat-transfer, and load terms.",
        "assumptions": [
            "Combustion and heat-transfer behavior is represented by lumped benchmark equations.",
            "Oxygen is treated as a controlled quality/safety variable, not a full emissions model.",
            "Economic scoring favors useful outlet heat while penalizing fuel and unsafe operation.",
        ],
        "best_use": "supervisory setpoint control, safety-aware temperature control, and energy tradeoff studies",
    },
    "crystallization": {
        "dynamics": "Moment dynamics and concentration balance map cooling-temperature action to crystal-size quality outputs such as Ln and CV.",
        "assumptions": [
            "The actuator is a normalized cooling-temperature command.",
            "Growth, nucleation, and solubility terms are benchmark-scale simplifications.",
            "Quality metrics are exposed through process info and tracking outputs.",
        ],
        "best_use": "PC-Gym-style crystallization tracking and single-actuator quality control",
    },
}

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

    for scenario in scenarios or SCENARIOS:
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
    if expected_scenario is not None and card["scenario"] != expected_scenario:
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


def render_model_card_markdown(card: Mapping) -> str:
    """Render a model-card mapping as human-readable Markdown."""

    scenario = str(card["scenario"])
    notes = HUMAN_MODEL_CARD_NOTES.get(scenario, {})
    lines = [
        f"# {card['name']} (`{scenario}`)",
        "",
        str(card.get("summary", "")).strip(),
        "",
        "## Recommended Use",
        "",
        notes.get("best_use", "process-control benchmark evaluation"),
        "",
        "## Dynamics And Assumptions",
        "",
        notes.get("dynamics", "Dynamics are implemented by the registered process model."),
        "",
    ]
    assumptions = notes.get("assumptions", ())
    if assumptions:
        lines.extend(_markdown_bullets(assumptions))
        lines.append("")
    lines.extend([
        "The executable source of truth is the model implementation under `aiogym.models.scenarios`.",
        "",
        "## Physical Metadata Status",
        "",
        f"- Parameter status: `{card.get('physical_metadata', {}).get('parameter_status', 'unknown')}`",
        f"- Fidelity: `{card.get('physical_metadata', {}).get('fidelity', 'unknown')}`",
        f"- Time unit: `{card.get('physical_metadata', {}).get('time_unit', 'unknown')}`",
        f"- Solver: `{card.get('solver', {}).get('method', 'unknown')}` with maximum step `{card.get('solver', {}).get('max_step', 'unknown')}`",
        "",
        "## State Vector",
        "",
    ])
    lines.extend(_markdown_table(
        ("Name", "Unit", "Bounds"),
        ((row.get("name", ""), row.get("unit", ""), _format_bounds(row.get("bounds"))) for row in card["states"]),
    ))
    lines.extend(["", "## Action Vector", ""])
    lines.extend(_markdown_table(
        ("Name", "Kind", "Index", "Bounds"),
        (
            (row.get("name", ""), row.get("kind", ""), row.get("index", ""), _format_bounds(row.get("bounds")))
            for row in card["actions"]
        ),
    ))
    if card.get("controlled_outputs"):
        lines.extend(["", "## Controlled Output Vector", ""])
        lines.extend(_markdown_table(
            ("Name", "Unit", "Bounds"),
            (
                (row.get("name", ""), row.get("unit", ""), _format_bounds(row.get("bounds")))
                for row in card["controlled_outputs"]
            ),
        ))
    if card.get("setpoints"):
        lines.extend(["", "## Setpoint Vector", ""])
        lines.extend(_markdown_table(
            ("Name", "Output", "Unit", "Bounds"),
            (
                (row.get("name", ""), row.get("output", ""), row.get("unit", ""), _format_bounds(row.get("bounds")))
                for row in card["setpoints"]
            ),
        ))
    lines.extend(["", "## Disturbances", ""])
    lines.extend(_markdown_table(
        ("Name", "Event", "Unit", "Bounds", "Default"),
        (
            (
                row.get("name", ""),
                row.get("event", ""),
                row.get("unit", ""),
                _format_bounds(row.get("bounds")),
                _format_value(card.get("disturbance_defaults", {}).get(row.get("name", ""), row.get("default", ""))),
            )
            for row in card.get("disturbances", ())
            if row.get("kind") != "setpoint"
        ),
    ))
    lines.extend(["", "## Parameters", ""])
    lines.extend(_markdown_table(
        ("Name", "Value", "Unit", "Bounds"),
        (
            (name, _format_value(row.get("value")), row.get("unit", ""), _format_bounds(row.get("bounds")))
            for name, row in sorted(card.get("parameters", {}).items())
        ),
    ))
    lines.extend(["", "## Safety And Operating Bounds", ""])
    lines.extend(_markdown_table(
        ("Constraint", "States", "Bounds"),
        (
            (row.get("name", ""), ", ".join(row.get("states", ())), _format_bounds(row.get("bounds")))
            for row in card.get("constraints", ())
        ),
    ))
    lines.extend(["", "## Benchmark Semantics", ""])
    lines.extend([
        f"- State vector length: {card['state_vector']['length']}",
        f"- Action vector length: {card['action_vector']['length']}",
        f"- Controlled output vector length: {card.get('controlled_output_vector', {}).get('length', len(card.get('controlled_outputs', [])))}",
        f"- Setpoint vector length: {card.get('setpoint_vector', {}).get('length', len(card.get('setpoints', [])))}",
        f"- Dynamics disturbances: {_format_value(card.get('dynamics_disturbances', []))}",
        f"- Micro integration step: {card.get('dt_micro')} s",
        f"- Energy is scored: {bool(card.get('energy_scored', True))}",
    ])
    lines.extend(["", "### Plant-Regime Randomization", ""])
    lines.extend(_markdown_table(
        ("Parameter", "Multiplier Bounds"),
        ((name, _format_bounds(bounds)) for name, bounds in sorted(card.get("plant_regime", {}).items())),
    ))
    lines.extend(["", "### Economic Configuration", ""])
    lines.extend(_markdown_table(
        ("Key", "Value"),
        ((name, _format_value(value)) for name, value in sorted(card.get("economic_config", {}).items())),
    ))
    layout = card.get("supervisory_layout", [])
    if layout:
        lines.extend(["", "### Supervisory Layout", ""])
        lines.extend(_markdown_table(
            ("Entry", "Meaning"),
            ((i, _format_value(row)) for i, row in enumerate(layout)),
        ))
    lines.append("")
    return "\n".join(lines)


def export_model_card_markdown(out_dir: str | Path, scenarios: Iterable[str] | None = None,
                               write_index: bool = True) -> dict:
    """Write one human-readable Markdown model card per scenario."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cards = collect_model_cards(scenarios)
    written = {}
    for scenario, card in cards.items():
        path = out / f"{scenario}.md"
        path.write_text(render_model_card_markdown(card))
        written[scenario] = str(path)
    manifest = {
        "schema_version": MODEL_CARD_SCHEMA_VERSION,
        "format": "markdown",
        "scenarios": list(cards),
        "cards": written,
    }
    if write_index:
        index_path = out / "README.md"
        index_lines = [
            "# AIO-Gym Model Cards",
            "",
            "Human-readable summaries generated from the registered process-model metadata.",
            "",
        ]
        for scenario, card in cards.items():
            index_lines.append(f"- [{card['name']} (`{scenario}`)]({scenario}.md)")
        index_lines.append("")
        index_path.write_text("\n".join(index_lines))
        manifest["index"] = str(index_path)
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


def _markdown_table(headers, rows):
    materialized = [tuple("" if cell is None else str(cell) for cell in row) for row in rows]
    if not materialized:
        return ["No entries."]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in materialized:
        out.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    return out


def _markdown_bullets(items):
    return [f"- {item}" for item in items]


def _escape_markdown_cell(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _format_bounds(bounds):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return ""
    lo, hi = bounds
    return f"[{_format_value(lo)}, {_format_value(hi)}]"


def _format_value(value):
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, np.generic):
        return _format_value(value.item())
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_format_value(v)}" for k, v in value.items()) + "}"
    return str(value)
