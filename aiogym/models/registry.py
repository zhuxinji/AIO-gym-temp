import copy
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .core import _is_model_instance
from .scenarios import (
    CascadeModel,
    QuadrupleModel,
    CSTRModel,
    HVACModel,
    ExtractionModel,
    FiredHeaterModel,
    CrystallizationModel,
)


MODELS = {
    "cascade": CascadeModel,
    "quadruple": QuadrupleModel,
    "cstr": CSTRModel,
    "hvac": HVACModel,
    "extraction": ExtractionModel,
    "heater": FiredHeaterModel,
    "crystallization": CrystallizationModel,
}
BUILTIN_MODELS = dict(MODELS)
SCENARIOS = list(MODELS.keys())


def gym_id_name(scenario, model_factory=None):
    factory = model_factory if model_factory is not None else MODELS[scenario]
    explicit = getattr(factory, "gym_id", None)
    if explicit:
        return explicit
    return "".join(part.capitalize() for part in scenario.split("_"))


def builtin_gym_ids():
    return {scenario: gym_id_name(scenario, factory) for scenario, factory in BUILTIN_MODELS.items()}


def _refresh_scenarios():
    SCENARIOS[:] = list(MODELS.keys())


def validate_model_contract(model):
    required_methods = (
        "actuator_counts", "initial_state", "action_vector", "action_vector_to_dict",
        "dynamics", "outputs", "measurement", "controlled_output", "setpoint_vector",
    )
    missing = [name for name in required_methods if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(f"custom process model is missing required methods: {', '.join(missing)}")
    schema_methods = ("state_schema", "action_schema", "disturbance_schema", "disturbance_defaults")
    missing_schema = [name for name in schema_methods if not callable(getattr(model, name, None))]
    if missing_schema:
        raise TypeError(
            "custom process model should inherit ProcessModelContract or implement schema methods: "
            + ", ".join(missing_schema)
        )
    if not hasattr(model, "p") or not isinstance(model.p, dict):
        raise TypeError("custom process model must expose a parameter dict as .p")
    if not hasattr(model, "scenario") or not getattr(model, "scenario"):
        raise TypeError("custom process model must expose a non-empty .scenario")
    if not hasattr(model, "n") or int(model.n) <= 0:
        raise TypeError("custom process model must expose a positive .n")
    if len(model.initial_state()) != len(model.state_schema()):
        raise TypeError("custom process model state_schema() length must match initial_state()")
    if len(model.action_schema()) != sum(model.actuator_counts()):
        raise TypeError("custom process model action_schema() length must match actuator_counts()")
    return model


def register_model(name, model_factory, *, replace=False):
    if not isinstance(name, str) or not name:
        raise ValueError("model name must be a non-empty string")
    if name in MODELS and not replace:
        raise ValueError(f"model '{name}' is already registered")
    if _is_model_instance(model_factory):
        validate_model_contract(model_factory)
        MODELS[name] = lambda: copy.deepcopy(model_factory)
    elif isinstance(model_factory, type) or callable(model_factory):
        validate_model_contract(model_factory())
        MODELS[name] = model_factory
    else:
        raise TypeError("model_factory must be a model instance, class, or zero-argument factory")
    _refresh_scenarios()


def unregister_model(name):
    if name in BUILTIN_MODELS:
        raise ValueError(f"built-in model '{name}' cannot be unregistered")
    MODELS.pop(name, None)
    _refresh_scenarios()


def make_model(scenario="cascade"):
    if _is_model_instance(scenario):
        return validate_model_contract(scenario)
    if isinstance(scenario, type) or (callable(scenario) and not isinstance(scenario, str)):
        return validate_model_contract(scenario())
    if scenario not in MODELS:
        raise ValueError(f"unknown process model '{scenario}'. Registered models: {', '.join(SCENARIOS)}")
    factory = MODELS[scenario]
    model = factory if _is_model_instance(factory) else factory()
    return validate_model_contract(model)


def obs_vector(model, levels, temps, t_cold, t_amb, h_sp, t_sp):
    """Matches obsVector() in controllers.js:
    obs = [levels(n), temps(n), t_sp(n), h_sp(controlled k), t_cold, t_amb]."""
    n = model.n
    o = []
    for i in range(n):
        o.append(levels[i] if i < len(levels) else 0.0)
    for i in range(n):
        o.append(temps[i])
    for i in range(n):
        o.append(t_sp[i])
    for i in model.controlled_levels():
        o.append(h_sp[i])
    o.append(t_cold)
    o.append(t_amb)
    return o


MODEL_CARD_SCHEMA_VERSION = "aiogym.model_card.v1"

HUMAN_MODEL_CARD_NOTES = {
    "cascade": {
        "dynamics": "Cascaded material balances move liquid through three heated tanks; energy balances mix inlet, interstage, ambient-loss, and heater terms.",
        "assumptions": [
            "Well-mixed tanks with lumped heat loss and actuator gains.",
            "Levels are clamped to the declared tank range after integration.",
            "Heat-transfer and flow coefficients are benchmark parameters, not plant-identification claims.",
        ],
        "best_use": "multi-loop level and temperature tracking, actuator allocation, and disturbance-rejection smoke tests",
    },
    "quadruple": {
        "dynamics": "Four interacting tank balances route two pump streams through upper and lower tanks, with a thermal state attached to each tank.",
        "assumptions": [
            "Hydraulic coupling is simplified to benchmark-scale orifice flow.",
            "Thermal dynamics share lumped heater and heat-loss terms.",
            "The model is intended for control comparison rather than hardware sizing.",
        ],
        "best_use": "interacting level control, MIMO setpoint tracking, and robustness tests",
    },
    "cstr": {
        "dynamics": "A two-state exothermic reactor tracks concentration and temperature with feed dilution, Arrhenius reaction rate, heat release, and cooling action.",
        "assumptions": [
            "Physical constants are simplified and partly lumped for a stable benchmark operating range.",
            "Cooling is represented as a normalized actuator rather than detailed jacket hydraulics.",
            "Economic value is production-oriented and should be compared only within the declared protocol.",
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
    if len(card["states"]) != int(card["state_vector"]["length"]):
        raise ValueError(f"{card['scenario']} state count does not match state_vector length")
    if len(card["actions"]) != int(card["action_vector"]["length"]):
        raise ValueError(f"{card['scenario']} action count does not match action_vector length")
    if not isinstance(card["parameters"], dict) or not card["parameters"]:
        raise ValueError(f"{card['scenario']} model card must include parameters")
    for name, row in card["parameters"].items():
        _require_model_card_fields(card["scenario"], f"parameter {name}", row, ("value", "unit", "bounds"))
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


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value
