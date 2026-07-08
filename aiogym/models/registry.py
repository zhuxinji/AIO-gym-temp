import copy

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
