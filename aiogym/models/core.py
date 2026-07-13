"""Native NumPy/pure-Python process models for AIO-Gym.

State layout:
  cascade / quadruple : x = [h0, T0, h1, T1, ...]   (level, temp interleaved)
  cstr                : x = [Ca, T]
  hvac                : x = [T0, T1]
  heater              : x = [T_firebox, T_out, O2]
Actions are the actuator vector in [0, 1]: [pumps..., valves..., heaters...].
"""
from __future__ import annotations
import copy
import math
from collections.abc import Mapping

RHO = 1000.0
CP = 4186.0
G = 9.81
RHO_CP = RHO * CP


def _maxv(a, b):
    return a if a > b else b


def _copy_value(v):
    return list(v) if isinstance(v, list) else v


def _is_model_instance(obj):
    return hasattr(obj, "derivatives") and hasattr(obj, "initial_state") and not isinstance(obj, (str, type))


class _NumericOps:
    @staticmethod
    def sqrt(v):
        return math.sqrt(v)

    @staticmethod
    def exp(v):
        return math.exp(v)

    @staticmethod
    def sin(v):
        return math.sin(v)

    @staticmethod
    def cos(v):
        return math.cos(v)

    @staticmethod
    def tan(v):
        return math.tan(v)

    @staticmethod
    def log(v):
        return math.log(v)

    @staticmethod
    def max(a, b):
        return _maxv(a, b)

    @staticmethod
    def smooth_max(a, b, eps=1e-6):
        return _maxv(a, b)

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def abs(v):
        return abs(v)

    @staticmethod
    def vector(values):
        return list(values)


def _casadi_ops(ca):
    class _CasadiOps:
        @staticmethod
        def sqrt(v):
            return ca.sqrt(v)

        @staticmethod
        def exp(v):
            return ca.exp(v)

        @staticmethod
        def sin(v):
            return ca.sin(v)

        @staticmethod
        def cos(v):
            return ca.cos(v)

        @staticmethod
        def tan(v):
            return ca.tan(v)

        @staticmethod
        def log(v):
            return ca.log(v)

        @staticmethod
        def max(a, b):
            return ca.fmax(a, b)

        @staticmethod
        def smooth_max(a, b, eps=1e-6):
            return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps ** 2))

        @staticmethod
        def min(a, b):
            return ca.fmin(a, b)

        @staticmethod
        def abs(v):
            return ca.fabs(v)

        @staticmethod
        def vector(values):
            return ca.vertcat(*values)

    return _CasadiOps


_NUMERIC_OPS = _NumericOps()


class ProcessModelContract:
    """Common read-only contract for process-model metadata.

    The numerical dynamics stay in each concrete model. This layer only exposes
    names, units, bounds, disturbances, and safety constraints for benchmark and
    custom-model tooling.
    """

    display_name = "Process model"
    summary = ""
    state_names = ()
    state_units = {}
    state_bounds = {}
    param_units = {}
    param_bounds = {}
    action_names = ()
    action_units = {}
    action_bounds = {}
    action_kinds = {}
    output_names = ()
    output_units = {}
    output_bounds = {}
    setpoint_names = ()
    setpoint_units = {}
    setpoint_bounds = {}
    default_y_sp = ()
    plant_regime = {}
    economic_config = {
        "temp_band": (),
        "level_band": (),
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.0,
        "w_viol": 0.0,
    }
    supervisory_layout = ()
    supports_generic_setpoint_randomization = True
    supports_integral_observation = True
    randomize_common_temperatures = True
    disturbance_attributes = {
        "t_cold": "t_cold",
        "t_amb": "t_amb",
        "extra_outflow": "extra_outflow",
        "Caf": "caf",
        "Tcool": "tcool",
        "pump_flow_factor": "pump_flow_factor",
        "heater_efficiency": "heater_efficiency",
        "heat_loss_factor": "heat_loss_factor",
        "heat_load": "heat_load",
        "hvac_efficiency": "hvac_efficiency",
        "growth_factor": "growth_factor",
        "nucleation_factor": "nucleation_factor",
        "solubility_bias": "solubility_bias",
    }
    input_disturbances = (
        {"name": "t_cold", "event": "cold_inlet_step", "unit": "degC", "bounds": (0.0, 40.0), "description": "inlet or cold-source temperature"},
        {"name": "t_amb", "event": "ambient_step", "unit": "degC", "bounds": (0.0, 45.0), "description": "ambient temperature"},
        {"name": "extra_outflow", "event": "demand_surge", "unit": "m3/s", "bounds": (0.0, 0.002), "default": 0.0, "description": "downstream demand outflow"},
    )
    event_disturbances = (
        {"name": "setpoint_move", "event": "setpoint_move", "kind": "setpoint", "description": "controlled-variable setpoint move"},
    )
    safety_constraints = ()

    def _extra_parameters(self):
        return {}

    def _schema_row(self, name, units, bounds):
        return {"name": name, "unit": units.get(name, ""), "bounds": bounds.get(name)}

    def state_schema(self):
        names = self._vector_names(self.state_names, "x", len(self.initial_state()))
        return [self._schema_row(name, self.state_units, self.state_bounds) for name in names]

    def action_schema(self):
        names = self._action_names()
        legacy_kinds = self._legacy_action_kinds()
        counters = {}
        rows = []
        for i, name in enumerate(names):
            kind = self.action_kinds.get(name, legacy_kinds[i] if i < len(legacy_kinds) else "input")
            counters.setdefault(kind, 0)
            rows.append({
                "name": name,
                "kind": kind,
                "index": i,
                "kind_index": counters[kind],
                "unit": self.action_units.get(name, "fraction"),
                "bounds": self.action_bounds.get(name, (0.0, 1.0)),
            })
            counters[kind] += 1
        return rows

    def action_dim(self):
        return len(self._action_names())

    def _action_names(self):
        if self.action_names:
            return list(self.action_names)
        if self.action_bounds:
            return list(self.action_bounds.keys())
        return [f"u{i}" for i in range(sum(self.actuator_counts()))]

    def _legacy_action_kinds(self):
        if not self.uses_legacy_actions():
            return []
        n_pumps, n_valves, n_heaters = self.actuator_counts()
        kinds = (["pump"] * n_pumps) + (["valve"] * n_valves) + (["heater"] * n_heaters)
        return kinds if len(kinds) == len(self._action_names()) else []

    def action_vector(self, act):
        """Return the flat actuator vector u = [pumps..., valves..., heaters...]."""
        if isinstance(act, Mapping):
            return (
                [float(v) for v in act.get("pumps", [])]
                + [float(v) for v in act.get("valves", [])]
                + [float(v) for v in act.get("heaters", [])]
            )
        return [float(v) for v in act]

    def action_vector_to_dict(self, u):
        """Map a flat actuator vector u back to the legacy physical action dict."""
        n_pumps, n_valves, n_heaters = self.actuator_counts()
        values = self.action_vector(u)
        expected = n_pumps + n_valves + n_heaters
        if len(values) != expected:
            raise ValueError(f"{self.scenario} expected {expected} action values, got {len(values)}")
        return {
            "pumps": list(values[:n_pumps]),
            "valves": list(values[n_pumps:n_pumps + n_valves]),
            "heaters": list(values[n_pumps + n_valves:]),
        }

    def default_action(self):
        u = [0.5] * self.action_dim()
        return self.action_vector_to_dict(u) if self.uses_legacy_actions() else u

    def uses_legacy_actions(self):
        return type(self).actuator_counts is not ProcessModelContract.actuator_counts

    def state_vector(self, x):
        """Return the generic state vector x used by controllers and simulators."""
        return [float(v) for v in x]

    def _vector_names(self, names, prefix: str, length: int):
        values = list(names)
        if len(values) != int(length):
            values = [f"{prefix}{i}" for i in range(int(length))]
        return values

    def dynamics(self, x, u, env=None, backend="numeric", ca=None):
        """Generic continuous dynamics dx/dt = f(x, u, env).

        ``backend="numeric"`` returns numbers for simulation. ``backend="casadi"``
        returns a CasADi expression graph for NMPC, using the same model formula.
        """
        if callable(getattr(self, "_dynamics", None)):
            if backend == "casadi":
                if ca is None:
                    raise ValueError("backend='casadi' requires the casadi module as ca=...")
                return self._dynamics(x, u, self.dynamics_disturbance_map(env), _casadi_ops(ca))
            if backend != "numeric":
                raise ValueError(f"unknown dynamics backend: {backend!r}")
            return self._dynamics(self.state_vector(x), self.action_vector(u), env or {}, _NUMERIC_OPS)
        if backend != "numeric":
            raise NotImplementedError(f"{self.scenario} does not support backend={backend!r} dynamics")
        return self.derivatives(self.state_vector(x), self.action_vector_to_dict(u), env or {})

    def derivatives(self, x, act, env):
        return self.dynamics(x, act, env)

    def outputs(self, x):
        """Semantic outputs derived from x.

        ``y`` is the generic controlled-output vector used for calculations.
        ``levels``/``temps`` are optional physical display channels for built-in
        process scenarios.
        """
        state = self.state_vector(x)
        display = self.display_outputs(state)
        out = {
            "x": state,
            "levels": list(display.get("levels", [])),
            "temps": list(display.get("temps", [])),
        }
        for key, value in display.items():
            if key not in out:
                out[key] = value
        if callable(getattr(self, "conc", None)):
            out["conc"] = self.conc(state)
        out["y"] = self.controlled_output(state)
        return out

    def measurement(self, x, env=None):
        """Measured state dict exposed to controllers.

        New controllers should use x/y generically; legacy display adapters can
        still inspect levels, temps, conc, and disturbance names.
        """
        return {**self.outputs(x), **dict(env or {})}

    def controlled_output(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return [x[i] for i in range(len(self.initial_state()))]
        return self.state_vector(x)

    def setpoint_vector(self, y_sp=None):
        return list(y_sp) if y_sp is not None else self.default_setpoint_vector()

    def default_setpoint_vector(self):
        if self.default_y_sp:
            return list(self.default_y_sp)
        values = []
        for row in self.setpoint_schema():
            bounds = row.get("bounds")
            if isinstance(bounds, (tuple, list)) and len(bounds) == 2 and bounds[0] is not None and bounds[1] is not None:
                values.append(0.5 * (float(bounds[0]) + float(bounds[1])))
            else:
                values.append(0.0)
        return values

    def env_setpoint_vector(self, options=None):
        """Return initial setpoints for an environment instance."""

        return self.default_setpoint_vector()

    def sample_env_setpoints(self, y_sp, rng, options=None):
        """Apply model-specific reset-time target sampling."""

        return list(y_sp)

    def setpoint_schema(self):
        output_rows = self.controlled_output_schema()
        names = self._vector_names(self.setpoint_names or self.output_names, "y_sp", len(output_rows))
        rows = []
        for i, output in enumerate(output_rows):
            output_name = output.get("name", f"y{i}")
            name = names[i] if i < len(names) else f"y_sp{i}"
            rows.append({
                "name": name,
                "unit": self.setpoint_units.get(name, self.output_units.get(output_name, output.get("unit", ""))),
                "bounds": self.setpoint_bounds.get(name, output.get("bounds")),
                "output": output_name,
            })
        return rows

    def controlled_output_schema(self):
        y0 = list(self.controlled_output(self.initial_state()))
        names = self._vector_names(self.output_names, "y", len(y0))
        legacy_bounds = self._legacy_controlled_output_bounds()
        rows = []
        for i, name in enumerate(names):
            bounds = self.output_bounds.get(name)
            if bounds is None and i < len(legacy_bounds):
                bounds = legacy_bounds[i]
            rows.append({"name": name, "unit": self.output_units.get(name, ""), "bounds": bounds})
        return rows

    def controlled_output_scales(self):
        scales = []
        for row in self.controlled_output_schema():
            bounds = row.get("bounds")
            scale = None
            if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lo, hi = bounds
                if lo is not None and hi is not None and float(hi) > float(lo):
                    scale = float(hi) - float(lo)
            scales.append(max(float(scale if scale is not None else 1.0), 1e-12))
        return scales

    def _legacy_controlled_output_bounds(self):
        return [None] * len(list(self.controlled_output(self.initial_state())))

    def dynamics_disturbance_specs(self):
        specs = []
        defaults = self.disturbance_defaults()
        for row in self.disturbance_schema():
            if row.get("kind") == "setpoint":
                continue
            name = row.get("name")
            if not name:
                continue
            default = defaults.get(name, row.get("default", self.p.get(name, 0.0)))
            if isinstance(default, (list, tuple)):
                specs.extend((name, i) for i in range(len(default)))
            else:
                specs.append((name, None))
        return tuple(specs)

    def dynamics_disturbance_names(self):
        return tuple(name if idx is None else f"{name}[{idx}]" for name, idx in self.dynamics_disturbance_specs())

    def dynamics_disturbance_map(self, d):
        values = {}
        for j, (name, idx) in enumerate(self.dynamics_disturbance_specs()):
            if idx is None:
                values[name] = d[j]
            else:
                values.setdefault(name, []).append(d[j])
        return values

    def disturbance_vector(self, values=None):
        merged = self.disturbance_defaults()
        merged.update(dict(values or {}))
        if "Caf" not in merged and "caf" in merged:
            merged["Caf"] = merged["caf"]
        if "Tcool" not in merged and "tcool" in merged:
            merged["Tcool"] = merged["tcool"]
        out = []
        for name, idx in self.dynamics_disturbance_specs():
            value = merged.get(name, self.p.get(name, 0.0))
            if idx is None:
                out.append(value)
            elif isinstance(value, (int, float)):
                out.append(value)
            else:
                out.append(value[idx])
        return out

    def energy_kw(self, u, backend="numeric", ca=None):
        return 0.0

    def action_energy_kw(self, act, x=None, env=None):
        """Return total action energy rate in kW for numeric environment steps."""

        return float(self.energy_kw(self.action_vector(act)))

    def economic_energy_kw(self, act, x=None, env=None):
        """Return the energy-rate term used by the model's economic objective."""

        return float(self.energy_kw(self.action_vector(act)))

    def economic_value(self, x, u, env=None, backend="numeric", ca=None):
        return 0.0

    def display_outputs(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return {"levels": [], "temps": [x[i] for i in range(len(self.initial_state()))]}
        return {"levels": [], "temps": list(x)}

    def parameter_schema(self):
        params = {k: _copy_value(v) for k, v in self.p.items()}
        params.update(self._extra_parameters())
        return {
            name: {"value": value, "unit": self.param_units.get(name, ""), "bounds": self.param_bounds.get(name)}
            for name, value in params.items()
        }

    def disturbance_schema(self):
        rows = []
        for row in self.input_disturbances + self.event_disturbances:
            out = dict(row)
            out.setdefault("name", out.get("event", "disturbance"))
            out.setdefault("dynamic", "event" in out)
            if "default" not in out and out["name"] in getattr(self, "p", {}):
                out["default"] = _copy_value(self.p[out["name"]])
            rows.append(out)
        return rows

    def disturbance_defaults(self):
        defaults = {}
        for row in self.disturbance_schema():
            if row.get("kind") == "setpoint":
                continue
            name = row.get("name")
            if not name:
                continue
            if "default" in row:
                defaults[name] = _copy_value(row["default"])
            elif name in getattr(self, "p", {}):
                defaults[name] = _copy_value(self.p[name])
            else:
                defaults[name] = 0.0
        return defaults

    def disturbance_attribute_map(self):
        defaults = self.runtime_env(self.disturbance_defaults())
        return {
            name: attr
            for name, attr in self.disturbance_attributes.items()
            if name in defaults
        }

    def runtime_env(self, disturbance_values):
        env = {name: _copy_value(value) for name, value in dict(disturbance_values or {}).items()}
        env.setdefault("t_cold", self.p.get("t_cold", 15.0))
        env.setdefault("t_amb", self.p.get("t_amb", 20.0))
        env.setdefault("extra_outflow", 0.0)
        return env

    def sample_disturbance(self, event, current, rng):
        if event == "cold_inlet_step":
            return float(max(2.0, min(35.0, float(current) + rng.uniform(-8, 8))))
        if event == "ambient_step":
            return float(max(0.0, min(40.0, float(current) + rng.uniform(-8, 12))))
        if event == "demand_surge":
            return float(abs(rng.uniform(0, 8e-4)))
        row = next((row for row in self.disturbance_schema() if row.get("event") == event), None)
        return self._sample_schema_disturbance(current, row, rng)

    def _sample_schema_disturbance(self, default, row, rng):
        if not row:
            return _copy_value(default)
        if row.get("values"):
            values = list(row["values"])
            return _copy_value(values[int(rng.integers(0, len(values)))])
        bounds = row.get("bounds")
        if (
            isinstance(bounds, (tuple, list))
            and len(bounds) == 2
            and bounds[0] is not None
            and bounds[1] is not None
        ):
            lo, hi = float(bounds[0]), float(bounds[1])
            if isinstance(default, (list, tuple)):
                return [float(rng.uniform(lo, hi)) for _ in default]
            return float(rng.uniform(lo, hi))
        return _copy_value(default)

    def process_constraint_info(self, x, levels, temps, env):
        return {}

    def process_info(self, x, levels, temps, env):
        return {}

    def constraint_penalty_scales(self):
        return {}

    def common_constraint_info(self, levels, temps):
        hmax = self.height_max
        return {
            "temp_high": max((max(0.0, t - 80.0) for t in temps), default=0.0),
            "temp_trip": max((max(0.0, t - 92.0) for t in temps), default=0.0),
            "level_high": max((max(0.0, levels[i] - 0.90 * hmax[i]) for i in range(len(levels))), default=0.0),
            "level_low": max((max(0.0, 0.15 * hmax[i] - levels[i]) for i in range(len(levels))), default=0.0),
        }

    def runaway_state(self, levels, temps):
        hmax = self.height_max
        return any(t > 92.0 for t in temps) or any(levels[i] > 0.97 * hmax[i] for i in range(len(levels)))

    def constraint_schema(self):
        return [dict(row) for row in self.safety_constraints]

    def model_card(self):
        return {
            "scenario": self.scenario,
            "name": self.display_name,
            "summary": self.summary,
            "states": self.state_schema(),
            "actions": self.action_schema(),
            "controlled_outputs": self.controlled_output_schema(),
            "setpoints": self.setpoint_schema(),
            "state_vector": {"name": "x", "length": len(self.initial_state())},
            "action_vector": {"name": "u", "length": self.action_dim()},
            "controlled_output_vector": {"name": "y", "length": len(self.controlled_output(self.initial_state()))},
            "setpoint_vector": {"name": "y_sp", "length": len(self.default_setpoint_vector())},
            "dynamics_disturbances": list(self.dynamics_disturbance_names()),
            "parameters": self.parameter_schema(),
            "disturbances": self.disturbance_schema(),
            "disturbance_defaults": self.disturbance_defaults(),
            "constraints": self.constraint_schema(),
            "plant_regime": copy.deepcopy(self.plant_regime),
            "economic_config": copy.deepcopy(self.economic_config),
            "supervisory_layout": [list(row) for row in self.supervisory_layout],
            "dt_micro": self.dt_micro,
            "energy_scored": bool(getattr(self, "energy_scored", True)),
        }

    def metadata(self):
        return self.model_card()

    @property
    def height_max(self):
        return [1.0] * int(self.n)

    def actuator_counts(self):
        return (0, 0, self.action_dim())

    def ideal_energy_kw(self, x, y_sp, env, act):
        return 0.0

class Integrator:
    """Fixed-step RK4 integrator for process models."""

    def __init__(self, model):
        self.model = model
        self.dt_micro = getattr(model, "dt_micro", 0.02) or 0.02
        self.reset()

    def reset(self, state=None):
        self.x = list(state) if state is not None else list(self.model.initial_state())
        self.t = 0.0

    def step(self, dt, act, env):
        m = self.model
        u = m.action_vector(act)
        f = lambda x: m.dynamics(x, u, env)
        nsub = max(1, math.ceil(dt / self.dt_micro - 1e-9))
        h = dt / nsub
        for _ in range(nsub):
            x = self.x
            k1 = f(x)
            k2 = f([v + 0.5 * h * k1[i] for i, v in enumerate(x)])
            k3 = f([v + 0.5 * h * k2[i] for i, v in enumerate(x)])
            k4 = f([v + h * k3[i] for i, v in enumerate(x)])
            self.x = [v + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i, v in enumerate(x)]
            if hasattr(m, "clamp_state"):
                self.x = m.clamp_state(self.x)
            self.t += h
        return self.x
