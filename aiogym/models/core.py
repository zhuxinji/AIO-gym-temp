"""Native (numpy/pure-python) port of the AIO-Gym plant models.

A faithful re-implementation of frontend/js/sim/models.js: same ODE right-hand
sides, same constants, same state layout, so a policy trained here transfers to
the browser sim and vice-versa. Parity against the JS source of truth is checked
by tests/test_parity.py (golden trajectories generated from the browser engine).

State layout (matches the JS):
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
    def max(a, b):
        return _maxv(a, b)

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
        def max(a, b):
            return ca.fmax(a, b)

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
        return [self._schema_row(name, self.state_units, self.state_bounds) for name in self.state_names]

    def action_schema(self):
        n_pumps, n_valves, n_heaters = self.actuator_counts()
        kinds = (["pump"] * n_pumps) + (["valve"] * n_valves) + (["heater"] * n_heaters)
        names = list(self.action_names)
        if len(names) != len(kinds):
            names = [f"{kind}_{i}" for i, kind in enumerate(kinds)]
        counters = {"pump": 0, "valve": 0, "heater": 0}
        rows = []
        for name, kind in zip(names, kinds):
            rows.append({"name": name, "kind": kind, "index": counters[kind], "unit": "fraction", "bounds": (0.0, 1.0)})
            counters[kind] += 1
        return rows

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

    def state_vector(self, x):
        """Return the generic state vector x used by controllers and simulators."""
        return [float(v) for v in x]

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
        """Semantic outputs derived from x, preserving legacy physical names."""
        state = self.state_vector(x)
        levels, temps = self.levels_temps(state)
        out = {"x": state, "levels": levels, "temps": temps}
        if callable(getattr(self, "conc", None)):
            out["conc"] = self.conc(state)
        out["y"] = self.controlled_output(state)
        return out

    def measurement(self, x, env=None):
        """Measured state dict exposed to controllers.

        New controllers can use x/y generically; legacy controllers can still use
        levels, temps, conc, and disturbance names.
        """
        return {**self.outputs(x), **dict(env or {})}

    def controlled_output(self, x, backend="numeric", ca=None):
        try:
            levels, temps = self.levels_temps(x, backend=backend, ca=ca)
        except TypeError:
            if backend != "numeric":
                raise
            levels, temps = self.levels_temps(x)
        return [levels[i] for i in self.controlled_levels()] + list(temps)

    def setpoint_vector(self, h_sp=None, t_sp=None):
        if h_sp is None:
            h_sp = {}
        if t_sp is None:
            t_sp = [0.0] * int(self.n)
        t_sp = list(t_sp)
        return [h_sp[i] for i in self.controlled_levels()] + [t_sp[i] for i in range(int(self.n))]

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

    def economic_value(self, x, u, env=None, backend="numeric", ca=None):
        return 0.0

    def levels_temps(self, x, backend="numeric", ca=None):
        return [], [x[i] for i in range(len(self.initial_state()))]

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
            "state_vector": {"name": "x", "length": len(self.initial_state())},
            "action_vector": {"name": "u", "length": sum(self.actuator_counts())},
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

    def controlled_levels(self):
        return []

    def default_setpoints(self):
        return {}, [0.0] * int(self.n)

    def heater_power(self, act):
        return 0.0

    def pump_power(self, act):
        return 0.0

    def ideal_power(self, levels, temps, t_sp, env, act):
        return 0.0
