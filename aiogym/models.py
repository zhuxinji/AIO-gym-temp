"""Native (numpy/pure-python) port of the AIO-Gym plant models.

A faithful re-implementation of frontend/js/sim/models.js: same ODE right-hand
sides, same constants, same state layout, so a policy trained here transfers to
the browser sim and vice-versa. Parity against the JS source of truth is checked
by tests/test_parity.py (golden trajectories generated from the browser engine).

State layout (matches the JS):
  cascade / quadruple : x = [h0, T0, h1, T1, ...]   (level, temp interleaved)
  cstr                : x = [Ca, T]
  hvac                : x = [T0, T1]
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


class CascadeModel(ProcessModelContract):
    scenario = "cascade"
    display_name = "Heated-tank cascade"
    summary = "Three interlinked heated tanks with level and temperature dynamics."
    n = 3
    dt_micro = 0.02
    state_names = ("h0", "T0", "h1", "T1", "h2", "T2")
    state_units = {"h0": "m", "h1": "m", "h2": "m", "T0": "degC", "T1": "degC", "T2": "degC"}
    state_bounds = {"h0": (0.0, 0.8), "h1": (0.0, 0.8), "h2": (0.0, 0.8), "T0": (0.0, 120.0), "T1": (0.0, 120.0), "T2": (0.0, 120.0)}
    action_names = ("feed_pump", "outlet_valve_0", "outlet_valve_1", "outlet_valve_2", "heater_0", "heater_1", "heater_2")
    plant_regime = {"ua_loss": (0.4, 2.6), "heater_max": (0.6, 1.15), "pump_flow_max": (0.7, 1.3), "cv_out": (0.7, 1.4)}
    economic_config = {
        "temp_band": [(34, 44), (48, 58), (60, 72)],
        "level_band": [(0.32, 0.58)] * 3,
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.6,
        "w_viol": 25.0,
    }
    supervisory_layout = (("t_sp", 0, 25, 80), ("t_sp", 1, 30, 82), ("t_sp", 2, 35, 85))
    param_units = {"area": "m2", "height_max": "m", "cv_out": "m2.5/s", "ua_loss": "W/K", "heater_max": "W", "pump_flow_max": "m3/s", "pump_power_max": "W", "t_cold": "degC", "t_amb": "degC", "h_floor": "m"}
    param_bounds = {"area": (0.01, 2.0), "height_max": (0.1, 5.0), "cv_out": (0.0, 0.02), "ua_loss": (0.0, 1000.0), "heater_max": (0.0, 500000.0), "pump_flow_max": (0.0, 0.02), "pump_power_max": (0.0, 10000.0), "t_cold": (0.0, 40.0), "t_amb": (0.0, 45.0), "h_floor": (1e-6, 0.1)}
    input_disturbances = ProcessModelContract.input_disturbances + (
        {"name": "pump_flow_factor", "event": "pump_capacity_shift", "unit": "fraction", "bounds": (0.4, 1.4), "default": 1.0, "description": "feed-pump flow capacity multiplier"},
        {"name": "heater_efficiency", "event": "heater_efficiency_shift", "unit": "fraction", "bounds": (0.4, 1.2), "default": 1.0, "description": "heater duty multiplier"},
        {"name": "heat_loss_factor", "event": "heat_loss_shift", "unit": "fraction", "bounds": (0.3, 3.0), "default": 1.0, "description": "ambient heat-loss multiplier"},
    )
    safety_constraints = ({"name": "level_bounds", "states": ("h0", "h1", "h2"), "bounds": (0.0, 0.8)}, {"name": "temperature_trip", "states": ("T0", "T1", "T2"), "bounds": (None, 92.0)})

    def __init__(self):
        self.p = dict(area=0.15, height_max=0.80, cv_out=0.0026, ua_loss=40.0,
                      heater_max=90000.0, pump_flow_max=0.0016, pump_power_max=1500.0,
                      t_cold=15.0, t_amb=20.0, h_floor=1e-3)

    def actuator_counts(self):
        return (1, 3, 3)

    @property
    def height_max(self):
        return [0.8, 0.8, 0.8]

    def pump_flow_factor(self, env=None):
        env = env or {}
        return env.get("pump_flow_factor", 1.0)

    def heater_efficiency(self, env=None):
        env = env or {}
        return env.get("heater_efficiency", 1.0)

    def heat_loss_factor(self, env=None):
        env = env or {}
        return env.get("heat_loss_factor", 1.0)

    def _flows(self, h, act, env):
        p = self.p
        qp = act["pumps"][0] * p["pump_flow_max"] * self.pump_flow_factor(env)
        qo = []
        for i in range(3):
            f = p["cv_out"] * act["valves"][i] * math.sqrt(_maxv(h[i], 0.0))
            if i == 2:
                f += env.get("extra_outflow", 0.0) or 0.0
            qo.append(f)
        return qp, qo

    def derivatives(self, x, act, env):
        return self.dynamics(x, act, env)

    def _dynamics(self, x, u, env, ops):
        p = self.p
        t_cold, t_amb = env["t_cold"], env["t_amb"]
        extra_outflow = env.get("extra_outflow", 0.0)
        pump_flow_factor = env.get("pump_flow_factor", 1.0)
        heater_efficiency = env.get("heater_efficiency", 1.0)
        heat_loss_factor = env.get("heat_loss_factor", 1.0)
        qp = u[0] * p["pump_flow_max"] * pump_flow_factor
        h = [x[0], x[2], x[4]]
        T = [x[1], x[3], x[5]]
        qo = [p["cv_out"] * u[1 + i] * ops.sqrt(ops.max(h[i], 1e-9)) for i in range(3)]
        qo[2] += extra_outflow
        dx = []
        for i in range(3):
            qin = qp if i == 0 else qo[i - 1]
            tin = t_cold if i == 0 else T[i - 1]
            vol = p["area"] * ops.max(h[i], p["h_floor"])
            pheat = u[4 + i] * p["heater_max"] * heater_efficiency
            qloss = p["ua_loss"] * heat_loss_factor * (T[i] - t_amb)
            dx += [(qin - qo[i]) / p["area"], qin * (tin - T[i]) / vol + (pheat - qloss) / (RHO_CP * vol)]
        return ops.vector(dx)

    def levels_temps(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return [x[0], x[2], x[4]], [x[1], x[3], x[5]]
        return [_maxv(x[0], 0.0), _maxv(x[2], 0.0), _maxv(x[4], 0.0)], [x[1], x[3], x[5]]

    def initial_state(self):
        return [0.30, 20.0, 0.30, 20.0, 0.30, 20.0]

    def clamp_state(self, x):
        return x

    def controlled_levels(self):
        return [0, 1, 2]

    def default_setpoints(self):
        return {0: 0.45, 1: 0.45, 2: 0.45}, [35.0, 50.0, 65.0]

    # ---- KPI support (mirrors models.js idealPower + heater/pump power) ----
    energy_scored = True

    def heater_power(self, act):
        return sum(u * self.p["heater_max"] for u in act["heaters"])

    def pump_power(self, act):
        return sum(u * self.p["pump_power_max"] for u in act["pumps"])

    def energy_kw(self, u, backend="numeric", ca=None):
        return sum(u[4 + i] * self.p["heater_max"] for i in range(3)) / 1000.0

    def ideal_power(self, levels, temps, t_sp, env, act):
        p = self.p
        q = act["pumps"][0] * p["pump_flow_max"] * self.pump_flow_factor(env)
        tot = 0.0
        for i in range(3):
            tin = env["t_cold"] if i == 0 else t_sp[i - 1]
            tot += _maxv(0.0, RHO_CP * q * (t_sp[i] - tin) + p["ua_loss"] * self.heat_loss_factor(env) * (t_sp[i] - env["t_amb"]))
        return tot


class QuadrupleModel(ProcessModelContract):
    scenario = "quadruple"
    display_name = "Johansson quadruple tank"
    summary = "Four-tank interacting level process with a thermal extension."
    n = 4
    dt_micro = 0.02
    state_names = ("h0", "T0", "h1", "T1", "h2", "T2", "h3", "T3")
    state_units = {"h0": "m", "h1": "m", "h2": "m", "h3": "m", "T0": "degC", "T1": "degC", "T2": "degC", "T3": "degC"}
    state_bounds = {"h0": (0.0, 0.8), "h1": (0.0, 0.8), "h2": (0.0, 0.8), "h3": (0.0, 0.8), "T0": (0.0, 120.0), "T1": (0.0, 120.0), "T2": (0.0, 120.0), "T3": (0.0, 120.0)}
    action_names = ("pump_0", "pump_1", "heater_0", "heater_1", "heater_2", "heater_3")
    plant_regime = {"ua_loss": (0.4, 2.6), "heater_max": (0.6, 1.15), "pump_flow_max": (0.7, 1.3), "a_out": (0.8, 1.25)}
    economic_config = {
        "temp_band": [(46, 58), (46, 58), (32, 46), (32, 46)],
        "level_band": [(0.32, 0.56)] * 2,
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.6,
        "w_viol": 25.0,
    }
    supervisory_layout = (("t_sp", 0, 25, 72), ("t_sp", 1, 25, 72), ("t_sp", 2, 20, 58), ("t_sp", 3, 20, 58))
    param_units = {"area": "m2", "height_max": "m", "a_out": "m2", "ua_loss": "W/K", "heater_max": "W", "pump_flow_max": "m3/s", "pump_power_max": "W", "t_cold": "degC", "t_amb": "degC", "h_floor": "m", "gamma1": "fraction", "gamma2": "fraction"}
    param_bounds = {"area": (0.01, 2.0), "height_max": (0.1, 5.0), "a_out": (0.0, 0.01), "ua_loss": (0.0, 1000.0), "heater_max": (0.0, 500000.0), "pump_flow_max": (0.0, 0.02), "pump_power_max": (0.0, 10000.0), "t_cold": (0.0, 40.0), "t_amb": (0.0, 45.0), "h_floor": (1e-6, 0.1), "gamma1": (0.05, 0.95), "gamma2": (0.05, 0.95)}
    input_disturbances = ProcessModelContract.input_disturbances + (
        {"name": "pump_flow_factor", "event": "pump_capacity_shift", "unit": "fraction", "bounds": (0.4, 1.4), "default": 1.0, "description": "pump flow-capacity multiplier"},
        {"name": "heater_efficiency", "event": "heater_efficiency_shift", "unit": "fraction", "bounds": (0.4, 1.2), "default": 1.0, "description": "heater duty multiplier"},
        {"name": "heat_loss_factor", "event": "heat_loss_shift", "unit": "fraction", "bounds": (0.3, 3.0), "default": 1.0, "description": "ambient heat-loss multiplier"},
    )
    safety_constraints = ({"name": "level_bounds", "states": ("h0", "h1", "h2", "h3"), "bounds": (0.0, 0.8)}, {"name": "temperature_trip", "states": ("T0", "T1", "T2", "T3"), "bounds": (None, 92.0)})

    def __init__(self, gamma1=0.70, gamma2=0.70):
        self.p = dict(area=0.06, height_max=0.80, a_out=[2.2e-4, 2.2e-4, 1.0e-4, 1.0e-4],
                      ua_loss=40.0, heater_max=[90000.0, 90000.0, 30000.0, 30000.0],
                      pump_flow_max=1.3e-3, pump_power_max=1200.0, t_cold=15.0, t_amb=20.0, h_floor=1e-3)
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def actuator_counts(self):
        return (2, 0, 4)

    @property
    def height_max(self):
        return [0.8, 0.8, 0.8, 0.8]

    def _out(self, h):
        return [a * math.sqrt(2 * G * _maxv(h[i], 0.0)) for i, a in enumerate(self.p["a_out"])]

    def _extra_parameters(self):
        return {"gamma1": self.gamma1, "gamma2": self.gamma2}

    def pump_flow_factor(self, env=None):
        env = env or {}
        return env.get("pump_flow_factor", 1.0)

    def heater_efficiency(self, env=None):
        env = env or {}
        return env.get("heater_efficiency", 1.0)

    def heat_loss_factor(self, env=None):
        env = env or {}
        return env.get("heat_loss_factor", 1.0)

    def _inflow(self, act, env, T, out):
        p = self.p
        flow_factor = self.pump_flow_factor(env)
        Q1 = act["pumps"][0] * p["pump_flow_max"] * flow_factor
        Q2 = act["pumps"][1] * p["pump_flow_max"] * flow_factor
        g1, g2, tc = self.gamma1, self.gamma2, env["t_cold"]
        return [
            [(g1 * Q1, tc), (out[2], T[2])],          # tank 1 <- pump1 + tank3
            [(g2 * Q2, tc), (out[3], T[3])],          # tank 2 <- pump2 + tank4
            [((1 - g2) * Q2, tc)],                     # tank 3 <- pump2
            [((1 - g1) * Q1, tc)],                     # tank 4 <- pump1
        ]

    def derivatives(self, x, act, env):
        return self.dynamics(x, act, env)

    def _dynamics(self, x, u, env, ops):
        p = self.p
        t_cold, t_amb = env["t_cold"], env["t_amb"]
        extra_outflow = env.get("extra_outflow", 0.0)
        pump_flow_factor = env.get("pump_flow_factor", 1.0)
        heater_efficiency = env.get("heater_efficiency", 1.0)
        heat_loss_factor = env.get("heat_loss_factor", 1.0)
        h = [x[0], x[2], x[4], x[6]]
        T = [x[1], x[3], x[5], x[7]]
        out = [p["a_out"][i] * ops.sqrt(2 * G * ops.max(h[i], 1e-9)) for i in range(4)]
        out[0] += extra_outflow
        Q1 = u[0] * p["pump_flow_max"] * pump_flow_factor
        Q2 = u[1] * p["pump_flow_max"] * pump_flow_factor
        g1, g2 = self.gamma1, self.gamma2
        inflow = [
            [(g1 * Q1, t_cold), (out[2], T[2])],
            [(g2 * Q2, t_cold), (out[3], T[3])],
            [((1 - g2) * Q2, t_cold)],
            [((1 - g1) * Q1, t_cold)],
        ]
        dx = []
        for i in range(4):
            qin = sum(q for q, _ in inflow[i])
            vol = p["area"] * ops.max(h[i], p["h_floor"])
            mix = sum(q * (tin - T[i]) for q, tin in inflow[i])
            pheat = u[2 + i] * p["heater_max"][i] * heater_efficiency
            qloss = p["ua_loss"] * heat_loss_factor * (T[i] - t_amb)
            dx += [(qin - out[i]) / p["area"], mix / vol + (pheat - qloss) / (RHO_CP * vol)]
        return ops.vector(dx)

    def levels_temps(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return [x[2 * i] for i in range(4)], [x[2 * i + 1] for i in range(4)]
        return [_maxv(x[2 * i], 0.0) for i in range(4)], [x[2 * i + 1] for i in range(4)]

    def initial_state(self):
        return [0.25, 20.0, 0.25, 20.0, 0.12, 20.0, 0.12, 20.0]

    def clamp_state(self, x):
        return x

    def controlled_levels(self):
        return [0, 1]

    def default_setpoints(self):
        return {0: 0.40, 1: 0.40}, [50.0, 50.0, 35.0, 35.0]

    # ---- KPI support ----
    energy_scored = True

    def heater_power(self, act):
        return sum(u * self.p["heater_max"][i] for i, u in enumerate(act["heaters"]))

    def pump_power(self, act):
        return sum(u * self.p["pump_power_max"] for u in act["pumps"])

    def energy_kw(self, u, backend="numeric", ca=None):
        return sum(u[2 + i] * self.p["heater_max"][i] for i in range(4)) / 1000.0

    def ideal_power(self, levels, temps, t_sp, env, act):
        p = self.p
        g1, g2, tc = self.gamma1, self.gamma2, env["t_cold"]
        flow_factor = self.pump_flow_factor(env)
        Q1 = act["pumps"][0] * p["pump_flow_max"] * flow_factor
        Q2 = act["pumps"][1] * p["pump_flow_max"] * flow_factor
        out = self._out(levels)
        inflow = [
            [(g1 * Q1, tc), (out[2], t_sp[2])],
            [(g2 * Q2, tc), (out[3], t_sp[3])],
            [((1 - g2) * Q2, tc)],
            [((1 - g1) * Q1, tc)],
        ]
        tot = 0.0
        for i in range(4):
            mix = sum(q * (t_sp[i] - tin) for q, tin in inflow[i])
            tot += _maxv(0.0, RHO_CP * mix + p["ua_loss"] * self.heat_loss_factor(env) * (t_sp[i] - env["t_amb"]))
        return tot


class CSTRModel(ProcessModelContract):
    scenario = "cstr"
    gym_id = "CSTR"
    display_name = "Exothermic CSTR"
    summary = "Two-state exothermic reactor with feed dilution and cooling-jacket action."
    n = 1
    dt_micro = 0.01
    state_names = ("Ca", "T")
    state_units = {"Ca": "mol/L", "T": "degC"}
    state_bounds = {"Ca": (0.0, 1.5), "T": (0.0, 200.0)}
    action_names = ("feed_pump", "cooling")
    plant_regime = {"Uc": (0.5, 1.6), "k0": (0.55, 1.7), "Hr": (0.85, 1.2)}
    economic_config = {
        "temp_band": [(None, 88.0)],
        "level_band": [],
        "value": "production",
        "w_value": 900.0,
        "w_energy": 0.4,
        "w_viol": 8.0,
    }
    supervisory_layout = (("t_sp", 0, 45, 90), ("mv", "pumps", 0, 0.3, 1.0))
    param_units = {"Dmax": "1/s", "Caf": "mol/L", "k0": "1/s", "EaR": "K", "Hr": "degC/(mol/L)", "Uc": "1/s", "Tcool": "degC", "cool_max": "W", "feed_power_max": "W", "t_cold": "degC", "t_amb": "degC", "h_floor": "m"}
    param_bounds = {"Dmax": (0.0, 1.0), "Caf": (0.0, 5.0), "k0": (0.0, 1e12), "EaR": (0.0, 50000.0), "Hr": (0.0, 1000.0), "Uc": (0.0, 10.0), "Tcool": (-20.0, 50.0), "cool_max": (0.0, 500000.0), "feed_power_max": (0.0, 10000.0), "t_cold": (0.0, 60.0), "t_amb": (0.0, 45.0), "h_floor": (1e-6, 0.1)}
    input_disturbances = (
        {"name": "t_cold", "event": "cold_inlet_step", "unit": "degC", "bounds": (0.0, 60.0), "description": "feed temperature"},
        {"name": "Caf", "event": "feed_concentration_step", "unit": "mol/L", "bounds": (0.0, 5.0), "description": "feed concentration"},
        {"name": "Tcool", "event": "coolant_temperature_step", "unit": "degC", "bounds": (-20.0, 50.0), "description": "coolant temperature"},
        {"name": "t_amb", "event": "ambient_step", "unit": "degC", "bounds": (0.0, 45.0), "description": "ambient temperature"},
    )
    safety_constraints = ({"name": "concentration_nonnegative", "states": ("Ca",), "bounds": (0.0, None)}, {"name": "temperature_trip", "states": ("T",), "bounds": (None, 92.0)}, {"name": "temperature_hard_cap", "states": ("T",), "bounds": (None, 200.0)})

    def __init__(self):
        self.p = dict(Dmax=0.02, Caf=1.0, k0=1e8, EaR=7000.0, Hr=120.0, Uc=0.05, Tcool=10.0,
                      cool_max=80000.0, feed_power_max=1200.0, t_cold=20.0, t_amb=20.0, h_floor=1e-3)

    def actuator_counts(self):
        return (1, 0, 1)

    @property
    def height_max(self):
        return [1.0]

    def _rate(self, Ca, T):
        p = self.p
        return p["k0"] * math.exp(-p["EaR"] / (T + 273.15)) * _maxv(Ca, 0.0)

    def feed_concentration(self, env=None):
        env = env or {}
        return env.get("Caf", env.get("caf", self.p["Caf"]))

    def coolant_temperature(self, env=None):
        env = env or {}
        return env.get("Tcool", env.get("t_cool", self.p["Tcool"]))

    def derivatives(self, x, act, env):
        return self.dynamics(x, act, env)

    def _dynamics(self, x, u, env, ops):
        p = self.p
        t_cold = env["t_cold"]
        Caf = env.get("Caf", env.get("caf", p["Caf"]))
        Tcool = env.get("Tcool", env.get("t_cool", p["Tcool"]))
        Ca, T = x[0], x[1]
        D = u[0] * p["Dmax"]
        uc = u[1]
        r = p["k0"] * ops.exp(-p["EaR"] / (T + 273.15)) * ops.max(Ca, 0.0)
        return ops.vector([
            D * (Caf - Ca) - r,
            D * (t_cold - T) + p["Hr"] * r - p["Uc"] * uc * (T - Tcool),
        ])

    def levels_temps(self, x, backend="numeric", ca=None):
        return [], [x[1]]

    def conc(self, x):
        return [_maxv(x[0], 0.0)]

    def initial_state(self):
        return [0.5, 50.0]

    def clamp_state(self, x):
        if x[0] < 0:
            x[0] = 0.0
        if x[0] > 1.5:
            x[0] = 1.5
        if x[1] < 0:
            x[1] = 0.0
        if x[1] > 200:
            x[1] = 200.0
        return x

    def controlled_levels(self):
        return []

    def default_setpoints(self):
        return {}, [60.0]

    # ---- KPI support (CSTR scores tracking + safety; no excess-energy term) ----
    energy_scored = False

    def heater_power(self, act):
        return act["heaters"][0] * self.p["cool_max"]

    def pump_power(self, act):
        return act["pumps"][0] * self.p["feed_power_max"]

    def energy_kw(self, u, backend="numeric", ca=None):
        return u[1] * self.p["cool_max"] / 1000.0

    def economic_value(self, x, u, env=None, backend="numeric", ca=None):
        env = env or {}
        return u[0] * self.p["Dmax"] * (env["Caf"] - x[0])

    def ideal_power(self, levels, temps, t_sp, env, act):
        return 0.0

    def production(self, x, act, env=None):
        """Reactant consumption-rate proxy used as the CSTR economic value."""
        Ca = _maxv(x[0], 0.0)
        D = act["pumps"][0] * self.p["Dmax"]
        return D * max(0.0, self.feed_concentration(env) - Ca)

    def conversion(self, x, env=None):
        Caf = max(self.feed_concentration(env), 1e-9)
        return max(0.0, min(1.0, (Caf - _maxv(x[0], 0.0)) / Caf))


class HVACModel(ProcessModelContract):
    scenario = "hvac"
    gym_id = "HVAC"
    display_name = "Two-zone HVAC"
    summary = "Two coupled thermal zones with bidirectional heat-pump actuation."
    n = 2
    dt_micro = 0.02
    state_names = ("T0", "T1")
    state_units = {"T0": "degC", "T1": "degC"}
    state_bounds = {"T0": (-20.0, 60.0), "T1": (-20.0, 60.0)}
    action_names = ("hvac_zone_0", "hvac_zone_1")
    plant_regime = {"Kc": (0.5, 1.7), "Ko": (0.5, 1.9), "C": (0.7, 1.4), "Pmax": (0.7, 1.2)}
    economic_config = {
        "temp_band": [(20.0, 24.0), (20.0, 24.0)],
        "level_band": [],
        "value": "none",
        "w_value": 0.0,
        "w_energy": 1.2,
        "w_viol": 14.0,
    }
    supervisory_layout = (("t_sp", 0, 18, 26), ("t_sp", 1, 18, 26))
    param_units = {"C": "J/K", "Pmax": "W", "Kc": "W/K", "Ko": "W/K", "t_cold": "degC", "t_amb": "degC", "h_floor": "m"}
    param_bounds = {"C": (100.0, 100000.0), "Pmax": (0.0, 20000.0), "Kc": (0.0, 1000.0), "Ko": (0.0, 1000.0), "t_cold": (-30.0, 50.0), "t_amb": (-30.0, 50.0), "h_floor": (1e-6, 0.1)}
    input_disturbances = (
        {"name": "t_amb", "event": "ambient_step", "unit": "degC", "bounds": (-30.0, 50.0), "description": "outdoor temperature"},
        {"name": "heat_load", "event": "internal_heat_load_step", "unit": "W", "bounds": (-1000.0, 2000.0), "default": [0.0, 0.0], "description": "internal heat load per zone"},
        {"name": "hvac_efficiency", "event": "hvac_efficiency_shift", "unit": "fraction", "bounds": (0.4, 1.3), "default": 1.0, "description": "HVAC delivered power multiplier"},
    )
    safety_constraints = ({"name": "comfort_band", "states": ("T0", "T1"), "bounds": (20.0, 24.0)}, {"name": "temperature_operating_bounds", "states": ("T0", "T1"), "bounds": (-20.0, 60.0)})

    def __init__(self):
        self.p = dict(C=6000.0, Pmax=1800.0, Kc=35.0, Ko=45.0, t_cold=5.0, t_amb=5.0, h_floor=1e-3)

    def actuator_counts(self):
        return (0, 0, 2)

    @property
    def height_max(self):
        return [1.0, 1.0]

    def _power(self, u):
        return (u - 0.5) * 2 * self.p["Pmax"]

    def hvac_efficiency(self, env=None):
        env = env or {}
        return env.get("hvac_efficiency", 1.0)

    def heat_loads(self, env=None):
        env = env or {}
        load = env.get("heat_load", [0.0, 0.0])
        if isinstance(load, (int, float)):
            return [float(load), float(load)]
        return [float(load[0]), float(load[1])]

    def derivatives(self, x, act, env):
        return self.dynamics(x, act, env)

    def _dynamics(self, x, u, env, ops):
        p = self.p
        Tout = env["t_amb"]
        heat_load = env.get("heat_load", [0.0, 0.0])
        if isinstance(heat_load, (int, float)):
            heat_load = [heat_load, heat_load]
        hvac_efficiency = env.get("hvac_efficiency", 1.0)
        P1 = (u[0] - 0.5) * 2 * p["Pmax"] * hvac_efficiency
        P2 = (u[1] - 0.5) * 2 * p["Pmax"] * hvac_efficiency
        return ops.vector([
            (P1 + heat_load[0] + p["Kc"] * (x[1] - x[0]) + p["Ko"] * (Tout - x[0])) / p["C"],
            (P2 + heat_load[1] + p["Kc"] * (x[0] - x[1]) + p["Ko"] * (Tout - x[1])) / p["C"],
        ])

    def levels_temps(self, x, backend="numeric", ca=None):
        return [], [x[0], x[1]]

    def initial_state(self):
        return [10.0, 10.0]

    def clamp_state(self, x):
        return x

    def controlled_levels(self):
        return []

    def default_setpoints(self):
        return {}, [22.0, 22.0]

    # ---- KPI support (HVAC scores tracking + safety; no excess-energy term) ----
    energy_scored = False

    def heater_power(self, act):
        return sum(abs(self._power(u)) for u in act["heaters"])

    def pump_power(self, act):
        return 0.0

    def energy_kw(self, u, backend="numeric", ca=None):
        ops = _casadi_ops(ca) if backend == "casadi" else _NUMERIC_OPS
        return sum(ops.abs((u[i] - 0.5) * 2 * self.p["Pmax"]) for i in range(2)) / 1000.0

    def ideal_power(self, levels, temps, t_sp, env, act):
        return 0.0


class ExtractionModel(ProcessModelContract):
    scenario = "extraction"
    display_name = "Multistage Extraction Column"
    summary = "Five-stage counter-current liquid-gas extraction column from the PC-Gym benchmark."
    n = 5
    dt_micro = 0.01
    state_names = ("CX1", "CY1", "CX2", "CY2", "CX3", "CY3", "CX4", "CY4", "CX5", "CY5")
    state_units = {name: "concentration" for name in state_names}
    state_bounds = {name: (0.0, 1.0) for name in state_names}
    action_names = ("liquid_flow_L", "gas_flow_G")
    param_units = {
        "Vl": "volume", "Vg": "volume", "m": "dimensionless", "Kla": "1/s",
        "e": "dimensionless", "CX0": "concentration", "CY6": "concentration",
        "Lmin": "flow", "Lmax": "flow", "Gmin": "flow", "Gmax": "flow",
        "pump_power_max": "W", "c_max": "concentration", "t_cold": "degC", "t_amb": "degC",
    }
    param_bounds = {
        "Vl": (1.0, 20.0), "Vg": (1.0, 20.0), "m": (0.01, 20.0), "Kla": (0.1, 20.0),
        "e": (0.1, 5.0), "CX0": (0.0, 1.0), "CY6": (0.0, 1.0),
        "Lmin": (0.0, 100.0), "Lmax": (0.0, 100.0), "Gmin": (0.0, 2000.0), "Gmax": (0.0, 2000.0),
        "pump_power_max": (0.0, 10000.0), "c_max": (0.0, 10.0), "t_cold": (0.0, 40.0), "t_amb": (0.0, 45.0),
    }
    plant_regime = {"Kla": (0.6, 1.5), "CX0": (0.75, 1.25), "CY6": (0.75, 1.25)}
    economic_config = {
        "temp_band": [(None, None), (None, None), (None, None), (None, None), (0.0035, 0.0048)],
        "level_band": [],
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.0,
        "w_viol": 25.0,
    }
    input_disturbances = (
        {"name": "CX0", "event": "feed_liquid_concentration_step", "unit": "concentration", "bounds": (0.3, 0.9), "description": "liquid feed concentration"},
        {"name": "CY6", "event": "gas_feed_concentration_step", "unit": "concentration", "bounds": (0.0, 0.12), "description": "gas/extract feed concentration"},
        {"name": "Kla", "event": "mass_transfer_shift", "unit": "1/s", "bounds": (2.0, 8.0), "description": "mass-transfer coefficient"},
    )
    safety_constraints = (
        {"name": "concentration_bounds", "states": state_names, "bounds": (0.0, 1.0)},
        {"name": "cx5_quality_band", "states": ("CX5",), "bounds": (0.0035, 0.0048)},
    )

    def __init__(self):
        self.p = dict(
            Vl=5.0, Vg=5.0, m=1.0, Kla=5.0, e=2.0,
            CX0=0.60, CY6=0.050, Lmin=5.0, Lmax=10.0, Gmin=500.0, Gmax=1000.0,
            pump_power_max=[1000.0, 1000.0], c_max=1.0, t_cold=0.0, t_amb=0.0,
        )

    def actuator_counts(self):
        return (2, 0, 0)

    def _flows_from_vector(self, u):
        p = self.p
        return (
            p["Lmin"] + u[0] * (p["Lmax"] - p["Lmin"]),
            p["Gmin"] + u[1] * (p["Gmax"] - p["Gmin"]),
        )

    def _action_from_flows(self, L, G):
        p = self.p
        return [
            max(0.0, min(1.0, (L - p["Lmin"]) / max(1e-9, p["Lmax"] - p["Lmin"]))),
            max(0.0, min(1.0, (G - p["Gmin"]) / max(1e-9, p["Gmax"] - p["Gmin"]))),
        ]

    def derivatives(self, x, act, env):
        return self.dynamics(x, act, env)

    def _dynamics(self, x, u, env, ops):
        p = self.p
        CX0 = env.get("CX0", p["CX0"])
        CY6 = env.get("CY6", p["CY6"])
        Kla = env.get("Kla", p["Kla"])
        L, Gf = self._flows_from_vector(u)
        dx = []
        for i in range(5):
            CX = x[2 * i]
            CY = x[2 * i + 1]
            CXprev = CX0 if i == 0 else x[2 * (i - 1)]
            CYnext = CY6 if i == 4 else x[2 * (i + 1) + 1]
            CXeq = (CY ** p["e"]) / p["m"]
            Fn = Kla * (CX - CXeq) * p["Vl"]
            dx.extend([
                (L * (CXprev - CX) - Fn) / p["Vl"],
                (Gf * (CYnext - CY) + Fn) / p["Vg"],
            ])
        return ops.vector(dx)

    def levels_temps(self, x, backend="numeric", ca=None):
        cx = [x[2 * i] for i in range(5)]
        if backend == "casadi":
            return [], cx
        return [], [max(0.0, min(1.0, v)) for v in cx]

    def outputs(self, x):
        state = self.state_vector(x)
        cx = [max(0.0, min(1.0, state[2 * i])) for i in range(5)]
        cy = [max(0.0, min(1.0, state[2 * i + 1])) for i in range(5)]
        return {
            "x": state,
            "y": list(cx),
            "levels": [],
            "temps": list(cx),
            "conc": list(cx),
            "conc_liq": cx,
            "x_liq": cx,
            "y_ext": cy,
        }

    def initial_state(self):
        return [0.431, 0.643, 0.243, 0.474, 0.097, 0.286, 0.027, 0.140, 0.007, 0.070]

    def clamp_state(self, x):
        return [max(0.0, min(1.0, v)) for v in x]

    def default_setpoints(self):
        return {}, [0.0, 0.0, 0.0, 0.0, 0.0040]

    energy_scored = False

    def pump_power(self, act):
        return sum(u * self.p["pump_power_max"][i] for i, u in enumerate(act["pumps"]))

    def energy_kw(self, u, backend="numeric", ca=None):
        return sum(u[i] * self.p["pump_power_max"][i] for i in range(2)) / 1000.0


MODELS = {"cascade": CascadeModel, "quadruple": QuadrupleModel, "cstr": CSTRModel, "hvac": HVACModel, "extraction": ExtractionModel}
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
