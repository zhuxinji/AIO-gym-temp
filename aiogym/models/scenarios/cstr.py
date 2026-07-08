import math

from ..core import G, RHO_CP, ProcessModelContract, _casadi_ops, _maxv


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
        "w_value": 1575.0,
        "w_energy": 0.7,
        "w_viol": 14.0,
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

    def sample_disturbance(self, event, current, rng):
        if event == "feed_concentration_step":
            return float(max(0.2, min(2.0, float(current) + rng.uniform(-0.25, 0.35))))
        if event == "coolant_temperature_step":
            return float(max(-5.0, min(35.0, float(current) + rng.uniform(-8, 10))))
        return super().sample_disturbance(event, current, rng)

    def process_constraint_info(self, x, levels, temps, env):
        ca = float(x[0])
        return {
            "cstr_ca_high": max(0.0, ca - 1.5),
            "cstr_ca_low": max(0.0, -ca),
            "cstr_temp_low": max(0.0, -temps[0]),
        }

    def process_info(self, x, levels, temps, env):
        return {
            "cstr_feed_conc": self.feed_concentration(env),
            "cstr_coolant_temp": self.coolant_temperature(env),
            "cstr_conversion": self.conversion(x, env),
        }

    def constraint_penalty_scales(self):
        return {"cstr_temp_low": 10.0, "cstr_ca_high": 1.0, "cstr_ca_low": 1.0}

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
