import math

from ..core import G, RHO_CP, ProcessModelContract, _casadi_ops, _maxv


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
        "w_energy": 0.7,
        "w_viol": 8.2,
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

    def sample_disturbance(self, event, current, rng):
        if event == "internal_heat_load_step":
            return [float(rng.uniform(-200, 900)), float(rng.uniform(-200, 900))]
        if event == "hvac_efficiency_shift":
            return float(max(0.6, min(1.2, float(current) + rng.uniform(-0.35, 0.20))))
        return super().sample_disturbance(event, current, rng)

    def process_constraint_info(self, x, levels, temps, env):
        lo_viol = max((20.0 - T for T in temps), default=0.0)
        hi_viol = max((T - 24.0 for T in temps), default=0.0)
        return {"hvac_comfort_low": max(0.0, lo_viol), "hvac_comfort_high": max(0.0, hi_viol)}

    def process_info(self, x, levels, temps, env):
        return {
            "hvac_heat_load": self.heat_loads(env),
            "hvac_efficiency": self.hvac_efficiency(env),
        }

    def constraint_penalty_scales(self):
        return {"hvac_comfort_low": 10.0, "hvac_comfort_high": 10.0}

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
        if backend == "casadi":
            return sum(ca.sqrt(((u[i] - 0.5) * 2 * self.p["Pmax"]) ** 2 + 1e-8) for i in range(2)) / 1000.0
        return sum(abs((u[i] - 0.5) * 2 * self.p["Pmax"]) for i in range(2)) / 1000.0

    def ideal_power(self, levels, temps, t_sp, env, act):
        return 0.0
