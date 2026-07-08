import math

from ..core import G, RHO_CP, ProcessModelContract, _casadi_ops, _maxv


class FiredHeaterModel(ProcessModelContract):
    scenario = "heater"
    display_name = "Fired Heater"
    summary = "Refinery fired heater with outlet-temperature and flue-O2 dynamics."
    n = 1
    dt_micro = 0.05
    state_names = ("T_firebox", "T_out", "O2")
    state_units = {"T_firebox": "degC", "T_out": "degC", "O2": "%"}
    state_bounds = {"T_firebox": (20.0, 1400.0), "T_out": (20.0, 650.0), "O2": (0.0, 20.9)}
    action_names = ("air_damper", "fuel_valve")
    param_units = {
        "Fmax": "kg/s", "lhv": "J/kg", "stoich": "kg_air/kg_fuel", "Amax": "kg/s",
        "cp_g": "J/kg/K", "UA": "W/K", "Cfb": "J/K", "Cc": "J/K",
        "Fp0": "kg/s", "cp_p": "J/kg/K", "tau_o2": "s",
        "t_cold": "degC", "t_amb": "degC", "h_floor": "fraction",
    }
    param_bounds = {
        "Fmax": (0.0, 10.0), "lhv": (1e6, 80e6), "stoich": (1.0, 40.0), "Amax": (0.0, 200.0),
        "cp_g": (100.0, 5000.0), "UA": (0.0, 500000.0), "Cfb": (1e5, 1e8), "Cc": (1e5, 1e8),
        "Fp0": (1.0, 300.0), "cp_p": (100.0, 5000.0), "tau_o2": (1.0, 200.0),
        "t_cold": (0.0, 500.0), "t_amb": (-30.0, 60.0), "h_floor": (1e-6, 0.1),
    }
    plant_regime = {"lhv": (0.82, 1.15), "Fp0": (0.8, 1.25), "UA": (0.75, 1.2)}
    economic_config = {
        "temp_band": [(362.0, 378.0)],
        "level_band": [(1.6, 5.5)],
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.35,
        "w_viol": 2100.0,
        "level_scale": 1.0,
    }
    supervisory_layout = (("t_sp", 0, 364.0, 372.0), ("h_sp", 0, 1.8, 5.0))
    input_disturbances = (
        {"name": "t_cold", "event": "cold_inlet_step", "unit": "degC", "bounds": (240.0, 330.0), "description": "process feed inlet temperature"},
        {"name": "t_amb", "event": "ambient_step", "unit": "degC", "bounds": (-10.0, 45.0), "description": "ambient temperature"},
        {"name": "extra_outflow", "event": "demand_surge", "unit": "throughput fraction", "bounds": (0.0, 0.002), "default": 0.0, "description": "feed-throughput demand surge"},
        {"name": "lhv_factor", "event": "fuel_lhv_shift", "unit": "fraction", "bounds": (0.5, 1.3), "default": 1.0, "description": "fuel heating-value multiplier"},
    )
    safety_constraints = (
        {"name": "outlet_temperature_band", "states": ("T_out",), "bounds": (362.0, 378.0)},
        {"name": "tube_skin_temperature_trip", "states": ("T_out",), "bounds": (None, 415.0)},
        {"name": "flue_o2_band", "states": ("O2",), "bounds": (1.6, 5.5)},
        {"name": "flue_o2_trip", "states": ("O2",), "bounds": (1.2, None)},
    )

    def __init__(self):
        self.p = dict(
            Fmax=1.0, lhv=46e6, stoich=17.2, Amax=40.0, cp_g=1400.0,
            UA=42e3, Cfb=3.5e6, Cc=7e6, Fp0=88.0, cp_p=2300.0, tau_o2=20.0,
            t_cold=280.0, t_amb=20.0, h_floor=1e-3,
        )

    def actuator_counts(self):
        return (0, 1, 1)

    @property
    def height_max(self):
        return [12.0]

    def runtime_env(self, disturbance_values):
        env = super().runtime_env(disturbance_values)
        env.setdefault("lhv_factor", 1.0)
        return env

    def sample_disturbance(self, event, current, rng):
        if event == "cold_inlet_step":
            return float(max(240.0, min(330.0, float(current) + rng.uniform(-20.0, 25.0))))
        if event == "ambient_step":
            return float(max(-10.0, min(45.0, float(current) + rng.uniform(-8.0, 12.0))))
        if event == "fuel_lhv_shift":
            return float(max(0.5, min(1.3, float(current) + rng.uniform(-0.18, 0.12))))
        return super().sample_disturbance(event, current, rng)

    def _combustion(self, x, u, env, ops):
        p = self.p
        fuel = u[1] * p["Fmax"]
        air = u[0] * p["Amax"]
        stoich_air = p["stoich"] * fuel
        completeness = 1.0 - ops.max(0.0, 1.0 - air / ops.max(stoich_air, 1e-6))
        duty = fuel * completeness * p["lhv"] * env.get("lhv_factor", 1.0)
        o2_eq = 20.9 * ops.max(0.0, air - stoich_air * completeness) / ops.max(air + fuel * completeness, 1e-6)
        flue_mass = air + fuel * completeness
        return fuel, air, duty, o2_eq, flue_mass

    def derivatives(self, x, act, env):
        return self.dynamics(x, act, env)

    def _dynamics(self, x, u, env, ops):
        p = self.p
        t_firebox, t_out, o2 = x[0], x[1], x[2]
        _, _, duty, o2_eq, flue_mass = self._combustion(x, u, env, ops)
        feed_rate = p["Fp0"] + env.get("extra_outflow", 0.0) * 30000.0
        feed_in = env["t_cold"]
        mean_process_temp = (feed_in + t_out) / 2.0
        return ops.vector([
            (duty - p["UA"] * (t_firebox - mean_process_temp) - flue_mass * p["cp_g"] * (t_firebox - env["t_amb"])) / p["Cfb"],
            (feed_rate * p["cp_p"] * (feed_in - t_out) + p["UA"] * (t_firebox - mean_process_temp)) / p["Cc"],
            (o2_eq - o2) / p["tau_o2"],
        ])

    def levels_temps(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return [x[2]], [x[1]]
        return [max(0.0, min(20.9, x[2]))], [x[1]]

    def initial_state(self):
        return [700.0, 350.0, 3.5]

    def clamp_state(self, x):
        return [
            max(20.0, min(1400.0, x[0])),
            max(20.0, min(650.0, x[1])),
            max(0.0, min(20.9, x[2])),
        ]

    def controlled_levels(self):
        return [0]

    def default_setpoints(self):
        return {0: 3.0}, [370.0]

    energy_scored = False
    kpi_level_scale = 25.0

    def cv_scales(self):
        return {"level": 1.2, "temp": 12.0}

    def mpc_init(self):
        return [0.30, 0.55]

    def heater_power(self, act):
        return act["heaters"][0] * self.p["Fmax"] * self.p["lhv"]

    def pump_power(self, act):
        return 0.0

    def energy_kw(self, u, backend="numeric", ca=None):
        return u[1] * self.p["Fmax"] * self.p["lhv"] / 1000.0

    def ideal_power(self, levels, temps, t_sp, env, act):
        return 0.0

    def process_constraint_info(self, x, levels, temps, env):
        o2 = float(levels[0]) if levels else 0.0
        t_out = float(temps[0]) if temps else 0.0
        return {
            "heater_o2_low": max(0.0, 1.2 - o2),
            "heater_o2_high": max(0.0, o2 - 12.0),
            "heater_outlet_high": max(0.0, t_out - 415.0),
        }

    def process_info(self, x, levels, temps, env):
        p = self.p
        return {
            "heater_firebox_temp": float(x[0]),
            "heater_feed_rate": p["Fp0"] + env.get("extra_outflow", 0.0) * 30000.0,
            "heater_lhv_factor": env.get("lhv_factor", 1.0),
        }

    def constraint_penalty_scales(self):
        return {
            "heater_o2_low": 1.0,
            "heater_o2_high": 1.0,
            "heater_o2_band_low": 1.0,
            "heater_o2_band_high": 1.0,
            "heater_temp_band_low": 10.0,
            "heater_temp_band_high": 10.0,
            "heater_outlet_high": 10.0,
        }

    def common_constraint_info(self, levels, temps):
        o2 = float(levels[0]) if levels else 0.0
        t_out = float(temps[0]) if temps else 0.0
        return {
            "heater_o2_band_low": max(0.0, 1.6 - o2),
            "heater_o2_band_high": max(0.0, o2 - 5.5),
            "heater_temp_band_low": max(0.0, 362.0 - t_out),
            "heater_temp_band_high": max(0.0, t_out - 378.0),
        }

    def runaway_state(self, levels, temps):
        o2 = float(levels[0]) if levels else 0.0
        t_out = float(temps[0]) if temps else 0.0
        return bool(o2 < 1.2 or t_out > 415.0)
