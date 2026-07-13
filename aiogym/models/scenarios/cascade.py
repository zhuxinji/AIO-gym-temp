import math

from ..core import RHO_CP, ProcessModelContract, _maxv


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
    output_names = ("tank_0_level", "tank_1_level", "tank_2_level", "tank_0_temperature", "tank_1_temperature", "tank_2_temperature")
    output_units = {"tank_0_level": "m", "tank_1_level": "m", "tank_2_level": "m", "tank_0_temperature": "degC", "tank_1_temperature": "degC", "tank_2_temperature": "degC"}
    output_bounds = {"tank_0_level": (0.0, 0.8), "tank_1_level": (0.0, 0.8), "tank_2_level": (0.0, 0.8), "tank_0_temperature": (25, 80), "tank_1_temperature": (30, 82), "tank_2_temperature": (35, 85)}
    default_y_sp = (0.45, 0.45, 0.45, 35.0, 50.0, 65.0)
    plant_regime = {"ua_loss": (0.4, 2.6), "heater_max": (0.6, 1.15), "pump_flow_max": (0.7, 1.3), "cv_out": (0.7, 1.4)}
    economic_config = {
        "temp_band": [(34, 44), (48, 58), (60, 72)],
        "level_band": [(0.32, 0.58)] * 3,
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.7,
        "w_viol": 29.0,
    }
    supervisory_layout = (("y_sp", 3, 25, 80), ("y_sp", 4, 30, 82), ("y_sp", 5, 35, 85))
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

    def sample_disturbance(self, event, current, rng):
        if event == "pump_capacity_shift":
            return float(max(0.6, min(1.3, float(current) + rng.uniform(-0.30, 0.30))))
        if event == "heater_efficiency_shift":
            return float(max(0.55, min(1.15, float(current) + rng.uniform(-0.35, 0.15))))
        if event == "heat_loss_shift":
            return float(max(0.5, min(2.4, float(current) + rng.uniform(-0.4, 1.2))))
        return super().sample_disturbance(event, current, rng)

    def process_info(self, x, levels, temps, env):
        return {
            "pump_flow_factor": env.get("pump_flow_factor", 1.0),
            "heater_efficiency": env.get("heater_efficiency", 1.0),
            "heat_loss_factor": env.get("heat_loss_factor", 1.0),
        }

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

    def display_outputs(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return {"levels": [x[0], x[2], x[4]], "temps": [x[1], x[3], x[5]]}
        return {"levels": [_maxv(x[0], 0.0), _maxv(x[2], 0.0), _maxv(x[4], 0.0)], "temps": [x[1], x[3], x[5]]}

    def initial_state(self):
        return [0.30, 20.0, 0.30, 20.0, 0.30, 20.0]

    def clamp_state(self, x):
        return x

    def controlled_output(self, x, backend="numeric", ca=None):
        display = self.display_outputs(x, backend=backend, ca=ca)
        levels, temps = display["levels"], display["temps"]
        return [levels[0], levels[1], levels[2], temps[0], temps[1], temps[2]]

    # ---- KPI support ----
    energy_scored = True

    def energy_kw(self, u, backend="numeric", ca=None):
        return sum(u[4 + i] * self.p["heater_max"] for i in range(3)) / 1000.0

    def action_energy_kw(self, act, x=None, env=None):
        u = self.action_vector(act)
        input_energy_kw = u[0] * self.p["pump_power_max"] / 1000.0
        thermal_energy_kw = sum(u[4 + i] * self.p["heater_max"] for i in range(3)) / 1000.0
        return input_energy_kw + thermal_energy_kw

    def ideal_energy_kw(self, x, y_sp, env, act):
        p = self.p
        u = self.action_vector(act)
        target = list(y_sp)
        output_targets = target[-int(self.n):]
        q = u[0] * p["pump_flow_max"] * self.pump_flow_factor(env)
        tot = 0.0
        for i in range(3):
            tin = env["t_cold"] if i == 0 else output_targets[i - 1]
            tot += _maxv(0.0, RHO_CP * q * (output_targets[i] - tin) + p["ua_loss"] * self.heat_loss_factor(env) * (output_targets[i] - env["t_amb"]))
        return tot / 1000.0
