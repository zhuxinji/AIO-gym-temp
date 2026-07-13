import math

from ..core import G, RHO_CP, ProcessModelContract, _maxv


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
    output_names = ("lower_tank_0_level", "lower_tank_1_level", "tank_0_temperature", "tank_1_temperature", "tank_2_temperature", "tank_3_temperature")
    output_units = {"lower_tank_0_level": "m", "lower_tank_1_level": "m", "tank_0_temperature": "degC", "tank_1_temperature": "degC", "tank_2_temperature": "degC", "tank_3_temperature": "degC"}
    output_bounds = {"lower_tank_0_level": (0.0, 0.8), "lower_tank_1_level": (0.0, 0.8), "tank_0_temperature": (25, 72), "tank_1_temperature": (25, 72), "tank_2_temperature": (20, 58), "tank_3_temperature": (20, 58)}
    default_y_sp = (0.40, 0.40, 50.0, 50.0, 35.0, 35.0)
    plant_regime = {"ua_loss": (0.4, 2.6), "heater_max": (0.6, 1.15), "pump_flow_max": (0.7, 1.3), "a_out": (0.8, 1.25)}
    economic_config = {
        "temp_band": [(46, 58), (46, 58), (32, 46), (32, 46)],
        "level_band": [(0.32, 0.56)] * 2,
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.7,
        "w_viol": 29.0,
    }
    supervisory_layout = (("y_sp", 2, 25, 72), ("y_sp", 3, 25, 72), ("y_sp", 4, 20, 58), ("y_sp", 5, 20, 58))
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

    def display_outputs(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return {"levels": [x[2 * i] for i in range(4)], "temps": [x[2 * i + 1] for i in range(4)]}
        return {"levels": [_maxv(x[2 * i], 0.0) for i in range(4)], "temps": [x[2 * i + 1] for i in range(4)]}

    def initial_state(self):
        return [0.25, 20.0, 0.25, 20.0, 0.12, 20.0, 0.12, 20.0]

    def clamp_state(self, x):
        return x

    def controlled_output(self, x, backend="numeric", ca=None):
        display = self.display_outputs(x, backend=backend, ca=ca)
        levels, temps = display["levels"], display["temps"]
        return [levels[0], levels[1], temps[0], temps[1], temps[2], temps[3]]

    # ---- KPI support ----
    energy_scored = True

    def energy_kw(self, u, backend="numeric", ca=None):
        return sum(u[2 + i] * self.p["heater_max"][i] for i in range(4)) / 1000.0

    def action_energy_kw(self, act, x=None, env=None):
        u = self.action_vector(act)
        input_energy_kw = sum(u[i] * self.p["pump_power_max"] for i in range(2)) / 1000.0
        thermal_energy_kw = sum(u[2 + i] * self.p["heater_max"][i] for i in range(4)) / 1000.0
        return input_energy_kw + thermal_energy_kw

    def ideal_energy_kw(self, x, y_sp, env, act):
        p = self.p
        g1, g2, tc = self.gamma1, self.gamma2, env["t_cold"]
        flow_factor = self.pump_flow_factor(env)
        u = self.action_vector(act)
        target = list(y_sp)
        output_targets = target[-int(self.n):]
        Q1 = u[0] * p["pump_flow_max"] * flow_factor
        Q2 = u[1] * p["pump_flow_max"] * flow_factor
        levels = self.display_outputs(self.state_vector(x))["levels"]
        out = self._out(levels)
        inflow = [
            [(g1 * Q1, tc), (out[2], output_targets[2])],
            [(g2 * Q2, tc), (out[3], output_targets[3])],
            [((1 - g2) * Q2, tc)],
            [((1 - g1) * Q1, tc)],
        ]
        tot = 0.0
        for i in range(4):
            mix = sum(q * (output_targets[i] - tin) for q, tin in inflow[i])
            tot += _maxv(0.0, RHO_CP * mix + p["ua_loss"] * self.heat_loss_factor(env) * (output_targets[i] - env["t_amb"]))
        return tot / 1000.0
