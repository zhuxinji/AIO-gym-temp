"""Fixed-setpoint PID baseline controller."""
from __future__ import annotations


def _clip01(v):
    return 0.0 if v < 0 else 1.0 if v > 1 else v


GAINS = {
    "cascade": {"level_pump": (8.0, 0.4, 0.0), "level_valve": (6.0, 0.3, 0.0), "temp": (0.06, 0.01, 0.0)},
    "quadruple": {"level_pump": (6.0, 0.25, 0.0), "level_valve": (0.0, 0.0, 0.0), "temp": (0.05, 0.012, 0.0)},
    "cstr": {"level_pump": (0.0, 0.0, 0.0), "level_valve": (0.0, 0.0, 0.0), "temp": (0.08, 0.02, 0.0)},
    "hvac": {"temp": (0.18, 0.03, 0.0)},
    "extraction": {"temp": (0.0, 0.25, 0.0)},
    "heater": {"level_pump": (0.0, 0.0, 0.0), "level_valve": (0.15, 0.05, 0.0), "temp": (0.035, 0.010, 0.0)},
    "crystallization": {"temp": (0.08, 0.02, 0.0)},
}
PAIRING = {
    "cascade": {"level": [("pump", 0, 0), ("valve", 0, 1), ("valve", 1, 2)],
                "temp": [(0, 0, False), (1, 1, False), (2, 2, False)], "demand_valve_index": 2, "holds": []},
    "quadruple": {"level": [("pump", 0, 0), ("pump", 1, 1)],
                  "temp": [(0, 0, False), (1, 1, False), (2, 2, False), (3, 3, False)], "demand_valve_index": None, "holds": []},
    "cstr": {"level": [], "temp": [(0, 0, True)], "demand_valve_index": None, "holds": [("pump", 0, 0.5)]},
    "hvac": {"level": [], "temp": [(0, 0, False), (1, 1, False)], "demand_valve_index": None, "holds": []},
    "extraction": {"level": [], "temp": [("pump", 0, 4, False)], "demand_valve_index": None, "holds": [("pump", 1, 0.0)]},
    "heater": {"level": [("valve", 0, 0)], "temp": [(0, 0, False)], "demand_valve_index": None, "holds": []},
    "crystallization": {"level": [], "temp": [(0, 0, True)], "demand_valve_index": None, "holds": []},
}


def _actuator_array(act, kind):
    return {"pump": act["pumps"], "valve": act["valves"], "heater": act["heaters"]}[kind]


def _parse_temp_pairing(row):
    if len(row) == 3:
        hi, ti, rev = row
        return "heater", hi, ti, rev
    if len(row) == 4:
        kind, ai, ti, rev = row
        return kind, ai, ti, rev
    raise ValueError(f"temperature pairing must have 3 or 4 fields, got {row!r}")


class PIDLoop:
    def __init__(self, g, reverse=False):
        self.kp, self.ki, self.kd = g
        self.reverse = reverse
        self.reset()

    def reset(self):
        self.i = 0.0
        self.prev = None

    def update(self, sp, meas, dt):
        e = (meas - sp) if self.reverse else (sp - meas)
        dmeas = 0.0 if (self.prev is None or dt <= 0) else (meas - self.prev) / dt
        self.prev = meas
        p = self.kp * e
        d = (1.0 if self.reverse else -1.0) * self.kd * dmeas
        i_cand = self.i + self.ki * e * dt
        raw = p + i_cand + d
        out = _clip01(raw)
        if not ((raw > 1 and e > 0) or (raw < 0 and e < 0)):
            self.i = i_cand
        return out


class PIDAgent:
    name = "PID"

    def __init__(self, model, gains=None, pairing=None):
        self.model = model
        self.nP, self.nV, self.nH = model.actuator_counts()
        self.gains = gains or GAINS[model.scenario]
        self.pairing = pairing or PAIRING[model.scenario]
        g = self.gains
        pr = self.pairing
        self.demand_idx = pr.get("demand_valve_index")
        self.demand_valve = 0.5
        self.holds = pr.get("holds", [])
        self.level_loops = [(kind, ai, li, PIDLoop(g["level_pump" if kind == "pump" else "level_valve"], False))
                            for (kind, ai, li) in pr.get("level", [])]
        self.temp_loops = [(kind, ai, ti, PIDLoop(g["temp"], rev)) for (kind, ai, ti, rev) in
                           (_parse_temp_pairing(row) for row in pr.get("temp", []))]

    def metadata(self):
        return {"name": self.name, "class": self.__class__.__name__,
                "kind": "fixed_setpoint_pid", "scenario": self.model.scenario,
                "action_mode": "actuator", "control_structure": "fixed_sp_pid",
                "gains": self.gains,
                "pairing": self.pairing}

    def reset(self):
        for *_, loop in self.level_loops:
            loop.reset()
        for *_, loop in self.temp_loops:
            loop.reset()

    def compute(self, meas, sp, dt):
        act = {"pumps": [0.0] * self.nP, "valves": [0.0] * self.nV, "heaters": [0.0] * self.nH}
        for kind, idx, value in self.holds:
            arr = _actuator_array(act, kind)
            if idx < len(arr):
                arr[idx] = value
        for kind, ai, li, loop in self.level_loops:
            out = loop.update(sp["h_sp"][li], meas["levels"][li], dt)
            _actuator_array(act, kind)[ai] = out
        if self.demand_idx is not None and self.nV:
            act["valves"][self.demand_idx] = self.demand_valve
        for kind, ai, ti, loop in self.temp_loops:
            _actuator_array(act, kind)[ai] = loop.update(sp["t_sp"][ti], meas["temps"][ti], dt)
        return act
