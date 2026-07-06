"""Lightweight nonlinear grid-search MPC controllers."""
from __future__ import annotations

import numpy as np

from aiogym.models import make_model


class FastGridCSTRMPC:
    """Nonlinear grid-search tracking MPC for the CSTR.

    This controller keeps feed pump fixed and optimizes the cooling actuator over
    a short receding horizon by simulating the nonlinear CSTR dynamics over a
    finite grid of candidate cooling values.
    """

    name = "FastGrid-CSTR-MPC"

    def __init__(self, horizon=4, grid=51, r_move=0.0, pump=0.5, nsub=2):
        self.p = make_model("cstr").p.copy()
        self.horizon = int(horizon)
        self.grid = np.linspace(0.0, 1.0, int(grid))
        self.r_move = float(r_move)
        self.pump = float(pump)
        self.nsub = int(nsub)
        self.u = 0.0

    def metadata(self):
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "kind": "nonlinear_grid_mpc",
            "scenario": "cstr",
            "action_mode": "actuator",
            "control_structure": "cstr_grid_mpc",
            "horizon": self.horizon,
            "grid": int(len(self.grid)),
            "r_move": self.r_move,
            "pump": self.pump,
            "nsub": self.nsub,
        }

    def reset(self):
        self.u = 0.0

    def compute(self, meas, sp, dt):
        p = self.p
        candidates = self.grid
        ca = np.full_like(candidates, meas["conc"][0], dtype=float)
        temp = np.full_like(candidates, meas["temps"][0], dtype=float)
        dilution = self.pump * p["Dmax"]
        h = dt / self.nsub
        cost = self.r_move * (candidates - self.u) ** 2

        for _ in range(self.horizon):
            for _ in range(self.nsub):
                rate = p["k0"] * np.exp(-p["EaR"] / (temp + 273.15)) * np.maximum(ca, 0.0)
                ca += h * (dilution * (p["Caf"] - ca) - rate)
                temp += h * (
                    dilution * (meas["t_cold"] - temp)
                    + p["Hr"] * rate
                    - p["Uc"] * candidates * (temp - p["Tcool"])
                )
                ca = np.maximum(ca, 0.0)
                temp = np.minimum(temp, 200.0)
            err = temp - sp["t_sp"][0]
            cost += err * err + np.where(temp > 90.0, 1000.0 * (temp - 90.0) ** 2, 0.0)

        self.u = float(candidates[int(np.argmin(cost))])
        return {"pumps": [self.pump], "valves": [], "heaters": [self.u]}
