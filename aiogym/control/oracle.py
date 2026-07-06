"""NMPC oracle — a true nonlinear MPC baseline (CasADi + IPOPT), in the spirit of
PC-Gym's do-mpc oracle (Bloor et al., arXiv:2410.22093). It's the "upper-bound"
controller RL/PID/APC-MPC are measured against.

Direct multiple-shooting transcription: RK4 over each control interval, IPOPT NLP.
Plant dynamics are supplied by the model contract through
``model.dynamics(..., backend="casadi")``. Two objectives:
  - "track":    Σ (x-x_sp)ᵀQ(x-x_sp) + ΔuᵀRΔu      (PC-Gym-style setpoint tracking)
  - "economic": maximize the economic stage profit (value − energy − violation),
                the right oracle for the economic scenarios (hugs the safe edge).

Usage:
    orc = NMPCOracle("cstr", horizon=20, mode="economic")
    act = orc.solve(x, t_cold, t_amb, t_sp, h_sp, disturbances=meas)
    # -> {"pumps":[...],"valves":[...],"heaters":[...]}
"""
from __future__ import annotations
import copy
import numpy as np

try:
    import casadi as ca
    _HAVE_CASADI = True
except Exception:                       # pragma: no cover
    _HAVE_CASADI = False

from ..models import make_model


def copy_economic_config(model):
    return copy.deepcopy(getattr(model, "economic_config", {}))


class NMPCOracle:
    def __init__(self, scenario="cstr", horizon=20, control_dt=0.5, mode="economic",
                 du_max=0.4, q_temp=1.0, q_level=50.0, r_move=0.05,
                 nsub_max=6, ipopt_max_iter=80, ipopt_tol=1e-4, warm_start=False):
        if not _HAVE_CASADI:
            raise RuntimeError("casadi not installed — pip install casadi")
        self.scenario = scenario
        self.model = make_model(scenario)
        self.p = self.model.p
        self.N = int(horizon)
        self.dt = float(control_dt)
        self.mode = mode
        self.du_max = du_max
        self.nsub_max = int(nsub_max)
        self.ipopt_max_iter = int(ipopt_max_iter)
        self.ipopt_tol = float(ipopt_tol)
        self.warm_start = bool(warm_start)
        nP, nV, nH = self.model.actuator_counts()
        self.nP, self.nV, self.nH = nP, nV, nH
        self.nu = nP + nV + nH
        self.nx = len(self.model.initial_state())
        self.ny = len(self.model.controlled_output(self.model.initial_state()))
        self.q_temp, self.q_level, self.r_move = q_temp, q_level, r_move
        self.econ = copy_economic_config(self.model)
        self.nd = len(self.model.dynamics_disturbance_names())
        # hard safety cap (below the 92°C runaway trip) so economic NMPC hugs the edge
        # without driving the plant unstable; HVAC has no runaway so cap loosely.
        self.t_safe = self._temperature_soft_cap()
        self.u_prev = np.full(self.nu, 0.5)
        self._warm = None
        self.last_error = None
        self._build()

    def _temperature_soft_cap(self):
        caps = []
        for row in self.model.constraint_schema():
            name = row.get("name", "")
            bounds = row.get("bounds")
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or bounds[1] is None:
                continue
            if "temperature" in name:
                caps.append(float(bounds[1]))
        if caps:
            return min(90.0, min(caps) - 2.0)
        return 40.0

    # one RK4 step of the model dynamics over the control interval. Substeps capped
    # for solve speed (a slightly coarse internal model is standard MPC practice).
    def _rk4(self, x, u, d):
        f = lambda xx: self.model.dynamics(xx, u, d, backend="casadi", ca=ca)
        nsub = max(1, min(self.nsub_max, int(round(self.dt / self.model.dt_micro))))
        h = self.dt / nsub
        for _ in range(nsub):
            k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def _stage_cost(self, x, u, sp, d):
        if self.mode == "economic":
            return -self._econ_profit(x, u, d)            # minimize -profit
        y = self.model.controlled_output(x, backend="casadi", ca=ca)
        n_level = len(self.model.controlled_levels())
        c = 0
        for i, yi in enumerate(y):
            weight = self.q_level if i < n_level else self.q_temp
            c += weight * (yi - sp["y_sp"][i]) ** 2
        return c

    def _econ_profit(self, x, u, d):
        cfg = self.econ
        levels, temps = self.model.levels_temps(x, backend="casadi", ca=ca)
        env = self.model.dynamics_disturbance_map(d)
        value = self.model.economic_value(x, u, env, backend="casadi", ca=ca)
        energy = self.model.energy_kw(u, backend="casadi", ca=ca)
        viol = 0
        for i, (lo, hi) in enumerate(cfg["temp_band"]):
            if lo is not None:
                viol += ca.fmax(0, lo - temps[i]) / 10.0
            if hi is not None:
                viol += ca.fmax(0, temps[i] - hi) / 10.0
        for j, i in enumerate(self.model.controlled_levels()):
            lo, hi = cfg["level_band"][j]
            if lo is not None:
                viol += ca.fmax(0, lo - levels[i]) / 0.1
            if hi is not None:
                viol += ca.fmax(0, levels[i] - hi) / 0.1
        return cfg["w_value"] * value - cfg["w_energy"] * energy - cfg["w_viol"] * viol

    def _build(self):
        N, nx, nu = self.N, self.nx, self.nu
        opti = ca.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        x0 = opti.parameter(nx)
        d = opti.parameter(self.nd)
        u_prev = opti.parameter(nu)
        ysp = opti.parameter(self.ny)
        sp = {"y_sp": [ysp[i] for i in range(self.ny)]}
        J = 0
        opti.subject_to(X[:, 0] == x0)
        slack = opti.variable(1, N)                                    # soft cap slack (feasibility)
        opti.subject_to(slack >= 0)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self._rk4(X[:, k], U[:, k], d))
            opti.subject_to(opti.bounded(0.0, U[:, k], 1.0))            # actuators in [0,1]
            up = u_prev if k == 0 else U[:, k - 1]
            opti.subject_to(opti.bounded(-self.du_max, U[:, k] - up, self.du_max))  # move limit
            _, temps = self.model.levels_temps(X[:, k + 1], backend="casadi", ca=ca)
            for temp in temps:                                         # soft safety cap on temperature outputs
                opti.subject_to(temp <= self.t_safe + slack[0, k])
            J += self._stage_cost(X[:, k], U[:, k], sp, d) + self.r_move * ca.sumsqr(U[:, k] - up)
        J += 1e4 * ca.sumsqr(slack)                                    # heavily discourage cap violation
        opti.minimize(J)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0,
                              "ipopt.max_iter": self.ipopt_max_iter,
                              "ipopt.acceptable_tol": self.ipopt_tol,
                              "ipopt.tol": self.ipopt_tol})
        self.opti, self.X, self.U, self.slack = opti, X, U, slack
        self.par = {"x0": x0, "d": d, "u_prev": u_prev, "ysp": ysp}

    def reset(self):
        self.u_prev = np.full(self.nu, 0.5)
        self._warm = None
        self.last_error = None

    def _disturbance_vector(self, t_cold, t_amb, disturbances=None):
        values = dict(disturbances or {})
        values.update({"t_cold": t_cold, "t_amb": t_amb})
        return self.model.disturbance_vector(values)

    def solve(self, x, t_cold, t_amb, t_sp, h_sp, disturbances=None):
        o = self.opti
        o.set_value(self.par["x0"], np.asarray(x, float))
        o.set_value(self.par["d"], self._disturbance_vector(t_cold, t_amb, disturbances))
        o.set_value(self.par["u_prev"], self.u_prev)
        o.set_value(self.par["ysp"], np.asarray(self.model.setpoint_vector(h_sp, t_sp), float))
        try:
            if (not self.warm_start) or self._warm is None:
                o.set_initial(self.U, np.tile(self.u_prev.reshape(-1, 1), (1, self.N)))
            else:
                Xw, Uw, Sw = self._warm
                Xw = np.column_stack([np.asarray(x, float), Xw[:, 2:], Xw[:, -1]])
                Uw = np.column_stack([Uw[:, 1:], Uw[:, -1]])
                Sw = np.asarray(Sw, float).reshape(1, -1)
                Sw = np.column_stack([Sw[:, 1:], Sw[:, -1]])
                o.set_initial(self.X, Xw)
                o.set_initial(self.U, Uw)
                o.set_initial(self.slack, Sw)
            sol = o.solve()
            Xv = np.asarray(sol.value(self.X), float)
            Uv = np.asarray(sol.value(self.U), float)
            Sv = np.asarray(sol.value(self.slack), float)
            if self.warm_start:
                self._warm = (Xv, Uv, Sv)
            u = np.clip(Uv[:, 0], 0.0, 1.0)
            self.last_error = None
        except Exception as e:
            self.last_error = e
            u = self.u_prev                                # keep last on solver failure
        self.u_prev = np.asarray(u, float).reshape(-1)
        return {"pumps": list(self.u_prev[:self.nP]),
                "valves": list(self.u_prev[self.nP:self.nP + self.nV]),
                "heaters": list(self.u_prev[self.nP + self.nV:])}


class OracleAgent:
    """Adapts NMPCOracle to the baselines agent interface compute(meas, sp, dt)."""
    name = "NMPC-oracle"

    def __init__(self, scenario, solve_every=1, **kw):
        self.orc = NMPCOracle(scenario, **kw)
        self.scenario = scenario
        self.model = self.orc.model
        self.solve_every = max(1, int(solve_every))
        self._ticks = 0
        self._last = None

    def metadata(self):
        return {"name": self.name, "class": self.__class__.__name__,
                "kind": "nonlinear_mpc_oracle", "scenario": self.scenario,
                "action_mode": "actuator", "control_structure": "nmpc_oracle",
                "solve_every": self.solve_every,
                "horizon": self.orc.N, "mode": self.orc.mode,
                "du_max": self.orc.du_max, "warm_start": self.orc.warm_start}

    def reset(self):
        self.orc.reset()
        self._ticks = 0
        self._last = None

    def _x_from_meas(self, meas):
        if "x" in meas:
            return list(meas["x"])
        if self.scenario == "cstr":
            return [meas["conc"][0], meas["temps"][0]]
        if self.scenario == "hvac":
            return list(meas["temps"])
        x = []                                             # cascade/quadruple: interleave h,T
        for i in range(self.model.n):
            x += [meas["levels"][i], meas["temps"][i]]
        return x

    def compute(self, meas, sp, dt):
        if self._last is None or self._ticks % self.solve_every == 0:
            self._last = self.orc.solve(self._x_from_meas(meas), meas["t_cold"], meas["t_amb"],
                                        sp["t_sp"], sp["h_sp"], disturbances=meas)
        self._ticks += 1
        return self._last
