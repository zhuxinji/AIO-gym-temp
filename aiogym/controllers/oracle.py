"""NMPC oracle — a true nonlinear MPC baseline (CasADi + IPOPT), in the spirit of
PC-Gym's do-mpc oracle (Bloor et al., arXiv:2410.22093). It's the "upper-bound"
controller RL/PID/APC-MPC are measured against.

Selectable multiple- or single-shooting transcription: RK4 over each control
interval, IPOPT NLP.
Plant dynamics are supplied by the model contract through
``model.dynamics(..., backend="casadi")``. Two objectives:
  - "tracking": Σ normalized(y-y_sp)ᵀQ normalized(y-y_sp) + ΔuᵀRΔu
                (PC-Gym-style setpoint tracking)
  - "economic": maximize the economic stage profit (value − energy − violation),
                the right oracle for the economic scenarios (hugs the safe edge).

Usage:
    orc = NMPCOracle("cstr", horizon=20, mode="economic")
    u = orc.solve(x, t_cold, t_amb, disturbances=meas, y_sp=y_sp)
"""
from __future__ import annotations
import copy
import math
import numpy as np

from .._internal.validation import nonnegative_float, positive_float, positive_int

try:
    import casadi as ca
    _HAVE_CASADI = True
except (ImportError, OSError):          # pragma: no cover
    _HAVE_CASADI = False

from ..models import make_model


def copy_economic_config(model):
    return copy.deepcopy(getattr(model, "economic_config", {}))


class NMPCOracle:
    def __init__(self, scenario="cstr", horizon=20, control_dt=0.5, mode="economic",
                 du_max=0.4, q_y=1.0, r_move=0.05,
                 nsub_max=6, ipopt_max_iter=80, ipopt_tol=1e-4, warm_start=False,
                 transcription="multiple_shooting", enforce_state_bounds=True,
                 enforce_temperature_cap=True, model=None):
        if not _HAVE_CASADI:
            raise RuntimeError("casadi not installed — pip install casadi")
        self.model = make_model(model if model is not None else scenario)
        self.scenario = self.model.scenario
        self.p = self.model.p
        self.N = positive_int("horizon", horizon)
        self.dt = positive_float("control_dt", control_dt)
        if mode not in {"economic", "tracking"}:
            raise ValueError("mode must be one of: economic, tracking")
        self.mode = mode
        self.du_max = nonnegative_float("du_max", du_max)
        self.nsub_max = positive_int("nsub_max", nsub_max)
        self.ipopt_max_iter = positive_int("ipopt_max_iter", ipopt_max_iter)
        self.ipopt_tol = positive_float("ipopt_tol", ipopt_tol)
        self.warm_start = bool(warm_start)
        if transcription not in {"multiple_shooting", "single_shooting"}:
            raise ValueError("transcription must be one of: multiple_shooting, single_shooting")
        self.transcription = transcription
        self.enforce_state_bounds = bool(enforce_state_bounds)
        self.enforce_temperature_cap = bool(enforce_temperature_cap)
        self.nu = self.model.action_dim()
        self.nx = len(self.model.initial_state())
        self.ny = len(self.model.controlled_output(self.model.initial_state()))
        self.y_scales = [max(float(value), 1e-12) for value in self.model.controlled_output_scales()]
        self.q_y = self._resolve_q_y(q_y)
        self.r_move = nonnegative_float("r_move", r_move)
        self.econ = copy_economic_config(self.model)
        self.nd = len(self.model.dynamics_disturbance_names())
        self.state_bounds = self._state_bounds() if self.enforce_state_bounds else [(None, None)] * self.nx
        self.state_scale, self.state_offset = self._state_scaling()
        # hard safety cap (below the 92°C runaway trip) so economic NMPC hugs the edge
        # without driving the plant unstable; HVAC has no runaway so cap loosely.
        self.t_safe = self._temperature_soft_cap() if self.enforce_temperature_cap else None
        self.u_init = self._initial_action()
        self.u_prev = self.u_init.copy()
        self._warm = None
        self.last_error = None
        self._build()

    def _resolve_q_y(self, q_y):
        values = q_y if isinstance(q_y, (list, tuple)) else [q_y]
        values = [float(v) for v in values]
        if len(values) == 1:
            values *= self.ny
        if len(values) != self.ny:
            raise ValueError(f"q_y must contain 1 or {self.ny} values, got {len(values)}")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("q_y values must be finite and non-negative")
        return values

    def _initial_action(self):
        initializer = getattr(self.model, "mpc_init", None)
        values = initializer() if callable(initializer) else [0.5] * self.nu
        action = np.asarray(values, dtype=float).reshape(-1)
        if len(action) != self.nu:
            raise ValueError(f"model.mpc_init() must contain {self.nu} values, got {len(action)}")
        if not np.all(np.isfinite(action)):
            raise ValueError("model.mpc_init() values must be finite")
        return np.clip(action, 0.0, 1.0)

    def _state_bounds(self):
        bounds = []
        for row in self.model.state_schema():
            value = row.get("bounds")
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                bounds.append((None, None))
                continue
            lo, hi = value
            bounds.append((None if lo is None else float(lo), None if hi is None else float(hi)))
        return bounds

    def _state_scaling(self):
        initial = np.asarray(self.model.initial_state(), dtype=float)
        scales = []
        offsets = []
        for value, (lo, hi) in zip(initial, self.state_bounds):
            if lo is not None and hi is not None:
                scales.append(max(hi - lo, 1e-6))
                offsets.append(lo)
            else:
                scales.append(max(abs(float(value)), 1.0))
                offsets.append(0.0)
        return np.asarray(scales, dtype=float), np.asarray(offsets, dtype=float)

    def _temperature_soft_cap(self):
        if getattr(self.model, "oracle_temperature_cap", True) is False:
            return None
        caps = []
        for row in self.model.constraint_schema():
            name = row.get("name", "")
            bounds = row.get("bounds")
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or bounds[1] is None:
                continue
            if "temperature" in name:
                caps.append(float(bounds[1]))
        if caps:
            return min(caps) - 2.0
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
        return self._tracking_state_cost(x, sp)

    def _tracking_state_cost(self, x, sp):
        y = self.model.controlled_output(x, backend="casadi", ca=ca)
        c = 0
        for i, yi in enumerate(y):
            weight = self.q_y[i] if i < len(self.q_y) else 1.0
            scale = self.y_scales[i] if i < len(self.y_scales) else 1.0
            c += weight * ((yi - sp["y_sp"][i]) / scale) ** 2
        return c

    def _econ_profit(self, x, u, d):
        cfg = self.econ
        display = self.model.display_outputs(x, backend="casadi", ca=ca)
        temps = display.get("temps", [])
        env = self.model.dynamics_disturbance_map(d)
        value = self.model.economic_value(x, u, env, backend="casadi", ca=ca)
        energy = self.model.energy_kw(u, backend="casadi", ca=ca)
        viol = 0
        for i, (lo, hi) in enumerate(cfg["temp_band"]):
            if lo is not None:
                viol += ca.fmax(0, lo - temps[i]) / 10.0
            if hi is not None:
                viol += ca.fmax(0, temps[i] - hi) / 10.0
        level_scale = cfg.get("level_scale", 0.1)
        y = self.model.controlled_output(x, backend="casadi", ca=ca)
        for j, (lo, hi) in enumerate(cfg["level_band"]):
            cv = y[j] if j < len(y) else 0
            if lo is not None:
                viol += ca.fmax(0, lo - cv) / level_scale
            if hi is not None:
                viol += ca.fmax(0, cv - hi) / level_scale
        return cfg["w_value"] * value - cfg["w_energy"] * energy - cfg["w_viol"] * viol

    def _build(self):
        if self.transcription == "single_shooting":
            self._build_single_shooting()
        else:
            self._build_multiple_shooting()

    def _solver_options(self):
        return {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0,
                "ipopt.max_iter": self.ipopt_max_iter,
                "ipopt.acceptable_tol": self.ipopt_tol,
                "ipopt.tol": self.ipopt_tol}

    def _constrain_state(self, opti, x):
        for i, (lo, hi) in enumerate(self.state_bounds):
            if lo is not None and hi is not None:
                opti.subject_to(opti.bounded(lo, x[i], hi))
            elif lo is not None:
                opti.subject_to(x[i] >= lo)
            elif hi is not None:
                opti.subject_to(x[i] <= hi)

    def _build_multiple_shooting(self):
        N, nx, nu = self.N, self.nx, self.nu
        opti = ca.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        opti.set_linear_scale(
            X,
            ca.repmat(ca.DM(self.state_scale), 1, N + 1),
            ca.repmat(ca.DM(self.state_offset), 1, N + 1),
        )
        x0 = opti.parameter(nx)
        d = opti.parameter(self.nd)
        u_prev = opti.parameter(nu)
        ysp = opti.parameter(self.ny)
        sp = {"y_sp": [ysp[i] for i in range(self.ny)]}
        J = 0
        opti.subject_to(X[:, 0] == x0)
        slack = opti.variable(1, N)                                    # soft cap slack (feasibility)
        opti.subject_to(slack >= 0)
        for i, (lo, hi) in enumerate(self.state_bounds):
            for k in range(N + 1):
                if lo is not None and hi is not None:
                    opti.subject_to(opti.bounded(lo, X[i, k], hi))
                elif lo is not None:
                    opti.subject_to(X[i, k] >= lo)
                elif hi is not None:
                    opti.subject_to(X[i, k] <= hi)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self._rk4(X[:, k], U[:, k], d))
            opti.subject_to(opti.bounded(0.0, U[:, k], 1.0))            # actuators in [0,1]
            up = u_prev if k == 0 else U[:, k - 1]
            opti.subject_to(opti.bounded(-self.du_max, U[:, k] - up, self.du_max))  # move limit
            temps = self.model.display_outputs(X[:, k + 1], backend="casadi", ca=ca).get("temps", [])
            if self.t_safe is not None:
                for temp in temps:                                     # soft safety cap on temperature outputs
                    opti.subject_to(temp <= self.t_safe + slack[0, k])
            J += self._stage_cost(X[:, k], U[:, k], sp, d) + self.r_move * ca.sumsqr(U[:, k] - up)
        if self.mode == "tracking":
            J += self._tracking_state_cost(X[:, N], sp)
        J += 1e4 * ca.sumsqr(slack)                                    # heavily discourage cap violation
        opti.minimize(J)
        opti.solver("ipopt", self._solver_options())
        self.opti, self.X, self.U, self.slack = opti, X, U, slack
        self.par = {"x0": x0, "d": d, "u_prev": u_prev, "ysp": ysp}

    def _build_single_shooting(self):
        N, nx, nu = self.N, self.nx, self.nu
        opti = ca.Opti()
        U = opti.variable(nu, N)
        x0 = opti.parameter(nx)
        d = opti.parameter(self.nd)
        u_prev = opti.parameter(nu)
        ysp = opti.parameter(self.ny)
        sp = {"y_sp": [ysp[i] for i in range(self.ny)]}
        slack = opti.variable(1, N)
        opti.subject_to(slack >= 0)
        x = x0
        J = 0
        for k in range(N):
            up = u_prev if k == 0 else U[:, k - 1]
            opti.subject_to(opti.bounded(0.0, U[:, k], 1.0))
            opti.subject_to(opti.bounded(-self.du_max, U[:, k] - up, self.du_max))
            J += self._stage_cost(x, U[:, k], sp, d) + self.r_move * ca.sumsqr(U[:, k] - up)
            x = self._rk4(x, U[:, k], d)
            self._constrain_state(opti, x)
            temps = self.model.display_outputs(x, backend="casadi", ca=ca).get("temps", [])
            if self.t_safe is not None:
                for temp in temps:
                    opti.subject_to(temp <= self.t_safe + slack[0, k])
        if self.mode == "tracking":
            J += self._tracking_state_cost(x, sp)
        J += 1e4 * ca.sumsqr(slack)
        opti.minimize(J)
        opti.solver("ipopt", self._solver_options())
        self.opti, self.X, self.U, self.slack = opti, None, U, slack
        self.par = {"x0": x0, "d": d, "u_prev": u_prev, "ysp": ysp}

    def _numeric_rk4(self, x, u, disturbances):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        nsub = max(1, min(self.nsub_max, int(round(self.dt / self.model.dt_micro))))
        h = self.dt / nsub
        for _ in range(nsub):
            k1 = np.asarray(self.model.dynamics(x, u, disturbances), dtype=float)
            k2 = np.asarray(self.model.dynamics(x + 0.5 * h * k1, u, disturbances), dtype=float)
            k3 = np.asarray(self.model.dynamics(x + 0.5 * h * k2, u, disturbances), dtype=float)
            k4 = np.asarray(self.model.dynamics(x + h * k3, u, disturbances), dtype=float)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if callable(getattr(self.model, "clamp_state", None)):
                x = np.asarray(self.model.clamp_state(list(x)), dtype=float)
        return x

    def _initial_state_guess(self, x, u, d):
        disturbances = self.model.dynamics_disturbance_map(d)
        cols = [np.asarray(x, dtype=float)]
        cur = np.asarray(x, dtype=float)
        for _ in range(self.N):
            cur = self._numeric_rk4(cur, u, disturbances)
            cols.append(cur)
        return np.column_stack(cols)

    def reset(self):
        self.u_prev = self.u_init.copy()
        self._warm = None
        self.last_error = None

    def _disturbance_vector(self, t_cold, t_amb, disturbances=None):
        values = dict(disturbances or {})
        values.update({"t_cold": t_cold, "t_amb": t_amb})
        return self.model.disturbance_vector(values)

    def solve(self, x, t_cold, t_amb, disturbances=None, y_sp=None):
        o = self.opti
        o.set_value(self.par["x0"], np.asarray(x, float))
        if self.nd:
            o.set_value(self.par["d"], self._disturbance_vector(t_cold, t_amb, disturbances))
        o.set_value(self.par["u_prev"], self.u_prev)
        target = self.model.setpoint_vector(y_sp)
        o.set_value(self.par["ysp"], np.asarray(target, float))
        try:
            dvec = self._disturbance_vector(t_cold, t_amb, disturbances)
            if (not self.warm_start) or self._warm is None:
                if self.X is not None:
                    o.set_initial(self.X, self._initial_state_guess(x, self.u_prev, dvec))
                o.set_initial(self.U, np.tile(self.u_prev.reshape(-1, 1), (1, self.N)))
                o.set_initial(self.slack, np.zeros((1, self.N)))
            else:
                Xw, Uw, Sw = self._warm
                if self.X is not None:
                    Xw = np.column_stack([np.asarray(x, float), Xw[:, 2:], Xw[:, -1]])
                    o.set_initial(self.X, Xw)
                Uw = np.column_stack([Uw[:, 1:], Uw[:, -1]])
                Sw = np.asarray(Sw, float).reshape(1, -1)
                Sw = np.column_stack([Sw[:, 1:], Sw[:, -1]])
                o.set_initial(self.U, Uw)
                o.set_initial(self.slack, Sw)
            sol = o.solve()
            Xv = None if self.X is None else np.asarray(sol.value(self.X), float).reshape(self.nx, self.N + 1)
            Uv = np.asarray(sol.value(self.U), float).reshape(self.nu, self.N)
            Sv = np.asarray(sol.value(self.slack), float).reshape(1, self.N)
            if self.warm_start:
                self._warm = (Xv, Uv, Sv)
            u = np.clip(Uv[:, 0], 0.0, 1.0)
            self.last_error = None
        except Exception as e:
            self.last_error = e
            u = self.u_prev                                # keep last on solver failure
        self.u_prev = np.asarray(u, float).reshape(-1)
        return list(self.u_prev)


class OracleAgent:
    """Controller-v1 wrapper around the nonlinear MPC oracle."""
    name = "NMPC-oracle"
    controller_api_version = "aiogym.controller.v1"
    action_mode = "actuator"
    control_structure = "nmpc_oracle"

    def __init__(self, scenario, solve_every=1, model=None, **kw):
        self.orc = NMPCOracle(scenario, model=model, **kw)
        self.scenario = self.orc.scenario
        self.model = self.orc.model
        self.solve_every = positive_int("solve_every", solve_every)
        self._ticks = 0
        self._last = None
        self.solve_count = 0
        self.solver_success_count = 0
        self.solver_failure_count = 0
        self.fallback_count = 0
        self.last_solver_error = None

    def metadata(self):
        return {"name": self.name, "class": self.__class__.__name__,
                "kind": "nonlinear_mpc_oracle", "scenario": self.scenario,
                "api": self.controller_api_version,
                "action_mode": self.action_mode, "control_structure": self.control_structure,
                "solve_every": self.solve_every,
                "horizon": self.orc.N, "control_dt": self.orc.dt, "mode": self.orc.mode,
                "du_max": self.orc.du_max, "q_y": list(self.orc.q_y),
                "r_move": self.orc.r_move, "nsub_max": self.orc.nsub_max,
                "ipopt_max_iter": self.orc.ipopt_max_iter,
                "warm_start": self.orc.warm_start,
                "transcription": self.orc.transcription,
                "enforce_state_bounds": self.orc.enforce_state_bounds,
                "enforce_temperature_cap": self.orc.enforce_temperature_cap,
                "diagnostics": self.diagnostics()}

    def diagnostics(self):
        return {
            "solve_count": int(self.solve_count),
            "solver_success_count": int(self.solver_success_count),
            "solver_failure_count": int(self.solver_failure_count),
            "fallback_count": int(self.fallback_count),
            "degraded": bool(self.solver_failure_count or self.fallback_count),
            "last_solver_error": self.last_solver_error,
        }

    def reset(self, seed=None):
        self.orc.reset()
        self._ticks = 0
        self._last = None
        self.solve_count = 0
        self.solver_success_count = 0
        self.solver_failure_count = 0
        self.fallback_count = 0
        self.last_solver_error = None

    def act(self, obs, context):
        action = self.compute(context.measurement, context.setpoint, context.control_dt)
        return np.asarray(self.model.action_vector(action), dtype=np.float32)

    def _x_from_meas(self, meas):
        if "x" in meas:
            return list(meas["x"])
        raise ValueError("oracle controller requires measurement['x']")

    def compute(self, meas, sp, dt):
        if self._last is None or self._ticks % self.solve_every == 0:
            self._last = self.orc.solve(self._x_from_meas(meas), meas.get("t_cold", 15.0), meas.get("t_amb", 20.0),
                                        disturbances=meas, y_sp=sp.get("y_sp"))
            self.solve_count += 1
            if self.orc.last_error is None:
                self.solver_success_count += 1
            else:
                self.solver_failure_count += 1
                self.fallback_count += 1
                self.last_solver_error = f"{self.orc.last_error.__class__.__name__}: {self.orc.last_error}"
        self._ticks += 1
        return self._last
