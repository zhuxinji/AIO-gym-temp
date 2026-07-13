"""Successive-linearization fixed-setpoint MPC baseline."""
from __future__ import annotations

import math

import numpy as np

from .._internal.validation import nonnegative_float, positive_float, positive_int


class MPCAgent:
    """Successive-linearization, velocity-form (M=1) constrained MPC."""
    name = "MPC"
    controller_api_version = "aiogym.controller.v1"
    action_mode = "actuator"
    control_structure = "fixed_sp_mpc"

    def __init__(self, model, Ts=0.5, P=40, move_supp=0.8, du_max=0.15, cv_scale=None):
        self.m = model
        self.nP, self.nV, self.nH = model.actuator_counts()
        self.nu = model.action_dim()
        self.nx = len(model.initial_state())
        self.ncv = len(model.controlled_output(model.initial_state()))
        self.Ts = positive_float("Ts", Ts)
        self.P = positive_int("P", P)
        self.move_supp = nonnegative_float("move_supp", move_supp)
        self.du_max = nonnegative_float("du_max", du_max)
        self.cv_scale = self._resolve_cv_scale(cv_scale)
        self.reset()

    def _resolve_cv_scale(self, cv_scale):
        if cv_scale is not None:
            values = cv_scale if isinstance(cv_scale, (list, tuple)) else [cv_scale]
            values = [float(v) for v in values]
            if len(values) == 1:
                values *= self.ncv
        else:
            values = [float(value) for value in self.m.controlled_output_scales()]
        if len(values) != self.ncv:
            raise ValueError(f"cv_scale must contain 1 or {self.ncv} values, got {len(values)}")
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("cv_scale values must be finite and positive")
        return values

    def metadata(self):
        return {"name": self.name, "class": self.__class__.__name__,
                "kind": "successive_linearization_mpc", "scenario": self.m.scenario,
                "api": self.controller_api_version,
                "action_mode": self.action_mode, "control_structure": self.control_structure,
                "Ts": self.Ts, "horizon": self.P,
                "move_supp": self.move_supp, "du_max": self.du_max,
                "cv_scale": self.cv_scale}

    def reset(self, seed=None):
        initializer = getattr(self.m, "mpc_init", None)
        initial_action = initializer() if callable(initializer) else self.m.default_action()
        self.u = np.asarray(self.m.action_vector(initial_action), dtype=np.float64)
        self._clock = 1e9

    def act(self, obs, context):
        action = self.compute(context.measurement, context.setpoint, context.control_dt)
        return np.asarray(self.m.action_vector(action), dtype=np.float32)

    def _unpack(self, u):
        return self.m.action_vector_to_dict(u)

    def _toX(self, meas):
        return np.asarray(meas["x"], dtype=np.float64)

    def _cv(self, x):
        return np.asarray(self.m.controlled_output(list(x)), dtype=np.float64)

    def _wcv(self):
        return np.array([1.0 / max(float(scale), 1e-12) ** 2 for scale in self.cv_scale], dtype=np.float64)

    def compute(self, meas, sp, dt):
        self._clock += dt
        if self._clock >= self.Ts:
            self._clock = 0.0
            self._solve(meas, sp)
        return self._unpack(self.u)

    def _solve(self, meas, sp):
        m, nx, nu, P, Ts = self.m, self.nx, self.nu, self.P, self.Ts
        env = {k: v for k, v in meas.items()
               if k not in ("x", "y", "levels", "temps", "conc")}
        env.setdefault("extra_outflow", 0.0)
        x0 = self._toX(meas)
        u0 = self.u.copy()
        f = lambda x: np.asarray(m.dynamics(list(x), u0, env), dtype=np.float64)
        f0 = f(x0)
        eps = 1e-5
        Ad = np.eye(nx)
        Bd = np.zeros((nx, nu))
        for j in range(nx):
            xp = x0.copy(); xp[j] += eps
            xm = x0.copy(); xm[j] -= eps
            Ad[:, j] += (f(xp) - f(xm)) / (2 * eps) * Ts
        for j in range(nu):
            up = u0.copy(); up[j] += eps
            um = u0.copy(); um[j] -= eps
            fp = np.asarray(m.dynamics(list(x0), up, env), dtype=np.float64)
            fm = np.asarray(m.dynamics(list(x0), um, env), dtype=np.float64)
            Bd[:, j] = (fp - fm) / (2 * eps) * Ts
        cv0 = self._cv(x0)
        nCV = len(cv0)
        C = np.zeros((nCV, nx))
        for j in range(nx):
            xp = x0.copy(); xp[j] += eps
            C[:, j] = (self._cv(xp) - cv0) / eps
        target = np.asarray(m.setpoint_vector(sp.get("y_sp")), dtype=np.float64)
        Wcv = self._wcv()
        c0 = (x0 + f0 * Ts) - Ad @ x0 - Bd @ u0
        xf = x0.copy()
        S = np.zeros((nx, nu))
        H = np.zeros((nu, nu))
        g = np.zeros(nu)
        for _ in range(P):
            xf = Ad @ xf + Bd @ u0 + c0
            S = Ad @ S + Bd
            G = C @ S
            e = cv0 + C @ (xf - x0) - target
            WG = Wcv[:, None] * G
            H += G.T @ WG
            g += G.T @ (Wcv * e)
        H += self.move_supp * np.eye(nu)
        du = np.linalg.solve(H, -g)
        du = np.clip(du, -self.du_max, self.du_max)
        self.u = np.clip(u0 + du, 0.0, 1.0)
