"""Observation, setpoint, and supervisory-action runtime helpers."""
from __future__ import annotations

import numpy as np

from .config import I_TEMP_MAX


class ObservationRuntimeMixin:
    def _split(self, action):
        a = np.clip(np.asarray(action, np.float64), 0.0, 1.0)
        return self.model.action_vector(a)

    def _obs(self):
        state = self.model.state_vector(self.integ.x)
        if self.noise:
            noisy = []
            for value, row in zip(state, self.model.state_schema()):
                bounds = row.get("bounds")
                scale = max(abs(float(value)), 1.0)
                if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                    lo, hi = bounds
                    if lo is not None and hi is not None and float(hi) > float(lo):
                        scale = float(hi) - float(lo)
                noisy.append(float(value) + float(self.np_random.normal(0, self.noise_pct * scale)))
            state = noisy
        o = state + list(self.y_sp) + list(self.model.disturbance_vector(self._env()))
        if self.integral_obs:
            o = o + [iy / I_TEMP_MAX for iy in self._iy]
        return np.asarray(o, dtype=np.float32)

    def _accumulate_integral(self):
        if not self.model.supports_integral_observation:
            return
        out = self.model.outputs(self.integ.x)
        y = list(out["y"])
        errors = [self.y_sp[i] - y[i] if i < len(y) else 0.0 for i in range(len(self.y_sp))]
        dt = self.control_dt
        self._iy = [float(np.clip(self._iy[i] + errors[i] * dt, -I_TEMP_MAX, I_TEMP_MAX)) for i in range(len(errors))]
    def _setpoint_bounds(self):
        bounds = []
        for row in self.model.setpoint_schema():
            raw = row.get("bounds")
            if isinstance(raw, (tuple, list)) and len(raw) == 2 and raw[0] is not None and raw[1] is not None:
                bounds.append((float(raw[0]), float(raw[1])))
            else:
                bounds.append((None, None))
        return bounds

    def _randomize_setpoints(self, rng):
        bounds = self._setpoint_bounds()
        next_sp = []
        for i, value in enumerate(self.y_sp):
            lo, hi = bounds[i] if i < len(bounds) else (None, None)
            trial = float(value * (1 + 0.10 * rng.uniform(-1, 1)))
            if lo is not None and hi is not None:
                trial = float(np.clip(trial, lo, hi))
            next_sp.append(trial)
        self.y_sp = next_sp
    def default_sp_action(self):
        """Normalized supervisory action that reproduces the default setpoints (= the
        fixed-SP PID baseline), the offline prior to learn from."""
        if self.layout is None:
            return None
        a = []
        for spec in self.layout:
            lo, hi = spec[-2], spec[-1]
            if spec[0] == "y_sp":
                v = self._ysp0[spec[1]]
            else:
                v = lo + 0.7 * (hi - lo)
            a.append(float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)))
        return np.array(a, np.float32)

    def _meas(self):
        """buildState-like dict the inner PID reads (true state)."""
        return self.model.measurement(self.integ.x, self._env())

    def _supervise(self, action):
        """Supervisory action = normalized setpoints -> set SPs, inner PID regulates
        to them; unregulated economic MVs ('mv') are applied directly."""
        a = np.clip(np.asarray(action, np.float64), 0.0, 1.0)
        mv = {}
        for i, spec in enumerate(self.layout):
            lo, hi = spec[-2], spec[-1]
            val = lo + float(a[i]) * (hi - lo)
            if spec[0] == "y_sp":
                self.y_sp[spec[1]] = val
            else:                                   # ("mv", u_index, lo, hi)
                mv[spec[1]] = val
        act = self.pid.compute(self._meas(), {"y_sp": self.y_sp}, self.control_dt)
        for u_index, value in mv.items():
            act[u_index] = value
        return act
