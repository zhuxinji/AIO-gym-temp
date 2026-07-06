"""AIOGymNativeEnv - a Gymnasium-first, native (numpy) env over the same plant
dynamics as the browser AIO-Gym. Fast, synchronous, seedable, and vectorizable,
built for offline-data generation and online RL training (SAC / RLPD), where a
browser-coupled env would be too slow and too loosely-coupled.

Physics parity with the browser JS is enforced by tests/test_parity.py.

Contract (matches the browser RL contract so ONNX policies are interchangeable):
  obs    = [levels(n), temps(n), t_sp(n), h_sp(controlled k), t_cold, t_amb]
  action = [pumps..., valves..., heaters...] in [0, 1]   (direct-actuator mode)

reward_mode:
  "kpi"      (default) reward = -(instantaneous KPI penalty) using the same
             tracking + excess-energy + safety scoring the gym/browser display
             (scoring.py), so the RL optimizes exactly what it is judged on.
  "economic" CSTR production-maximisation (legacy economic demo).
  "track"    plain setpoint tracking (legacy).

dynamic=True injects within-episode disturbances (setpoint steps, cold-inlet
steps, ambient drift, demand surges) on top of domain-randomised start points.
The policy OBSERVES the changed conditions (t_cold / t_amb / setpoints are all in
obs), so this trains the online adaptation a fixed-tuning MPC can't match.
"""
from __future__ import annotations
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .models import make_model, obs_vector
from .kernel import Integrator
from .scoring import KPIScorer

# Advisory/interlock limits mirror frontend/js/sim/alarms.js (LIMITS).
T_HIGH, T_TRIP = 80.0, 92.0
H_HIGH_FRAC, H_LOW_FRAC, H_OVERFLOW_FRAC = 0.90, 0.15, 0.97
I_TEMP_MAX, I_LEVEL_MAX = 300.0, 8.0          # anti-windup clamp + obs normalizer for integral error

# ---- Environment wrapper ----
class AIOGymNativeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario="cascade", control_dt=0.5, episode_steps=600,
                 reward_mode="kpi", dynamic=True, randomize=True, randomize_setpoints=True,
                 randomize_plant=False, plant_drift=False, integral_obs=False, action_mode="actuator",
                 noise=False, noise_pct=0.01, custom_reward=None, custom_model=None,
                 terminate_on_runaway=False, reward_scale=0.03, w_prod=1000.0, w_energy=2.0, w_constraint=8.0):
        super().__init__()
        self.model = make_model(custom_model if custom_model is not None else scenario)
        self.scenario = self.model.scenario
        self.control_dt = float(control_dt)
        self.episode_steps = int(episode_steps)
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale          # keep Q-magnitudes sane -> stable critic
        self.dynamic = dynamic
        self.randomize_plant = randomize_plant    # per-episode operating-regime variation
        self.plant_drift = plant_drift            # slow within-episode parameter drift
        self.randomize = randomize
        self.randomize_setpoints = randomize_setpoints
        self.noise = noise                        # measurement noise on observed levels/temps
        self.noise_pct = noise_pct                # std as a fraction of the per-quantity scale
        self.custom_reward = custom_reward         # optional callable(env, levels, temps, act) -> float
        self.terminate_on_runaway = terminate_on_runaway
        # legacy economic-mode weights (CSTR)
        self.w_prod, self.w_energy, self.w_constraint = w_prod, w_energy, w_constraint

        self._p_nominal = {k: (list(v) if isinstance(v, list) else v) for k, v in self.model.p.items()}
        self._regime = copy.deepcopy(getattr(self.model, "plant_regime", {}))
        self._econ = copy.deepcopy(getattr(self.model, "economic_config", {}))
        self._disturbance_defaults = self.model.disturbance_defaults()
        self._disturbance_by_event = {
            row["event"]: row
            for row in self.model.disturbance_schema()
            if row.get("event")
        }
        self._reset_disturbance_values()
        self.integ = Integrator(self.model)
        self.scorer = KPIScorer(self.model)
        nP, nV, nH = self.model.actuator_counts()
        self.nP, self.nV, self.nH = nP, nV, nH
        self.nu = nP + nV + nH
        hsp, tsp = self.model.default_setpoints()
        self._hsp0 = [hsp.get(i, 0.0) for i in range(self.model.n)]
        self._tsp0 = list(tsp)
        self._tcold0 = float(self._disturbance_defaults.get("t_cold", self.model.p.get("t_cold", 15.0)))
        self._tamb0 = float(self._disturbance_defaults.get("t_amb", self.model.p.get("t_amb", 20.0)))
        self._caf0 = float(self._disturbance_defaults.get("Caf", self.model.p.get("Caf", 1.0)))
        self._tcool0 = float(self._disturbance_defaults.get("Tcool", self.model.p.get("Tcool", 10.0)))
        self._pump_flow_factor0 = float(self._disturbance_defaults.get("pump_flow_factor", 1.0))
        self._heater_efficiency0 = float(self._disturbance_defaults.get("heater_efficiency", 1.0))
        self._heat_loss_factor0 = float(self._disturbance_defaults.get("heat_loss_factor", 1.0))
        self._heat_load0 = list(self._disturbance_defaults.get("heat_load", [0.0, 0.0]))
        self._hvac_efficiency0 = float(self._disturbance_defaults.get("hvac_efficiency", 1.0))

        # integral-of-error obs (the I-term a memoryless policy otherwise lacks): lets
        # the RL policy do offset-free tracking like PID + adapt under operating-regime drift.
        self.integral_obs = integral_obs
        self.nctrl = len(self.model.controlled_levels())
        obs_dim = 3 * self.model.n + self.nctrl + 2
        if integral_obs:
            obs_dim += self.model.n + self.nctrl       # integral temp error + integral level error
        # supervisory (RL-on-PID): action = setpoints, an inner PID does the regulation.
        self.action_mode = action_mode
        if action_mode == "setpoint":
            self.layout = list(getattr(self.model, "supervisory_layout", ()))
            if not self.layout:
                raise ValueError(f"setpoint action_mode has no supervisory layout for '{self.scenario}'")
        else:
            self.layout = None
        if self.layout is not None:
            from .control.baselines import PIDAgent
            self.pid = PIDAgent(self.model)
            act_dim = len(self.layout)
        else:
            self.pid = None
            act_dim = self.nu
        self.action_space = spaces.Box(0.0, 1.0, (act_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self._k = 0

    # ---- helpers ----
    def _copy_disturbance_value(self, value):
        return copy.deepcopy(value)

    def _reset_disturbance_values(self):
        self._disturbance_values = {
            name: self._copy_disturbance_value(value)
            for name, value in self._disturbance_defaults.items()
        }

    def _sync_known_disturbances(self):
        known = {
            "t_cold": self.t_cold,
            "t_amb": self.t_amb,
            "extra_outflow": self.extra_outflow,
            "Caf": self.caf,
            "Tcool": self.tcool,
            "pump_flow_factor": self.pump_flow_factor,
            "heater_efficiency": self.heater_efficiency,
            "heat_loss_factor": self.heat_loss_factor,
            "heat_load": list(self.heat_load),
            "hvac_efficiency": self.hvac_efficiency,
        }
        for name in self._disturbance_defaults:
            if name in known:
                self._disturbance_values[name] = self._copy_disturbance_value(known[name])

    def _env(self):
        env = {
            name: self._copy_disturbance_value(value)
            for name, value in self._disturbance_values.items()
        }
        env.update({"t_cold": self.t_cold, "t_amb": self.t_amb, "extra_outflow": self.extra_outflow})
        if self.scenario == "cstr":
            env.update({"Caf": self.caf, "Tcool": self.tcool})
        elif self.scenario in ("cascade", "quadruple"):
            env.update({"pump_flow_factor": self.pump_flow_factor,
                        "heater_efficiency": self.heater_efficiency,
                        "heat_loss_factor": self.heat_loss_factor})
        elif self.scenario == "hvac":
            env.update({"heat_load": list(self.heat_load), "hvac_efficiency": self.hvac_efficiency})
        return env

    def _split(self, action):
        a = np.clip(np.asarray(action, np.float64), 0.0, 1.0)
        return self.model.action_vector_to_dict(a)

    def _obs(self):
        out = self.model.outputs(self.integ.x)
        levels, temps = out["levels"], out["temps"]
        if self.noise:                            # measurement noise on observed state (reward uses true state)
            rng = self.np_random
            levels = [l + float(rng.normal(0, self.noise_pct * 0.5)) for l in levels]
            temps = [t + float(rng.normal(0, self.noise_pct * 10.0)) for t in temps]
        o = obs_vector(self.model, levels, temps, self.t_cold, self.t_amb, self.h_sp, self.t_sp)
        if self.integral_obs:
            o = o + [it / I_TEMP_MAX for it in self._itemp] + [il / I_LEVEL_MAX for il in self._ilevel]
        return np.asarray(o, dtype=np.float32)

    def _accumulate_integral(self):
        out = self.model.outputs(self.integ.x)
        levels, temps = out["levels"], out["temps"]
        ctrl = self.model.controlled_levels()
        dt = self.control_dt
        self._itemp = [float(np.clip(self._itemp[i] + (self.t_sp[i] - temps[i]) * dt, -I_TEMP_MAX, I_TEMP_MAX))
                       for i in range(self.model.n)]
        self._ilevel = [float(np.clip(self._ilevel[j] + (self.h_sp[i] - levels[i]) * dt, -I_LEVEL_MAX, I_LEVEL_MAX))
                        for j, i in enumerate(ctrl)]

    # ---- operating-regime variation ----
    def _restore_nominal(self):
        for k, v in self._p_nominal.items():
            self.model.p[k] = (list(v) if isinstance(v, list) else v)

    def _apply_regime(self):
        """Scale plant params by per-episode multipliers (fouling / ageing / gain drift)."""
        self._regime_mult = self._sample_regime_mult()
        self._apply_mult(self._regime_mult)

    def _sample_regime_mult(self):
        rng = self.np_random
        return {
            k: float(rng.uniform(lo, hi))
            for k, (lo, hi) in self._regime.items()
            if k in self._p_nominal
        }

    def _init_regime_state(self):
        self._regime_mult = {k: 1.0 for k in self._regime if k in self._p_nominal}
        self._regime_target = dict(self._regime_mult)

    def _apply_plant_drift(self):
        if not self.plant_drift or not self._regime_mult:
            return
        if not self._regime_target:
            self._regime_target = self._sample_regime_mult()
        alpha = min(0.03, max(0.002, 4.0 / max(1, self.episode_steps)))
        next_mult = {}
        settled = True
        for k, current in self._regime_mult.items():
            target = self._regime_target.get(k, current)
            value = current + alpha * (target - current)
            lo, hi = self._regime[k]
            value = float(np.clip(value, lo, hi))
            next_mult[k] = value
            settled = settled and abs(value - target) <= 0.01 * max(1.0, abs(target))
        self._regime_mult = next_mult
        if settled:
            self._regime_target = self._sample_regime_mult()
        self._apply_mult(self._regime_mult)

    def _apply_mult(self, mult):
        for k, m in mult.items():
            nom = self._p_nominal[k]
            self.model.p[k] = [x * m for x in nom] if isinstance(nom, list) else nom * m

    # ---- disturbance scheduler (the "adaptation" dimension) ----
    def _disturbance_names(self):
        return [
            row["event"]
            for row in self.model.disturbance_schema()
            if row.get("dynamic", False) and row.get("event")
        ]

    def _schedule_disturbances(self):
        self._dist_events = []
        if not self.dynamic:
            return
        rng = self.np_random
        names = self._disturbance_names()
        if not names:
            return
        for _ in range(int(rng.integers(1, 4))):
            t = int(rng.integers(int(0.15 * self.episode_steps), max(2, self.episode_steps)))
            self._dist_events.append((t, names[int(rng.integers(0, len(names)))]))

    def _apply_disturbance(self, event):
        rng = self.np_random
        if event == "cold_inlet_step":
            self.t_cold = float(np.clip(self._tcold0 + rng.uniform(-8, 8), 2, 35))
        elif event == "ambient_step":
            self.t_amb = float(np.clip(self._tamb0 + rng.uniform(-8, 12), 0, 40))
        elif event == "demand_surge":
            self.extra_outflow = float(abs(rng.uniform(0, 8e-4)))
        elif event == "feed_concentration_step":
            self.caf = float(np.clip(self._caf0 + rng.uniform(-0.25, 0.35), 0.2, 2.0))
        elif event == "coolant_temperature_step":
            self.tcool = float(np.clip(self._tcool0 + rng.uniform(-8, 10), -5, 35))
        elif event == "pump_capacity_shift":
            self.pump_flow_factor = float(np.clip(self._pump_flow_factor0 + rng.uniform(-0.30, 0.30), 0.6, 1.3))
        elif event == "heater_efficiency_shift":
            self.heater_efficiency = float(np.clip(self._heater_efficiency0 + rng.uniform(-0.35, 0.15), 0.55, 1.15))
        elif event == "heat_loss_shift":
            self.heat_loss_factor = float(np.clip(self._heat_loss_factor0 + rng.uniform(-0.4, 1.2), 0.5, 2.4))
        elif event == "internal_heat_load_step":
            self.heat_load = [float(rng.uniform(-200, 900)), float(rng.uniform(-200, 900))]
        elif event == "hvac_efficiency_shift":
            self.hvac_efficiency = float(np.clip(self._hvac_efficiency0 + rng.uniform(-0.35, 0.20), 0.6, 1.2))
        elif event == "setpoint_move":
            for i in self.model.controlled_levels():
                self.h_sp[i] = float(np.clip(self.h_sp[i] * (1 + 0.15 * rng.uniform(-1, 1)), 0.15, 0.70))
            self.t_sp = [float(np.clip(t * (1 + 0.10 * rng.uniform(-1, 1)), 15.0, 85.0)) for t in self.t_sp]
        else:
            self._apply_schema_disturbance(event)
        self._sync_known_disturbances()

    def _apply_schema_disturbance(self, event):
        row = self._disturbance_by_event.get(event)
        if not row or row.get("kind") == "setpoint":
            return
        name = row.get("name")
        if not name:
            return
        default = self._disturbance_values.get(name, self._disturbance_defaults.get(name, row.get("default", 0.0)))
        self._disturbance_values[name] = self._sample_schema_disturbance(default, row)

    def _sample_schema_disturbance(self, default, row):
        rng = self.np_random
        if row.get("values"):
            values = list(row["values"])
            return self._copy_disturbance_value(values[int(rng.integers(0, len(values)))])
        bounds = row.get("bounds")
        if (
            isinstance(bounds, (tuple, list))
            and len(bounds) == 2
            and bounds[0] is not None
            and bounds[1] is not None
        ):
            lo, hi = float(bounds[0]), float(bounds[1])
            if isinstance(default, (list, tuple)):
                return [float(rng.uniform(lo, hi)) for _ in default]
            return float(rng.uniform(lo, hi))
        return self._copy_disturbance_value(default)

    def _reward_done(self, act):
        out = self.model.outputs(self.integ.x)
        levels, temps = out["levels"], out["temps"]
        hmax = self.model.height_max
        ctrl = self.model.controlled_levels()
        # tracking error, normalised (level scale 0.1 m, temp scale 10 degC) for info/legacy
        track = sum(abs(levels[i] - self.h_sp[i]) / 0.1 for i in ctrl)
        track += sum(abs(temps[i] - self.t_sp[i]) / 10.0 for i in range(self.model.n))
        con = 0.0
        for i, h in enumerate(levels):
            if h > H_HIGH_FRAC * hmax[i]:
                con += (h - H_HIGH_FRAC * hmax[i]) / (0.1 * hmax[i])
            elif h < H_LOW_FRAC * hmax[i]:
                con += (H_LOW_FRAC * hmax[i] - h) / (0.1 * hmax[i])
        for T in temps:
            if T > T_HIGH:
                con += (T - T_HIGH) / 10.0
        runaway = any(T > T_TRIP for T in temps) or any(levels[i] > H_OVERFLOW_FRAC * hmax[i] for i in range(len(levels)))

        # Always accumulate the KPI scorer (independent of reward_mode) so any agent
        # PID / MPC / RL can be ranked by env.scorer.report()["score"].
        heat_w = self.model.heater_power(act)
        ideal_w = self.model.ideal_power(levels, temps, self.t_sp, self._env(), act)
        pen = self.scorer.step_penalty(levels, temps, self.h_sp, self.t_sp,
                                       heat_w, ideal_w, runaway, self.control_dt)

        prod = 0.0
        if self.reward_mode == "economic":
            profit, prod = self._economic_profit(act, levels, temps, runaway)
            reward = profit * self.reward_scale          # scaled for stable critic; profit reported raw
        elif self.reward_mode == "kpi":
            reward = -pen * self.reward_scale            # -(instantaneous KPI penalty)
            profit = 0.0
        else:
            energy = sum(act["heaters"]) + 0.3 * sum(act["pumps"])
            reward = -(track + 0.03 * energy + 5.0 * con)
            profit = 0.0

        if self.custom_reward is not None:               # user-supplied reward overrides
            reward = float(self.custom_reward(self, levels, temps, act))

        terminated = bool(self.terminate_on_runaway and runaway)
        if terminated:
            reward -= 50.0
        # per-constraint violation amounts (PC-Gym-style tracking for safe-RL benchmarking)
        cons_info = {"temp_high": max((T - T_HIGH for T in temps), default=0.0),
                     "temp_trip": max((T - T_TRIP for T in temps), default=0.0),
                     "level_high": max((levels[i] - H_HIGH_FRAC * hmax[i] for i in range(len(levels))), default=0.0),
                     "level_low": max((H_LOW_FRAC * hmax[i] - levels[i] for i in range(len(levels))), default=0.0)}
        process_extra = {}
        if self.scenario == "cstr":
            ca = float(self.integ.x[0])
            cons_info.update({
                "cstr_ca_high": max(0.0, ca - 1.5),
                "cstr_ca_low": max(0.0, -ca),
                "cstr_temp_low": max(0.0, -temps[0]),
            })
            process_extra = {
                "cstr_feed_conc": self.caf,
                "cstr_coolant_temp": self.tcool,
                "cstr_conversion": self.model.conversion(self.integ.x, self._env()),
            }
        elif self.scenario in ("cascade", "quadruple"):
            process_extra = {
                "pump_flow_factor": self.pump_flow_factor,
                "heater_efficiency": self.heater_efficiency,
                "heat_loss_factor": self.heat_loss_factor,
            }
        elif self.scenario == "hvac":
            lo_viol = max((20.0 - T for T in temps), default=0.0)
            hi_viol = max((T - 24.0 for T in temps), default=0.0)
            cons_info.update({"hvac_comfort_low": max(0.0, lo_viol), "hvac_comfort_high": max(0.0, hi_viol)})
            process_extra = {
                "hvac_heat_load": list(self.heat_load),
                "hvac_efficiency": self.hvac_efficiency,
            }
        heat_kw = self.model.heater_power(act) / 1000.0
        pump_kw = self.model.pump_power(act) / 1000.0
        info = {"track": track, "constraint": con, "prod": prod, "profit": profit,
                "heat_kw": heat_kw, "pump_kw": pump_kw, "energy_kw": heat_kw + pump_kw,
                "runaway": runaway, "cons_info": cons_info, "cons_violated": any(v > 0 for v in cons_info.values()),
                "levels": levels, "temps": temps}
        if self.randomize_plant or self.plant_drift:
            info["plant_mult"] = dict(getattr(self, "_regime_mult", {}))
        if process_extra:
            info.update(process_extra)
        return float(reward), terminated, info

    def _economic_profit(self, act, levels, temps, runaway):
        """Economic objective: value minus energy cost minus soft-band violation.

        The optimum hugs a band/constraint edge that drifts with operating regime, so fixed-SP control is
        suboptimal. Returns (profit, production)."""
        cfg = self._econ
        ctrl = self.model.controlled_levels()
        value = prod = 0.0
        if cfg["value"] == "production" and hasattr(self.model, "production"):
            prod = self.model.production(self.integ.x, act, self._env())
            value = prod
        energy_kw = self.model.heater_power(act) / 1000.0
        viol = 0.0
        for i, (lo, hi) in enumerate(cfg["temp_band"]):
            if lo is not None and temps[i] < lo:
                viol += (lo - temps[i]) / 10.0
            if hi is not None and temps[i] > hi:
                viol += (temps[i] - hi) / 10.0
        for j, i in enumerate(ctrl):
            lo, hi = cfg["level_band"][j]
            if lo is not None and levels[i] < lo:
                viol += (lo - levels[i]) / 0.1
            if hi is not None and levels[i] > hi:
                viol += (levels[i] - hi) / 0.1
        profit = cfg["w_value"] * value - cfg["w_energy"] * energy_kw - cfg["w_viol"] * viol
        if runaway:
            profit -= 50.0
        return profit, prod

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self._restore_nominal()
        self._init_regime_state()
        if self.randomize_plant:
            self._apply_regime()        # this episode's operating regime
            self._regime_target = self._sample_regime_mult()
        elif self.plant_drift:
            self._regime_target = self._sample_regime_mult()
        x0 = list(self.model.initial_state())
        self.h_sp = list(self._hsp0)
        self.t_sp = list(self._tsp0)
        self._reset_disturbance_values()
        self.t_cold = float(self._tcold0)
        self.t_amb = float(self._tamb0)
        self.caf = float(self._caf0)
        self.tcool = float(self._tcool0)
        self.pump_flow_factor = float(self._pump_flow_factor0)
        self.heater_efficiency = float(self._heater_efficiency0)
        self.heat_loss_factor = float(self._heat_loss_factor0)
        self.heat_load = list(self._heat_load0)
        self.hvac_efficiency = float(self._hvac_efficiency0)
        self.extra_outflow = 0.0
        if self.randomize:
            for j in range(len(x0)):
                x0[j] *= 1.0 + 0.08 * float(rng.uniform(-1, 1))
            self.t_cold = float(np.clip(self._tcold0 + rng.uniform(-5, 5), 2, 35))
            self.t_amb = float(np.clip(self._tamb0 + rng.uniform(-5, 8), 0, 40))
        self._sync_known_disturbances()
        if self.randomize_setpoints:
            for i in self.model.controlled_levels():
                self.h_sp[i] = float(np.clip(self.h_sp[i] * (1 + 0.15 * rng.uniform(-1, 1)), 0.15, 0.70))
            self.t_sp = [float(np.clip(t * (1 + 0.10 * rng.uniform(-1, 1)), 15.0, 85.0)) for t in self.t_sp]
        self.integ.reset(x0)
        self.scorer.reset()
        if self.pid is not None:
            self.pid.reset()
        self._itemp = [0.0] * self.model.n
        self._ilevel = [0.0] * self.nctrl
        self._k = 0
        self._schedule_disturbances()
        return self._obs(), {}

    def default_sp_action(self):
        """Normalized supervisory action that reproduces the default setpoints (= the
        fixed-SP PID baseline), the offline prior to learn from."""
        if self.layout is None:
            return None
        a = []
        for spec in self.layout:
            lo, hi = spec[-2], spec[-1]
            if spec[0] == "t_sp":
                v = self._tsp0[spec[1]]
            elif spec[0] == "h_sp":
                v = self._hsp0[spec[1]] or 0.45
            else:
                v = lo + 0.7 * (hi - lo)
            a.append(float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)))
        return np.array(a, np.float32)

    def _meas(self):
        """buildState-like dict the inner PID reads (true state)."""
        return self.model.measurement(self.integ.x, {"t_cold": self.t_cold, "t_amb": self.t_amb})

    def _supervise(self, action):
        """Supervisory action = normalized setpoints -> set SPs, inner PID regulates
        to them; unregulated economic MVs ('mv') are applied directly."""
        a = np.clip(np.asarray(action, np.float64), 0.0, 1.0)
        mv = {}
        for i, spec in enumerate(self.layout):
            lo, hi = spec[-2], spec[-1]
            val = lo + float(a[i]) * (hi - lo)
            if spec[0] == "t_sp":
                self.t_sp[spec[1]] = val
            elif spec[0] == "h_sp":
                self.h_sp[spec[1]] = val
            else:                                   # ("mv", kind, idx, lo, hi)
                mv[(spec[1], spec[2])] = val
        act = self.pid.compute(self._meas(), {"h_sp": self.h_sp, "t_sp": self.t_sp}, self.control_dt)
        for (kind, idx), v in mv.items():
            act[kind][idx] = v
        return {"pumps": list(act["pumps"]), "valves": list(act["valves"]), "heaters": list(act["heaters"])}

    def step(self, action):
        act = self._supervise(action) if self.pid is not None else self._split(action)
        for (t, event) in self._dist_events:
            if t == self._k:
                self._apply_disturbance(event)
        self._apply_plant_drift()
        self.integ.step(self.control_dt, act, self._env())
        self._accumulate_integral()
        self._k += 1
        reward, terminated, info = self._reward_done(act)
        truncated = self._k >= self.episode_steps
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        pass
