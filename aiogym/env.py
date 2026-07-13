"""AIOGymNativeEnv - a Gymnasium-first, native (numpy) env over the same plant
dynamics as the browser AIO-Gym. Fast, synchronous, seedable, and vectorizable,
built for offline-data generation and online RL training (SAC / RLPD), where a
browser-coupled env would be too slow and too loosely-coupled.

Physics parity with the browser JS is enforced by tests/test_parity.py.

Contract:
  generic obs    = [x, y_sp, disturbances]
  generic action = u in [0, 1]
  legacy built-ins keep the browser-compatible physical observation/action adapter.

reward_mode:
  "kpi"      (default) reward = -(instantaneous KPI penalty) using the same
             tracking + excess-energy + safety KPI the gym/browser display
             (evaluation.metrics.kpi), so the RL optimizes exactly what it is judged on.
  "economic" CSTR production-maximisation (legacy economic demo).
  "tracking" PC-Gym-style setpoint tracking: reward = -(normalized squared
             SP error + input move penalty).

dynamic=True injects within-episode disturbances (setpoint steps, cold-inlet
steps, ambient drift, demand surges) on top of domain-randomised start points.
The policy OBSERVES the changed conditions (t_cold / t_amb / setpoints are all in
obs), so this trains the online adaptation a fixed-tuning MPC can't match.
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .models import apply_model_params, make_model
from .models import Integrator
from .models.adapters import BrowserModelAdapter
from .evaluation.metrics.kpi import KPIScorer
from .objectives import stage_reward
from ._deprecations import warn_deprecated

# Advisory/interlock limits mirror frontend/js/sim/alarms.js (LIMITS).
T_HIGH, T_TRIP = 80.0, 92.0
H_HIGH_FRAC, H_LOW_FRAC, H_OVERFLOW_FRAC = 0.90, 0.15, 0.97
I_TEMP_MAX, I_LEVEL_MAX = 300.0, 8.0          # anti-windup clamp + obs normalizer for integral error


def _validated_range(name, value):
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name} must contain two numeric bounds") from ex
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two bounds")
    if not all(np.isfinite(item) for item in values):
        raise ValueError(f"{name} bounds must be finite")
    if values[0] > values[1]:
        raise ValueError(f"{name} lower bound must not exceed its upper bound")
    return values


# ---- Environment wrapper ----
class AIOGymNativeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario="cascade", control_dt=0.5, episode_steps=600,
                 reward_mode="kpi", dynamic=True, randomize=True, randomize_setpoints=True,
                 randomize_plant=False, plant_drift=False, integral_obs=False, action_mode="actuator",
                 noise=False, noise_pct=0.01, custom_reward=None, custom_stage_reward=None,
                 custom_model=None, model_params=None,
                 terminate_on_runaway=False, reward_scale=0.03, w_prod=1000.0, w_energy=2.0, w_constraint=8.0,
                 tracking_q_y=1.0, tracking_r_move=0.05,
                 crystal_ln_sp=None, crystal_cv_sp=None, crystal_random_targets=False,
                 crystal_ln_range=(10.0, 11.5), crystal_cv_range=(0.75, 0.95)):
        super().__init__()
        self.model = apply_model_params(
            make_model(custom_model if custom_model is not None else scenario),
            model_params,
        )
        self._browser_adapter = BrowserModelAdapter(self.model)
        self.scenario = self.model.scenario
        self.control_dt = float(control_dt)
        self.episode_steps = int(episode_steps)
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        if reward_mode == "track":
            warn_deprecated('reward_mode="track"', 'reward_mode="tracking"')
            reward_mode = "tracking"
        if reward_mode not in {"kpi", "economic", "tracking"}:
            raise ValueError("reward_mode must be one of: economic, kpi, tracking")
        if action_mode not in {"actuator", "setpoint"}:
            raise ValueError("action_mode must be one of: actuator, setpoint")
        self.noise_pct = float(noise_pct)
        if not np.isfinite(self.noise_pct) or self.noise_pct < 0:
            raise ValueError("noise_pct must be finite and non-negative")
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale          # keep Q-magnitudes sane -> stable critic
        self.tracking_q_y = self._resolve_tracking_q_y(tracking_q_y)
        self.tracking_r_move = self._nonnegative("tracking_r_move", tracking_r_move)
        self.dynamic = dynamic
        self.randomize_plant = randomize_plant    # per-episode operating-regime variation
        self.plant_drift = plant_drift            # slow within-episode parameter drift
        self.randomize = randomize
        self.randomize_setpoints = randomize_setpoints
        self.noise = noise                        # measurement noise on observed levels/temps
        # Measurement noise standard deviation as a fraction of each quantity scale.
        if custom_reward is not None and not callable(custom_reward):
            raise TypeError("custom_reward must be callable")
        if custom_stage_reward is not None and not callable(custom_stage_reward):
            raise TypeError("custom_stage_reward must be callable")
        if custom_reward is not None and custom_stage_reward is not None:
            raise ValueError("custom_reward and custom_stage_reward cannot be used together")
        self.custom_reward = custom_reward         # legacy callable(env, levels, temps, act)
        self.custom_stage_reward = custom_stage_reward
        self.terminate_on_runaway = terminate_on_runaway
        # legacy economic-mode weights (CSTR)
        self.w_prod, self.w_energy, self.w_constraint = w_prod, w_energy, w_constraint
        self.crystal_ln_sp = crystal_ln_sp
        self.crystal_cv_sp = crystal_cv_sp
        self.crystal_random_targets = bool(crystal_random_targets)
        self.crystal_ln_range = _validated_range("crystal_ln_range", crystal_ln_range)
        self.crystal_cv_range = _validated_range("crystal_cv_range", crystal_cv_range)
        self._model_env_options = {
            "crystal_ln_sp": crystal_ln_sp,
            "crystal_cv_sp": crystal_cv_sp,
            "crystal_random_targets": bool(crystal_random_targets),
            "crystal_ln_range": self.crystal_ln_range,
            "crystal_cv_range": self.crystal_cv_range,
        }

        self._p_nominal = {k: (list(v) if isinstance(v, list) else v) for k, v in self.model.p.items()}
        self._regime = copy.deepcopy(getattr(self.model, "plant_regime", {}))
        self._econ = copy.deepcopy(getattr(self.model, "economic_config", {}))
        self._disturbance_defaults = self.model.runtime_env(self.model.disturbance_defaults())
        self._disturbance_attrs = self.model.disturbance_attribute_map()
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
        self.nu = self.model.action_dim()
        y_sp = list(self.model.env_setpoint_vector(self._model_env_options))
        self._ysp0 = list(y_sp)

        # integral-of-error obs (the I-term a memoryless policy otherwise lacks): lets
        # the RL policy do offset-free tracking like PID + adapt under operating-regime drift.
        self.integral_obs = integral_obs
        self.nctrl = len(self.model.legacy_observation_level_target_slots())
        model_obs = self.model.env_observation(
            self.model.initial_state(),
            self.model.default_action(),
            self.model.runtime_env(self._disturbance_defaults),
            self._ysp0,
        )
        self._uses_model_observation = model_obs is not None
        if self._uses_model_observation:
            obs_dim = len(model_obs)
        elif self._browser_adapter.enabled:
            obs_dim = self._browser_adapter.observation_dim
        else:
            obs_dim = len(self.model.initial_state()) + len(self._ysp0) + len(self.model.dynamics_disturbance_names())
        if integral_obs and self.model.supports_integral_observation:
            obs_dim += len(self._ysp0)       # integral controlled-output error
        # supervisory (RL-on-PID): action = setpoints, an inner PID does the regulation.
        self.action_mode = action_mode
        if action_mode == "setpoint":
            self.layout = list(getattr(self.model, "supervisory_layout", ()))
            if not self.layout:
                raise ValueError(f"setpoint action_mode has no supervisory layout for '{self.scenario}'")
        else:
            self.layout = None
        if self.layout is not None:
            from .controllers.pid import PIDAgent
            self.pid = PIDAgent(self.model)
            act_dim = len(self.layout)
        else:
            self.pid = None
            act_dim = self.nu
        self.action_space = spaces.Box(0.0, 1.0, (act_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self._k = 0
        self.last_act = self.model.default_action()
        self.previous_act = copy.deepcopy(self.last_act)

    def _resolve_tracking_q_y(self, q_y):
        values = q_y if isinstance(q_y, (list, tuple)) else [q_y]
        values = [float(value) for value in values]
        if len(values) == 1:
            values *= len(self.model.controlled_output(self.model.initial_state()))
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("tracking_q_y values must be finite and non-negative")
        return values

    def _nonnegative(self, name, value):
        number = float(value)
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return number

    # ---- helpers ----
    def _copy_disturbance_value(self, value):
        return copy.deepcopy(value)

    def _reset_disturbance_values(self):
        self._disturbance_values = {
            name: self._copy_disturbance_value(value)
            for name, value in self._disturbance_defaults.items()
        }
        for name, attr in self._disturbance_attrs.items():
            if name in self._disturbance_values:
                setattr(self, attr, self._copy_disturbance_value(self._disturbance_values[name]))

    def _set_disturbance_value(self, name, value):
        self._disturbance_values[name] = self._copy_disturbance_value(value)
        attr = self._disturbance_attrs.get(name)
        if attr:
            setattr(self, attr, self._copy_disturbance_value(value))

    def _sync_known_disturbances(self):
        for name in self._disturbance_defaults:
            attr = self._disturbance_attrs.get(name)
            if attr and hasattr(self, attr):
                self._disturbance_values[name] = self._copy_disturbance_value(getattr(self, attr))

    def _env(self):
        self._sync_known_disturbances()
        return self.model.runtime_env(self._disturbance_values)

    def _legacy_obs_setpoints(self):
        return self._browser_adapter.setpoints(self.y_sp)

    def legacy_observation_setpoints(self):
        """Return browser-observation target slices derived from generic y_sp."""
        return self._legacy_obs_setpoints()

    def _split(self, action):
        a = np.clip(np.asarray(action, np.float64), 0.0, 1.0)
        return self._browser_adapter.action(a)

    def _obs(self):
        if self._uses_model_observation:
            o = self.model.env_observation(self.integ.x, self.last_act, self._env(), self.y_sp)
            return np.asarray(o, dtype=np.float32)
        if not self._browser_adapter.enabled:
            o = self.model.state_vector(self.integ.x) + list(self.y_sp) + list(self.model.disturbance_vector(self._env()))
            if self.integral_obs:
                o = o + [iy / I_TEMP_MAX for iy in self._iy]
            return np.asarray(o, dtype=np.float32)
        out = self.model.outputs(self.integ.x)
        levels, temps = out["levels"], out["temps"]
        if self.noise:                            # measurement noise on observed state (reward uses true state)
            rng = self.np_random
            levels = [l + float(rng.normal(0, self.noise_pct * 0.5)) for l in levels]
            temps = [t + float(rng.normal(0, self.noise_pct * 10.0)) for t in temps]
        o = self._browser_adapter.observation(levels, temps, self.t_cold, self.t_amb, self.y_sp)
        if self.integral_obs:
            o = o + [iy / I_TEMP_MAX for iy in self._iy]
        return np.asarray(o, dtype=np.float32)

    def _accumulate_integral(self):
        if not self.model.supports_integral_observation:
            return
        out = self.model.outputs(self.integ.x)
        y = list(out.get("y", self.model.controlled_output(self.integ.x)))
        errors = [self.y_sp[i] - y[i] if i < len(y) else 0.0 for i in range(len(self.y_sp))]
        dt = self.control_dt
        self._iy = [float(np.clip(self._iy[i] + errors[i] * dt, -I_TEMP_MAX, I_TEMP_MAX)) for i in range(len(errors))]

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
        if event == "setpoint_move":
            self._randomize_setpoints(rng)
        else:
            row = self._disturbance_by_event.get(event)
            if row and row.get("kind") != "setpoint":
                name = row.get("name")
                default = self._disturbance_defaults.get(name, row.get("default", 0.0))
                self._set_disturbance_value(name, self.model.sample_disturbance(event, default, rng))

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

    def evaluate_transition(self, state, action, next_state, *, setpoint=None,
                            disturbance=None, previous_action=None):
        """Evaluate a predicted actuator transition without advancing the environment.

        Omitted context values use the environment's current snapshot. During a
        multi-step rollout, pass each candidate action as ``previous_action`` for
        the following transition so the move penalty follows the candidate path.
        """

        if self.action_mode != "actuator":
            raise ValueError(
                "evaluate_transition requires action_mode='actuator'; setpoint actions "
                "depend on the stateful inner PID"
            )
        if self.custom_reward is not None:
            raise ValueError(
                "evaluate_transition cannot use stateful custom_reward; "
                "configure custom_stage_reward instead"
            )
        action = self._validated_action(action)
        return self._evaluate_model_transition(
            state,
            self._split(action),
            next_state,
            setpoint=setpoint,
            disturbance=disturbance,
            previous_action=previous_action,
        )

    def _evaluate_model_transition(self, state, action, next_state, *, setpoint=None,
                                   disturbance=None, previous_action=None):
        return stage_reward(
            self.model,
            state,
            action,
            next_state,
            setpoint=self.y_sp if setpoint is None else setpoint,
            disturbance=self._env() if disturbance is None else disturbance,
            previous_action=self.previous_act if previous_action is None else previous_action,
            reward_mode=self.reward_mode,
            reward_scale=self.reward_scale,
            tracking_q_y=self.tracking_q_y,
            tracking_r_move=self.tracking_r_move,
            terminate_on_runaway=self.terminate_on_runaway,
            economic_config=self._econ,
            reward_override=self.custom_stage_reward,
        )

    def _reward_done(self, state, act):
        result = self._evaluate_model_transition(state, act, self.integ.x)
        self.scorer.accumulate(result.kpi, self.control_dt)

        reward = result.reward
        info = dict(result.info)
        if self.custom_reward is not None:               # legacy stateful override
            reward = float(self.custom_reward(self, info["levels"], info["temps"], act))
            if result.terminated:
                reward -= 50.0
        if self.randomize_plant or self.plant_drift:
            info["plant_mult"] = dict(getattr(self, "_regime_mult", {}))
        return float(reward), result.terminated, info

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
        self.y_sp = list(self._ysp0)
        self._reset_disturbance_values()
        if self.randomize:
            for j in range(len(x0)):
                x0[j] *= 1.0 + 0.08 * float(rng.uniform(-1, 1))
            if self.model.randomize_common_temperatures:
                if "t_cold" in self._disturbance_values:
                    base = float(self._disturbance_defaults["t_cold"])
                    self._set_disturbance_value("t_cold", float(np.clip(base + rng.uniform(-5, 5), 2, 35)))
                if "t_amb" in self._disturbance_values:
                    base = float(self._disturbance_defaults["t_amb"])
                    self._set_disturbance_value("t_amb", float(np.clip(base + rng.uniform(-5, 8), 0, 40)))
        self._sync_known_disturbances()
        self.y_sp = self.model.sample_env_setpoints(self.y_sp, rng, self._model_env_options)
        if self.randomize_setpoints and self.model.supports_generic_setpoint_randomization:
            self._randomize_setpoints(rng)
        self.integ.reset(x0)
        self.scorer.reset()
        if self.pid is not None:
            self.pid.reset()
        self._iy = [0.0] * len(self.y_sp)
        self._k = 0
        self.last_act = self.model.default_action()
        self.previous_act = copy.deepcopy(self.last_act)
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
            else:                                   # ("mv", kind, idx, lo, hi)
                mv[(spec[1], spec[2])] = val
        act = self.pid.compute(self._meas(), {"y_sp": self.y_sp}, self.control_dt)
        for (kind, idx), v in mv.items():
            act[kind][idx] = v
        return {"pumps": list(act["pumps"]), "valves": list(act["valves"]), "heaters": list(act["heaters"])}

    def _validated_action(self, action):
        try:
            values = np.asarray(action, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as ex:
            raise ValueError("action must be a numeric vector") from ex
        expected = int(self.action_space.shape[0])
        if values.size != expected:
            raise ValueError(f"action must contain {expected} values, got {values.size}")
        if not np.all(np.isfinite(values)):
            raise ValueError("action values must be finite")
        return values

    def step(self, action):
        action = self._validated_action(action)
        act = self._supervise(action) if self.pid is not None else self._split(action)
        state = list(self.integ.x)
        self.last_act = act
        for (t, event) in self._dist_events:
            if t == self._k:
                self._apply_disturbance(event)
        self._apply_plant_drift()
        self.integ.step(self.control_dt, act, self._env())
        self._accumulate_integral()
        self._k += 1
        reward, terminated, info = self._reward_done(state, act)
        self.previous_act = copy.deepcopy(act)
        truncated = self._k >= self.episode_steps
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        pass


def make_env(model: str = "cascade", objective: str | Mapping[str, Any] | None = None,
             seed: int | None = None, config: str | Path | Mapping[str, Any] | None = None,
             *, protocol: str | Mapping[str, Any] | None = None,
             **overrides) -> AIOGymNativeEnv:
    """Compatibility entrypoint for :mod:`aiogym.env_factory`."""

    warn_deprecated("aiogym.env.make_env", "aiogym.make_env")

    from .env_factory import make_env as _make_env

    return _make_env(
        model=model,
        objective=objective,
        seed=seed,
        config=config,
        protocol=protocol,
        **overrides,
    )
