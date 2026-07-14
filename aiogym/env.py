"""AIOGymNativeEnv - a Gymnasium-first native process-control environment.

Fast, synchronous, seedable, and vectorizable for benchmark evaluation,
offline-data generation, and online RL training.

Contract:
  generic obs    = [x, y_sp, disturbances]
  generic action = u in [0, 1]

reward_mode:
  "kpi"      (default) reward = -(instantaneous KPI penalty) using the same
             tracking + excess-energy + safety KPI
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

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .models import apply_model_params, make_model
from .models import Integrator
from .evaluation.metrics.kpi import KPIScorer
from .evaluation.objectives import stage_reward

# Shared advisory/interlock limits for process safety scoring.
T_HIGH, T_TRIP = 80.0, 92.0
H_HIGH_FRAC, H_LOW_FRAC, H_OVERFLOW_FRAC = 0.90, 0.15, 0.97
I_TEMP_MAX, I_LEVEL_MAX = 300.0, 8.0          # anti-windup clamp + obs normalizer for integral error

_DIRECT_ENV_DEFAULTS = {
    "dynamic": True,
    "randomize": True,
    "randomize_setpoints": True,
    "randomize_plant": False,
    "plant_drift": False,
    "integral_obs": False,
    "action_mode": "actuator",
    "noise": False,
    "noise_pct": 0.01,
    "terminate_on_runaway": False,
}


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

    def __init__(self, scenario="cascade", control_dt=None, episode_steps=None, task=None,
                 reward_mode="kpi", dynamic=None, randomize=None, randomize_setpoints=None,
                 randomize_plant=None, plant_drift=None, integral_obs=None, action_mode=None,
                 noise=None, noise_pct=None, custom_stage_reward=None,
                 custom_model=None, model_params=None,
                 terminate_on_runaway=None, reward_scale=0.03, w_prod=1000.0, w_energy=2.0, w_constraint=8.0,
                 tracking_q_y=1.0, tracking_r_move=0.05,
                 crystal_ln_sp=None, crystal_cv_sp=None, crystal_random_targets=False,
                 crystal_ln_range=(10.0, 11.5), crystal_cv_range=(0.75, 0.95)):
        super().__init__()
        base_model = make_model(custom_model if custom_model is not None else scenario)
        self.scenario = base_model.scenario
        self.task_profile = None
        task_defaults = {}
        if task is not None:
            from .evaluation.task_profiles import load_task_profile, task_environment

            self.task_profile = load_task_profile(task, scenario=self.scenario)
            task_defaults = task_environment(self.task_profile)
        resolved_model_params = dict(
            (self.task_profile or {}).get("model_params", {})
        )
        resolved_model_params.update(dict(model_params or {}))
        self.model = apply_model_params(base_model, resolved_model_params)
        self._task_initial_state = copy.deepcopy(
            (self.task_profile or {}).get("initialization", {}).get("state")
        )
        task_setpoints = (self.task_profile or {}).get("setpoints", {})
        self._task_initial_setpoint = copy.deepcopy(task_setpoints.get("initial"))
        self._task_setpoint_events = {
            int(event["at_step"]): list(event["values"])
            for event in task_setpoints.get("schedule", [])
        }
        self._task_disturbance_events = {}
        for event in (self.task_profile or {}).get("disturbances", []):
            self._task_disturbance_events.setdefault(int(event["at_step"]), []).append({
                "name": str(event["name"]),
                "value": copy.deepcopy(event["value"]),
            })
        resolved_control_dt = (
            control_dt if control_dt is not None else task_defaults.get("control_dt", 0.5)
        )
        resolved_episode_steps = (
            episode_steps if episode_steps is not None else task_defaults.get("episode_steps", 600)
        )
        resolved_conditions = {}
        for name, explicit in {
            "dynamic": dynamic,
            "randomize": randomize,
            "randomize_setpoints": randomize_setpoints,
            "randomize_plant": randomize_plant,
            "plant_drift": plant_drift,
            "integral_obs": integral_obs,
            "action_mode": action_mode,
            "noise": noise,
            "noise_pct": noise_pct,
            "terminate_on_runaway": terminate_on_runaway,
        }.items():
            resolved_conditions[name] = (
                explicit
                if explicit is not None
                else task_defaults.get(name, _DIRECT_ENV_DEFAULTS[name])
            )
        dynamic = bool(resolved_conditions["dynamic"])
        randomize = bool(resolved_conditions["randomize"])
        randomize_setpoints = bool(resolved_conditions["randomize_setpoints"])
        randomize_plant = bool(resolved_conditions["randomize_plant"])
        plant_drift = bool(resolved_conditions["plant_drift"])
        integral_obs = bool(resolved_conditions["integral_obs"])
        action_mode = resolved_conditions["action_mode"]
        noise = bool(resolved_conditions["noise"])
        noise_pct = resolved_conditions["noise_pct"]
        terminate_on_runaway = bool(resolved_conditions["terminate_on_runaway"])
        self.control_dt = float(resolved_control_dt)
        self.episode_steps = int(resolved_episode_steps)
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
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
        if custom_stage_reward is not None and not callable(custom_stage_reward):
            raise TypeError("custom_stage_reward must be callable")
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
        self._disturbance_schema_by_name = {
            row["name"]: row for row in self.model.disturbance_schema() if row.get("name")
        }
        for events in self._task_disturbance_events.values():
            for event in events:
                self._validate_task_disturbance(event["name"], event["value"])
        self._reset_disturbance_values()
        self.integ = Integrator(self.model)
        self.scorer = KPIScorer(self.model)
        self.nu = self.model.action_dim()
        y_sp = list(
            self._task_initial_setpoint
            if self._task_initial_setpoint is not None
            else self.model.env_setpoint_vector(self._model_env_options)
        )
        if len(y_sp) != len(self.model.controlled_output(self.model.initial_state())):
            raise ValueError(
                f"task setpoint length {len(y_sp)} does not match controlled-output length "
                f"{len(self.model.controlled_output(self.model.initial_state()))}"
            )
        if self._task_initial_state is not None and len(self._task_initial_state) != len(self.model.initial_state()):
            raise ValueError(
                f"task initial-state length {len(self._task_initial_state)} does not match state length "
                f"{len(self.model.initial_state())}"
            )
        for at_step, values in self._task_setpoint_events.items():
            if len(values) != len(y_sp):
                raise ValueError(
                    f"task setpoint event at step {at_step} has {len(values)} values; expected {len(y_sp)}"
                )
        self._ysp0 = list(y_sp)

        # integral-of-error obs (the I-term a memoryless policy otherwise lacks): lets
        # the RL policy do offset-free tracking like PID + adapt under operating-regime drift.
        self.integral_obs = integral_obs
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
        self.last_act = self.model.action_vector(self.model.default_action())
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

    def _validate_task_disturbance(self, name, value):
        if name not in self._disturbance_defaults:
            available = ", ".join(sorted(self._disturbance_defaults)) or "none"
            raise ValueError(f"unknown task disturbance {name!r}; available: {available}")
        row = self._disturbance_schema_by_name.get(name, {})
        bounds = row.get("bounds")
        values = value if isinstance(value, list) else [value]
        if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            lo, hi = bounds
            for item in values:
                number = float(item)
                if lo is not None and number < float(lo):
                    raise ValueError(f"task disturbance {name!r} is below its lower bound {lo}")
                if hi is not None and number > float(hi):
                    raise ValueError(f"task disturbance {name!r} is above its upper bound {hi}")

    def _sync_known_disturbances(self):
        for name in self._disturbance_defaults:
            attr = self._disturbance_attrs.get(name)
            if attr and hasattr(self, attr):
                self._disturbance_values[name] = self._copy_disturbance_value(getattr(self, attr))

    def _env(self):
        self._sync_known_disturbances()
        return self.model.runtime_env(self._disturbance_values)

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

        info = dict(result.info)
        if self.randomize_plant or self.plant_drift:
            info["plant_mult"] = dict(getattr(self, "_regime_mult", {}))
        return float(result.reward), result.terminated, info

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
        x0 = list(
            self._task_initial_state
            if self._task_initial_state is not None
            else self.model.initial_state()
        )
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
        self.last_act = self.model.action_vector(self.model.default_action())
        self.previous_act = copy.deepcopy(self.last_act)
        self._schedule_disturbances()
        # A task event at t=0 is part of the initial controller context. Applying
        # it before the first observation avoids an artificial one-sample delay
        # in paper-style reference-step experiments.
        if 0 in self._task_setpoint_events:
            self.y_sp = list(self._task_setpoint_events[0])
        for event in self._task_disturbance_events.get(0, []):
            self._set_disturbance_value(event["name"], event["value"])
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
            else:                                   # ("mv", u_index, lo, hi)
                mv[spec[1]] = val
        act = self.pid.compute(self._meas(), {"y_sp": self.y_sp}, self.control_dt)
        for u_index, value in mv.items():
            act[u_index] = value
        return act

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
        if self._k in self._task_setpoint_events:
            self.y_sp = list(self._task_setpoint_events[self._k])
        for (t, event) in self._dist_events:
            if t == self._k:
                self._apply_disturbance(event)
        for event in self._task_disturbance_events.get(self._k, []):
            self._set_disturbance_value(event["name"], event["value"])
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
