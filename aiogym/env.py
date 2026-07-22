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
  "tracking" setpoint tracking: reward = -(normalized squared SP error
             + input move penalty + nominal steady-input deviation penalty).

``auto_events`` controls generic automatically generated within-episode events; it
does not enable or disable the process model's physical dynamics. The model is
integrated on every step for both values. ``auto_events=True`` can inject setpoint
steps, cold-inlet steps, ambient drift, or demand surges on top of domain-
randomised start points. Named tasks normally set it to ``False`` and declare
their own deterministic event schedules. The policy observes changed conditions
(t_cold / t_amb / setpoints are all in obs).
"""
from __future__ import annotations
import copy

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ._internal.config import resolve_auto_events
from .models import apply_model_params, make_model
from .models.integration import Integrator
from .evaluation.metrics.kpi import KPIScorer

from ._environment.config import (
    H_HIGH_FRAC,
    H_LOW_FRAC,
    H_OVERFLOW_FRAC,
    I_LEVEL_MAX,
    I_TEMP_MAX,
    T_HIGH,
    T_TRIP,
    DIRECT_ENV_DEFAULTS as _DIRECT_ENV_DEFAULTS,
    validated_range as _validated_range,
)
from ._environment.disturbances import DisturbanceRuntimeMixin
from ._environment.observations import ObservationRuntimeMixin
from ._environment.transitions import TransitionRuntimeMixin

# ---- Environment wrapper ----
class AIOGymNativeEnv(
    DisturbanceRuntimeMixin,
    ObservationRuntimeMixin,
    TransitionRuntimeMixin,
    gym.Env,
):
    metadata = {"render_modes": []}

    def __init__(self, scenario="cascade", control_dt=None, episode_steps=None, task=None,
                 reward_mode="kpi", auto_events=None, dynamic=None, randomize=None,
                 randomize_setpoints=None,
                 randomize_plant=None, plant_drift=None, integral_obs=None, action_mode=None,
                 noise=None, noise_pct=None, custom_stage_reward=None,
                 model_params=None,
                 terminate_on_runaway=None, reward_scale=0.03, w_prod=1000.0, w_energy=2.0, w_constraint=8.0,
                 tracking_q_y=1.0, tracking_r_move=1.0, tracking_r_steady=1.0,
                 crystal_ln_sp=None, crystal_cv_sp=None, crystal_random_targets=False,
                 crystal_ln_range=(10.0, 11.5), crystal_cv_range=(0.75, 0.95)):
        super().__init__()
        base_model = make_model(scenario)
        self.scenario = base_model.scenario
        from .models.tasks import resolve_environment_options

        auto_events = resolve_auto_events(
            auto_events,
            dynamic,
            warn_legacy=dynamic is not None,
        )

        self.task_profile, environment_options = resolve_environment_options(
            scenario=self.scenario,
            task=task,
            explicit={
                "control_dt": control_dt,
                "episode_steps": episode_steps,
                "action_mode": action_mode,
                "auto_events": auto_events,
                "randomize": randomize,
                "randomize_setpoints": randomize_setpoints,
                "randomize_plant": randomize_plant,
                "plant_drift": plant_drift,
                "integral_obs": integral_obs,
                "terminate_on_runaway": terminate_on_runaway,
                "noise": noise,
                "noise_pct": noise_pct,
                "model_params": model_params,
            },
            defaults=_DIRECT_ENV_DEFAULTS,
            default_control_dt=0.5,
            default_episode_steps=600,
        )
        self.model = apply_model_params(base_model, environment_options["model_params"])
        if self.task_profile is not None:
            from .models.tasks import configure_model_for_task

            configure_model_for_task(self.model, self.task_profile)
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
        auto_events = environment_options["auto_events"]
        randomize = environment_options["randomize"]
        randomize_setpoints = environment_options["randomize_setpoints"]
        randomize_plant = environment_options["randomize_plant"]
        plant_drift = environment_options["plant_drift"]
        integral_obs = environment_options["integral_obs"]
        action_mode = environment_options["action_mode"]
        noise = environment_options["noise"]
        terminate_on_runaway = environment_options["terminate_on_runaway"]
        self.control_dt = environment_options["control_dt"]
        self.episode_steps = environment_options["episode_steps"]
        if reward_mode not in {"kpi", "economic", "tracking"}:
            raise ValueError("reward_mode must be one of: economic, kpi, tracking")
        if action_mode not in {"actuator", "setpoint"}:
            raise ValueError("action_mode must be one of: actuator, setpoint")
        self.noise_pct = environment_options["noise_pct"]
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale          # keep Q-magnitudes sane -> stable critic
        self.tracking_q_y = self._resolve_tracking_q_y(tracking_q_y)
        self.tracking_r_move = self._nonnegative("tracking_r_move", tracking_r_move)
        self.tracking_r_steady = self._nonnegative("tracking_r_steady", tracking_r_steady)
        self.auto_events = auto_events
        self.dynamic = auto_events  # deprecated read compatibility
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

    def step(self, action):
        action = self._validated_action(action)
        act = self._supervise(action) if self.pid is not None else self._split(action)
        state = list(self.integ.x)
        self.last_act = act
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
        # Stage the next step's scheduled reference before returning its
        # observation. The controller therefore sees an event at step k before
        # selecting u_k, while the transition just completed is still scored
        # against the reference that was active when its action was selected.
        if self._k in self._task_setpoint_events:
            self.y_sp = list(self._task_setpoint_events[self._k])
        truncated = self._k >= self.episode_steps
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        pass
