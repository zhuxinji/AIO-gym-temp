"""Transition validation and reward/termination evaluation."""
from __future__ import annotations

import numpy as np

from ..evaluation.objectives import stage_reward


class TransitionRuntimeMixin:
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
            dt=self.control_dt,
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
