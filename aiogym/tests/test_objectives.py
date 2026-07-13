"""Contracts for pure stage-reward evaluation."""
from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from aiogym.env import AIOGymNativeEnv
from aiogym.models import SCENARIOS
from aiogym.objectives import stage_reward


class StageRewardContractTests(unittest.TestCase):
    def test_stage_reward_matches_environment_step(self):
        for scenario in SCENARIOS:
            for reward_mode in ("tracking", "kpi", "economic"):
                with self.subTest(scenario=scenario, reward_mode=reward_mode):
                    env = AIOGymNativeEnv(
                        scenario,
                        reward_mode=reward_mode,
                        dynamic=False,
                        randomize=False,
                        randomize_setpoints=False,
                        randomize_plant=False,
                        plant_drift=False,
                        episode_steps=2,
                    )
                    env.reset(seed=0)
                    state = list(env.integ.x)
                    previous_action = copy.deepcopy(env.previous_act)
                    setpoint = list(env.y_sp)
                    disturbance = copy.deepcopy(env._env())
                    action = np.full(env.action_space.shape, 0.4, dtype=np.float32)
                    physical_action = env._split(action)

                    _, environment_reward, terminated, _, environment_info = env.step(action)
                    score_after_step = env.scorer.report()
                    expected = stage_reward(
                        env.model,
                        state,
                        physical_action,
                        env.integ.x,
                        setpoint=setpoint,
                        disturbance=disturbance,
                        previous_action=previous_action,
                        reward_mode=reward_mode,
                        reward_scale=env.reward_scale,
                        tracking_q_y=env.tracking_q_y,
                        tracking_r_move=env.tracking_r_move,
                        terminate_on_runaway=env.terminate_on_runaway,
                        economic_config=env._econ,
                    )

                    self.assertTrue(math.isclose(expected.reward, environment_reward, rel_tol=1e-12, abs_tol=1e-12))
                    self.assertIs(expected.terminated, terminated)
                    self.assertTrue(math.isclose(
                        expected.info["tracking_cost"],
                        environment_info["tracking_cost"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ))
                    self.assertTrue(math.isclose(
                        expected.info["energy_kw"],
                        environment_info["energy_kw"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ))
                    self.assertEqual(env.scorer.report(), score_after_step)

    def test_stage_reward_repeated_calls_are_side_effect_free(self):
        env = AIOGymNativeEnv(
            "cstr",
            reward_mode="tracking",
            dynamic=False,
            randomize=False,
            randomize_setpoints=False,
            randomize_plant=False,
            plant_drift=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)
        action = env._split(np.array([0.25, 0.75], dtype=np.float32))
        kwargs = {
            "setpoint": list(env.y_sp),
            "disturbance": copy.deepcopy(env._env()),
            "previous_action": copy.deepcopy(env.previous_act),
            "reward_mode": env.reward_mode,
            "reward_scale": env.reward_scale,
            "tracking_q_y": tuple(env.tracking_q_y),
            "tracking_r_move": env.tracking_r_move,
            "terminate_on_runaway": env.terminate_on_runaway,
            "economic_config": copy.deepcopy(env._econ),
        }
        scorer_before = env.scorer.report()
        previous_action_before = copy.deepcopy(env.previous_act)
        state_before = list(env.integ.x)

        first = stage_reward(env.model, state, action, state, **kwargs)
        second = stage_reward(env.model, state, action, state, **kwargs)

        self.assertEqual(first, second)
        self.assertEqual(env.scorer.report(), scorer_before)
        self.assertEqual(env.previous_act, previous_action_before)
        self.assertEqual(env.integ.x, state_before)

    def test_environment_evaluates_candidate_actions_with_explicit_context(self):
        env = AIOGymNativeEnv(
            "cstr",
            reward_mode="tracking",
            dynamic=False,
            randomize=False,
            randomize_setpoints=False,
            randomize_plant=False,
            plant_drift=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)
        candidate = np.array([0.2, 0.8], dtype=np.float32)
        scorer_before = env.scorer.report()
        state_before = list(env.integ.x)

        with_default_previous = env.evaluate_transition(state, candidate, state)
        with_candidate_previous = env.evaluate_transition(
            state,
            candidate,
            state,
            setpoint=list(env.y_sp),
            disturbance=copy.deepcopy(env._env()),
            previous_action=candidate,
        )

        self.assertGreater(with_default_previous.info["tracking_move_cost"], 0.0)
        self.assertEqual(with_candidate_previous.info["tracking_move_cost"], 0.0)
        self.assertEqual(env.scorer.report(), scorer_before)
        self.assertEqual(env.integ.x, state_before)

    def test_environment_rejects_setpoint_actions_for_transition_evaluation(self):
        env = AIOGymNativeEnv(
            "cstr",
            action_mode="setpoint",
            dynamic=False,
            randomize=False,
            randomize_setpoints=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)

        with self.assertRaisesRegex(ValueError, "action_mode='actuator'"):
            env.evaluate_transition(state, env.action_space.sample(), state)

    def test_custom_stage_reward_is_shared_by_step_and_prediction(self):
        calls = []

        def custom_stage_reward(state, action, next_state, context):
            calls.append((state, action, next_state, context))
            context.disturbance["local_only"] = 1.0
            context.info["local_only"] = True
            return 4.0 - abs(next_state[0] - context.setpoint[0])

        env = AIOGymNativeEnv(
            "cstr",
            reward_mode="tracking",
            custom_stage_reward=custom_stage_reward,
            dynamic=False,
            randomize=False,
            randomize_setpoints=False,
            randomize_plant=False,
            plant_drift=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)
        previous_action = copy.deepcopy(env.previous_act)
        setpoint = list(env.y_sp)
        disturbance = copy.deepcopy(env._env())
        action = np.array([0.3, 0.7], dtype=np.float32)

        _, environment_reward, _, _, environment_info = env.step(action)
        predicted = env.evaluate_transition(
            state,
            action,
            env.integ.x,
            setpoint=setpoint,
            disturbance=disturbance,
            previous_action=previous_action,
        )

        self.assertEqual(len(calls), 2)
        self.assertTrue(math.isclose(environment_reward, predicted.reward, rel_tol=1e-12, abs_tol=1e-12))
        self.assertNotIn("local_only", environment_info)
        self.assertNotIn("local_only", env._env())

    def test_custom_stage_reward_receives_terminal_penalty_once(self):
        env = AIOGymNativeEnv(
            "cstr",
            custom_stage_reward=lambda *_: 7.0,
            terminate_on_runaway=True,
            dynamic=False,
            randomize=False,
            randomize_setpoints=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)
        result = env.evaluate_transition(
            state,
            np.array([0.5, 0.5], dtype=np.float32),
            [state[0], 100.0],
        )

        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, -43.0)

    def test_stateful_custom_reward_is_not_available_to_planners(self):
        env = AIOGymNativeEnv(
            "cstr",
            custom_reward=lambda *_: 3.0,
            dynamic=False,
            randomize=False,
            randomize_setpoints=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)

        with self.assertRaisesRegex(ValueError, "custom_stage_reward"):
            env.evaluate_transition(state, np.array([0.5, 0.5], dtype=np.float32), state)
        _, reward, _, _, _ = env.step(np.array([0.5, 0.5], dtype=np.float32))
        self.assertEqual(reward, 3.0)

    def test_custom_reward_contract_rejects_ambiguous_or_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            AIOGymNativeEnv(
                "cstr",
                custom_reward=lambda *_: 1.0,
                custom_stage_reward=lambda *_: 2.0,
            )

        env = AIOGymNativeEnv(
            "cstr",
            custom_stage_reward=lambda *_: float("nan"),
            dynamic=False,
            randomize=False,
            randomize_setpoints=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)
        with self.assertRaisesRegex(ValueError, "finite scalar"):
            env.evaluate_transition(state, np.array([0.5, 0.5], dtype=np.float32), state)


if __name__ == "__main__":
    unittest.main()
