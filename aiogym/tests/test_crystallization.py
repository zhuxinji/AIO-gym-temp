#!/usr/bin/env python3
"""Smoke tests for the crystallization backend scenario."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aiogym.controllers import make_controller
from aiogym.env import AIOGymNativeEnv
from aiogym.evaluation import BenchmarkProtocol, evaluate_controller
from aiogym.models import SCENARIOS, make_model


def _make_env(**kwargs):
    return AIOGymNativeEnv(
        "crystallization",
        action_mode="actuator",
        dynamic=False,
        randomize=False,
        randomize_setpoints=False,
        **kwargs,
    )


def test_registered_model_contract():
    assert "crystallization" in SCENARIOS
    model = make_model("crystallization")
    assert model.scenario == "crystallization"
    assert model.actuator_counts() == (0, 0, 1)
    assert model.observation_dim() == 13


def test_reset_contract():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (13,)
    assert env.observation_space.shape == (13,)
    assert env.action_space.shape == (1,)
    assert np.all(np.isfinite(obs))
    assert np.allclose(obs[-3:], [1.0, 1.0, 0.0])


def test_action_to_temperature_mapping():
    model = make_model("crystallization")
    for aT, Tc in ((0.0, 30.0), (0.5, 35.0), (1.0, 40.0)):
        act = {"pumps": [], "valves": [], "heaters": [aT]}
        assert np.isclose(model.action_to_tc(act), Tc)


def test_unified_reward_modes_are_finite():
    model = make_model("crystallization")
    assert not callable(getattr(model, "reward_terms", None))
    for reward_mode in ("track", "kpi", "economic"):
        env = _make_env(reward_mode=reward_mode, crystal_ln_sp=10.5, crystal_cv_sp=0.85, episode_steps=3)
        obs, _ = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert not terminated
        assert not truncated
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert np.isfinite(info["track"])
        assert "profit" in info


def test_tracking_controllers_build():
    pid = make_controller("pid", scenario="crystallization")
    mpc = make_controller("mpc", scenario="crystallization")
    oracle = make_controller("oracle", scenario="crystallization")
    assert pid.metadata()["scenario"] == "crystallization"
    assert pid.metadata()["pairing"]["temp"] == [[0, 0, True]]
    assert mpc.metadata()["scenario"] == "crystallization"
    assert mpc.metadata()["horizon"] == 2
    assert oracle.metadata()["scenario"] == "crystallization"
    assert oracle.metadata()["horizon"] == 2
    assert oracle.metadata()["mode"] == "track"


def test_pid_tracking_rollout():
    protocol = BenchmarkProtocol.tracking(
        "crystallization",
        action_mode="actuator",
        episode_steps=3,
        control_dt=1.0,
        dynamic=False,
        randomize=False,
        randomize_setpoints=False,
        randomize_plant=False,
        plant_drift=False,
    )
    result = evaluate_controller(
        make_controller("pid", scenario="crystallization"),
        protocol.make_env(),
        episodes=1,
        seed=0,
        protocol=protocol,
    )
    assert result["name"] == "PID"
    assert np.isfinite(result["tracking_iae"])
    assert result["controller"]["scenario"] == "crystallization"


def test_nominal_rollout_no_nan():
    env = _make_env(episode_steps=60)
    obs, _ = env.reset(seed=0)
    info = {}
    for _ in range(60):
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert not terminated
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert np.isfinite(info["Ln"])
        assert np.isfinite(info["CV"])
        assert np.isfinite(info["Tc"])
        assert 30.0 <= info["Tc"] <= 40.0
    assert truncated
    for key in ("Ln", "CV", "Tc", "c", "S", "Ceq", "B0", "Ginf"):
        assert key in info


def test_custom_setpoint_observation():
    env = _make_env(crystal_ln_sp=10.5, crystal_cv_sp=0.85)
    obs, _ = env.reset(seed=0)
    assert np.isclose(obs[7], 10.5)
    assert np.isclose(obs[8], 0.85)


def test_random_target_observation():
    env = _make_env(
        crystal_random_targets=True,
        crystal_ln_range=(10.25, 10.25),
        crystal_cv_range=(0.82, 0.82),
    )
    obs, _ = env.reset(seed=0)
    assert np.isclose(obs[7], 10.25)
    assert np.isclose(obs[8], 0.82)


if __name__ == "__main__":
    test_registered_model_contract()
    test_reset_contract()
    test_action_to_temperature_mapping()
    test_unified_reward_modes_are_finite()
    test_tracking_controllers_build()
    test_pid_tracking_rollout()
    test_nominal_rollout_no_nan()
    test_custom_setpoint_observation()
    test_random_target_observation()
    print("ALL CRYSTALLIZATION TESTS PASS OK")
