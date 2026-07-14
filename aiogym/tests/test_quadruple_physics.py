"""Physical and task acceptance tests for the Johansson quadruple tank."""
from __future__ import annotations

import numpy as np
import pytest

import aiogym
from aiogym.controllers import make_controller
from aiogym.evaluation import evaluate_controller
from aiogym.models import Integrator, apply_model_params, make_model


def test_reference_parameters_and_minimum_phase_configuration():
    model = make_model("quadruple")
    assert model.p == {
        "tank_area": [28.0, 32.0, 28.0, 32.0],
        "outlet_area": [0.071, 0.057, 0.071, 0.057],
        "pump_gain": [3.33, 3.35],
        "gamma": [0.70, 0.60],
        "gravity": 981.0,
        "max_voltage": 10.0,
        "max_level": 20.0,
        "nominal_voltage": [3.0, 3.0],
    }
    assert model.phase_configuration == "minimum-phase"
    nonminimum = apply_model_params(model, {"gamma": [0.43, 0.34]})
    assert nonminimum.phase_configuration == "nonminimum-phase"


def test_nominal_state_is_exact_nonlinear_equilibrium():
    model = make_model("quadruple")
    dx = model.dynamics(
        model.initial_state(), model.default_action(), model.disturbance_defaults()
    )
    assert np.max(np.abs(dx)) < 1e-12
    assert model.initial_state() == pytest.approx([
        12.2629675195507,
        12.783158403008972,
        1.6339411322567796,
        1.409044702533737,
    ])


@pytest.mark.parametrize(
    "state,action",
    [
        ([12.0, 13.0, 2.0, 1.5], [0.3, 0.3]),
        ([5.0, 17.0, 8.0, 3.0], [0.1, 0.8]),
        ([19.0, 4.0, 0.2, 12.0], [1.0, 0.0]),
    ],
)
def test_total_volume_balance_closes(state, action):
    model = make_model("quadruple")
    assert abs(model.mass_balance_residual(state, action)) < 1e-12


def test_pump_routing_matches_physical_diagonals():
    model = make_model("quadruple")
    x = model.initial_state()
    env = model.disturbance_defaults()
    base = np.asarray(model.dynamics(x, [0.3, 0.3], env))
    pump_1 = np.asarray(model.dynamics(x, [0.4, 0.3], env)) - base
    pump_2 = np.asarray(model.dynamics(x, [0.3, 0.4], env)) - base
    assert pump_1[0] > 0 and pump_1[3] > 0
    assert pump_1[1] == pytest.approx(0.0) and pump_1[2] == pytest.approx(0.0)
    assert pump_2[1] > 0 and pump_2[2] > 0
    assert pump_2[0] == pytest.approx(0.0) and pump_2[3] == pytest.approx(0.0)


def test_rk4_solution_is_step_converged_for_reference_transient():
    model = make_model("quadruple")
    env = model.disturbance_defaults()
    coarse = Integrator(model, max_step=0.1)
    fine = Integrator(model, max_step=0.05)
    for _ in range(60):
        coarse.step(1.0, [0.36, 0.24], env)
        fine.step(1.0, [0.36, 0.24], env)
    assert np.max(np.abs(np.asarray(coarse.x) - np.asarray(fine.x))) < 1e-8


def test_reference_task_applies_equilibrium_and_setpoint_schedule():
    env = aiogym.make_env(
        "quadruple", objective="tracking", task="minimum-phase-tracking"
    )
    env.reset(seed=0)
    assert env.control_dt == 1.0 and env.episode_steps == 600
    assert env.integ.x == pytest.approx(env.model.initial_state())
    initial = list(env.y_sp)
    for _ in range(120):
        env.step([0.3, 0.3])
    assert env.y_sp == initial
    env.step([0.3, 0.3])
    assert env.y_sp == pytest.approx([initial[0] + 1.0, initial[1] - 1.0])


def test_reference_pi_bias_and_closed_loop_smoke():
    env = aiogym.make_env(
        "quadruple", objective="tracking", task="minimum-phase-tracking"
    )
    controller = make_controller("pid", scenario="quadruple")
    env.reset(seed=0)
    nominal_action = controller.compute(
        env.model.measurement(env.integ.x, env._env()), {"y_sp": env.y_sp}, 1.0
    )
    assert env.model.action_vector(nominal_action) == pytest.approx([0.3, 0.3])
    result = evaluate_controller(controller, env, episodes=1, seed_list=[0])
    assert result["tracking_cost"] < 0.2
    assert result["constraint_violation_count"] == 0
    assert result["normalized_score"] > 99.0


def test_nonminimum_phase_task_has_rhp_zero_and_matched_controller_profile():
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple", task="nonminimum-phase-classic", episode_steps=2
    )
    env = protocol.make_env()
    env.reset(seed=0)
    zeros = env.model.transmission_zeros()
    assert env.model.phase_configuration == "nonminimum-phase"
    assert max(zeros) > 0.0
    assert env.integ.x == pytest.approx([
        12.441864220183483,
        13.166812925360238,
        4.730260670665678,
        4.986334403669725,
    ])
    controller = make_controller(
        "pid",
        model=env.model,
        scenario="quadruple",
        config={"profile": "quadruple-nonminimum-phase"},
    )
    nominal_action = controller.compute(
        env.model.measurement(env.integ.x, env._env()), {"y_sp": env.y_sp}, 1.0
    )
    assert env.model.action_vector(nominal_action) == pytest.approx([0.315, 0.315])


@pytest.mark.parametrize(
    "task,horizon,phase,initial_action",
    [
        ("pminus-reference-step", 360, "minimum-phase", [0.3, 0.3]),
        ("pplus-reference-step", 3600, "nonminimum-phase", [0.315, 0.315]),
    ],
)
def test_paper_reference_steps_are_active_in_initial_context(task, horizon, phase, initial_action):
    protocol = aiogym.BenchmarkProtocol.tracking("quadruple", task=task)
    env = protocol.make_env()
    env.reset(seed=0)
    equilibrium = env.model.initial_state()
    assert env.episode_steps == horizon
    assert env.model.phase_configuration == phase
    assert env.y_sp == pytest.approx([equilibrium[0] + 2.0, equilibrium[1]])
    assert env.model.default_action() == pytest.approx(initial_action)


def test_zero_boundary_task_places_transmission_zero_at_origin():
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple", task="zero-boundary-stress", episode_steps=2
    )
    env = protocol.make_env()
    env.reset(seed=0)
    assert env.model.phase_configuration == "zero-at-origin"
    assert min(abs(value) for value in env.model.transmission_zeros()) < 1e-12
    dx = env.model.dynamics(env.integ.x, env.model.default_action(), env._env())
    assert max(abs(value) for value in dx) < 1e-12


def test_task_disturbance_schedule_is_executable_and_reproducible():
    env = aiogym.make_env(
        "quadruple",
        objective="robustness",
        task="disturbance-rejection",
        episode_steps=422,
    )
    env.reset(seed=0)
    for _ in range(121):
        env.step([0.3, 0.3])
    assert env._env()["pump_flow_factor"] == pytest.approx(0.8)
    for _ in range(180):
        env.step([0.3, 0.3])
    assert env._env()["pump_flow_factor"] == pytest.approx(1.0)
    for _ in range(121):
        env.step([0.3, 0.3])
    assert env._env()["outlet_area_factor"] == pytest.approx(0.8)
