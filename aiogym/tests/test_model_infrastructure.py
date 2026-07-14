"""Tests for physical metadata, task profiles, and generic model readiness."""
from __future__ import annotations

import pytest
from types import SimpleNamespace

import aiogym


def test_all_builtins_have_non_mutating_parameter_profiles():
    assert set(aiogym.list_parameter_profiles()) == set(aiogym.SCENARIOS)
    for scenario in aiogym.SCENARIOS:
        model = aiogym.make_model(scenario)
        before = dict(model.p)
        profile = aiogym.load_parameter_profile(scenario)
        card = model.model_card()
        assert profile["scenario"] == scenario
        expected_status = "reference-parameterized" if scenario == "quadruple" else "legacy-unverified"
        assert profile["status"] == expected_status
        assert model.p == before
        assert card["physical_metadata"]["parameter_status"] == expected_status
        assert card["solver"]["method"] == "rk4"
        assert card["solver"]["max_step"] == pytest.approx(model.dt_micro)
        if scenario != "quadruple":
            assert all(row["status"] == "legacy-unverified" for row in card["parameters"].values())


def test_all_builtins_pass_generic_readiness_checks():
    for scenario in aiogym.SCENARIOS:
        report = aiogym.validate_model_readiness(scenario)
        assert report["passed"], report
        expected_not_checked = (
            []
            if scenario == "quadruple"
            else ["mass_balance", "energy_balance", "reference_parameter_fidelity"]
        )
        assert report["not_checked"] == expected_not_checked
        if scenario == "quadruple":
            assert report["not_applicable"] == ["energy_balance"]


def test_models_use_single_dynamics_and_flat_action_contract():
    for scenario in aiogym.SCENARIOS:
        model = aiogym.make_model(scenario)
        assert not hasattr(model, "derivatives")
        assert not hasattr(model, "action_vector_to_dict")
        assert not hasattr(model, "uses_legacy_actions")
        assert isinstance(model.default_action(), list)
        assert len(model.default_action()) == model.action_dim()
        with pytest.raises(ValueError, match="numeric vector"):
            model.action_vector({"pumps": [0.5]})


def test_task_profiles_are_opt_in_and_explicit_timing_wins():
    assert {
        f"{scenario}/legacy-default" for scenario in aiogym.SCENARIOS
    }.issubset(set(aiogym.list_task_profiles()))
    assert "quadruple/minimum-phase-classic" in aiogym.list_task_profiles()
    assert "quadruple/nonminimum-phase-classic" in aiogym.list_task_profiles()
    assert {
        "quadruple/pminus-reference-step",
        "quadruple/pplus-reference-step",
        "quadruple/zero-boundary-stress",
        "quadruple/disturbance-rejection",
    }.issubset(set(aiogym.list_task_profiles()))

    direct_default = aiogym.AIOGymNativeEnv("cstr")
    task_env = aiogym.AIOGymNativeEnv("cstr", task="legacy-default")
    overridden = aiogym.AIOGymNativeEnv(
        "cstr", task="legacy-default", control_dt=0.25, episode_steps=12
    )
    assert (direct_default.control_dt, direct_default.episode_steps) == (0.5, 600)
    assert (task_env.control_dt, task_env.episode_steps) == (0.5, 600)
    assert task_env.task_profile["name"] == "legacy-default"
    assert (overridden.control_dt, overridden.episode_steps) == (0.25, 12)

    protocol_default = aiogym.BenchmarkProtocol.tracking("cstr")
    protocol_task = aiogym.BenchmarkProtocol.tracking("cstr", task="legacy-default")
    factory_task = aiogym.make_env("cstr", objective="tracking", task="legacy-default")
    assert (protocol_default.control_dt, protocol_default.episode_steps) == (0.5, 400)
    assert (protocol_task.control_dt, protocol_task.episode_steps) == (0.5, 600)
    assert protocol_task.metadata()["task"]["name"] == "legacy-default"
    assert (factory_task.control_dt, factory_task.episode_steps) == (0.5, 600)


def test_task_owns_conditions_while_objective_owns_scoring():
    task = aiogym.load_task_profile("quadruple/nonminimum-phase-classic")
    protocol = aiogym.BenchmarkProtocol.kpi(
        "quadruple", task=task, episode_steps=7
    )
    metadata = protocol.metadata()
    assert protocol.objective == "kpi"
    assert protocol.env_reward_mode == "kpi"
    assert protocol.dynamic is False
    assert protocol.randomize is False
    assert protocol.noise is False
    assert protocol.episode_steps == 7
    assert protocol.model_params["gamma"] == [0.43, 0.34]
    assert metadata["task_identity"]["name"] == "nonminimum-phase-classic"
    assert len(metadata["task_identity"]["profile_hash"]) == 64
    with pytest.raises(ValueError, match="does not support objective"):
        aiogym.BenchmarkProtocol.economic("quadruple", task=task)


def test_profile_scenario_mismatches_are_rejected():
    with pytest.raises(ValueError, match="expected task"):
        aiogym.load_task_profile("cstr/legacy-default", scenario="hvac")
    with pytest.raises(ValueError, match="expected parameter profile"):
        aiogym.load_parameter_profile("cstr", scenario="hvac")


def test_explicit_suite_cases_keep_task_identity_and_controller_profiles():
    from aiogym.cli.suite_benchmark import build_cases

    args = SimpleNamespace(
        suite="quadruple-phase-comparison",
        scenarios=None,
        objectives=None,
        controllers=None,
        seed_list=None,
        seed=7,
        episodes=1,
        episode_steps=None,
        control_dt=None,
        sb3_path=None,
        sb3_algo="sac",
        onnx_path=None,
    )
    _, cases = build_cases(args)
    assert len(cases) == 6
    assert {case["task"] for case in cases} == {
        "minimum-phase-classic", "nonminimum-phase-classic",
    }
    minimum_pid = next(
        case for case in cases
        if case["task"] == "minimum-phase-classic" and case["controller"] == "pid"
    )
    nonminimum_pid = next(
        case for case in cases
        if case["task"] == "nonminimum-phase-classic" and case["controller"] == "pid"
    )
    assert minimum_pid["protocol"].episode_steps == 600
    assert nonminimum_pid["protocol"].episode_steps == 1800
    assert minimum_pid["controller_config"]["profile"] == "quadruple-minimum-phase"
    assert nonminimum_pid["controller_config"]["profile"] == "quadruple-nonminimum-phase"


def test_named_task_suite_inherits_task_timing_instead_of_legacy_defaults():
    from aiogym.cli.suite_benchmark import build_cases

    args = SimpleNamespace(
        suite="quadruple-disturbance-rejection",
        scenarios=None,
        objectives=None,
        controllers="pid",
        seed_list=None,
        seed=7,
        episodes=1,
        episode_steps=None,
        control_dt=None,
        sb3_path=None,
        sb3_algo="sac",
        onnx_path=None,
    )
    _, cases = build_cases(args)
    assert len(cases) == 1
    assert cases[0]["protocol"].episode_steps == 900
    assert cases[0]["protocol"].control_dt == 1.0


def test_leaderboard_ranks_restart_for_each_task():
    from aiogym.evaluation.artifacts import _leaderboard

    rows = []
    for task, pid_cost, mpc_cost in (
        ("minimum-phase-classic", 1.0, 2.0),
        ("nonminimum-phase-classic", 5.0, 4.0),
    ):
        for controller, value in (("PID", pid_cost), ("MPC", mpc_cost)):
            rows.append({
                "scenario": "quadruple",
                "task": task,
                "objective": "tracking",
                "controller": controller,
                "status": "passed",
                "metric": "tracking_cost",
                "tracking_cost": value,
            })
    board = _leaderboard(rows)
    winners = {(row["task"], row["controller"]) for row in board if row["rank"] == 1}
    assert winners == {
        ("minimum-phase-classic", "PID"),
        ("nonminimum-phase-classic", "MPC"),
    }


def test_tracking_artifact_count_distinguishes_tasks_within_one_scenario():
    from aiogym.evaluation.reports import _tracking_benchmark_case_count

    rows = [
        {
            "scenario": "quadruple",
            "task": task,
            "objective": "tracking",
            "status": "passed",
        }
        for task in ("pminus-reference-step", "pplus-reference-step")
    ]
    assert _tracking_benchmark_case_count(rows) == 2
