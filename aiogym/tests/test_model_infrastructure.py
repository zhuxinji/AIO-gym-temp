"""Tests for physical metadata, task profiles, and generic model readiness."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

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
    assert minimum_pid["controller_config"]["profile"] == "quadruple-minimum-phase-benchmark"
    assert nonminimum_pid["controller_config"]["profile"] == "quadruple-nonminimum-phase-benchmark"


def test_quadruple_suite_runs_all_three_controllers_on_every_formal_task():
    from aiogym.cli.suite_benchmark import build_cases

    args = SimpleNamespace(
        suite="quadruple",
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
    assert len(cases) == 18
    by_task = {}
    for case in cases:
        by_task.setdefault(case["task"], set()).add(case["controller"])
    assert by_task == {
        "minimum-phase-classic": {"pid", "mpc", "oracle"},
        "nonminimum-phase-classic": {"pid", "mpc", "oracle"},
        "pminus-reference-step": {"pid", "mpc", "oracle"},
        "pplus-reference-step": {"pid", "mpc", "oracle"},
        "zero-boundary-stress": {"pid", "mpc", "oracle"},
        "disturbance-rejection": {"pid", "mpc", "oracle"},
    }
    assert {case["objective"] for case in cases} == {"tracking", "robustness"}
    assert "legacy-default" not in by_task
    pid_profiles = {
        case["task"]: case["controller_config"]["profile"]
        for case in cases
        if case["controller"] == "pid"
    }
    assert pid_profiles == {
        "minimum-phase-classic": "quadruple-minimum-phase-benchmark",
        "nonminimum-phase-classic": "quadruple-nonminimum-phase-benchmark",
        "pminus-reference-step": "quadruple-minimum-phase-benchmark",
        "pplus-reference-step": "quadruple-nonminimum-phase-benchmark",
        "zero-boundary-stress": "quadruple-zero-boundary",
        "disturbance-rejection": "quadruple-disturbance-rejection",
    }
    mpc_profiles = {
        case["task"]: case["controller_config"]["profile"]
        for case in cases
        if case["controller"] == "mpc"
    }
    assert mpc_profiles == {
        "minimum-phase-classic": "quadruple-minimum-phase",
        "nonminimum-phase-classic": "quadruple-nonminimum-phase",
        "pminus-reference-step": "quadruple-minimum-phase",
        "pplus-reference-step": "quadruple-nonminimum-phase",
        "zero-boundary-stress": "quadruple-zero-boundary",
        "disturbance-rejection": "quadruple-disturbance-rejection",
    }
    oracle_profiles = {
        case["task"]: case["controller_config"]["profile"]
        for case in cases
        if case["controller"] == "oracle"
    }
    assert oracle_profiles == {
        "minimum-phase-classic": "quadruple-minimum-phase",
        "nonminimum-phase-classic": "quadruple-nonminimum-phase",
        "pminus-reference-step": "quadruple-minimum-phase",
        "pplus-reference-step": "quadruple-nonminimum-phase",
        "zero-boundary-stress": "quadruple-zero-boundary",
        "disturbance-rejection": "quadruple-disturbance-rejection",
    }


def test_quadruple_paper_reference_suite_contains_only_paper_pi_cases():
    from aiogym.cli.suite_benchmark import build_cases, builtin_suites

    args = SimpleNamespace(
        suite="quadruple-paper-reference",
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
    assert [(case["task"], case["controller"]) for case in cases] == [
        ("pminus-reference-step", "pid"),
        ("pplus-reference-step", "pid"),
    ]
    assert [case["controller_config"]["profile"] for case in cases] == [
        "quadruple-minimum-phase",
        "quadruple-nonminimum-phase",
    ]
    assert "quadruple" in builtin_suites()
    assert "quadruple-all" not in builtin_suites()


def test_tracking_plots_show_task_identity_setpoints_and_controls(tmp_path):
    from aiogym.evaluation.artifacts import plot_results, write_benchmark_artifacts

    task = "minimum-phase-classic"
    rollout = {
        "name": "PID",
        "scenario": "quadruple",
        "task": task,
        "objective": "tracking",
        "rollout": [
            {
                "time": 0.0,
                "next_state": [12.0, 13.0, 2.0, 1.5],
                "action": [0.3, 0.3],
                "info": {"y": [12.0, 13.0], "y_sp": [13.0, 12.0]},
            },
            {
                "time": 1.0,
                "next_state": [12.1, 12.9, 2.1, 1.6],
                "action": [0.4, 0.2],
                "info": {"y": [12.1, 12.9], "y_sp": [13.0, 12.0]},
            },
        ],
    }
    rows = [
        {
            "scenario": "quadruple",
            "task": task,
            "objective": "tracking",
            "controller": "PID",
            "status": "passed",
            "metric": "tracking_cost",
            "tracking_cost": 1.0,
            "runtime_seconds_per_step": 0.001,
        },
        {
            "scenario": "quadruple",
            "task": "zero-boundary-stress",
            "objective": "tracking",
            "controller": "PID",
            "status": "passed",
            "metric": "tracking_cost",
            "tracking_cost": 2.0,
            "runtime_seconds_per_step": 0.001,
        },
    ]
    payload = {
        "benchmark": "benchmark_suite",
        "suite": "plot-smoke",
        "suite_config": {"scenarios": ["quadruple"]},
        "rows": rows,
        "results": [],
        "report": {},
        "rollouts": [rollout],
    }
    artifacts = write_benchmark_artifacts(tmp_path, payload)
    payload["artifacts"] = artifacts
    (tmp_path / "benchmark.json").write_text(json.dumps(payload))
    figures = plot_results(tmp_path)

    comparison = (tmp_path / "figures" / "tracking_comparison.svg").read_text()
    assert "Task" in comparison
    assert "minimum-phase-classic" in comparison
    assert "zero-boundary-stress" in comparison
    leaderboard = (tmp_path / "figures" / "leaderboard.svg").read_text()
    assert "minimum-phase-classic / tracking" in leaderboard
    assert "tracking / quadruple / minimum-phase-classic" not in leaderboard
    controls = figures["tracking_control_by_scenario"]
    assert set(controls) == {"quadruple / minimum-phase-classic"}
    control_svg = (tmp_path / "figures" / "tracking_control_quadruple-minimum-phase-classic.svg").read_text()
    assert "quadruple / minimum-phase-classic tracking control" in control_svg
    assert "Setpoint" in control_svg
    assert 'stroke-dasharray="6 5"' in control_svg
    assert "pump_1_voltage" in control_svg
    assert "PID" in control_svg
    assert "Time (s)" in control_svg
    assert ">0.25</text>" in control_svg
    assert 'font-size="18" text-anchor="middle"' in control_svg
    assert 'font-size="20" text-anchor="start"' in control_svg


def test_summary_bars_preserve_small_nonzero_value_labels(tmp_path):
    from aiogym.evaluation.plots import plot_summary

    path = tmp_path / "small-values.svg"
    plot_summary([
        {
            "name": "tiny-but-nonzero",
            "objective": "tracking",
            "tracking_cost": 0.004321,
            "tracking_mse": 0.0001234,
            "tracking_iae": 0.01234,
        }
    ], path, "quadruple")
    svg = path.read_text()
    assert ">0.004321</text>" in svg
    assert ">0.0001234</text>" in svg
    assert ">0.01234</text>" in svg
    assert ">0.00</text>" not in svg


def test_quadruple_paper_reference_plot_uses_four_paper_signals(tmp_path):
    from aiogym.evaluation.plots import plot_tracking_control

    rollout = {
        "name": "PID",
        "rollout": [
            {
                "time": 0.0,
                "next_state": [12.0, 13.0, 2.0, 1.5],
                "action": [0.3, 0.3],
                "info": {"y": [12.0, 13.0], "y_sp": [14.0, 13.0]},
            },
            {
                "time": 1.0,
                "next_state": [12.1, 13.0, 2.1, 1.6],
                "action": [0.4, 0.2],
                "info": {"y": [12.1, 13.0], "y_sp": [14.0, 13.0]},
            },
        ],
    }
    path = tmp_path / "paper-reference.svg"
    plot_tracking_control(
        [rollout], path, "quadruple", task="pminus-reference-step"
    )
    svg = path.read_text()
    assert "h1 (cm)" in svg and "h2 (cm)" in svg
    assert "h3 (cm)" not in svg and "h4 (cm)" not in svg
    assert "pump_1_voltage" in svg and "pump_2_voltage" in svg


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


def test_economic_suites_exclude_models_without_meaningful_economics():
    from aiogym.cli.suite_benchmark import build_cases, load_suite

    valid_economic = {"cascade", "cstr", "hvac", "heater"}
    args = SimpleNamespace(
        suite="standard-baselines",
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
    tracking_scenarios = {
        case["scenario"] for case in cases if case["objective"] == "tracking"
    }
    economic_scenarios = {
        case["scenario"] for case in cases if case["objective"] == "economic"
    }
    assert tracking_scenarios == set(aiogym.SCENARIOS)
    assert economic_scenarios == valid_economic
    assert {case["protocol"].episode_steps for case in cases} == {80}
    assert {case["protocol"].control_dt for case in cases} == {0.5}

    args.suite = "all-actuator"
    _, all_cases = build_cases(args)
    assert {
        case["scenario"] for case in all_cases if case["objective"] == "tracking"
    } == set(aiogym.SCENARIOS)
    assert {
        case["scenario"] for case in all_cases if case["objective"] == "economic"
    } == valid_economic

    args.suite = "core"
    _, core_cases = build_cases(args)
    assert {
        case["scenario"] for case in core_cases if case["objective"] == "tracking"
    } == {"cascade", "quadruple", "cstr", "hvac"}
    assert {
        case["scenario"] for case in core_cases if case["objective"] == "economic"
    } == {"cascade", "cstr", "hvac"}

    assert set(load_suite("economic-actuator")["scenarios"]) == valid_economic
    assert set(load_suite("economic-supervisory")["scenarios"]) == valid_economic


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
