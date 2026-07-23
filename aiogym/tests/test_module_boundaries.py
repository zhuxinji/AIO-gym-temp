"""Checks for responsibility-based package boundaries."""

import importlib.util


def test_controller_public_api_exposes_current_implementations():
    import aiogym.controllers as facade
    from aiogym.controllers import adapters, configs, contracts, registry

    assert facade.ControllerContext is contracts.ControllerContext
    assert facade.Controller is contracts.Controller
    assert facade.PolicyController is adapters.PolicyController
    assert facade.SB3PolicyController is adapters.SB3PolicyController
    assert facade.as_controller is adapters.as_controller
    assert facade.load_controller_config is configs.load_controller_config
    assert facade.make_controller is registry.make_controller
    assert not hasattr(facade, "registered_controllers")


def test_evaluation_public_api_exposes_current_implementations():
    import aiogym.evaluation as public
    import aiogym.evaluation.protocols as protocols
    from aiogym.evaluation import cases, execution
    from aiogym.evaluation import objective_specs

    assert protocols.BenchmarkCase is cases.BenchmarkCase
    assert protocols.EnvironmentSpec is cases.EnvironmentSpec
    assert protocols.ObjectiveSpec is objective_specs.ObjectiveSpec
    assert public.evaluate_controller is execution.evaluate_controller
    assert public.rollout_controller is execution.rollout_controller
    assert importlib.util.find_spec("aiogym.evaluation.core") is None


def test_model_core_uses_backend_and_integrator_implementations():
    import aiogym.models as public
    import aiogym.models.core as core
    from aiogym.models import backends, integration

    assert core.Integrator is integration.Integrator
    assert public.Integrator is integration.Integrator
    assert core._NUMERIC_OPS is backends._NUMERIC_OPS
    assert core._casadi_ops is backends._casadi_ops
    assert core._maxv is backends._maxv


def test_environment_class_composes_focused_runtime_mixins():
    from aiogym._environment.disturbances import DisturbanceRuntimeMixin
    from aiogym._environment.observations import ObservationRuntimeMixin
    from aiogym._environment.transitions import TransitionRuntimeMixin
    from aiogym.env import AIOGymNativeEnv

    assert AIOGymNativeEnv._env is DisturbanceRuntimeMixin._env
    assert AIOGymNativeEnv._obs is ObservationRuntimeMixin._obs
    assert AIOGymNativeEnv.evaluate_transition is TransitionRuntimeMixin.evaluate_transition


def test_suite_cli_uses_suite_modules():
    from aiogym.cli import suite_benchmark as facade
    from aiogym.evaluation import suite

    assert not hasattr(facade, "expand_scenarios")
    assert not hasattr(facade, "controller_config_for")
    assert not hasattr(facade, "SUMMARY_COLUMNS")
    assert facade.load_suite("core") == suite.load_suite("core")


def test_removed_evaluation_modules_are_absent():
    for module in (
        "aiogym.evaluation.aggregation",
        "aiogym.evaluation.artifact_checks",
        "aiogym.evaluation.artifact_plotting",
        "aiogym.evaluation.artifact_tables",
        "aiogym.evaluation.artifact_writers",
        "aiogym.evaluation.artifacts",
        "aiogym.evaluation.benchmark",
        "aiogym.evaluation.evaluator",
        "aiogym.evaluation.metadata",
        "aiogym.evaluation.plots",
        "aiogym.evaluation.report_rendering",
        "aiogym.evaluation.reports",
        "aiogym.evaluation.rollouts",
        "aiogym.evaluation.rows",
        "aiogym.evaluation.runner",
        "aiogym.evaluation.suite_cases",
        "aiogym.evaluation.suite_loading",
        "aiogym.evaluation.suite_results",
        "aiogym.evaluation.task_profiles",
        "aiogym.evaluation.task_acceptance",
    ):
        assert importlib.util.find_spec(module) is None
