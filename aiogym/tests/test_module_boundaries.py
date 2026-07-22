"""Compatibility checks for responsibility-based package boundaries."""


def test_controller_facade_reexports_focused_implementations():
    import aiogym.controllers as facade
    from aiogym.controllers import adapters, configs, contracts, registry

    assert facade.ControllerContext is contracts.ControllerContext
    assert facade.Controller is contracts.Controller
    assert facade.PolicyController is adapters.PolicyController
    assert facade.SB3PolicyController is adapters.SB3PolicyController
    assert facade.as_controller is adapters.as_controller
    assert facade.load_controller_config is configs.load_controller_config
    assert facade.make_controller is registry.make_controller
    assert facade.registered_controllers is registry.registered_controllers


def test_evaluation_facades_reexport_focused_implementations():
    import aiogym.evaluation as public
    import aiogym.evaluation.core as core
    import aiogym.evaluation.protocols as protocols
    from aiogym.evaluation import aggregation, cases, evaluator, metric_catalog
    from aiogym.evaluation import objective_specs, rollouts

    assert protocols.BenchmarkCase is cases.BenchmarkCase
    assert protocols.EnvironmentSpec is cases.EnvironmentSpec
    assert protocols.ObjectiveSpec is objective_specs.ObjectiveSpec
    assert protocols.METRIC_DEFINITIONS is metric_catalog.METRIC_DEFINITIONS
    assert core.evaluate_controller is evaluator.evaluate_controller
    assert core.rollout_controller is rollouts.rollout_controller
    assert core.result_schema is aggregation.result_schema
    assert core.build_evaluation_report is aggregation.build_evaluation_report
    assert public.evaluate_controller is evaluator.evaluate_controller
    assert public.rollout_controller is rollouts.rollout_controller


def test_model_core_reexports_backend_and_integrator_compatibility():
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


def test_suite_cli_facade_delegates_to_suite_modules():
    from aiogym.cli import suite_benchmark as facade
    from aiogym.evaluation import suite_cases, suite_loading, suite_results

    assert facade.expand_scenarios is suite_loading.expand_scenarios
    assert facade.controller_config_for is suite_cases.controller_config_for
    assert facade.SUMMARY_COLUMNS is suite_results.SUMMARY_COLUMNS
    assert facade.build_summary_table is suite_results.build_summary_table
    assert facade.load_suite("core") == suite_loading.load_suite("core")


def test_artifact_and_report_facades_reexport_focused_implementations():
    import aiogym.evaluation.artifacts as artifacts
    import aiogym.evaluation.reports as reports
    from aiogym.evaluation import artifact_checks, artifact_plotting, artifact_tables
    from aiogym.evaluation import artifact_writers, report_rendering

    assert artifacts.plot_results is artifact_plotting.plot_results
    assert artifacts._leaderboard is artifact_tables._leaderboard
    assert artifacts._write_benchmark_artifacts is artifact_writers._write_benchmark_artifacts
    assert reports.render_benchmark_report is report_rendering.render_benchmark_report
    assert reports.check_benchmark_artifacts is artifact_checks.check_benchmark_artifacts
    assert reports._tracking_benchmark_case_count is report_rendering._tracking_benchmark_case_count
