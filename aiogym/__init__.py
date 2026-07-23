"""aiogym — native Gymnasium environments for process-control research.

Fast, synchronous, seedable, and vectorizable for controller benchmarking,
offline-data generation, and RL training (SAC / RLPD / Cal-QL).

    import gymnasium as gym, aiogym          # registers the ids on import
    env = gym.make("AIOGym/Cascade-v0")
    # or:  from aiogym import AIOGymNativeEnv;  env = AIOGymNativeEnv("cstr")
"""
from .models import (
    MODEL_CARD_SCHEMA_VERSION,
    PARAMETER_PROFILE_SCHEMA_VERSION,
    TASK_PROFILE_SCHEMA_VERSION,
    Integrator,
    ProcessModelContract,
    builtin_gym_ids,
    collect_model_cards,
    define_model,
    export_model_cards,
    make_model,
    list_parameter_profiles,
    load_parameter_profile,
    load_task_profile,
    register_model,
    unregister_model,
    task_operation,
    validate_model_card,
    validate_model_readiness,
    validate_parameter_profile,
    validate_task_profile,
)
from .env import AIOGymNativeEnv
from .env_factory import make_env
from .evaluation.objectives import StageRewardContext, StageRewardResult, stage_reward
from .evaluation import (
    ARTIFACT_CHECK_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    BenchmarkCase,
    BenchmarkProtocol,
    EnvironmentSpec,
    ObjectiveSpec,
    build_evaluation_report,
    check_benchmark_artifacts,
    evaluate_controller,
    plot_results,
    render_benchmark_report,
    reward_mode_for_objective,
    objective_spec,
    resolve_objective,
    run_benchmark,
)
from .controllers import (
    load_controller_config,
    make_controller,
    register_controller,
    unregister_controller,
)
from .catalog import list_controllers, list_scenarios, list_suites, list_tasks

__all__ = ["AIOGymNativeEnv", "StageRewardContext", "StageRewardResult", "stage_reward",
           "make_model", "register_model", "unregister_model", "Integrator",
           "ProcessModelContract", "define_model",
           "PARAMETER_PROFILE_SCHEMA_VERSION", "list_parameter_profiles", "load_parameter_profile",
           "validate_parameter_profile", "validate_model_readiness",
           "BenchmarkProtocol", "BenchmarkCase", "EnvironmentSpec",
           "ObjectiveSpec", "objective_spec", "resolve_objective",
           "reward_mode_for_objective",
           "build_evaluation_report", "evaluate_controller",
           "TASK_PROFILE_SCHEMA_VERSION", "load_task_profile", "task_operation", "validate_task_profile",
           "make_controller", "register_controller", "unregister_controller",
           "load_controller_config",
           "list_controllers", "list_scenarios", "list_suites", "list_tasks",
           "MODEL_CARD_SCHEMA_VERSION", "collect_model_cards", "export_model_cards",
           "validate_model_card",
           "make_env", "plot_results", "run_benchmark",
           "REPORT_SCHEMA_VERSION", "render_benchmark_report",
           "ARTIFACT_CHECK_SCHEMA_VERSION", "check_benchmark_artifacts"]

from gymnasium.envs.registration import register, registry
from ._internal.identifiers import canonical_scenario_id

for _scn, _name in builtin_gym_ids().items():
    _env_id = f"AIOGym/{_name}-v0"
    if _env_id not in registry:
        register(id=f"AIOGym/{_name}-v0", entry_point="aiogym.env:AIOGymNativeEnv",
                 kwargs={"scenario": canonical_scenario_id(_scn)})
