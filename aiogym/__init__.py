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
    SCENARIOS,
    Integrator,
    ProcessModelContract,
    builtin_gym_ids,
    collect_model_cards,
    define_model,
    export_model_card_markdown,
    export_model_cards,
    make_model,
    list_parameter_profiles,
    load_parameter_profile,
    render_model_card_markdown,
    register_model,
    unregister_model,
    validate_model_card,
    validate_model_readiness,
    validate_parameter_profile,
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
    TASK_PROFILE_SCHEMA_VERSION,
    build_evaluation_report,
    check_benchmark_artifacts,
    evaluate_controller,
    list_task_profiles,
    load_task_profile,
    plot_results,
    render_benchmark_report,
    objective_spec,
    resolve_objective,
    run_benchmark,
    task_operation,
    validate_task_profile,
)
from .controllers import (
    load_controller_config,
    make_controller,
    register_controller,
    registered_controllers,
    unregister_controller,
)

__all__ = ["AIOGymNativeEnv", "StageRewardContext", "StageRewardResult", "stage_reward",
           "make_model", "register_model", "unregister_model", "Integrator", "SCENARIOS",
           "ProcessModelContract", "define_model",
           "PARAMETER_PROFILE_SCHEMA_VERSION", "list_parameter_profiles", "load_parameter_profile",
           "validate_parameter_profile", "validate_model_readiness",
           "BenchmarkProtocol", "BenchmarkCase", "EnvironmentSpec",
           "ObjectiveSpec", "objective_spec", "resolve_objective",
           "build_evaluation_report", "evaluate_controller",
           "TASK_PROFILE_SCHEMA_VERSION", "list_task_profiles", "load_task_profile", "task_operation", "validate_task_profile",
           "make_controller", "register_controller", "registered_controllers", "unregister_controller",
           "load_controller_config",
           "MODEL_CARD_SCHEMA_VERSION", "collect_model_cards", "export_model_card_markdown", "export_model_cards",
           "render_model_card_markdown", "validate_model_card",
           "make_env", "plot_results", "run_benchmark",
           "REPORT_SCHEMA_VERSION", "render_benchmark_report",
           "ARTIFACT_CHECK_SCHEMA_VERSION", "check_benchmark_artifacts"]

from gymnasium.envs.registration import register, registry

for _scn, _name in builtin_gym_ids().items():
    _env_id = f"AIOGym/{_name}-v0"
    if _env_id not in registry:
        register(id=f"AIOGym/{_name}-v0", entry_point="aiogym.env:AIOGymNativeEnv",
                 kwargs={"scenario": _scn})
