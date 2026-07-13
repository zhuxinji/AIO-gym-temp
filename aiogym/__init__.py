"""aiogym — native (numpy) Gymnasium env over the AIO-Gym plant dynamics.

A fast, synchronous, seedable, vectorizable re-implementation of the browser
AIO-Gym physics (frontend/js/sim), built for RL training (SAC / RLPD / Cal-QL).
Dynamics parity with the browser JS is enforced by tests/test_parity.py.

    import gymnasium as gym, aiogym          # registers the ids on import
    env = gym.make("AIOGym/Cascade-v0")
    # or:  from aiogym import AIOGymNativeEnv;  env = AIOGymNativeEnv("cstr")
"""
from .models import (
    MODEL_CARD_SCHEMA_VERSION,
    SCENARIOS,
    Integrator,
    ProcessModelContract,
    builtin_gym_ids,
    collect_model_cards,
    define_model,
    export_model_card_markdown,
    export_model_cards,
    make_model,
    obs_vector,
    render_model_card_markdown,
    register_model,
    unregister_model,
    validate_model_card,
)
from .env import AIOGymNativeEnv
from .env_factory import make_env
from .objectives import StageRewardContext, StageRewardResult, stage_reward
from .evaluation import (
    ARTIFACT_CHECK_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkProtocol,
    build_evaluation_report,
    check_benchmark_artifacts,
    evaluate_controller,
    plot_results,
    render_benchmark_report,
    run_benchmark,
)
from .controllers import (
    load_controller_config,
    make_controller,
    register_controller,
    registered_controllers,
    unregister_controller,
)

__all__ = ["AIOGymNativeEnv", "StageRewardContext", "StageRewardResult", "stage_reward",
           "make_model", "register_model", "unregister_model", "obs_vector", "Integrator", "SCENARIOS",
           "ProcessModelContract", "define_model",
           "BenchmarkProtocol", "BenchmarkConfig", "build_evaluation_report", "evaluate_controller",
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
