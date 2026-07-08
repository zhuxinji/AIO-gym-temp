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
    export_model_cards,
    make_model,
    obs_vector,
    register_model,
    unregister_model,
    validate_model_card,
)
from .env import AIOGymNativeEnv
from .evaluation import (
    ARTIFACT_CHECK_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkProtocol,
    build_evaluation_report,
    check_benchmark_artifacts,
    evaluate_controller,
    render_benchmark_report,
)
from .controllers import make_controller, register_controller, registered_controllers, load_controller_config
from .api import make_env, plot_results, run_benchmark

__all__ = ["AIOGymNativeEnv", "make_model", "register_model", "unregister_model", "obs_vector", "Integrator", "SCENARIOS",
           "ProcessModelContract", "BenchmarkProtocol", "BenchmarkConfig", "build_evaluation_report", "evaluate_controller",
           "make_controller", "register_controller", "registered_controllers", "load_controller_config",
           "MODEL_CARD_SCHEMA_VERSION", "collect_model_cards", "export_model_cards", "validate_model_card",
           "make_env", "plot_results", "run_benchmark",
           "REPORT_SCHEMA_VERSION", "render_benchmark_report",
           "ARTIFACT_CHECK_SCHEMA_VERSION", "check_benchmark_artifacts"]

try:
    from gymnasium.envs.registration import register

    for _scn, _name in builtin_gym_ids().items():
        register(id=f"AIOGym/{_name}-v0", entry_point="aiogym.env:AIOGymNativeEnv",
                 kwargs={"scenario": _scn})
except Exception:
    # gymnasium not installed yet — the env class is still importable directly
    pass
