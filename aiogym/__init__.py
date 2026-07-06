"""aiogym — native (numpy) Gymnasium env over the AIO-Gym plant dynamics.

A fast, synchronous, seedable, vectorizable re-implementation of the browser
AIO-Gym physics (frontend/js/sim), built for RL training (SAC / RLPD / Cal-QL).
Dynamics parity with the browser JS is enforced by tests/test_parity.py.

    import gymnasium as gym, aiogym          # registers the ids on import
    env = gym.make("AIOGym/Cascade-v0")
    # or:  from aiogym import AIOGymNativeEnv;  env = AIOGymNativeEnv("cstr")
"""
from .models import make_model, register_model, unregister_model, obs_vector, SCENARIOS, ProcessModelContract, builtin_gym_ids
from .kernel import Integrator
from .env import AIOGymNativeEnv
from .evaluation import BenchmarkConfig, BenchmarkProtocol, build_evaluation_report, evaluate_controller
from .controllers import make_controller, register_controller, registered_controllers, load_controller_config

__all__ = ["AIOGymNativeEnv", "make_model", "register_model", "unregister_model", "obs_vector", "Integrator", "SCENARIOS",
           "ProcessModelContract", "BenchmarkProtocol", "BenchmarkConfig", "build_evaluation_report", "evaluate_controller",
           "make_controller", "register_controller", "registered_controllers", "load_controller_config"]

try:
    from gymnasium.envs.registration import register

    for _scn, _name in builtin_gym_ids().items():
        register(id=f"AIOGym/{_name}-v0", entry_point="aiogym.env:AIOGymNativeEnv",
                 kwargs={"scenario": _scn})
except Exception:
    # gymnasium not installed yet — the env class is still importable directly
    pass
