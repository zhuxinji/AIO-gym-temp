"""Unified controller interface and registry for AIO-Gym."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np



CONTROLLER_API_VERSION = "aiogym.controller.v1"


def _call_compatible(func, calls, label: str):
    """Call the first signature-compatible argument set without masking errors."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        args, kwargs = calls[0]
        return func(*args, **kwargs)
    for args, kwargs in calls:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return func(*args, **kwargs)
    raise TypeError(f"{label} has an unsupported call signature")


@dataclass(frozen=True)
class ControllerContext:
    """Per-step information exposed to controllers."""

    measurement: Mapping[str, Any]
    setpoint: Mapping[str, Any]
    info: Mapping[str, Any]
    action_mode: str
    control_dt: float
    env: Any = None


class Controller(Protocol):
    name: str
    action_mode: str
    control_structure: str
    controller_api_version: str

    def reset(self, seed: int | None = None) -> None:
        ...

    def act(self, obs: np.ndarray, context: ControllerContext) -> np.ndarray:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


def build_context(env, info: Mapping[str, Any] | None = None) -> ControllerContext:
    return ControllerContext(
        measurement=make_meas(env),
        setpoint={"y_sp": list(getattr(env, "y_sp", []))},
        info=dict(info or {}),
        action_mode=getattr(env, "action_mode", "actuator"),
        control_dt=float(env.control_dt),
        env=env,
    )


def make_meas(env):
    """Build a measured state dictionary from the native environment."""

    return env.model.measurement(env.integ.x, env._env())


def validate_action(action: Any, env, controller_name: str) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = env.action_space.shape[0]
    if out.shape[0] != expected:
        raise ValueError(
            f"{controller_name} produced {out.shape[0]} actions for a "
            f"{expected}-action {getattr(env, 'action_mode', 'unknown')!r} environment"
        )
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{controller_name} produced a non-finite action")
    return out


class PolicyController:
    """Adapter for learned policies with predict(obs) or act(obs)."""

    controller_api_version = CONTROLLER_API_VERSION

    def __init__(self, policy, name: str | None = None, action_mode: str = "actuator",
                 control_structure: str = "learned_policy"):
        self.policy = policy
        self.name = name or getattr(policy, "name", policy.__class__.__name__)
        self.action_mode = action_mode
        self.control_structure = control_structure

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self.policy, "reset"):
            _call_compatible(
                self.policy.reset,
                (((), {"seed": seed}), ((), {})),
                f"{self.policy.__class__.__name__}.reset",
            )

    def act(self, obs: np.ndarray, context: ControllerContext) -> np.ndarray:
        if hasattr(self.policy, "predict"):
            out = _call_compatible(
                self.policy.predict,
                (((obs,), {"deterministic": True}), ((obs,), {})),
                f"{self.policy.__class__.__name__}.predict",
            )
            return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)
        if hasattr(self.policy, "act"):
            out = _call_compatible(
                self.policy.act,
                (
                    ((obs,), {"deterministic": True}),
                    ((obs, context), {}),
                    ((obs,), {}),
                ),
                f"{self.policy.__class__.__name__}.act",
            )
            return np.asarray(out, dtype=np.float32)
        raise TypeError(f"{self.policy!r} has neither predict(obs) nor act(obs)")

    def metadata(self) -> dict[str, Any]:
        data = _metadata(self.policy)
        data.setdefault("name", self.name)
        data.setdefault("class", self.policy.__class__.__name__)
        data["api"] = self.controller_api_version
        data["adapter"] = self.__class__.__name__
        data["action_mode"] = self.action_mode
        data.setdefault("control_structure", self.control_structure)
        return data


class SB3PolicyController(PolicyController):
    """Stable-Baselines3 policy adapter with optional lazy loading."""

    @classmethod
    def load(cls, path: str, algo: str = "sac", **kw):
        algo_key = algo.lower()
        if algo_key == "sac":
            from stable_baselines3 import SAC
            policy = SAC.load(path)
        elif algo_key == "ppo":
            from stable_baselines3 import PPO
            policy = PPO.load(path)
        elif algo_key == "td3":
            from stable_baselines3 import TD3
            policy = TD3.load(path)
        else:
            raise ValueError(f"unsupported SB3 algorithm: {algo}")
        return cls(policy, name=kw.pop("name", f"SB3-{algo_key.upper()}"), **kw)


def as_controller(agent, action_mode: str = "actuator", name: str | None = None,
                  control_structure: str | None = None) -> Controller:
    if getattr(agent, "controller_api_version", None) == CONTROLLER_API_VERSION:
        return agent
    if hasattr(agent, "predict") or hasattr(agent, "act"):
        return PolicyController(
            agent,
            name=name,
            action_mode=action_mode,
            control_structure=control_structure or _metadata(agent).get("control_structure", "learned_policy"),
        )
    raise TypeError(f"{agent!r} is not a supported controller or policy")


ControllerFactory = Callable[..., Controller]
_REGISTRY: dict[str, ControllerFactory] = {}
BUILTIN_CONTROLLERS: dict[str, ControllerFactory] = {}


def register_controller(name: str, factory: ControllerFactory, *, replace: bool = False) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("controller name must be a non-empty string")
    if not callable(factory):
        raise TypeError("controller factory must be callable")
    key = name.lower()
    if key in _REGISTRY and not replace:
        raise ValueError(f"controller '{key}' is already registered")
    _REGISTRY[key] = factory


def registered_controllers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def unregister_controller(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("controller name must be a non-empty string")
    key = name.lower()
    if key in BUILTIN_CONTROLLERS:
        _REGISTRY[key] = BUILTIN_CONTROLLERS[key]
    else:
        _REGISTRY.pop(key, None)


def make_controller(name: str, model=None, scenario: str | None = None,
                    config: Mapping[str, Any] | None = None, policy=None) -> Controller:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown controller {name!r}; available: {', '.join(registered_controllers())}")
    requested_scenario = scenario or dict(config or {}).get("scenario")
    cfg = _merged_controller_config(key, requested_scenario, config)
    if model is None:
        from ..models import make_model

        model = make_model(requested_scenario or cfg.pop("scenario", "cstr"))
    return _REGISTRY[key](model=model, scenario=requested_scenario or getattr(model, "scenario", None),
                          config=cfg, policy=policy)


def load_controller_config(name: str, scenario: str | None = None,
                           profile: str | None = None) -> dict[str, Any]:
    key = name.lower()
    path = Path(__file__).resolve().parent / "configs" / f"{key}.json"
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    params = dict(data.get("parameters", {}))
    if scenario:
        params.update(data.get("scenarios", {}).get(scenario, {}))
    profile_data = data.get("profiles", {}).get(profile, {}) if profile else {}
    params.update(profile_data.get("parameters", {}))
    if scenario:
        params.update(profile_data.get("scenarios", {}).get(scenario, {}))
    out = {k: v for k, v in data.items() if k not in {"parameters", "profiles"}}
    out["parameters"] = params
    return out


def _merged_controller_config(name: str, scenario: str | None,
                              config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    override = dict(config or {})
    explicit_profile = override.pop("profile", None)
    base = load_controller_config(name, scenario, profile=explicit_profile)
    params = dict(base.get("parameters", {}))
    params.update(override.pop("parameters", {}))
    flat = {k: v for k, v in override.items() if k not in _CONFIG_META_KEYS}
    params.update(flat)
    merged = {k: v for k, v in base.items() if k != "parameters"}
    for k, v in override.items():
        if k in _CONFIG_META_KEYS:
            merged[k] = v
    if explicit_profile is not None:
        merged["profile"] = explicit_profile
    merged["parameters"] = params
    return merged


_CONFIG_META_KEYS = {
    "action_mode", "control_structure", "name", "class", "adapter",
    "scenario", "scenarios", "profile",
}


def _controller_params(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    params = dict(cfg.get("parameters", {}))
    for k, v in cfg.items():
        if k not in _CONFIG_META_KEYS and k != "parameters":
            params[k] = v
    return params


def _metadata(obj) -> dict[str, Any]:
    if hasattr(obj, "metadata"):
        return dict(obj.metadata())
    if hasattr(obj, "protocol_metadata"):
        return dict(obj.protocol_metadata())
    return {"name": getattr(obj, "name", obj.__class__.__name__),
            "class": obj.__class__.__name__}


def _pid_factory(model=None, scenario=None, config=None, policy=None):
    from .pid import PIDAgent

    cfg = dict(config or {})
    agent = PIDAgent(model, **_controller_params(cfg))
    agent.control_structure = cfg.get("control_structure", "fixed_sp_pid")
    return agent


def _mpc_factory(model=None, scenario=None, config=None, policy=None):
    from .mpc import MPCAgent

    cfg = dict(config or {})
    agent = MPCAgent(model, **_controller_params(cfg))
    agent.control_structure = cfg.get("control_structure", "fixed_sp_mpc")
    return agent


def _oracle_factory(model=None, scenario=None, config=None, policy=None):
    from .oracle import OracleAgent

    cfg = dict(config or {})
    agent = OracleAgent(scenario or model.scenario, model=model, **_controller_params(cfg))
    agent.control_structure = cfg.get("control_structure", "nmpc_oracle")
    return agent


def _policy_factory(model=None, scenario=None, config=None, policy=None):
    cfg = dict(config or {})
    params = _controller_params(cfg)
    pol = policy or params.pop("policy", None)
    if pol is None:
        raise ValueError("policy controller requires a policy object")
    params.setdefault("action_mode", cfg.get("action_mode", "actuator"))
    params.setdefault("control_structure", cfg.get("control_structure", "learned_policy"))
    return PolicyController(pol, **params)


def _sb3_factory(model=None, scenario=None, config=None, policy=None):
    cfg = dict(config or {})
    params = _controller_params(cfg)
    params.setdefault("action_mode", cfg.get("action_mode", "setpoint"))
    params.setdefault("control_structure", cfg.get("control_structure", "sb3_policy"))
    pol = policy or params.pop("policy", None)
    if pol is not None:
        return SB3PolicyController(pol, **params)
    path = params.pop("path")
    algo = params.pop("algo", "sac")
    return SB3PolicyController.load(path, algo=algo, **params)


def _onnx_factory(model=None, scenario=None, config=None, policy=None):
    from .onnx import ONNXPolicyController

    cfg = dict(config or {})
    params = _controller_params(cfg)
    path = params.pop("path", None)
    if not path:
        raise ValueError("onnx controller requires a policy path")
    action_mode = params.pop("action_mode", cfg.get("action_mode", "setpoint"))
    params.setdefault("name", cfg.get("name", "ONNX-policy"))
    params.setdefault("control_structure", cfg.get("control_structure", "onnx_policy"))
    expected_action_dim = (
        len(getattr(model, "supervisory_layout", ()))
        if action_mode == "setpoint"
        else model.action_dim()
    )
    return ONNXPolicyController.load(
        path,
        action_mode=action_mode,
        expected_action_dim=expected_action_dim,
        scenario=scenario or model.scenario,
        **params,
    )


register_controller("pid", _pid_factory)
register_controller("mpc", _mpc_factory)
register_controller("oracle", _oracle_factory)
register_controller("policy", _policy_factory)
register_controller("sb3", _sb3_factory)
register_controller("onnx", _onnx_factory)
BUILTIN_CONTROLLERS.update(_REGISTRY)
