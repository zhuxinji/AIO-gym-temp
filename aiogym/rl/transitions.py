"""Transition records and collection utilities for model-based learning."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .._validation import nonnegative_int, positive_int
from .._deprecations import warn_deprecated


TRANSITION_SCHEMA_VERSION = "aiogym.transition.v1"


@dataclass(frozen=True)
class Transition:
    """One environment transition with both policy observations and plant state."""

    obs: Sequence[float]
    state: Sequence[float]
    action: Sequence[float]
    reward: float
    next_obs: Sequence[float]
    next_state: Sequence[float]
    terminated: bool
    truncated: bool
    setpoint: Sequence[float] = ()
    disturbance: Mapping[str, Any] = field(default_factory=dict)
    info: Mapping[str, Any] = field(default_factory=dict)
    episode: int = 0
    step: int = 0
    schema_version: str = TRANSITION_SCHEMA_VERSION

    def __post_init__(self):
        for name in ("obs", "state", "action", "next_obs", "next_state", "setpoint"):
            object.__setattr__(self, name, _finite_vector(name, getattr(self, name)))
        if len(self.obs) != len(self.next_obs):
            raise ValueError("obs and next_obs must have the same length")
        if len(self.state) != len(self.next_state):
            raise ValueError("state and next_state must have the same length")
        reward = float(self.reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "episode", nonnegative_int("episode", self.episode))
        object.__setattr__(self, "step", nonnegative_int("step", self.step))
        object.__setattr__(self, "disturbance", copy.deepcopy(dict(self.disturbance)))
        object.__setattr__(self, "info", copy.deepcopy(dict(self.info)))
        if self.schema_version != TRANSITION_SCHEMA_VERSION:
            raise ValueError(f"unsupported transition schema: {self.schema_version!r}")

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode": self.episode,
            "step": self.step,
            "obs": list(self.obs),
            "state": list(self.state),
            "action": list(self.action),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "next_obs": list(self.next_obs),
            "next_state": list(self.next_state),
            "setpoint": list(self.setpoint),
            "disturbance": _plain(self.disturbance),
            "info": _plain(self.info),
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "Transition":
        data = {key: value for key, value in row.items() if key in _TRANSITION_FIELDS}
        setpoint = data.get("setpoint", ())
        if isinstance(setpoint, Mapping):
            data["setpoint"] = setpoint.get("y_sp", ())
        data.setdefault("episode", 0)
        data.setdefault("step", 0)
        data.setdefault("schema_version", TRANSITION_SCHEMA_VERSION)
        return cls(**data)


class TransitionDataset:
    """Validated in-memory transitions with supervised-learning views."""

    def __init__(self, transitions: Iterable[Transition | Mapping[str, Any]] = ()):
        self._items: list[Transition] = []
        self._dims: tuple[int, int, int] | None = None
        self.extend(transitions)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]

    @property
    def observation_dim(self) -> int | None:
        return self._dims[0] if self._dims is not None else None

    @property
    def state_dim(self) -> int | None:
        return self._dims[1] if self._dims is not None else None

    @property
    def action_dim(self) -> int | None:
        return self._dims[2] if self._dims is not None else None

    @property
    def next_episode(self) -> int:
        return max((item.episode for item in self._items), default=-1) + 1

    def append(self, transition: Transition | Mapping[str, Any]) -> None:
        item = transition if isinstance(transition, Transition) else Transition.from_mapping(transition)
        dims = (len(item.obs), len(item.state), len(item.action))
        if self._dims is None:
            self._dims = dims
        elif dims != self._dims:
            raise ValueError(
                "transition dimensions must match the dataset: "
                f"expected obs/state/action={self._dims}, got {dims}"
            )
        self._items.append(item)

    def extend(self, transitions: Iterable[Transition | Mapping[str, Any]]) -> None:
        for transition in transitions:
            self.append(transition)

    def supervised_arrays(self, source: str = "state") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current values, actions, and next values for dynamics training."""

        if source in {"obs", "observation"}:
            current_name, next_name = "obs", "next_obs"
            width = self.observation_dim or 0
        elif source == "state":
            current_name, next_name = "state", "next_state"
            width = self.state_dim or 0
        else:
            raise ValueError("source must be 'state' or 'observation'")
        if not self._items:
            return (
                np.empty((0, width), dtype=np.float32),
                np.empty((0, self.action_dim or 0), dtype=np.float32),
                np.empty((0, width), dtype=np.float32),
            )
        current = np.asarray([getattr(item, current_name) for item in self._items], dtype=np.float32)
        actions = np.asarray([item.action for item in self._items], dtype=np.float32)
        next_values = np.asarray([getattr(item, next_name) for item in self._items], dtype=np.float32)
        return current, actions, next_values

    def rl_tuples(self):
        """Return the legacy ``(obs, action, reward, next_obs, terminated)`` view."""

        warn_deprecated("TransitionDataset.rl_tuples()", "TransitionDataset.to_rows() or supervised_arrays()")

        return [
            (
                np.asarray(item.obs, dtype=np.float32),
                np.asarray(item.action, dtype=np.float32),
                item.reward,
                np.asarray(item.next_obs, dtype=np.float32),
                float(item.terminated),
            )
            for item in self._items
        ]

    def to_rows(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> "TransitionDataset":
        return cls(Transition.from_mapping(row) for row in rows)

    @classmethod
    def from_rollout(cls, rollout, *, episode: int = 0) -> "TransitionDataset":
        """Build a dataset from a ``rollout_controller`` artifact or its rows."""

        episode = nonnegative_int("episode", episode)
        rows = rollout.get("rollout", ()) if isinstance(rollout, Mapping) else rollout
        normalized = []
        for row in rows:
            data = dict(row)
            data["episode"] = episode
            normalized.append(data)
        return cls.from_rows(normalized)


def collect_transitions(env, policy=None, *, episodes: int = 1, seed: int = 0,
                        max_steps: int | None = None,
                        dataset: TransitionDataset | None = None) -> TransitionDataset:
    """Collect direct-actuator transitions from random actions or a controller."""

    episodes = positive_int("episodes", episodes)
    if max_steps is not None:
        max_steps = positive_int("max_steps", max_steps)
    if getattr(env, "action_mode", "actuator") != "actuator":
        raise ValueError("model-based transition collection requires action_mode='actuator'")

    from ..controllers import as_controller, build_context, validate_action

    controller = None if policy is None else as_controller(policy, action_mode="actuator")
    output = dataset if dataset is not None else TransitionDataset()
    first_episode = output.next_episode

    for episode_offset in range(episodes):
        episode_index = first_episode + episode_offset
        episode_seed = int(seed) + episode_offset
        obs, reset_info = env.reset(seed=episode_seed)
        env.action_space.seed(episode_seed)
        if controller is not None:
            controller.reset(seed=episode_seed)
        info = dict(reset_info or {})
        step_index = 0
        done = False
        while not done and (max_steps is None or step_index < max_steps):
            if controller is None:
                action = env.action_space.sample()
            else:
                context = build_context(env, info)
                action = validate_action(controller.act(obs, context), env, controller.name)
            state = list(env.integ.x)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            output.append(Transition(
                obs=obs,
                state=state,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                next_obs=next_obs,
                next_state=list(env.integ.x),
                setpoint=list(getattr(env, "y_sp", [])),
                disturbance=copy.deepcopy(env._env()),
                info=next_info,
                episode=episode_index,
                step=step_index,
            ))
            obs = next_obs
            info = dict(next_info or {})
            done = bool(terminated or truncated)
            step_index += 1
    return output


def _finite_vector(name: str, values: Sequence[float]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in np.asarray(values).reshape(-1))
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name} must be a numeric vector") from ex
    if not all(np.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector




def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


_TRANSITION_FIELDS = {
    "schema_version",
    "episode",
    "step",
    "obs",
    "state",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_obs",
    "next_state",
    "setpoint",
    "disturbance",
    "info",
}
