"""Reinforcement-learning algorithms and training entrypoints.

Training dependencies such as torch are optional, so algorithm classes are
loaded lazily when requested.
"""

_RLPD_EXPORTS = {"Actor", "Critic", "ReplayBuffer", "RLPD"}
_TRANSITION_EXPORTS = {
    "TRANSITION_SCHEMA_VERSION",
    "Transition",
    "TransitionDataset",
    "collect_transitions",
}
__all__ = sorted(_RLPD_EXPORTS | _TRANSITION_EXPORTS)


def __getattr__(name):
    if name in _RLPD_EXPORTS:
        from . import rlpd
        return getattr(rlpd, name)
    if name in _TRANSITION_EXPORTS:
        from . import transitions
        return getattr(transitions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
