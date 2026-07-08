"""Reinforcement-learning algorithms and training entrypoints.

Training dependencies such as torch are optional, so algorithm classes are
loaded lazily when requested.
"""

__all__ = ["Actor", "Critic", "ReplayBuffer", "RLPD"]


def __getattr__(name):
    if name in __all__:
        from . import rlpd
        return getattr(rlpd, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
