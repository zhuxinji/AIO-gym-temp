"""Centralized deprecation messages for compatibility surfaces."""
from __future__ import annotations

import warnings


def warn_deprecated(old: str, new: str, *, removal: str = "0.3") -> None:
    warnings.warn(
        f"{old} is deprecated; use {new}. It is scheduled for removal in aiogym {removal}.",
        DeprecationWarning,
        stacklevel=3,
    )
