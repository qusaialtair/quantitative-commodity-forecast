"""Shared environment-variable parsers for QCTF Model engine modules."""
from __future__ import annotations

import os


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean from an environment variable.

    Truthy values: ``1``, ``true``, ``yes``, ``y``, ``on`` (case-insensitive).
    Returns *default* when the variable is unset.
    """
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    """Parse a float from an environment variable, falling back on parse errors."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
