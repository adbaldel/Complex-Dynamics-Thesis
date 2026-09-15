"""Shared parameter validation for numerical and plotting functions."""

from __future__ import annotations

import operator

import numpy as np


def count(value: int, name: str, *, minimum: int = 0) -> int:
    """Validate an integer iteration or sample count."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def positive(value: float, name: str) -> float:
    """Validate a positive, finite numerical threshold."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return value
