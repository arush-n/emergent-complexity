"""Three-dimensional grid types and shape validation."""

from __future__ import annotations

import operator
from typing import TypeAlias

import jax
import jax.numpy as jnp

Grid3D: TypeAlias = jax.Array


def _positive_dimension(value: int, name: str) -> int:
    try:
        dimension = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if dimension < 1:
        raise ValueError(f"{name} must be positive")
    return dimension


def validate_dimensions_3d(depth: int, height: int, width: int) -> None:
    """Validate positive ``(depth, height, width)`` dimensions."""

    _positive_dimension(depth, "depth")
    _positive_dimension(height, "height")
    _positive_dimension(width, "width")


def empty_grid_3d(depth: int, height: int, width: int) -> Grid3D:
    """Return an all-dead uint8 grid with shape ``(depth, height, width)``."""

    validate_dimensions_3d(depth, height, width)
    return jnp.zeros((depth, height, width), dtype=jnp.uint8)
