"""Grid helpers.

The numerical convention is a two-dimensional ``(height, width)`` JAX array.
Cell values are stored as ``uint8`` values: zero is dead and one is alive.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

Grid = jax.Array


def validate_dimensions(height: int, width: int) -> None:
    """Raise ``ValueError`` when a grid dimension is not a positive integer."""

    if not isinstance(height, int) or not isinstance(width, int):
        raise TypeError("height and width must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")


def empty_grid(height: int, width: int, *, dtype: Any = jnp.uint8) -> Grid:
    """Create an all-dead grid with shape ``(height, width)``."""

    validate_dimensions(height, width)
    return jnp.zeros((height, width), dtype=dtype)


def as_grid(grid: Any) -> Grid:
    """Convert input data to the canonical binary ``uint8`` grid representation."""

    array = jnp.asarray(grid)
    if array.ndim != 2:
        raise ValueError(f"a single grid must have two dimensions, got shape {array.shape}")
    return (array != 0).astype(jnp.uint8)
