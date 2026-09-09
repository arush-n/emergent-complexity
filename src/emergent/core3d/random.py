"""Explicit-key random grids and rules for 3D experiments."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .grid import Grid3D, validate_dimensions_3d
from .rules import Rule3D


def _validate_density(density: float) -> float:
    if not isinstance(density, (int, float)):
        raise TypeError("density must be a number")
    value = float(density)
    if not 0.0 <= value <= 1.0:
        raise ValueError("density must be between 0 and 1")
    return value


def _random_grid_3d(key: Any, depth: int, height: int, width: int, density: float) -> Grid3D:
    return jax.random.bernoulli(key, p=density, shape=(depth, height, width)).astype(jnp.uint8)


random_grid_3d_jit = jax.jit(
    _random_grid_3d,
    static_argnames=("depth", "height", "width"),
)


def random_grid_3d(
    key: Any,
    depth: int,
    height: int,
    width: int,
    density: float,
) -> Grid3D:
    """Generate a reproducible Bernoulli grid with shape ``(D, H, W)``."""

    validate_dimensions_3d(depth, height, width)
    return random_grid_3d_jit(key, depth, height, width, _validate_density(density))


_random_grids_3d_jit = jax.jit(
    jax.vmap(_random_grid_3d, in_axes=(0, None, None, None, None)),
    static_argnames=("depth", "height", "width"),
)


def generate_random_grids_3d(
    keys: Any,
    depth: int,
    height: int,
    width: int,
    density: float,
) -> Grid3D:
    """Generate one 3D grid per key, returning ``(batch, D, H, W)``."""

    validate_dimensions_3d(depth, height, width)
    value = _validate_density(density)
    key_array = jnp.asarray(keys)
    if key_array.ndim < 1:
        raise ValueError("keys must contain a batch of JAX PRNG keys")
    return _random_grids_3d_jit(key_array, depth, height, width, value)


def random_rule_3d(key: Any) -> Rule3D:
    """Generate a uniformly random 54-bit 3D rule."""

    bits = jax.random.bernoulli(key, p=0.5, shape=(54,))
    host_bits = jax.device_get(bits).tolist()
    return Rule3D(
        tuple(bool(bit) for bit in host_bits[:27]),
        tuple(bool(bit) for bit in host_bits[27:]),
    )


def random_rules_3d(key: Any, count: int) -> list[Rule3D]:
    """Generate ``count`` independent random 3D rules."""

    if not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    keys = jax.random.split(key, count)
    bits = jax.vmap(lambda item_key: jax.random.bernoulli(item_key, p=0.5, shape=(54,)))(keys)
    host_bits = jax.device_get(bits).tolist()
    return [
        Rule3D(
            tuple(bool(bit) for bit in row[:27]),
            tuple(bool(bit) for bit in row[27:]),
        )
        for row in host_bits
    ]
