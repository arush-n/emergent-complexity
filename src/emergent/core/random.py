"""Explicit-key random initial conditions and random rule generation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .grid import Grid, validate_dimensions
from .rules import Rule


def key_from_seed(seed: int) -> jax.Array:
    """Create a modern JAX PRNG key from an integer seed."""

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if hasattr(jax.random, "key"):
        return jax.random.key(seed)
    return jax.random.PRNGKey(seed)


def _validate_density(density: float) -> None:
    if not isinstance(density, (int, float)):
        raise TypeError("density must be a number")
    if not 0.0 <= float(density) <= 1.0:
        raise ValueError("density must be between 0 and 1")


def _random_grid(key: Any, height: int, width: int, density: float) -> Grid:
    return jax.random.bernoulli(key, p=density, shape=(height, width)).astype(jnp.uint8)


random_grid_jit = jax.jit(_random_grid, static_argnames=("height", "width"))


def random_grid(key: Any, height: int, width: int, density: float) -> Grid:
    """Generate an independent Bernoulli grid using the supplied JAX key."""

    validate_dimensions(height, width)
    _validate_density(density)
    return random_grid_jit(key, height, width, float(density))


_random_grids_jit = jax.jit(
    jax.vmap(_random_grid, in_axes=(0, None, None, None)),
    static_argnames=("height", "width"),
)


def generate_random_grids(
    keys: Any,
    height: int,
    width: int,
    density: float,
) -> Grid:
    """Generate one grid per key, returning ``(batch, height, width)``."""

    validate_dimensions(height, width)
    _validate_density(density)
    key_array = jnp.asarray(keys)
    if key_array.ndim < 1:
        raise ValueError("keys must contain a batch of JAX PRNG keys")
    return _random_grids_jit(key_array, height, width, float(density))


def random_grid_from_seed(
    seed: int,
    height: int,
    width: int,
    density: float,
) -> Grid:
    """Convenience wrapper for deterministic seed-based initialization."""

    return random_grid(key_from_seed(seed), height, width, density)


def random_rule(key: Any) -> Rule:
    """Generate a uniformly random binary outer-totalistic rule."""

    bits = jax.random.bernoulli(key, p=0.5, shape=(18,))
    host_bits = jax.device_get(bits).tolist()
    return Rule(
        tuple(bool(bit) for bit in host_bits[:9]),
        tuple(bool(bit) for bit in host_bits[9:]),
    )


def random_rules(key: Any, count: int) -> list[Rule]:
    """Generate ``count`` independent random rules using split keys."""

    if not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    keys = jax.random.split(key, count)
    bits = jax.vmap(lambda item_key: jax.random.bernoulli(item_key, p=0.5, shape=(18,)))(keys)
    host_bits = jax.device_get(bits).tolist()
    return [
        Rule(
            tuple(bool(bit) for bit in row[:9]),
            tuple(bool(bit) for bit in row[9:]),
        )
        for row in host_bits
    ]
