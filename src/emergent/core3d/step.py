"""Pure JAX transition functions for 3D binary cellular automata."""

from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp

from .grid import Grid3D

_OFFSETS = tuple(
    (dz, dy, dx) for dz, dy, dx in product((-1, 0, 1), repeat=3) if (dz, dy, dx) != (0, 0, 0)
)


def count_neighbors_3d(grid: Grid3D) -> Grid3D:
    """Count live cells in the 26-cell toroidal Moore neighborhood.

    The offset tuple is static Python configuration. Every grid operation in
    the sum is a JAX array operation and is compiled when this function is
    jitted or called from a JAX control-flow primitive.
    """

    values = jnp.asarray(grid, dtype=jnp.uint8)
    counts = jnp.zeros_like(values, dtype=jnp.uint8)
    for offset in _OFFSETS:
        counts = counts + jnp.roll(values, shift=offset, axis=(0, 1, 2))
    return counts


def step_3d(grid: Grid3D, birth_mask: Grid3D, survival_mask: Grid3D) -> Grid3D:
    """Advance one 3D binary grid by one generation."""

    values = jnp.asarray(grid, dtype=jnp.uint8)
    birth = jnp.asarray(birth_mask, dtype=jnp.uint8)
    survival = jnp.asarray(survival_mask, dtype=jnp.uint8)
    counts = count_neighbors_3d(values)
    return jnp.where(values != 0, survival[counts], birth[counts]).astype(jnp.uint8)


count_neighbors_3d_jit = jax.jit(count_neighbors_3d)
step_3d_jit = jax.jit(step_3d)
batched_step_3d = jax.jit(jax.vmap(step_3d, in_axes=(0, None, None)))
