"""Pure JAX transition functions for binary Moore-neighborhood automata."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .grid import Grid


def neighbor_count(grid: Grid) -> Grid:
    """Count live cells in the eight-cell Moore neighborhood.

    Boundaries are toroidal: values shifted off one edge re-enter at the
    opposite edge. The center cell is deliberately excluded.
    """

    values = jnp.asarray(grid, dtype=jnp.uint8)
    return (
        jnp.roll(values, shift=(-1, -1), axis=(0, 1))
        + jnp.roll(values, shift=(-1, 0), axis=(0, 1))
        + jnp.roll(values, shift=(-1, 1), axis=(0, 1))
        + jnp.roll(values, shift=(0, -1), axis=(0, 1))
        + jnp.roll(values, shift=(0, 1), axis=(0, 1))
        + jnp.roll(values, shift=(1, -1), axis=(0, 1))
        + jnp.roll(values, shift=(1, 0), axis=(0, 1))
        + jnp.roll(values, shift=(1, 1), axis=(0, 1))
    )


def step(grid: Grid, birth_mask: Grid, survival_mask: Grid) -> Grid:
    """Advance one binary grid by one generation.

    ``birth_mask`` and ``survival_mask`` are length-nine arrays indexed by the
    live-neighbor count. No Python-side state is read or modified.
    """

    values = jnp.asarray(grid, dtype=jnp.uint8)
    birth = jnp.asarray(birth_mask, dtype=jnp.uint8)
    survival = jnp.asarray(survival_mask, dtype=jnp.uint8)
    counts = neighbor_count(values)
    return jnp.where(values != 0, survival[counts], birth[counts]).astype(jnp.uint8)


neighbor_count_jit = jax.jit(neighbor_count)
step_jit = jax.jit(step)
batched_step = jax.jit(jax.vmap(step, in_axes=(0, None, None)))
