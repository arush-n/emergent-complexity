"""Small measurements useful for UI status and simulator debugging."""

from __future__ import annotations

import jax.numpy as jnp

from .grid import Grid


def alive_count(grid: Grid) -> jnp.ndarray:
    """Return the number of non-zero cells."""

    return jnp.sum(jnp.asarray(grid) != 0, dtype=jnp.int32)


def alive_fraction(grid: Grid) -> jnp.ndarray:
    """Return the fraction of cells that are alive."""

    values = jnp.asarray(grid)
    return jnp.mean(values != 0)


def changed_cell_count(previous: Grid, current: Grid) -> jnp.ndarray:
    """Return the number of cells whose state differs between two grids."""

    return jnp.sum(jnp.asarray(previous) != jnp.asarray(current), dtype=jnp.int32)


def transition_counts(previous: Grid, current: Grid) -> jnp.ndarray:
    """Return ``[alive, changed, births, deaths]`` for one transition."""

    before = jnp.asarray(previous) != 0
    after = jnp.asarray(current) != 0
    return jnp.asarray(
        [
            jnp.sum(after, dtype=jnp.int32),
            jnp.sum(before != after, dtype=jnp.int32),
            jnp.sum(~before & after, dtype=jnp.int32),
            jnp.sum(before & ~after, dtype=jnp.int32),
        ],
        dtype=jnp.int32,
    )


def has_changed(previous: Grid, current: Grid) -> jnp.ndarray:
    """Return a scalar boolean indicating whether any cell changed."""

    return jnp.any(jnp.asarray(previous) != jnp.asarray(current))
