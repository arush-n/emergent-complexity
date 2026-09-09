"""Small descriptive measurements for 3D simulator diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from .grid import Grid3D


def alive_count_3d(grid: Grid3D) -> Grid3D:
    """Return the number of live voxels."""

    return jnp.sum(jnp.asarray(grid, dtype=jnp.uint8), dtype=jnp.int32)


def alive_fraction_3d(grid: Grid3D) -> Grid3D:
    """Return the fraction of live voxels."""

    values = jnp.asarray(grid, dtype=jnp.uint8)
    return alive_count_3d(values) / values.size


def transition_counts_3d(previous: Grid3D, current: Grid3D) -> Grid3D:
    """Return ``[alive, changed, births, deaths]`` for one transition."""

    before = jnp.asarray(previous, dtype=jnp.uint8)
    after = jnp.asarray(current, dtype=jnp.uint8)
    births = (before == 0) & (after != 0)
    deaths = (before != 0) & (after == 0)
    return jnp.asarray(
        [
            jnp.sum(after, dtype=jnp.int32),
            jnp.sum(before != after, dtype=jnp.int32),
            jnp.sum(births, dtype=jnp.int32),
            jnp.sum(deaths, dtype=jnp.int32),
        ],
        dtype=jnp.int32,
    )
