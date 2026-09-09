"""One-pass JAX stepping with spatially scoped pair-specific Life rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ...core.step import neighbor_count, neighbor_count_jit, step_jit
from .canonical import shape_key_sort_key
from .interaction import PairInteraction, PairKey


def _pair_sort_key(pair: PairKey) -> tuple[tuple[bytes, int, int], tuple[bytes, int, int]]:
    return shape_key_sort_key(pair.species_a), shape_key_sort_key(pair.species_b)


def resolve_owner_map(
    zones: Mapping[PairKey, Any] | Iterable[tuple[PairKey, Any]],
    *,
    strengths: Mapping[PairKey, float],
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, list[PairKey]]:
    """Resolve overlapping zones by strength, then pair identity.

    Returned owner slots are ordered by canonical pair identity.  Because
    equal-strength candidates are visited in that order and only strict
    improvements replace an owner, exact ties resolve to the lexicographically
    first pair on every platform.
    """

    height, width = grid_shape
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError("grid_shape dimensions must be positive integers")
    zone_items = list(zones.items()) if isinstance(zones, Mapping) else list(zones)
    merged: dict[PairKey, np.ndarray] = {}
    for pair, zone in zone_items:
        if not isinstance(pair, PairKey):
            raise TypeError("zone keys must be PairKey values")
        values = np.asarray(zone, dtype=bool)
        if values.shape != (height, width):
            raise ValueError("each interaction zone must match grid_shape")
        if pair not in strengths:
            raise KeyError(f"missing interaction strength for {pair!r}")
        merged[pair] = values.copy() if pair not in merged else merged[pair] | values

    ordered_pairs = sorted(
        (pair for pair, zone in merged.items() if np.any(zone)),
        key=_pair_sort_key,
    )
    owner_map = np.full((height, width), -1, dtype=np.int32)
    best_strength = np.full((height, width), -np.inf, dtype=np.float64)
    for owner_index, pair in enumerate(ordered_pairs):
        strength = float(strengths[pair])
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("interaction strengths must be finite and non-negative")
        zone = merged[pair]
        replace = zone & (strength > best_strength)
        owner_map[replace] = owner_index
        best_strength[replace] = strength
    return owner_map, ordered_pairs


def pair_rule_arrays(
    pair_keys: Iterable[PairKey],
    interactions: Mapping[PairKey, PairInteraction],
    *,
    capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack cached pair rules into ``(pair, count)`` NumPy arrays.

    ``capacity`` optionally pads the tables with inert zero rows.  The engine
    uses a stable capacity so JAX does not recompile merely because the number
    of active pair identities changed between generations.
    """

    ordered = list(pair_keys)
    if capacity is None:
        table_size = len(ordered)
    else:
        if not isinstance(capacity, int) or capacity < len(ordered) or capacity < 1:
            raise ValueError("capacity must be at least one and contain every pair key")
        table_size = capacity
    if not ordered and capacity is None:
        return np.zeros((0, 9), dtype=np.uint8), np.zeros((0, 9), dtype=np.uint8)
    birth = np.zeros((table_size, 9), dtype=np.uint8)
    survival = np.zeros((table_size, 9), dtype=np.uint8)
    if ordered:
        birth[: len(ordered)] = np.asarray(
            [interactions[pair].birth_mask for pair in ordered],
            dtype=np.uint8,
        )
        survival[: len(ordered)] = np.asarray(
            [interactions[pair].survival_mask for pair in ordered],
            dtype=np.uint8,
        )
    if birth.shape != (table_size, 9) or survival.shape != (table_size, 9):
        raise ValueError("cached pair masks must have shape (pair_count, 9)")
    return birth, survival


def _step_with_interactions(
    grid: Any,
    owner_map: Any,
    pair_birth_masks: Any,
    pair_survival_masks: Any,
    base_birth: Any,
    base_survival: Any,
) -> jax.Array:
    values = jnp.asarray(grid, dtype=jnp.uint8)
    owners = jnp.asarray(owner_map, dtype=jnp.int32)
    birth_masks = jnp.asarray(pair_birth_masks, dtype=jnp.uint8)
    survival_masks = jnp.asarray(pair_survival_masks, dtype=jnp.uint8)
    birth = jnp.asarray(base_birth, dtype=jnp.uint8)
    survival = jnp.asarray(base_survival, dtype=jnp.uint8)

    # A zero-pair call is the exact native path.  In addition to being cheaper,
    # this avoids indexing an empty pair-mask table when no encounter exists.
    if birth_masks.shape[0] == 0:
        return step_jit(values, birth, survival)

    counts = neighbor_count_jit(values)
    base_result = jnp.where(values != 0, survival[counts], birth[counts]).astype(jnp.uint8)
    pair_index = jnp.clip(owners, 0, birth_masks.shape[0] - 1)
    local_birth = birth_masks[pair_index, counts]
    local_survival = survival_masks[pair_index, counts]
    local_result = jnp.where(values != 0, local_survival, local_birth).astype(jnp.uint8)
    return jnp.where(owners >= 0, local_result, base_result).astype(jnp.uint8)


step_with_interactions = jax.jit(_step_with_interactions)
experimental_step = step_with_interactions


def _step_with_interactions_single_batch(
    grid: Any,
    owner_map: Any,
    pair_birth_masks: Any,
    pair_survival_masks: Any,
    base_birth: Any,
    base_survival: Any,
) -> jax.Array:
    """Step one batch member; the outer vmap supplies environment parallelism."""

    values = jnp.asarray(grid, dtype=jnp.uint8)
    owners = jnp.asarray(owner_map, dtype=jnp.int32)
    birth_masks = jnp.asarray(pair_birth_masks, dtype=jnp.uint8)
    survival_masks = jnp.asarray(pair_survival_masks, dtype=jnp.uint8)
    birth = jnp.asarray(base_birth, dtype=jnp.uint8)
    survival = jnp.asarray(base_survival, dtype=jnp.uint8)

    counts = neighbor_count(values)
    base_result = jnp.where(values != 0, survival[counts], birth[counts]).astype(jnp.uint8)
    pair_index = jnp.clip(owners, 0, birth_masks.shape[0] - 1)
    local_birth = birth_masks[pair_index, counts]
    local_survival = survival_masks[pair_index, counts]
    local_result = jnp.where(values != 0, local_survival, local_birth).astype(jnp.uint8)
    return jnp.where(owners >= 0, local_result, base_result).astype(jnp.uint8)


step_batch_with_interactions = jax.jit(
    jax.vmap(
        _step_with_interactions_single_batch,
        in_axes=(0, 0, 0, 0, None, None),
    )
)
