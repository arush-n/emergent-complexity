"""Fixed-shape JAX batches, scalar-compatible chemistry, and exact cycle detection."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..engine import RNAChemistryEngine


@jax.jit
def batched_neighbor_count(grids: jax.Array) -> jax.Array:
    """Count Moore neighbors for a batch in one compiled array program.

    ``core.step.neighbor_count`` intentionally accepts one two-dimensional
    grid.  The continuous search has a fixed leading environment axis, so it
    is cheaper to express the eight rolls directly over the trailing spatial
    axes than to build a nested Python ``vmap`` call for every launch.
    """

    values = jnp.asarray(grids, dtype=jnp.uint8)
    return sum(
        jnp.roll(values, shift=(row, col), axis=(-2, -1))
        for row in (-1, 0, 1)
        for col in (-1, 0, 1)
        if row or col
    )


@jax.jit
def step_fields(grids: jax.Array, rules: jax.Array) -> jax.Array:
    """Apply one 18-bit B/S rule per cell; count neighbors once per world."""
    counts = batched_neighbor_count(grids).astype(jnp.uint32)
    channels = counts + jnp.where(grids != 0, 9, 0).astype(jnp.uint32)
    return ((rules >> channels) & 1).astype(jnp.uint8)


def rule_code(birth: np.ndarray, survival: np.ndarray) -> int:
    return sum(int(bit) << i for i, bit in enumerate(np.r_[birth, survival]))


def pack_grids(grids: np.ndarray) -> np.ndarray:
    """Pack binary grids along their flattened cell axis for compact traces."""

    values = np.asarray(grids, dtype=np.uint8)
    if values.ndim < 2:
        raise ValueError("grids must have at least two dimensions")
    return np.packbits(values.reshape(*values.shape[:-2], -1), axis=-1, bitorder="little")


def unpack_grids(packed: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Restore grids written by :func:`pack_grids` without changing bits."""

    values = np.unpackbits(np.asarray(packed), axis=-1, bitorder="little")
    cell_count = height * width
    return values[..., :cell_count].reshape(*values.shape[:-1], height, width).astype(np.uint8)


def prepare(engine: RNAChemistryEngine, grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Reuse the scalar engine's chemical law, omitting cumulative metric history."""
    config = engine.config
    due = (
        engine.generation >= config.warmup_steps
        and (engine.generation - config.warmup_steps) % config.detect_every == 0
    )
    sites = []
    effects = []
    owners = np.full(grid.shape, -1, dtype=np.int32)
    if due and config.interactions_enabled:
        _, observations, _ = engine._observe_current(grid)
        _, sites, owners, effects = engine._prepare_interactions(observations)
    if config.binding_lifetime_mode == "energy":
        sites = engine._apply_binding_lifetime(sites)
        owners, effects = engine._resolve_active_sites(sites)
    codes = np.asarray(
        [rule_code(effect.birth_mask, effect.survival_mask) for effect in effects],
        dtype=np.uint32,
    )
    result = np.full(grid.shape, rule_code(engine.base_birth, engine.base_survival), np.uint32)
    if codes.size:
        mask = owners >= 0
        result[mask] = codes[owners[mask]]
    return result, sum(bool(np.any(site.zone)) for site in sites)


def state_bytes(engine: RNAChemistryEngine, grid: np.ndarray) -> bytes:
    """Serialize everything affecting future dynamics, excluding reporting caches.

    Persistent binding insertion order is retained because the scalar engine
    iterates it. Time is represented by warmup countdown and detection phase.
    Pickle is used only as an internal byte encoding, never to load input files.
    """
    config = engine.config
    phase = (
        max(0, config.warmup_steps - engine.generation),
        (engine.generation - config.warmup_steps) % config.detect_every,
    )
    bound = [
        (
            key,
            remaining,
            np.packbits(active.zone).tobytes(),
            rule_code(active.effect.birth_mask, active.effect.survival_mask),
            float(active.effect.strength),
            active.effect.sort_key,
        )
        for key, (active, remaining) in engine._bound_sites.items()
    ]
    return pickle.dumps((phase, np.packbits(grid).tobytes(), bound), protocol=5)


@dataclass
class ExactCycle:
    """Online Brent detection: exact equality, any period, one anchor in memory.

    Detection may occur after the first recurrence. The reported period is
    exact; no digest collision or bounded-period assumption is involved.
    """

    anchor: bytes
    power: int = 1
    distance: int = 0

    def observe(self, state: bytes) -> int | None:
        self.distance += 1
        if state == self.anchor:
            return self.distance
        if self.distance == self.power:
            self.anchor = state
            self.power *= 2
            self.distance = 0
        return None
