"""Fixed-shape JAX batches, scalar-compatible chemistry, and exact cycle detection."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ...canonical import ShapeKey
from ...components import detect_components_batch
from ..chemistry import PairChemistry
from ..engine import ActiveSite, RNAChemistryEngine


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


# The continuous runner has already copied the previous grids to the host for
# tracing and terminal checks, so it never needs the old device buffer after a
# transition. Donation lets XLA reuse that buffer for the next state. Keep the
# regular non-donating entry point above for replay/tests and callers that may
# legitimately retain their input array.
step_fields_donated = jax.jit(step_fields, donate_argnums=(0,))


@jax.jit
def unchanged_batch(before: jax.Array, after: jax.Array) -> jax.Array:
    """Return an exact no-change flag for every fixed-shape environment."""

    return jnp.all(
        jnp.asarray(before, dtype=jnp.uint8) == jnp.asarray(after, dtype=jnp.uint8),
        axis=(-2, -1),
    )


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


@dataclass(frozen=True)
class PreparedFields:
    """One environment's dense JAX input plus lightweight chemistry evidence."""

    rules: np.ndarray
    active_zone_count: int
    shape_keys: tuple[ShapeKey, ...]
    pair_chemistries: tuple[PairChemistry, ...]
    active_sites: tuple[ActiveSite, ...]


def prepare_details(
    engine: RNAChemistryEngine,
    grid: np.ndarray,
    *,
    components: list[object] | None = None,
) -> PreparedFields:
    """Prepare one rule field and retain evidence for search annotations.

    Component detection is supplied by ``prepare_batch`` when possible.  The
    scalar path remains public for the engine/replay tests and for callers
    that are not running a lockstep batch.
    """

    config = engine.config
    due = (
        engine.generation >= config.warmup_steps
        and (engine.generation - config.warmup_steps) % config.detect_every == 0
    )
    sites = []
    effects = []
    owners = np.full(grid.shape, -1, dtype=np.int32)
    observations = []
    pair_chemistries = []
    if due and config.interactions_enabled:
        if components is None:
            _, observations, _ = engine._observe_current(grid)
        else:
            _, observations, _ = engine._observe_components(components)
        pair_chemistries, sites, owners, effects = engine._prepare_interactions(observations)
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
    active_sites = tuple(sites)
    return PreparedFields(
        rules=result,
        active_zone_count=sum(bool(np.any(site.zone)) for site in active_sites),
        shape_keys=tuple(item[1] for item in observations),
        pair_chemistries=tuple(pair_chemistries),
        active_sites=active_sites,
    )


def prepare(engine: RNAChemistryEngine, grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Scalar-compatible wrapper around :func:`prepare_details`."""

    details = prepare_details(engine, grid)
    return details.rules, details.active_zone_count


def prepare_batch(
    engines: Sequence[RNAChemistryEngine],
    grids: np.ndarray,
    *,
    components_by_environment: Sequence[list[object] | None] | None = None,
) -> list[PreparedFields]:
    """Detect toroidal components once per compatible batch, then prepare rules.

    Chemistry itself remains per-environment because sequences and successful
    sites are ragged.  The expensive grid labeling is nevertheless performed
    in one C-backed batched call for each detector configuration, and the
    resulting dense rule fields are handed to the single JAX transition.
    """

    if not engines:
        return []
    values = (
        np.asarray(grids, dtype=np.uint8)
        if isinstance(grids, np.ndarray)
        else np.asarray(jax.device_get(grids), dtype=np.uint8)
    )
    if values.ndim != 3 or len(engines) != values.shape[0]:
        raise ValueError("engines and grids must describe a matching (batch, height, width)")
    if components_by_environment is not None and len(components_by_environment) != len(engines):
        raise ValueError("components_by_environment must match the engine batch")
    due_groups: dict[tuple[int, str], list[int]] = {}
    for index, engine in enumerate(engines):
        config = engine.config
        due = (
            engine.generation >= config.warmup_steps
            and (engine.generation - config.warmup_steps) % config.detect_every == 0
        )
        if due and config.interactions_enabled:
            due_groups.setdefault(
                (config.min_component_cells, config.component_backend), []
            ).append(index)
    detected: dict[int, list[object]] = {}
    if components_by_environment is not None:
        detected.update(
            {
                index: components
                for index, components in enumerate(components_by_environment)
                if components is not None
            }
        )
    else:
        for (minimum_cells, backend), indices in due_groups.items():
            component_batches = detect_components_batch(
                values[indices],
                min_component_cells=minimum_cells,
                backend=backend,
            )
            detected.update(
                {index: components for index, components in zip(indices, component_batches)}
            )
    return [
        prepare_details(engine, values[index], components=detected.get(index))
        for index, engine in enumerate(engines)
    ]


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


def grid_state_bytes(grid: np.ndarray) -> bytes:
    """Serialize only a fixed-shape grid for exact spatial recurrence checks."""

    values = np.asarray(grid, dtype=np.uint8)
    return np.packbits(np.ascontiguousarray(values).reshape(-1), bitorder="little").tobytes()


def morphology_state_bytes(counts: Mapping[ShapeKey, int]) -> bytes:
    """Serialize the exact multiset of canonical morphologies in a grid."""

    ordered = tuple(
        (key.height, key.width, key.packed, int(count))
        for key, count in sorted(
            counts.items(), key=lambda item: (item[0].height, item[0].width, item[0].packed)
        )
    )
    return pickle.dumps(ordered, protocol=5)


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
