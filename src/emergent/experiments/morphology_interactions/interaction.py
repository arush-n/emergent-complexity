"""Deterministic morphology-pair interactions and spatial encounter zones."""

from __future__ import annotations

import hashlib
import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.rules import Rule, format_rule, masks_to_rule, parse_rule, rule_to_int
from .canonical import (
    ShapeKey,
    canonical_matrix_from_grid,
    shape_key_from_matrix,
    shape_key_sort_key,
)
from .components import Component
from .encoding import MorphologyEncoder, deterministic_uint64

CHANNEL_COUNT = 18
NEIGHBOR_COUNT = 9
STRUCTURED_WEIGHT_SCALE = 2.0


def _as_rule(rule: Rule | str) -> Rule:
    if isinstance(rule, str):
        return parse_rule(rule)
    if not isinstance(rule, Rule):
        raise TypeError("base_rule must be a Rule or rule string")
    return rule


def _non_negative_integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _make_structured_weights(universe_seed: int, identity_dim: int) -> np.ndarray:
    values = np.asarray(
        [
            deterministic_uint64(universe_seed, "structured-weight", index) / float(2**64)
            for index in range(CHANNEL_COUNT * identity_dim * identity_dim)
        ],
        dtype=np.float64,
    )
    # Encodings are unit-norm vectors.  A modest fixed gain keeps the
    # structured landscape visible through the default 0.35 rule-projection
    # threshold while preserving the requested beta/sqrt(d) bilinear form.
    values = (STRUCTURED_WEIGHT_SCALE * (2.0 * values - 1.0)).reshape(
        (CHANNEL_COUNT, identity_dim, identity_dim)
    )
    weights = 0.5 * (values + np.swapaxes(values, 1, 2))
    weights.setflags(write=False)
    return weights


@dataclass(frozen=True)
class InteractionUniverse:
    """The fixed mathematical law queried by all species in one run."""

    universe_seed: int
    identity_dim: int = 32
    beta: float = 1.0
    encoder: MorphologyEncoder = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.universe_seed, int):
            raise TypeError("universe_seed must be an integer")
        if not isinstance(self.identity_dim, int) or self.identity_dim < 6:
            raise ValueError("identity_dim must be at least 6")
        if not isinstance(self.beta, (int, float)) or float(self.beta) < 0:
            raise ValueError("beta must be a non-negative number")
        encoder = MorphologyEncoder(self.universe_seed, self.identity_dim)
        weights = _make_structured_weights(self.universe_seed, self.identity_dim)
        object.__setattr__(self, "encoder", encoder)
        object.__setattr__(self, "weights", weights)

    @property
    def seed(self) -> int:
        """Short alias used in manifests and small analysis scripts."""

        return self.universe_seed


def make_interaction_universe(
    universe_seed: int | None = None,
    *,
    seed: int | None = None,
    identity_dim: int = 32,
    beta: float = 1.0,
) -> InteractionUniverse:
    """Construct one universe and generate its fixed frequencies and matrices."""

    if universe_seed is None:
        if seed is None:
            raise TypeError("universe_seed or seed is required")
        universe_seed = seed
    elif seed is not None and int(seed) != int(universe_seed):
        raise ValueError("universe_seed and seed must agree when both are supplied")
    return InteractionUniverse(universe_seed, identity_dim=identity_dim, beta=beta)


def structured_interaction(
    z_a: Any,
    z_b: Any,
    weights: Any,
    *,
    beta: float = 1.0,
) -> np.ndarray:
    """Evaluate the symmetric bilinear interaction landscape ``u(A, B)``."""

    first = np.asarray(z_a, dtype=np.float64)
    second = np.asarray(z_b, dtype=np.float64)
    matrices = np.asarray(weights, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("z_a and z_b must be one-dimensional vectors of equal size")
    if matrices.shape != (CHANNEL_COUNT, first.size, first.size):
        raise ValueError("weights must have shape (18, identity_dim, identity_dim)")
    if not isinstance(beta, (int, float)) or float(beta) < 0:
        raise ValueError("beta must be a non-negative number")
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
    # Use one byte-stable operand order as well as symmetric matrices.  The
    # mathematical bilinear form is symmetric either way, but this makes the
    # public function exactly array-equal under argument reversal despite
    # floating-point reduction order.
    first_bytes = np.ascontiguousarray(first).tobytes()
    second_bytes = np.ascontiguousarray(second).tobytes()
    if second_bytes < first_bytes:
        first, second = second, first
    bilinear = np.einsum("i,kij,j->k", first, symmetric, second, optimize=False)
    return np.tanh(float(beta) / np.sqrt(float(first.size)) * bilinear).astype(np.float64)


def _pair_digest_word(
    universe_seed: int,
    key_a: ShapeKey,
    key_b: ShapeKey,
    channel: int,
) -> int:
    pair = make_pair_key(key_a, key_b)
    digest = hashlib.blake2b(digest_size=8, person=b"morph-scramble")
    digest.update((universe_seed & ((1 << 64) - 1)).to_bytes(8, "little"))
    digest.update(pair.species_a.to_bytes())
    digest.update(pair.species_b.to_bytes())
    digest.update(int(channel).to_bytes(4, "little", signed=False))
    return int.from_bytes(digest.digest(), "little", signed=False)


def scrambled_interaction(
    key_a: ShapeKey,
    key_b: ShapeKey,
    *,
    universe_seed: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return a symmetric deterministic pseudorandom vector in ``[-1, 1]``."""

    if universe_seed is None:
        if seed is None:
            raise TypeError("universe_seed or seed is required")
        universe_seed = seed
    elif seed is not None and int(seed) != int(universe_seed):
        raise ValueError("universe_seed and seed must agree when both are supplied")
    words = np.asarray(
        [
            _pair_digest_word(universe_seed, key_a, key_b, channel)
            for channel in range(CHANNEL_COUNT)
        ],
        dtype=np.uint64,
    )
    values = words.astype(np.float64) / float(2**64)
    return (2.0 * values - 1.0).astype(np.float64)


def interpolate_interaction(
    structured: Any,
    scrambled: Any,
    *,
    alpha: float,
) -> np.ndarray:
    """Interpolate structured and scrambled laws at a configured ``alpha``."""

    if not isinstance(alpha, (int, float)) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    first = np.asarray(structured, dtype=np.float64)
    second = np.asarray(scrambled, dtype=np.float64)
    if first.shape != (CHANNEL_COUNT,) or second.shape != (CHANNEL_COUNT,):
        raise ValueError("interaction vectors must contain exactly 18 channels")
    return ((1.0 - float(alpha)) * first + float(alpha) * second).astype(np.float64)


def interaction_vector_to_rule(
    vector: Any,
    base_rule: Rule | str = "B3/S23",
    *,
    max_rule_changes: int = 2,
    interaction_threshold: float = 0.35,
) -> tuple[Rule, int]:
    """Map at most the strongest bounded interaction channels into a rule."""

    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (CHANNEL_COUNT,):
        raise ValueError("interaction vector must contain exactly 18 channels")
    if not np.all(np.isfinite(values)) or np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("interaction vector values must be finite and lie in [-1, 1]")
    changes = _non_negative_integer(max_rule_changes, "max_rule_changes")
    if (
        not isinstance(interaction_threshold, (int, float))
        or not 0.0 <= float(interaction_threshold) <= 1.0
    ):
        raise ValueError("interaction_threshold must be between 0 and 1")

    base = _as_rule(base_rule)
    base_bits = np.asarray(base.birth + base.survival, dtype=np.uint8)
    desired = values > 0.0
    candidates = [
        (abs(float(values[channel])), channel)
        for channel in range(CHANNEL_COUNT)
        if 0.0 < abs(float(values[channel])) >= float(interaction_threshold)
        and bool(desired[channel]) != bool(base_bits[channel])
    ]
    # Sort strongest first; channel order resolves exact ties reproducibly.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[:changes]
    updated = base_bits.copy()
    for _, channel in selected:
        updated[channel] = 1 if desired[channel] else 0
    rule = masks_to_rule(updated[:NEIGHBOR_COUNT], updated[NEIGHBOR_COUNT:])
    return rule, rule_to_int(rule)


@dataclass
class PairInteraction:
    """Cached permanent law for one unordered pair of species identities."""

    species_a: ShapeKey
    species_b: ShapeKey
    structured_vector: np.ndarray
    scrambled_vector: np.ndarray
    final_vector: np.ndarray
    birth_mask: np.ndarray
    survival_mask: np.ndarray
    rule_id: int
    strength: float
    local_rule: Rule | None = None
    first_seen_generation: int = 0
    encounters: int = 0

    @property
    def pair_key(self) -> PairKey:
        return PairKey(self.species_a, self.species_b)

    @property
    def local_rule_text(self) -> str:
        """Return the stable B/S representation used in artifacts."""

        if self.local_rule is None:
            return ""
        return format_rule(self.local_rule)


@dataclass(frozen=True, order=True)
class PairKey:
    """Unordered pair of exact shape identities."""

    species_a: ShapeKey
    species_b: ShapeKey

    def __post_init__(self) -> None:
        if not isinstance(self.species_a, ShapeKey) or not isinstance(self.species_b, ShapeKey):
            raise TypeError("pair keys must contain ShapeKey values")
        first, second = sorted(
            (self.species_a, self.species_b),
            key=shape_key_sort_key,
        )
        object.__setattr__(self, "species_a", first)
        object.__setattr__(self, "species_b", second)


def make_pair_key(key_a: ShapeKey, key_b: ShapeKey) -> PairKey:
    """Construct the stable unordered cache key for two species."""

    return PairKey(key_a, key_b)


def make_pair_interaction(
    key_a: ShapeKey,
    key_b: ShapeKey,
    z_a: Any,
    z_b: Any,
    *,
    universe: InteractionUniverse,
    alpha: float,
    base_rule: Rule | str = "B3/S23",
    max_rule_changes: int = 2,
    interaction_threshold: float = 0.35,
    first_seen_generation: int = 0,
) -> PairInteraction:
    """Calculate a pair law once, using canonical ordering for symmetry."""

    pair = make_pair_key(key_a, key_b)
    vector_by_key = {
        key_a: np.asarray(z_a, dtype=np.float64),
        key_b: np.asarray(z_b, dtype=np.float64),
    }
    first_vector = vector_by_key[pair.species_a]
    second_vector = vector_by_key[pair.species_b]
    structured = structured_interaction(
        first_vector,
        second_vector,
        universe.weights,
        beta=universe.beta,
    )
    scrambled = scrambled_interaction(
        pair.species_a,
        pair.species_b,
        universe_seed=universe.universe_seed,
    )
    final = interpolate_interaction(structured, scrambled, alpha=alpha)
    local_rule, rule_id = interaction_vector_to_rule(
        final,
        base_rule,
        max_rule_changes=max_rule_changes,
        interaction_threshold=interaction_threshold,
    )
    birth_mask = np.asarray(local_rule.birth, dtype=np.uint8)
    survival_mask = np.asarray(local_rule.survival, dtype=np.uint8)
    for value in (structured, scrambled, final, birth_mask, survival_mask):
        value.setflags(write=False)
    return PairInteraction(
        species_a=pair.species_a,
        species_b=pair.species_b,
        structured_vector=structured,
        scrambled_vector=scrambled,
        final_vector=final,
        birth_mask=birth_mask,
        survival_mask=survival_mask,
        rule_id=rule_id,
        strength=float(np.mean(np.abs(final))),
        local_rule=local_rule,
        first_seen_generation=first_seen_generation,
        encounters=0,
    )


def _key_and_matrix(shape: ShapeKey | Any) -> tuple[ShapeKey, np.ndarray]:
    if isinstance(shape, ShapeKey):
        from .canonical import matrix_from_shape_key

        return shape, matrix_from_shape_key(shape)
    matrix = canonical_matrix_from_grid(shape)
    return shape_key_from_matrix(matrix), matrix


def interaction(
    shape_a: ShapeKey | Any,
    shape_b: ShapeKey | Any,
    *,
    seed: int = 42,
    alpha: float = 0.5,
    beta: float = 1.0,
    base_rule: Rule | str = "B3/S23",
    identity_dim: int = 32,
    max_rule_changes: int = 2,
    interaction_threshold: float = 0.35,
) -> PairInteraction:
    """Convenience API used by small deterministic interaction tests."""

    key_a, _ = _key_and_matrix(shape_a)
    key_b, _ = _key_and_matrix(shape_b)
    universe = make_interaction_universe(seed, identity_dim=identity_dim, beta=beta)
    z_a = universe.encoder.encode(key_a)
    z_b = universe.encoder.encode(key_b)
    return make_pair_interaction(
        key_a,
        key_b,
        z_a,
        z_b,
        universe=universe,
        alpha=alpha,
        base_rule=base_rule,
        max_rule_changes=max_rule_changes,
        interaction_threshold=interaction_threshold,
    )


calculate_interaction = interaction


def calculate_interaction_vector(
    shape_a: ShapeKey | Any,
    shape_b: ShapeKey | Any,
    **kwargs: Any,
) -> np.ndarray:
    """Return only the final 18-channel vector from :func:`interaction`."""

    return interaction(shape_a, shape_b, **kwargs).final_vector.copy()


interaction_vector = calculate_interaction_vector


def toroidal_dilate(mask: Any, radius: int) -> np.ndarray:
    """Dilate a 2D mask with a toroidal Chebyshev neighborhood."""

    radius = _non_negative_integer(radius, "radius")
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    result = np.zeros_like(values, dtype=bool)
    for row_delta in range(-radius, radius + 1):
        for col_delta in range(-radius, radius + 1):
            result |= np.roll(values, shift=(row_delta, col_delta), axis=(0, 1))
    return result


def component_pair_is_close(
    component_a: Component,
    component_b: Component,
    grid_shape: tuple[int, int],
    interaction_radius: int = 2,
) -> bool:
    """Return whether two components are within toroidal Moore distance."""

    radius = _non_negative_integer(interaction_radius, "interaction_radius")
    height, width = grid_shape
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError("grid_shape dimensions must be positive integers")
    second = component_b.coordinates
    for row, col in component_a.coordinates:
        row_distance = np.minimum(np.abs(second[:, 0] - row), height - np.abs(second[:, 0] - row))
        col_distance = np.minimum(np.abs(second[:, 1] - col), width - np.abs(second[:, 1] - col))
        if np.any(np.maximum(row_distance, col_distance) <= radius):
            return True
    return False


def find_interacting_pairs(
    components: list[Component] | tuple[Component, ...],
    grid_shape: tuple[int, int],
    interaction_radius: int = 2,
) -> list[tuple[int, int]]:
    """Return deterministic component-index pairs that are physically close.

    A component-owner grid turns the spatial search into local neighborhood
    lookups rather than an all-components/all-components comparison.
    """

    radius = _non_negative_integer(interaction_radius, "interaction_radius")
    height, width = grid_shape
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError("grid_shape dimensions must be positive integers")
    owner = np.full((height, width), -1, dtype=np.int32)
    for component_index, component in enumerate(components):
        owner[component.coordinates[:, 0], component.coordinates[:, 1]] = component_index

    pairs: set[tuple[int, int]] = set()
    for component_index, component in enumerate(components):
        for row, col in component.coordinates:
            for row_delta in range(-radius, radius + 1):
                for col_delta in range(-radius, radius + 1):
                    other_index = int(
                        owner[(int(row) + row_delta) % height, (int(col) + col_delta) % width]
                    )
                    if other_index >= 0 and other_index != component_index:
                        pairs.add(tuple(sorted((component_index, other_index))))
    return sorted(pairs)


def build_interaction_zone(
    component_a: Component,
    component_b: Component,
    grid_shape: tuple[int, int],
    *,
    interaction_radius: int = 2,
    effect_padding: int = 1,
) -> np.ndarray:
    """Build the toroidal local effect mask for one component encounter."""

    radius = _non_negative_integer(interaction_radius, "interaction_radius")
    padding = _non_negative_integer(effect_padding, "effect_padding")
    height, width = grid_shape
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError("grid_shape dimensions must be positive integers")
    first_coordinates = {tuple(value) for value in component_a.coordinates.tolist()}
    second_coordinates = {tuple(value) for value in component_b.coordinates.tolist()}

    def near(source: set[tuple[int, int]], target: set[tuple[int, int]]) -> set[tuple[int, int]]:
        result: set[tuple[int, int]] = set()
        for row, col in source:
            if any(
                ((row + row_delta) % height, (col + col_delta) % width) in target
                for row_delta in range(-radius, radius + 1)
                for col_delta in range(-radius, radius + 1)
            ):
                result.add((row, col))
        return result

    encounter = near(first_coordinates, second_coordinates) | near(
        second_coordinates,
        first_coordinates,
    )
    zone_coordinates = {
        ((row + row_delta) % height, (col + col_delta) % width)
        for row, col in encounter
        for row_delta in range(-padding, padding + 1)
        for col_delta in range(-padding, padding + 1)
    }
    zone = np.zeros((height, width), dtype=bool)
    if zone_coordinates:
        rows, cols = zip(*zone_coordinates)
        zone[np.asarray(rows), np.asarray(cols)] = True
    return zone


def pair_rule_text(pair_interaction: PairInteraction) -> str:
    """Return the local rule text for concise reports."""

    return pair_interaction.local_rule_text
