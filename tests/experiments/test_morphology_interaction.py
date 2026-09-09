from __future__ import annotations

import numpy as np

from emergent.core.rules import parse_rule, rule_to_masks
from emergent.core.step import step_jit
from emergent.experiments.morphology_interactions.canonical import canonicalize_grid
from emergent.experiments.morphology_interactions.components import Component
from emergent.experiments.morphology_interactions.config import MorphologyExperimentConfig
from emergent.experiments.morphology_interactions.engine import MorphologyInteractionEngine
from emergent.experiments.morphology_interactions.interaction import (
    CHANNEL_COUNT,
    build_interaction_zone,
    interaction,
    interaction_vector_to_rule,
    make_pair_key,
    scrambled_interaction,
)
from emergent.experiments.morphology_interactions.local_step import resolve_owner_map


def test_interaction_is_byte_stable_and_symmetric() -> None:
    first = np.asarray([[1, 1], [1, 0]], dtype=np.uint8)
    second = np.asarray([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)

    forward = interaction(first, second, seed=42, alpha=0.5)
    reverse = interaction(second, first, seed=42, alpha=0.5)
    repeated = interaction(first, second, seed=42, alpha=0.5)

    np.testing.assert_array_equal(forward.final_vector, reverse.final_vector)
    np.testing.assert_array_equal(forward.final_vector, repeated.final_vector)
    np.testing.assert_array_equal(forward.scrambled_vector, reverse.scrambled_vector)
    assert forward.final_vector.shape == (CHANNEL_COUNT,)
    assert np.all((-1.0 <= forward.final_vector) & (forward.final_vector <= 1.0))


def test_scrambled_law_depends_on_exact_key_and_universe_seed() -> None:
    shape_a = np.asarray([[1, 1], [1, 0]], dtype=np.uint8)
    shape_b = np.asarray([[1, 1]], dtype=np.uint8)
    key_a = canonicalize_grid(shape_a)
    key_b = canonicalize_grid(shape_b)

    first = scrambled_interaction(key_a, key_b, universe_seed=42)
    second = scrambled_interaction(key_b, key_a, universe_seed=42)
    other_seed = scrambled_interaction(key_a, key_b, universe_seed=43)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other_seed)


def test_rule_projection_changes_at_most_the_requested_channels() -> None:
    vector = np.linspace(-1.0, 1.0, CHANNEL_COUNT)
    base = parse_rule("B3/S23")

    rule, rule_id = interaction_vector_to_rule(
        vector,
        base,
        max_rule_changes=2,
        interaction_threshold=0.0,
    )
    changed = sum(a != b for a, b in zip(base.birth + base.survival, rule.birth + rule.survival))

    assert changed <= 2
    assert 0 <= rule_id < 2**18


def test_default_structured_landscape_reaches_local_rule_projection() -> None:
    block = np.ones((2, 2), dtype=np.uint8)

    pair = interaction(block, block, seed=42, alpha=0.0)
    base = parse_rule("B3/S23")
    base_bits = np.asarray(base.birth + base.survival)

    assert pair.local_rule_text != "B3/S23"
    assert np.count_nonzero(np.r_[pair.birth_mask, pair.survival_mask] != base_bits) <= 2


def test_zone_dilation_wraps_and_overlaps_resolve_by_pair_identity() -> None:
    first = Component(np.asarray([[2, 0]], dtype=np.int64), 1)
    second = Component(np.asarray([[2, 7]], dtype=np.int64), 1)
    zone = build_interaction_zone(
        first,
        second,
        (6, 8),
        interaction_radius=1,
        effect_padding=1,
    )
    assert zone[2, 0]
    assert zone[2, 7]
    assert zone[1, 0]
    assert zone[3, 7]

    key_a = canonicalize_grid(np.asarray([[1]], dtype=np.uint8))
    key_b = canonicalize_grid(np.asarray([[1, 1]], dtype=np.uint8))
    key_c = canonicalize_grid(np.asarray([[1], [1]], dtype=np.uint8))
    pair_one = make_pair_key(key_a, key_b)
    pair_two = make_pair_key(key_a, key_c)
    zones = {
        pair_one: np.ones((4, 4), dtype=bool),
        pair_two: np.ones((4, 4), dtype=bool),
    }
    strengths = {pair_one: 0.5, pair_two: 0.5}
    owner_map, ordered = resolve_owner_map(
        zones,
        strengths=strengths,
        grid_shape=(4, 4),
    )
    expected = ordered.index(min(ordered, key=lambda pair: (pair.species_a, pair.species_b)))
    assert np.all(owner_map == expected)


def test_active_rule_changes_are_confined_to_the_interaction_zone() -> None:
    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[8, 2] = 1
    grid[8, 4] = 1
    config = {
        "width": 20,
        "height": 20,
        "seed": 4,
        "steps": 1,
        "warmup_steps": 0,
        "alpha": 0.5,
        "interaction_radius": 2,
        "effect_padding": 1,
    }
    result = MorphologyInteractionEngine(
        MorphologyExperimentConfig(**config),
        initial_grid=grid,
    ).step()
    birth, survival = rule_to_masks(parse_rule("B3/S23"))
    native = np.asarray(step_jit(grid, birth, survival))
    changed = result.grid != native

    assert len(result.active_interactions) == 1
    assert np.any(changed)
    assert np.all(result.owner_map[changed] >= 0)
