from __future__ import annotations

import numpy as np

from emergent.core.rules import parse_rule, rule_to_masks
from emergent.core.step import step_jit
from emergent.experiments.morphology_interactions.config import MorphologyExperimentConfig
from emergent.experiments.morphology_interactions.engine import MorphologyInteractionEngine
from emergent.experiments.morphology_interactions.interaction import interaction
from emergent.experiments.morphology_interactions.species import SpeciesRegistry


def _initial_encounter_grid() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=np.uint8)
    grid[4:6, 3:5] = 1
    grid[4:6, 7:9] = 1
    return grid


def test_registry_and_reappearing_pair_recover_the_same_law() -> None:
    registry = SpeciesRegistry()
    key_grid = np.asarray([[1, 1], [1, 0]], dtype=np.uint8)
    from emergent.experiments.morphology_interactions.canonical import canonicalize_grid

    key = canonicalize_grid(key_grid)
    first, is_new = registry.observe(key, generation=0, cell_count=3)
    second, is_new_again = registry.observe(key, generation=10, cell_count=3)
    law_first = interaction(key_grid, np.asarray([[1, 1]], dtype=np.uint8), seed=42)
    law_again = interaction(key_grid, np.asarray([[1, 1]], dtype=np.uint8), seed=42)

    assert is_new is True
    assert is_new_again is False
    assert first is second
    assert second.observations == 2
    assert second.species_id == 0
    np.testing.assert_array_equal(law_first.final_vector, law_again.final_vector)
    assert law_first.rule_id == law_again.rule_id


def test_interactions_disabled_replays_native_cgol_bit_for_bit() -> None:
    initial = _initial_encounter_grid()
    config = MorphologyExperimentConfig(
        width=16,
        height=16,
        seed=7,
        steps=6,
        warmup_steps=0,
        interactions_enabled=False,
    )
    result = MorphologyInteractionEngine(config, initial_grid=initial).run()
    birth, survival = rule_to_masks(parse_rule("B3/S23"))
    expected = initial.copy()
    for _ in range(config.steps):
        expected = np.asarray(step_jit(expected, birth, survival))

    np.testing.assert_array_equal(result.final_grid, expected)


def test_complete_engine_replay_has_identical_deterministic_outputs() -> None:
    initial = _initial_encounter_grid()
    config = MorphologyExperimentConfig(
        width=16,
        height=16,
        seed=11,
        steps=8,
        warmup_steps=0,
        alpha=0.5,
        snapshot_every=3,
    )
    first = MorphologyInteractionEngine(config, initial_grid=initial).run()
    second = MorphologyInteractionEngine(config, initial_grid=initial).run()

    np.testing.assert_array_equal(first.final_grid, second.final_grid)
    assert first.generation_records == second.generation_records
    assert list(first.snapshots) == list(second.snapshots)
    for generation in first.snapshots:
        np.testing.assert_array_equal(first.snapshots[generation], second.snapshots[generation])
    assert list(first.species_registry.keys()) == list(second.species_registry.keys())
    assert list(first.interaction_cache.keys()) == list(second.interaction_cache.keys())
    for pair in first.interaction_cache:
        np.testing.assert_array_equal(
            first.interaction_cache[pair].final_vector,
            second.interaction_cache[pair].final_vector,
        )
        assert first.interaction_cache[pair].encounters == second.interaction_cache[pair].encounters


def test_sparse_metrics_do_not_lose_species_discoveries_from_summary() -> None:
    initial = np.zeros((8, 8), dtype=np.uint8)
    initial[3:5, 3:5] = 1
    config = MorphologyExperimentConfig(
        width=8,
        height=8,
        seed=19,
        steps=2,
        warmup_steps=0,
        interactions_enabled=False,
        metrics_every=2,
    )

    result = MorphologyInteractionEngine(config, initial_grid=initial).run()

    assert len(result.species_registry) == 1
    assert result.summary["total_new_species"] == len(result.species_registry)
    assert sum(int(row["new_species_this_step"]) for row in result.generation_records) == 0
