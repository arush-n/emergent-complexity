from __future__ import annotations

from dataclasses import replace

import numpy as np

from emergent.experiments.morphology_interactions.config import MorphologyExperimentConfig
from emergent.experiments.morphology_interactions.engine import MorphologyInteractionEngine
from emergent.experiments.morphology_interactions.parallel import run_parallel


def _encounter_grids() -> np.ndarray:
    grids = np.zeros((3, 20, 20), dtype=np.uint8)
    grids[0, 8, 2] = 1
    grids[0, 8, 4] = 1
    grids[1, 8:10, 2:4] = 1
    grids[1, 8:10, 5:7] = 1
    rows, cols = np.indices((20, 20))
    grids[2] = (((rows * 17 + cols * 31) % 23) < 2).astype(np.uint8)
    return grids


def test_parallel_runner_matches_independent_scalar_engines() -> None:
    config = MorphologyExperimentConfig(
        width=20,
        height=20,
        seed=13,
        steps=5,
        warmup_steps=0,
        alpha=0.5,
        snapshot_every=100,
    )
    initial_grids = _encounter_grids()
    scalar_results = [
        MorphologyInteractionEngine(
            replace(config, seed=config.seed + index),
            initial_grid=initial_grids[index],
        ).run()
        for index in range(initial_grids.shape[0])
    ]

    parallel = run_parallel(
        config,
        num_envs=3,
        initial_grids=initial_grids,
        batch_size=2,
        host_workers=2,
        pair_capacity=32,
        record_snapshots=True,
    )

    for index, scalar in enumerate(scalar_results):
        np.testing.assert_array_equal(parallel.final_grids[index], scalar.final_grid)
        assert parallel.engines[index].metrics.records == scalar.generation_records
        assert list(parallel.engines[index].species_registry) == list(scalar.species_registry)
        assert list(parallel.engines[index].interaction_cache) == list(scalar.interaction_cache)
        for pair in scalar.interaction_cache:
            actual = parallel.engines[index].interaction_cache[pair]
            expected = scalar.interaction_cache[pair]
            assert actual.rule_id == expected.rule_id
            assert actual.encounters == expected.encounters
            np.testing.assert_array_equal(actual.final_vector, expected.final_vector)

    assert any(len(engine.interaction_cache) for engine in parallel.engines)


def test_parallel_random_initial_batch_matches_native_seeded_grids() -> None:
    config = MorphologyExperimentConfig(
        width=18,
        height=18,
        seed=21,
        density=0.12,
        steps=0,
    )
    result = run_parallel(
        config,
        num_envs=4,
        batch_size=4,
        host_workers=1,
        pair_capacity=16,
    )

    for index, engine in enumerate(result.engines):
        expected_config = replace(config, seed=config.seed + index)
        expected = MorphologyInteractionEngine(expected_config).initial_grid
        np.testing.assert_array_equal(engine.initial_grid, expected)


def test_parallel_control_uses_native_batched_conway() -> None:
    config = MorphologyExperimentConfig(
        width=16,
        height=16,
        seed=31,
        steps=6,
        warmup_steps=0,
        interactions_enabled=False,
    )
    initial_grids = _encounter_grids()[:, :16, :16]
    parallel = run_parallel(
        config,
        num_envs=3,
        initial_grids=initial_grids,
        host_workers=1,
        pair_capacity=16,
    )
    scalar = [
        MorphologyInteractionEngine(
            replace(config, seed=config.seed + index),
            initial_grid=initial_grids[index],
        ).run()
        for index in range(3)
    ]

    for index, expected in enumerate(scalar):
        np.testing.assert_array_equal(parallel.final_grids[index], expected.final_grid)


def test_no_metric_control_skips_morphology_analysis() -> None:
    config = MorphologyExperimentConfig(
        width=16,
        height=16,
        seed=31,
        steps=3,
        warmup_steps=0,
        interactions_enabled=False,
    )
    result = run_parallel(
        config,
        num_envs=3,
        initial_grids=_encounter_grids()[:, :16, :16],
        host_workers=1,
        pair_capacity=16,
        collect_metrics=False,
    )

    assert all(not engine.species_registry for engine in result.engines)
    assert all(not engine.metrics.records for engine in result.engines)
