from __future__ import annotations

import numpy as np

from emergent.experiments.morphology_interactions.rna_chemistry.config import RNAExperimentConfig
from emergent.experiments.morphology_interactions.rna_chemistry.engine import RNAChemistryEngine


def test_complete_rna_engine_replay_is_identical() -> None:
    initial = np.zeros((20, 20), dtype=np.uint8)
    initial[8:10, 3:5] = 1
    initial[8:10, 7:9] = 1
    config = RNAExperimentConfig(
        width=20,
        height=20,
        steps=5,
        warmup_steps=0,
        calibration_size=32,
        snapshot_every=2,
    )
    first = RNAChemistryEngine(config, initial_grid=initial).run()
    second = RNAChemistryEngine(config, initial_grid=initial).run()

    np.testing.assert_array_equal(first.final_grid, second.final_grid)
    assert first.generation_records == second.generation_records
    assert list(first.snapshots) == list(second.snapshots)
    for generation in first.snapshots:
        np.testing.assert_array_equal(first.snapshots[generation], second.snapshots[generation])
    assert list(first.species_registry) == list(second.species_registry)
    assert list(first.pair_cache) == list(second.pair_cache)
