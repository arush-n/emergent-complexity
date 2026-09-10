from __future__ import annotations

import csv

import numpy as np

from emergent.core.rules import parse_rule, rule_to_masks
from emergent.core.step import step_jit
from emergent.experiments.morphology_interactions.rna_chemistry.config import RNAExperimentConfig
from emergent.experiments.morphology_interactions.rna_chemistry.engine import (
    RNAChemistryEngine,
    write_artifacts,
)


def _initial_grid() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=np.uint8)
    grid[5:7, 3:5] = 1
    grid[5:7, 8:10] = 1
    return grid


def test_interactions_disabled_is_bit_for_bit_native_conway() -> None:
    initial = _initial_grid()
    config = RNAExperimentConfig(
        width=16,
        height=16,
        steps=6,
        warmup_steps=0,
        interactions_enabled=False,
        calibration_size=32,
    )
    result = RNAChemistryEngine(config, initial_grid=initial).run()
    birth, survival = rule_to_masks(parse_rule("B3/S23"))
    expected = initial.copy()
    for _ in range(config.steps):
        expected = np.asarray(step_jit(expected, birth, survival))
    np.testing.assert_array_equal(result.final_grid, expected)


def test_engine_writes_compact_rna_artifacts(tmp_path) -> None:
    config = RNAExperimentConfig(
        width=16,
        height=16,
        steps=2,
        warmup_steps=0,
        interactions_enabled=False,
        calibration_size=32,
    )
    result = RNAChemistryEngine(config, initial_grid=_initial_grid()).run()
    target = write_artifacts(result, tmp_path / "rna")

    assert (target / "manifest.json").exists()
    assert (target / "generations.csv").exists()
    assert (target / "species.csv").exists()
    assert (target / "interactions.csv").exists()
    assert (target / "snapshots.npz").exists()

    with (target / "species.csv").open(newline="", encoding="utf-8") as handle:
        species_fields = csv.DictReader(handle).fieldnames
    assert species_fields is not None
    assert {
        "perimeter",
        "sequence_entropy",
        "accessible_fraction",
        "unique_kmer_count",
        "unique_reaction_motif_count",
    }.issubset(species_fields)

    with (target / "interactions.csv").open(newline="", encoding="utf-8") as handle:
        interaction_fields = csv.DictReader(handle).fieldnames
    assert interaction_fields is not None
    assert {
        "first_seen",
        "encounters",
        "total_paired_bases",
        "total_site_effect_area",
        "local_rule",
    }.issubset(interaction_fields)


def test_sparse_metric_sampling_keeps_unsampled_discoveries() -> None:
    config = RNAExperimentConfig(
        width=16,
        height=16,
        steps=3,
        warmup_steps=0,
        metrics_every=3,
        interactions_enabled=False,
        calibration_size=32,
    )
    result = RNAChemistryEngine(config, initial_grid=_initial_grid()).run()

    assert len(result.generation_records) == 1
    assert result.summary["unique_species_total"] > 0
    assert result.summary["unique_chemical_sequences_total"] > 0
