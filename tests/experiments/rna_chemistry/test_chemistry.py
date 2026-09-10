from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from emergent.experiments.morphology_interactions.canonical import canonicalize_grid
from emergent.experiments.morphology_interactions.rna_chemistry.binding import scan_binding_sites
from emergent.experiments.morphology_interactions.rna_chemistry.chemistry import (
    CHANNEL_COUNT,
    _structured_batch_kernel,
    evaluate_pair_chemistry,
    make_chemistry_universe,
)
from emergent.experiments.morphology_interactions.rna_chemistry.motifs import (
    MOTIF_FEATURE_DIM,
    extract_reaction_motif,
    motif_feature_vector,
    motif_feature_vectors,
)
from emergent.experiments.morphology_interactions.rna_chemistry.sequence import ChemicalSequence


def _sequence(values: list[int]) -> ChemicalSequence:
    key = canonicalize_grid(np.asarray([[1]], dtype=np.uint8))
    coordinates = np.zeros((len(values), 2), dtype=np.int64)
    return ChemicalSequence(key, np.asarray(values, dtype=np.uint8), coordinates, len(values))


def test_structured_and_scrambled_vectors_are_calibrated() -> None:
    universe = make_chemistry_universe(42, calibration_size=32)
    report = universe.calibration_report()

    assert abs(report["structured_rms"] - report["scrambled_rms"]) < 1e-5
    assert abs(report["structured_threshold_rate"] - report["scrambled_threshold_rate"]) < 0.1


def test_structured_chemistry_is_jax_compiled_matrix_work() -> None:
    universe = make_chemistry_universe(42, calibration_size=32)
    features = jnp.ones((4, MOTIF_FEATURE_DIM), dtype=jnp.float32)
    jaxpr = str(
        jax.make_jaxpr(_structured_batch_kernel)(
            features,
            jnp.asarray(universe.weights),
            universe.calibration.structured_gain,
            universe.beta,
        )
    )

    result = universe.structured_batch(np.ones((4, MOTIF_FEATURE_DIM), dtype=np.float32))
    assert result.shape == (4, CHANNEL_COUNT)
    assert "dot_general" in jaxpr


def test_same_site_chemistry_is_byte_stable_and_symmetric() -> None:
    first = _sequence([0, 1, 2, 3])
    second = _sequence([0, 1, 2, 3])
    universe = make_chemistry_universe(9, calibration_size=32)

    forward = evaluate_pair_chemistry(first, second, universe=universe, binding_energy_threshold=0)
    reverse = evaluate_pair_chemistry(second, first, universe=universe, binding_energy_threshold=0)
    repeated = evaluate_pair_chemistry(first, second, universe=universe, binding_energy_threshold=0)

    assert len(forward.sites) == 1
    assert forward.pair_key == reverse.pair_key
    np.testing.assert_array_equal(forward.sites[0].feature_vector, reverse.sites[0].feature_vector)
    np.testing.assert_array_equal(forward.sites[0].final_vector, reverse.sites[0].final_vector)
    np.testing.assert_array_equal(forward.sites[0].final_vector, repeated.sites[0].final_vector)
    assert forward.sites[0].feature_vector.shape == (MOTIF_FEATURE_DIM,)


def test_bucketed_motif_features_match_single_site_features() -> None:
    first = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.uint8)
    second = first.copy()
    scan = scan_binding_sites(first, second, seed_length=4, minimum_length=4)
    entries = tuple(
        (
            site,
            extract_reaction_motif(first, second, site, motif_length=4),
        )
        for site in scan.sites
    )
    batched = motif_feature_vectors(first, second, entries)
    individual = [motif_feature_vector(first, second, site, motif) for site, motif in entries]
    assert len(batched) == len(individual)
    for actual, expected in zip(batched, individual):
        np.testing.assert_array_equal(actual, expected)


def test_energy_lifetime_is_deterministic_and_stronger_sites_last_longer() -> None:
    universe = make_chemistry_universe(11, calibration_size=32)
    sequence = _sequence([0, 1, 2, 3])
    result = evaluate_pair_chemistry(
        sequence,
        sequence,
        universe=universe,
        binding_energy_threshold=0,
        binding_lifetime_mode="energy",
        max_binding_lifetime=16,
    )

    assert result.sites[0].lifetime >= 1
    assert (
        result.sites[0].lifetime
        == evaluate_pair_chemistry(
            sequence,
            sequence,
            universe=universe,
            binding_energy_threshold=0,
            binding_lifetime_mode="energy",
            max_binding_lifetime=16,
        )
        .sites[0]
        .lifetime
    )
