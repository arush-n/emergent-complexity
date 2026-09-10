from __future__ import annotations

import numpy as np

from emergent.experiments.morphology_interactions.canonical import canonicalize_grid
from emergent.experiments.morphology_interactions.components import Component
from emergent.experiments.morphology_interactions.rna_chemistry.chemistry import (
    evaluate_pair_chemistry,
    make_chemistry_universe,
)
from emergent.experiments.morphology_interactions.rna_chemistry.effects import (
    resolve_site_owner_map,
    site_effect_zone,
    site_interaction_to_effect,
)
from emergent.experiments.morphology_interactions.rna_chemistry.sequence import ChemicalSequence


def _site_interaction():
    key = canonicalize_grid(np.asarray([[1, 1, 1, 1]], dtype=np.uint8))
    coordinates = np.asarray([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int64)
    sequence = ChemicalSequence(key, np.asarray([0, 1, 2, 3], dtype=np.uint8), coordinates, 4)
    universe = make_chemistry_universe(4, calibration_size=32)
    chemistry = evaluate_pair_chemistry(
        sequence, sequence, universe=universe, binding_energy_threshold=0
    )
    return chemistry.sites[0], sequence


def test_one_site_gets_only_a_bounded_rule_change() -> None:
    interaction, _ = _site_interaction()
    effect = site_interaction_to_effect(interaction, max_rule_changes=1)
    assert effect.birth_mask.shape == (9,)
    assert effect.survival_mask.shape == (9,)
    assert 0 <= effect.rule_id < 2**18


def test_site_zone_is_local_and_spatially_mapped() -> None:
    interaction, sequence = _site_interaction()
    first = Component(np.asarray([[8, 4], [8, 5], [8, 6], [8, 7]], dtype=np.int64), 4)
    second = Component(np.asarray([[8, 10], [8, 11], [8, 12], [8, 13]], dtype=np.int64), 4)
    zone = site_effect_zone(
        first,
        second,
        sequence,
        sequence,
        interaction,
        (20, 20),
        interaction_radius=3,
        effect_padding=1,
    )

    assert zone.shape == (20, 20)
    assert np.any(zone)
    assert np.count_nonzero(zone) < 20 * 20


def test_site_overlap_ties_resolve_deterministically() -> None:
    interaction, _ = _site_interaction()
    first = site_interaction_to_effect(interaction, max_rule_changes=1)
    second = site_interaction_to_effect(interaction, max_rule_changes=1)
    zones = [(first, np.ones((6, 6), dtype=bool)), (second, np.ones((6, 6), dtype=bool))]
    owner_map, ordered = resolve_site_owner_map(zones, grid_shape=(6, 6))

    assert len(ordered) == 1
    assert np.all(owner_map == 0)
