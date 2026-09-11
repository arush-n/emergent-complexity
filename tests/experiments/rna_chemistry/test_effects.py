from __future__ import annotations

import numpy as np

from emergent.experiments.morphology_interactions.canonical import canonicalize_grid
from emergent.experiments.morphology_interactions.components import Component
from emergent.experiments.morphology_interactions.interaction import (
    build_interaction_zone,
    toroidal_dilate,
)
from emergent.experiments.morphology_interactions.rna_chemistry.chemistry import (
    evaluate_pair_chemistry,
    make_chemistry_universe,
)
from emergent.experiments.morphology_interactions.rna_chemistry.effects import (
    component_site_coordinate,
    component_site_coordinates,
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


def test_sparse_site_dilation_matches_dense_reference() -> None:
    interaction, sequence = _site_interaction()
    first = Component(np.asarray([[8, 4], [8, 5], [8, 6], [8, 7]], dtype=np.int64), 4)
    second = Component(np.asarray([[8, 10], [8, 11], [8, 12], [8, 13]], dtype=np.int64), 4)
    grid_shape = (20, 20)
    encounter = build_interaction_zone(
        first,
        second,
        grid_shape,
        interaction_radius=3,
        effect_padding=1,
    )
    mapped_first = component_site_coordinates(first, sequence, grid_shape)
    mapped_second = component_site_coordinates(second, sequence, grid_shape)
    anchors = np.zeros(grid_shape, dtype=bool)
    for coordinate in np.concatenate(
        (
            mapped_first[interaction.site.start_a : interaction.site.end_a],
            mapped_second[interaction.site.start_b : interaction.site.end_b],
        )
    ):
        if np.all(coordinate >= 0):
            anchors[tuple(coordinate)] = True
    expected = toroidal_dilate(anchors, 4) & encounter
    if not np.any(expected):
        expected = encounter
    actual = site_effect_zone(
        first,
        second,
        sequence,
        sequence,
        interaction,
        grid_shape,
        interaction_radius=3,
        effect_padding=1,
        encounter_zone=encounter,
        mapped_coordinates_a=mapped_first,
        mapped_coordinates_b=mapped_second,
    )
    np.testing.assert_array_equal(actual, expected)


def test_component_site_coordinates_are_reused_for_all_positions() -> None:
    _, sequence = _site_interaction()
    component = Component(np.asarray([[8, 4], [8, 5], [8, 6], [8, 7]], dtype=np.int64), 4)

    mapped = component_site_coordinates(component, sequence, (20, 20))

    np.testing.assert_array_equal(mapped, [[8, 4], [8, 5], [8, 6], [8, 7]])
    for position, expected in enumerate(((8, 4), (8, 5), (8, 6), (8, 7))):
        assert component_site_coordinate(component, sequence, position, (20, 20)) == expected


def test_site_overlap_ties_resolve_deterministically() -> None:
    interaction, _ = _site_interaction()
    first = site_interaction_to_effect(interaction, max_rule_changes=1)
    second = site_interaction_to_effect(interaction, max_rule_changes=1)
    zones = [(first, np.ones((6, 6), dtype=bool)), (second, np.ones((6, 6), dtype=bool))]
    owner_map, ordered = resolve_site_owner_map(zones, grid_shape=(6, 6))

    assert len(ordered) == 1
    assert np.all(owner_map == 0)
