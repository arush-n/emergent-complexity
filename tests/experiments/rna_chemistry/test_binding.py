from __future__ import annotations

import numpy as np

from emergent.experiments.morphology_interactions.rna_chemistry.binding import (
    find_binding_sites,
    scan_binding_sites,
)


def test_antiparallel_canonical_seed_and_stack_score() -> None:
    # ACGU is its own reverse complement, so it pairs perfectly with itself.
    sequence = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    result = scan_binding_sites(sequence, sequence)

    assert result.candidate_seed_count == 1
    assert len(result.sites) == 1
    site = result.sites[0]
    assert (site.start_a, site.end_a, site.start_b, site.end_b) == (0, 4, 0, 4)
    assert site.canonical_pairs == 4
    assert site.wobble_pairs == 0
    assert site.mismatches == 0
    assert site.pair_score == -10.0
    assert site.stack_score == -1.5
    assert site.total_score == -11.5


def test_wobble_can_be_disabled_without_affecting_canonical_pairs() -> None:
    # GGGG versus CCCC is canonical; UUUU versus GGGG relies on wobble only.
    wobble_a = np.asarray([3, 3, 3, 3], dtype=np.uint8)
    wobble_b = np.asarray([2, 2, 2, 2], dtype=np.uint8)
    assert len(find_binding_sites(wobble_a, wobble_b, allow_gu_wobble=True)) == 1
    assert len(find_binding_sites(wobble_a, wobble_b, allow_gu_wobble=False)) == 0


def test_accessibility_reduces_binding_magnitude_deterministically() -> None:
    sequence = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    exposed = scan_binding_sites(sequence, sequence)
    inaccessible = scan_binding_sites(
        sequence,
        sequence,
        accessibility_a=np.full(4, 0.25),
        accessibility_b=np.ones(4),
    )

    assert inaccessible.sites[0].total_score > exposed.sites[0].total_score
    assert inaccessible.sites[0].pair_score == exposed.sites[0].pair_score


def test_seed_scanning_is_repeatable() -> None:
    first = scan_binding_sites("ACGUACGU", "ACGUACGU")
    second = scan_binding_sites("ACGUACGU", "ACGUACGU")
    assert first == second
