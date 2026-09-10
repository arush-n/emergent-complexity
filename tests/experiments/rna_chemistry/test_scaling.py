from __future__ import annotations

from emergent.experiments.morphology_interactions.rna_chemistry.audit import audit_interaction_space


def test_interaction_audit_is_repeatable_and_reports_jax_backend() -> None:
    first = audit_interaction_space(
        seed=17,
        pair_count=32,
        shape_count=16,
        minimum_cells=2,
        maximum_cells=12,
        calibration_size=32,
    )
    second = audit_interaction_space(
        seed=17,
        pair_count=32,
        shape_count=16,
        minimum_cells=2,
        maximum_cells=12,
        calibration_size=32,
    )

    assert first.summary == second.summary
    assert first.rows == second.rows
    assert first.summary["unique_shape_keys"] == 16
    assert first.summary["jax_backend"] in {"cpu", "gpu", "tpu"}


def test_longer_sequences_have_more_available_seed_opportunities() -> None:
    result = audit_interaction_space(
        seed=5,
        pair_count=128,
        shape_count=32,
        minimum_cells=1,
        maximum_cells=32,
        calibration_size=32,
    )
    short = [row for row in result.rows if row["length_product"] <= 16]
    long = [row for row in result.rows if row["length_product"] >= 256]

    assert short and long
    assert sum(row["expected_seed_count"] for row in long) / len(long) > sum(
        row["expected_seed_count"] for row in short
    ) / len(short)
