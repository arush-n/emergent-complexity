from __future__ import annotations

import numpy as np

from emergent.experiments.morphology_interactions.rna_chemistry.sensitivity import (
    run_sensitivity,
)


def test_rna_sensitivity_excludes_disconnecting_bridge_removals() -> None:
    rows = run_sensitivity(
        np.asarray([[1, 1, 1]], dtype=np.uint8),
        np.asarray([[1]], dtype=np.uint8),
        alphas=(0.0,),
        calibration_size=32,
    )

    assert rows
    labels = {row["perturbation"] for row in rows}
    assert "remove:0,1" not in labels
    assert any(label.startswith("remove:") for label in labels)
    assert all("interaction_distance" in row for row in rows)


def test_rna_sensitivity_is_repeatable() -> None:
    shape_a = np.asarray([[1, 1], [1, 0]], dtype=np.uint8)
    shape_b = np.asarray([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    first = run_sensitivity(shape_a, shape_b, alphas=(0.0, 1.0), calibration_size=32)
    second = run_sensitivity(shape_a, shape_b, alphas=(0.0, 1.0), calibration_size=32)

    assert first == second
