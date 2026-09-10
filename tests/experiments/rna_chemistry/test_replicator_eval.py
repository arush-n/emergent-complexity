from __future__ import annotations

import numpy as np

from emergent.experiments.morphology_interactions.canonical import shape_key_from_matrix
from emergent.experiments.morphology_interactions.replicator_search import Candidate
from emergent.experiments.morphology_interactions.rna_chemistry.config import RNAExperimentConfig
from emergent.experiments.morphology_interactions.rna_chemistry.replicator_eval import (
    evaluate_candidate,
)


def test_replicator_evaluator_reports_no_inserted_replication_action() -> None:
    matrix = np.asarray([[1, 1], [1, 1]], dtype=np.uint8)
    candidate = Candidate(shape_key_from_matrix(matrix), matrix)
    config = RNAExperimentConfig(
        width=16,
        height=16,
        steps=4,
        warmup_steps=0,
        interactions_enabled=False,
        calibration_size=32,
    )

    evaluation = evaluate_candidate(candidate, config)

    assert evaluation.max_exact_copy_count == 1
    assert not evaluation.is_replication_like
    assert not evaluation.is_sustained
