import json

import numpy as np

from emergent.experiments.morphology_interactions.canonical import shape_key_from_matrix
from emergent.experiments.morphology_interactions.rna_chemistry.config import RNAExperimentConfig
from emergent.experiments.morphology_interactions.rna_chemistry.search.screen import (
    collect_leads,
    screen_batch,
)


def test_isolated_controls_prune_stable_and_extinct_without_replication_claims():
    keys = [
        shape_key_from_matrix(np.ones((2, 2), np.uint8)),
        shape_key_from_matrix(np.ones((1, 1), np.uint8)),
    ]
    config = RNAExperimentConfig(width=16, height=16, calibration_size=32)
    results = screen_batch(keys, config, steps=20)
    assert len(results) == 8
    assert {row["condition"] for row in results} == {"native", "structured", "mixed", "scrambled"}
    assert all(row["first_double"] is None for row in results)
    assert all(row["steps_executed"] < 20 for row in results)
    assert all(row["outcome"] == "periodic" for row in results[:4])
    assert all(row["outcome"] == "extinct" for row in results[4:])
    assert all(not row["confirmed_replicator"] for row in results)


def test_leads_deduplicate_shapes_and_keep_replay_origin(tmp_path):
    worker = tmp_path / "worker_000"
    worker.mkdir()
    event = {
        "event": "copy_growth_lead", "shape": {"height": 2, "width": 2, "packed": "f0"},
        "copies": 2, "trial": 1, "trace_tick": 7,
    }
    (worker / "events.jsonl").write_text(
        json.dumps(event) + "\n" + json.dumps(event | {"copies": 4, "trace_tick": 12})
        + '\n{"event":'
    )
    leads = collect_leads(tmp_path, minimum_cells=4)
    assert len(leads) == 1
    assert leads[0]["copies"] == 4
    assert leads[0]["source_trace_tick"] == 12
    assert collect_leads(tmp_path, minimum_cells=5) == []
