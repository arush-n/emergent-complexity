import numpy as np
import pytest

from emergent.core.rules import parse_rule, rule_to_masks
from emergent.core.step import batched_step
from emergent.experiments.morphology_interactions.rna_chemistry.config import RNAExperimentConfig
from emergent.experiments.morphology_interactions.rna_chemistry.engine import RNAChemistryEngine
from emergent.experiments.morphology_interactions.rna_chemistry.search.replay import verify
from emergent.experiments.morphology_interactions.rna_chemistry.search.run import parser, run_worker
from emergent.experiments.morphology_interactions.rna_chemistry.search.runtime import (
    ExactCycle,
    pack_grids,
    prepare,
    rule_code,
    state_bytes,
    step_fields,
    unpack_grids,
)


def test_field_kernel_matches_native_batch():
    grids = np.random.default_rng(17).integers(0, 2, size=(8, 16, 16), dtype=np.uint8)
    birth, survival = rule_to_masks(parse_rule("B3/S23"))
    fields = np.full(grids.shape, rule_code(birth, survival), dtype=np.uint32)
    np.testing.assert_array_equal(step_fields(grids, fields), batched_step(grids, birth, survival))


def test_trace_grid_bitpacking_is_exact():
    grids = np.random.default_rng(18).integers(0, 2, size=(3, 5, 7, 9), dtype=np.uint8)
    encoded = pack_grids(grids)
    assert encoded.nbytes < grids.nbytes
    np.testing.assert_array_equal(unpack_grids(encoded, height=7, width=9), grids)


@pytest.mark.parametrize("lifetime", ["instant", "energy"])
def test_parallel_preparation_matches_scalar_chemistry(lifetime):
    config = RNAExperimentConfig(
        width=16,
        height=16,
        warmup_steps=0,
        calibration_size=32,
        binding_lifetime_mode=lifetime,
        interaction_radius=3,
    )
    grid = (np.random.default_rng(91).random((16, 16)) < 0.18).astype(np.uint8)
    scalar = RNAChemistryEngine(config, initial_grid=grid)
    parallel = RNAChemistryEngine(config, initial_grid=grid, universe=scalar.universe)
    saw_sites = 0
    for _ in range(16):
        fields, sites = prepare(parallel, grid)
        saw_sites += sites
        grid = np.asarray(step_fields(grid[None], fields[None]))[0]
        parallel.generation += 1
        expected = scalar.step()
        np.testing.assert_array_equal(grid, expected.grid)
        assert state_bytes(parallel, grid) == state_bytes(scalar, expected.grid)
    assert saw_sites > 0


def test_cycle_detection_handles_long_period_and_transient_exactly():
    cycle = ExactCycle(b"initial")
    detected = None
    for i in range(200):
        state = f"transient{i}".encode() if i < 13 else str((i - 13) % 17).encode()
        detected = cycle.observe(state)
        if detected is not None:
            break
    assert detected == 17


def test_warmup_and_detection_phase_are_part_of_terminal_identity():
    config = RNAExperimentConfig(
        width=8, height=8, warmup_steps=3, detect_every=2, calibration_size=32
    )
    grid = np.zeros((8, 8), np.uint8)
    engine = RNAChemistryEngine(config, initial_grid=grid)
    initial = state_bytes(engine, grid)
    engine.generation = 1
    assert state_bytes(engine, grid) != initial
    engine.generation = 3
    initial = state_bytes(engine, grid)
    engine.generation = 4
    assert state_bytes(engine, grid) != initial
    engine.generation = 5
    assert state_bytes(engine, grid) == initial


def test_refilled_search_trace_replays_chemistry_and_every_transition(tmp_path):
    args = parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--worker-id",
            "0",
            "--workers",
            "1",
            "--batch-size",
            "3",
            "--size",
            "8",
            "--max-ticks",
            "12",
            "--density",
            "0.02",
            "--trace-chunk",
            "5",
            "--component-backend",
            "python",
        ]
    )
    run_worker(args)
    result = verify(tmp_path / "worker_000")
    assert result["verified_transitions"] == 36
    assert result["verified_chunks"] == 3
    import json

    events = [
        json.loads(line)
        for line in (tmp_path / "worker_000" / "events.jsonl").read_text().splitlines()
    ]
    assert sum(event["event"] == "start" for event in events) > 3
