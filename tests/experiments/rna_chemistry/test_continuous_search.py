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
    grid_state_bytes,
    pack_grids,
    prepare,
    prepare_batch,
    rule_code,
    state_bytes,
    step_fields,
    unchanged_batch,
    unpack_grids,
)
from emergent.experiments.morphology_interactions.rna_chemistry.search.schedule import (
    plan_for_trial,
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


def test_unchanged_batch_is_an_exact_jax_predicate():
    before = np.zeros((3, 5, 5), dtype=np.uint8)
    after = before.copy()
    after[1, 2, 2] = 1
    after[2, 0, 0] = 1
    np.testing.assert_array_equal(unchanged_batch(before, after), [True, False, False])


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


def test_batched_preparation_matches_scalar_rule_fields():
    config = RNAExperimentConfig(
        width=16,
        height=16,
        warmup_steps=0,
        calibration_size=32,
        component_backend="python",
        binding_lifetime_mode="instant",
    )
    grids = np.random.default_rng(92).random((4, 16, 16)) < 0.18
    grids = grids.astype(np.uint8)
    universe = RNAChemistryEngine(config, initial_grid=grids[0]).universe
    scalar_engines = [
        RNAChemistryEngine(config, initial_grid=grid, universe=universe) for grid in grids
    ]
    batch_engines = [
        RNAChemistryEngine(config, initial_grid=grid, universe=universe) for grid in grids
    ]
    scalar = [prepare(engine, grid) for engine, grid in zip(scalar_engines, grids)]
    batch = prepare_batch(batch_engines, grids)
    for (scalar_rules, scalar_sites), details in zip(scalar, batch):
        np.testing.assert_array_equal(details.rules, scalar_rules)
        assert details.active_zone_count == scalar_sites


def test_trial_schedule_keys_configs_and_replacements_are_deterministic():
    base = RNAExperimentConfig(width=8, height=8, calibration_size=32)
    plans = [
        plan_for_trial(base, trial=index, schedule_index=index, mode="rotating")
        for index in range(10)
    ]
    assert len({plan.trial_key for plan in plans}) == len(plans)
    assert len({plan.initial_seed for plan in plans}) == len(plans)
    assert all(
        plans[index].strategy.key != plans[index - 1].strategy.key for index in range(1, 10)
    )
    replacement = plan_for_trial(
        base,
        trial=11,
        schedule_index=0,
        mode="rotating",
        previous_strategy=plans[0].strategy.key,
    )
    assert replacement.strategy.key != plans[0].strategy.key
    fixed = [
        plan_for_trial(base, trial=index, schedule_index=index, mode="fixed")
        for index in range(4)
    ]
    assert len({plan.config_key for plan in fixed}) == 1
    assert len({plan.trial_key for plan in fixed}) == len(fixed)


def test_cycle_detection_handles_long_period_and_transient_exactly():
    cycle = ExactCycle(b"initial")
    detected = None
    for i in range(200):
        state = f"transient{i}".encode() if i < 13 else str((i - 13) % 17).encode()
        detected = cycle.observe(state)
        if detected is not None:
            break
    assert detected == 17


def test_grid_only_cycle_detection_catches_repeating_spatial_state():
    state_a = np.zeros((4, 4), dtype=np.uint8)
    state_a[1, 1] = 1
    state_b = np.zeros((4, 4), dtype=np.uint8)
    state_b[2, 2] = 1
    cycle = ExactCycle(grid_state_bytes(state_a))
    assert cycle.observe(grid_state_bytes(state_b)) is None
    assert cycle.observe(grid_state_bytes(state_a)) is None
    assert cycle.observe(grid_state_bytes(state_b)) == 2


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


def test_search_evicts_exactly_unchanged_grid_during_warmup(tmp_path):
    args = parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--worker-id",
            "0",
            "--workers",
            "1",
            "--batch-size",
            "1",
            "--size",
            "8",
            "--seed",
            "0",
            "--density",
            "0",
            "--warmup-steps",
            "3",
            "--max-ticks",
            "1",
            "--strategy-schedule",
            "rotating",
            "--component-backend",
            "python",
        ]
    )
    run_worker(args)
    import json

    events = [
        json.loads(line)
        for line in (tmp_path / "worker_000" / "events.jsonl").read_text().splitlines()
    ]
    terminal = next(event for event in events if event["event"] == "terminal")
    assert terminal["reason"] == "unchanged"
    assert terminal["unchanged_grid"] is True
    assert terminal["failure"] is True


def test_search_evicts_exact_repeating_grid_before_full_chemistry_state(tmp_path):
    args = parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--worker-id",
            "0",
            "--workers",
            "1",
            "--batch-size",
            "1",
            "--size",
            "8",
            "--density",
            "0.08",
            "--seed",
            "907",
            "--warmup-steps",
            "0",
            "--strategy-schedule",
            "fixed",
            "--disable-interactions",
            "--component-backend",
            "python",
            "--max-ticks",
            "3",
        ]
    )
    run_worker(args)
    import json

    events = [
        json.loads(line)
        for line in (tmp_path / "worker_000" / "events.jsonl").read_text().splitlines()
    ]
    terminal = next(event for event in events if event["event"] == "terminal")
    assert terminal["reason"] == "grid_repeat"
    assert terminal["grid_period"] == 2
    assert terminal["failure"] is True


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
            "--warmup-steps",
            "0",
            "--strategy-schedule",
            "fixed",
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
    terminals = [event for event in events if event["event"] == "terminal"]
    assert terminals
    assert all(event["failure"] and event["outcome"] == "failure" for event in terminals)
    replacements = [
        event
        for event in events
        if event["event"] == "start" and event["replaces_trial"] is not None
    ]
    assert replacements
    assert len({event["trial_key"] for event in events if event["event"] == "start"}) == sum(
        event["event"] == "start" for event in events
    )
