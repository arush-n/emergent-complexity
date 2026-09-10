from __future__ import annotations

import numpy as np

from emergent.core.rules import parse_rule, rule_to_masks
from emergent.core.step import step_jit
from emergent.experiments.morphology_interactions.canonical import canonicalize_grid
from emergent.experiments.morphology_interactions.replicator_search import (
    Candidate,
    ReplicatorSearchConfig,
    evaluate_candidate_state,
    initial_population,
    matrix_to_text,
    mutate_candidate,
    run_replicator_search,
    write_search_artifacts,
)


def test_exact_copy_evidence_is_detected_from_a_supplied_state() -> None:
    candidate_matrix = np.ones((2, 2), dtype=np.uint8)
    candidate = Candidate(canonicalize_grid(candidate_matrix), candidate_matrix)
    grid = np.zeros((16, 16), dtype=np.uint8)
    grid[3:5, 3:5] = 1
    grid[9:11, 9:11] = 1

    evaluation = evaluate_candidate_state(
        candidate,
        grid,
        generation=7,
        component_backend="python",
    )

    assert evaluation.best_generation == 7
    assert evaluation.best_copy_count == 2
    assert evaluation.best_purity == 1.0
    assert evaluation.best_mass_balance == 1.0
    assert evaluation.best_similarity == 1.0
    assert evaluation.best_repeat_count == 2
    assert evaluation.best_repeat_purity == 1.0
    assert evaluation.best_repeat_key == candidate.key
    assert evaluation.best_score == 1.05
    assert evaluation.is_replicator
    assert evaluation.is_fission_like


def test_fission_like_lead_does_not_count_as_self_replication() -> None:
    candidate_matrix = np.ones((2, 2), dtype=np.uint8)
    candidate = Candidate(canonicalize_grid(candidate_matrix), candidate_matrix)
    grid = np.zeros((12, 16), dtype=np.uint8)
    grid[3, 3:6] = 1
    grid[8, 9:12] = 1

    evaluation = evaluate_candidate_state(
        candidate,
        grid,
        generation=4,
        component_backend="python",
    )

    assert evaluation.best_copy_count == 0
    assert evaluation.best_repeat_count == 2
    assert evaluation.best_repeat_purity == 1.0
    assert evaluation.best_repeat_key != candidate.key
    assert not evaluation.is_replicator
    assert evaluation.is_fission_like


def test_population_and_mutation_are_deterministic_and_connected() -> None:
    config = ReplicatorSearchConfig(
        seed=11,
        candidate_size=6,
        world_size=24,
        population_size=8,
        elite_count=3,
        generations=1,
        evaluation_steps=2,
        initial_live_cells=6,
        component_backend="python",
    )
    first = initial_population(config)
    second = initial_population(config)

    assert [candidate.key for candidate in first] == [candidate.key for candidate in second]
    for child_index, parent in enumerate(first):
        child = mutate_candidate(
            parent,
            config,
            search_generation=0,
            child_index=child_index,
        )
        assert np.count_nonzero(child.matrix) >= 1
        assert child.matrix.max() == 1
        assert max(child.matrix.shape) <= config.candidate_size
        assert matrix_to_text(child.matrix).count("#") == child.cell_count


def test_small_search_replays_exactly(tmp_path) -> None:
    config = ReplicatorSearchConfig(
        seed=3,
        candidate_size=5,
        world_size=20,
        population_size=6,
        elite_count=2,
        generations=2,
        evaluation_steps=3,
        initial_live_cells=5,
        component_backend="python",
    )

    first = run_replicator_search(config)
    second = run_replicator_search(config)

    assert first.found == second.found
    assert first.found_search_generation == second.found_search_generation
    assert first.best_candidate.key == second.best_candidate.key
    np.testing.assert_array_equal(first.best_candidate.matrix, second.best_candidate.matrix)
    np.testing.assert_array_equal(first.best_state, second.best_state)
    assert first.best_evaluation == second.best_evaluation
    assert first.history == second.history
    assert first.trace == second.trace
    assert first.evaluations == second.evaluations
    assert first.terminal_reports == second.terminal_reports

    output_dir = write_search_artifacts(first, tmp_path / "search", timestamp="test")
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "history.csv").exists()
    assert (output_dir / "trace.csv").exists()
    assert (output_dir / "terminal.csv").exists()
    assert (output_dir / "identity_audit.json").exists()
    assert len(first.trace) >= len(first.history)
    with np.load(output_dir / "best_state.npz") as artifact:
        np.testing.assert_array_equal(artifact["grid"], first.best_state)

    birth, survival = rule_to_masks(parse_rule("B3/S23"))
    replay = np.zeros_like(first.best_state)
    row = (replay.shape[0] - first.best_candidate.matrix.shape[0]) // 2
    col = (replay.shape[1] - first.best_candidate.matrix.shape[1]) // 2
    height, width = first.best_candidate.matrix.shape
    replay[row : row + height, col : col + width] = first.best_candidate.matrix
    for _ in range(first.best_evaluation.best_generation):
        replay = np.asarray(step_jit(replay, birth, survival), dtype=np.uint8)
    np.testing.assert_array_equal(replay, first.best_state)


def test_terminal_rollouts_stop_stable_and_periodic_worlds() -> None:
    stable_config = ReplicatorSearchConfig(
        seed=12,
        candidate_size=4,
        world_size=16,
        population_size=1,
        elite_count=1,
        generations=0,
        evaluation_steps=8,
        initial_live_cells=4,
        component_backend="python",
    )
    stable_result = run_replicator_search(stable_config)
    stable_candidate = next(
        candidate
        for candidate in initial_population(stable_config)
        if candidate.matrix.shape == (2, 2)
    )
    stable_report = stable_result.terminal_reports[stable_candidate.key]

    assert stable_report.reason == "stable"
    assert stable_report.generation == 1
    assert stable_report.period == 1

    periodic_config = ReplicatorSearchConfig(
        seed=12,
        candidate_size=4,
        world_size=16,
        population_size=2,
        elite_count=1,
        generations=0,
        evaluation_steps=8,
        initial_live_cells=4,
        component_backend="python",
    )
    periodic_result = run_replicator_search(periodic_config)
    blinker = next(
        candidate for candidate in initial_population(periodic_config) if candidate.cell_count == 3
    )
    periodic_report = periodic_result.terminal_reports[blinker.key]

    assert periodic_report.reason == "cycle_2"
    assert periodic_report.generation == 2
    assert periodic_report.period == 2
