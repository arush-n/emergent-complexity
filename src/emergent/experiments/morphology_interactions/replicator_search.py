"""Deterministic evolutionary search for self-replicating native-Life shapes.

This is an experiment-local search, not a change to the simulator.  Candidate
patterns are rolled out with the repository's native ``B3/S23`` transition.
The search only calls a candidate a replicator when a later state contains at
least two distinct components with the candidate's exact canonical
``ShapeKey`` and sufficiently little non-copy debris.

Example::

    python -m emergent.experiments.morphology_interactions.replicator_search \
        --candidate-size 9 --world-size 64 --population-size 32 \
        --generations 20 --evaluation-steps 32
"""

from __future__ import annotations

import argparse
import csv
import json
import operator
import platform
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ...core.rules import format_rule, parse_rule, rule_to_masks
from ...core.step import batched_step
from ...io.patterns import get_pattern
from .canonical import (
    ShapeKey,
    canonical_matrix_from_grid,
    canonicalize_component,
    matrix_from_shape_key,
    shape_key_from_matrix,
    shape_key_sort_key,
)
from .components import MOORE_OFFSETS, detect_components, detect_components_batch
from .encoding import audit_encodings, deterministic_uint64

NATIVE_RULE = "B3/S23"
TraceRow = dict[str, int | float | bool | str]


@dataclass(frozen=True)
class ReplicatorSearchConfig:
    """Immutable resource and scoring controls for one search universe."""

    seed: int = 42
    candidate_size: int = 9
    world_size: int = 64
    population_size: int = 32
    elite_count: int = 8
    generations: int = 20
    evaluation_steps: int = 32
    mutations_per_child: int = 2
    initial_live_cells: int = 8
    min_purity: float = 0.90
    component_backend: str = "auto"
    rotation_invariant: bool = True
    reflection_invariant: bool = False
    terminate_on_terminal: bool = True
    max_cycle_period: int = 2

    def __post_init__(self) -> None:
        integer_fields = (
            "seed",
            "candidate_size",
            "world_size",
            "population_size",
            "elite_count",
            "generations",
            "evaluation_steps",
            "mutations_per_child",
            "initial_live_cells",
            "max_cycle_period",
        )
        for name in integer_fields:
            value = getattr(self, name)
            try:
                normalized = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be an integer") from exc
            if normalized < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, normalized)
        if self.candidate_size < 1:
            raise ValueError("candidate_size must be positive")
        if self.world_size < self.candidate_size + 2:
            raise ValueError("world_size must leave a margin around each candidate")
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
        if not 1 <= self.elite_count <= self.population_size:
            raise ValueError("elite_count must be between 1 and population_size")
        if self.initial_live_cells < 1:
            raise ValueError("initial_live_cells must be positive")
        if self.max_cycle_period < 1:
            raise ValueError("max_cycle_period must be positive")
        if (
            not isinstance(self.min_purity, (int, float))
            or not 0.0 <= float(self.min_purity) <= 1.0
        ):
            raise ValueError("min_purity must be between 0 and 1")
        object.__setattr__(self, "min_purity", float(self.min_purity))
        if self.component_backend not in {"auto", "python", "scipy"}:
            raise ValueError("component_backend must be one of: auto, python, scipy")
        for name in (
            "rotation_invariant",
            "reflection_invariant",
            "terminate_on_terminal",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.reflection_invariant and not self.rotation_invariant:
            raise ValueError("reflection_invariant requires rotation_invariant")

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible search parameters."""

        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    """One connected candidate represented by exact identity and matrix."""

    key: ShapeKey
    matrix: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.matrix, dtype=np.uint8)
        if values.ndim != 2 or values.size == 0 or not np.any(values):
            raise ValueError("candidate matrix must be a non-empty nonzero matrix")
        if shape_key_from_matrix(values) != self.key:
            raise ValueError("candidate matrix and key must agree")
        values = (values != 0).astype(np.uint8, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "matrix", values)

    @property
    def cell_count(self) -> int:
        """Return the number of live cells in the candidate."""

        return int(self.matrix.sum())


@dataclass(frozen=True)
class ReplicatorEvaluation:
    """Best replication evidence observed during one candidate rollout."""

    candidate_key: ShapeKey
    initial_cells: int
    best_generation: int
    best_copy_count: int
    best_purity: float
    best_approximate_purity: float
    best_repeat_count: int
    best_repeat_purity: float
    best_repeat_key: ShapeKey | None
    best_mass_balance: float
    best_similarity: float
    best_score: float
    is_replicator: bool
    is_fission_like: bool


@dataclass(frozen=True)
class RolloutTermination:
    """Reason one candidate environment stopped advancing."""

    candidate_key: ShapeKey
    generation: int
    reason: str
    period: int | None = None


@dataclass
class ReplicatorSearchResult:
    """Search output with compact history and evaluated candidate cache."""

    config: ReplicatorSearchConfig
    best_candidate: Candidate
    best_evaluation: ReplicatorEvaluation
    best_state: np.ndarray
    found: bool
    found_search_generation: int | None
    history: list[TraceRow]
    trace: list[TraceRow]
    evaluations: dict[ShapeKey, ReplicatorEvaluation]
    elapsed_seconds: float
    terminal_reports: dict[ShapeKey, RolloutTermination] = field(default_factory=dict)


def _is_connected(coordinates: set[tuple[int, int]]) -> bool:
    """Check local non-toroidal Moore connectivity for a candidate."""

    if not coordinates:
        return False
    pending = [min(coordinates)]
    visited = {pending[0]}
    while pending:
        row, col = pending.pop()
        for row_delta, col_delta in MOORE_OFFSETS:
            neighbor = (row + row_delta, col + col_delta)
            if neighbor in coordinates and neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) == len(coordinates)


def _candidate_from_matrix(matrix: Any, config: ReplicatorSearchConfig) -> Candidate:
    """Canonicalize one connected matrix and enforce the search size bound."""

    values = canonical_matrix_from_grid(
        matrix,
        rotation_invariant=config.rotation_invariant,
        reflection_invariant=config.reflection_invariant,
    )
    if max(values.shape) > config.candidate_size:
        raise ValueError("candidate exceeds candidate_size")
    key = shape_key_from_matrix(values)
    return Candidate(key, values)


def _frontier(coordinates: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return sorted empty Moore neighbors around a connected shape."""

    candidates = {
        (row + row_delta, col + col_delta)
        for row, col in coordinates
        for row_delta, col_delta in MOORE_OFFSETS
        if (row + row_delta, col + col_delta) not in coordinates
    }
    return sorted(candidates)


def _within_candidate_size(coordinates: set[tuple[int, int]], size: int) -> bool:
    """Return whether coordinates fit in the configured local canvas."""

    rows = [row for row, _ in coordinates]
    cols = [col for _, col in coordinates]
    return max(rows) - min(rows) + 1 <= size and max(cols) - min(cols) + 1 <= size


def _matrix_from_coordinates(coordinates: set[tuple[int, int]]) -> np.ndarray:
    """Trim a coordinate set into a binary local matrix."""

    rows = np.asarray([row for row, _ in coordinates], dtype=np.int64)
    cols = np.asarray([col for _, col in coordinates], dtype=np.int64)
    shifted_rows = rows - rows.min()
    shifted_cols = cols - cols.min()
    matrix = np.zeros((int(shifted_rows.max()) + 1, int(shifted_cols.max()) + 1), dtype=np.uint8)
    matrix[shifted_rows, shifted_cols] = 1
    return matrix


def _grow_initial_candidate(
    config: ReplicatorSearchConfig,
    population_index: int,
) -> Candidate:
    """Grow one deterministic connected random candidate from a seed cell."""

    coordinates = {(0, 0)}
    maximum_cells = min(config.initial_live_cells, config.candidate_size**2)
    target_cells = 1 + deterministic_uint64(
        config.seed, "replicator-initial-size", population_index
    ) % max(1, maximum_cells)
    for growth_index in range(target_cells - 1):
        frontier = [
            coordinate
            for coordinate in _frontier(coordinates)
            if _within_candidate_size(coordinates | {coordinate}, config.candidate_size)
        ]
        if not frontier:
            break
        choice = deterministic_uint64(
            config.seed,
            "replicator-initial-growth",
            population_index * max(1, config.initial_live_cells) + growth_index,
        ) % len(frontier)
        coordinates.add(frontier[int(choice)])
    return _candidate_from_matrix(_matrix_from_coordinates(coordinates), config)


def _seeded_patterns(config: ReplicatorSearchConfig) -> list[Candidate]:
    """Return small native patterns useful as explicit search controls."""

    candidates: list[Candidate] = []
    for name in ("block", "blinker", "glider"):
        pattern = np.asarray(get_pattern(name), dtype=np.uint8)
        if max(pattern.shape) <= config.candidate_size:
            candidates.append(_candidate_from_matrix(pattern, config))
    return candidates


def initial_population(config: ReplicatorSearchConfig) -> list[Candidate]:
    """Create a deterministic population with known-pattern controls."""

    population: list[Candidate] = []
    by_key: dict[ShapeKey, Candidate] = {}
    for candidate in _seeded_patterns(config):
        if candidate.key not in by_key:
            by_key[candidate.key] = candidate
            population.append(candidate)
    attempts = 0
    max_attempts = max(100, config.population_size * 100)
    while len(population) < config.population_size and attempts < max_attempts:
        candidate = _grow_initial_candidate(config, attempts)
        if candidate.key not in by_key:
            by_key[candidate.key] = candidate
            population.append(candidate)
        attempts += 1
    # Tiny candidate canvases have a finite identity space.  Duplicates are
    # preferable to silently returning a smaller population in that case.
    while len(population) < config.population_size:
        population.append(_grow_initial_candidate(config, len(population)))
    return population


def mutate_candidate(
    parent: Candidate,
    config: ReplicatorSearchConfig,
    *,
    search_generation: int,
    child_index: int,
) -> Candidate:
    """Apply deterministic connected add/remove mutations to one parent."""

    coordinates = {tuple(value) for value in np.argwhere(parent.matrix != 0).tolist()}
    for mutation_index in range(config.mutations_per_child):
        draw_index = (
            search_generation * max(1, config.population_size) * max(1, config.mutations_per_child)
            + child_index * max(1, config.mutations_per_child)
            + mutation_index
        )
        operation = (
            deterministic_uint64(config.seed, "replicator-mutation-operation", draw_index) % 2
        )
        if operation == 0:
            choices = [
                coordinate
                for coordinate in _frontier(coordinates)
                if _within_candidate_size(coordinates | {coordinate}, config.candidate_size)
            ]
            if choices:
                choice = deterministic_uint64(
                    config.seed, "replicator-mutation-choice", draw_index
                ) % len(choices)
                coordinates.add(choices[int(choice)])
        else:
            choices = [
                coordinate
                for coordinate in sorted(coordinates)
                if len(coordinates) > 1 and _is_connected(coordinates - {coordinate})
            ]
            if choices:
                choice = deterministic_uint64(
                    config.seed, "replicator-mutation-choice", draw_index
                ) % len(choices)
                coordinates.remove(choices[int(choice)])
    return _candidate_from_matrix(_matrix_from_coordinates(coordinates), config)


def _matrix_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return centered Dice similarity between two canonical matrices."""

    height = max(first.shape[0], second.shape[0])
    width = max(first.shape[1], second.shape[1])
    first_padded = np.zeros((height, width), dtype=np.uint8)
    second_padded = np.zeros((height, width), dtype=np.uint8)
    first_padded[: first.shape[0], : first.shape[1]] = first
    second_padded[: second.shape[0], : second.shape[1]] = second
    intersection = int(np.count_nonzero(first_padded & second_padded))
    denominator = int(first.sum()) + int(second.sum())
    return 0.0 if denominator == 0 else 2.0 * intersection / denominator


def _observation(
    candidate: Candidate,
    grid: np.ndarray,
    components: Iterable[Any],
    generation: int,
    min_purity: float,
    *,
    rotation_invariant: bool,
    reflection_invariant: bool,
) -> ReplicatorEvaluation:
    """Score one state against a candidate's exact identity."""

    component_list = list(components)
    target_components = []
    component_keys: list[ShapeKey] = []
    similarities: list[float] = []
    best_similarity = 0.0
    for component in component_list:
        key = canonicalize_component(
            component,
            grid_shape=grid.shape,
            rotation_invariant=rotation_invariant,
            reflection_invariant=reflection_invariant,
        )
        matrix = matrix_from_shape_key(key)
        component_keys.append(key)
        similarity = _matrix_similarity(candidate.matrix, matrix)
        similarities.append(similarity)
        best_similarity = max(best_similarity, similarity)
        if key == candidate.key:
            target_components.append(component)
    copy_count = len(target_components)
    initial_cells = candidate.cell_count
    alive_cells = int(np.count_nonzero(grid))
    target_cells = copy_count * initial_cells
    purity = min(1.0, target_cells / alive_cells) if alive_cells else 0.0
    expected_cells = max(1, 2 * initial_cells)
    mass_balance = max(0.0, 1.0 - abs(alive_cells - expected_cells) / expected_cells)
    similarity_mass = min(2.0, sum(sorted(similarities, reverse=True)[:2]))
    approximate_target_cells = initial_cells * similarity_mass
    approximate_purity = min(1.0, approximate_target_cells / alive_cells) if alive_cells else 0.0
    copy_score = similarity_mass / 2.0
    # Exact copies determine the hit flag.  Similarity-weighted copy mass is a
    # search heuristic so evolution can prefer promising near-copies before a
    # candidate satisfies the exact ShapeKey criterion.
    near_copy_score = copy_score * approximate_purity * mass_balance
    repeat_counts = Counter(component_keys)
    repeated_keys = [key for key, count in repeat_counts.items() if count >= 2]
    if repeated_keys:
        repeat_count = max(repeat_counts[key] for key in repeated_keys)
        repeat_key = min(
            (key for key in repeated_keys if repeat_counts[key] == repeat_count),
            key=shape_key_sort_key,
        )
        repeated_cells = repeat_count * int(matrix_from_shape_key(repeat_key).sum())
        repeat_purity = min(1.0, repeated_cells / alive_cells) if alive_cells else 0.0
    else:
        repeat_count = 0
        repeat_key = None
        repeat_purity = 0.0
    repeat_score = min(1.0, repeat_count / 2.0) * repeat_purity * mass_balance
    score = max(near_copy_score, repeat_score) + 0.05 * best_similarity * max(
        copy_score,
        min(1.0, repeat_count / 2.0),
    )
    return ReplicatorEvaluation(
        candidate_key=candidate.key,
        initial_cells=initial_cells,
        best_generation=generation,
        best_copy_count=copy_count,
        best_purity=purity,
        best_approximate_purity=approximate_purity,
        best_repeat_count=repeat_count,
        best_repeat_purity=repeat_purity,
        best_repeat_key=repeat_key,
        best_mass_balance=mass_balance,
        best_similarity=best_similarity,
        best_score=score,
        is_replicator=copy_count >= 2 and purity >= min_purity,
        is_fission_like=generation > 0 and repeat_count >= 2 and repeat_purity >= min_purity,
    )


def evaluate_candidate_state(
    candidate: Candidate,
    grid: Any,
    *,
    generation: int = 0,
    min_purity: float = 0.90,
    component_backend: str = "auto",
    rotation_invariant: bool = True,
    reflection_invariant: bool = False,
) -> ReplicatorEvaluation:
    """Score one explicitly supplied state against a candidate.

    This small public hook makes a claimed hit independently inspectable: a
    caller can load a saved grid, run the same toroidal component detector,
    and evaluate the exact-copy criterion without rerunning the evolutionary
    search.
    """

    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    values = np.asarray(grid)
    if values.ndim != 2 or any(dimension < 1 for dimension in values.shape):
        raise ValueError("grid must be a non-empty two-dimensional array")
    try:
        generation_value = operator.index(generation)
    except TypeError as exc:
        raise TypeError("generation must be an integer") from exc
    if generation_value < 0:
        raise ValueError("generation must be non-negative")
    if not isinstance(min_purity, (int, float)) or not np.isfinite(min_purity):
        raise ValueError("min_purity must be a finite number between 0 and 1")
    if not 0.0 <= float(min_purity) <= 1.0:
        raise ValueError("min_purity must be between 0 and 1")
    if not isinstance(rotation_invariant, bool) or not isinstance(reflection_invariant, bool):
        raise TypeError("rotation_invariant and reflection_invariant must be booleans")
    if reflection_invariant and not rotation_invariant:
        raise ValueError("reflection_invariant requires rotation_invariant")
    components = detect_components(values, min_component_cells=1, backend=component_backend)
    return _observation(
        candidate,
        values,
        components,
        generation_value,
        float(min_purity),
        rotation_invariant=rotation_invariant,
        reflection_invariant=reflection_invariant,
    )


def _evaluation_rank(
    evaluation: ReplicatorEvaluation,
) -> tuple[int, int, float, int, int, float, float, float, float, int]:
    """Return the deterministic order used for elite selection."""

    return (
        int(evaluation.is_replicator),
        int(evaluation.is_fission_like),
        evaluation.best_score,
        evaluation.best_copy_count,
        evaluation.best_repeat_count,
        evaluation.best_purity,
        evaluation.best_repeat_purity,
        evaluation.best_mass_balance,
        evaluation.best_similarity,
        -evaluation.best_generation,
    )


def _better_evaluation(
    candidate: Candidate,
    evaluation: ReplicatorEvaluation,
    best_candidate: Candidate,
    best_evaluation: ReplicatorEvaluation,
) -> bool:
    """Compare two candidate/evaluation pairs with key tie-breaking."""

    first_rank = _evaluation_rank(evaluation)
    second_rank = _evaluation_rank(best_evaluation)
    if first_rank != second_rank:
        return first_rank > second_rank
    return shape_key_sort_key(candidate.key) < shape_key_sort_key(best_candidate.key)


def _worlds_for_candidates(candidates: list[Candidate], world_size: int) -> np.ndarray:
    """Place canonical candidates at the center of independent toroidal worlds."""

    worlds = np.zeros((len(candidates), world_size, world_size), dtype=np.uint8)
    for index, candidate in enumerate(candidates):
        row = (world_size - candidate.matrix.shape[0]) // 2
        col = (world_size - candidate.matrix.shape[1]) // 2
        height, width = candidate.matrix.shape
        worlds[index, row : row + height, col : col + width] = candidate.matrix
    return worlds


def _trace_row(
    search_generation: int,
    candidate: Candidate,
    evaluation: ReplicatorEvaluation,
) -> TraceRow:
    """Serialize one candidate's best rollout evidence for ``trace.csv``."""

    return {
        "search_generation": search_generation,
        "candidate_height": candidate.key.height,
        "candidate_width": candidate.key.width,
        "candidate_packed_hex": candidate.key.packed_hex,
        "candidate_cells": candidate.cell_count,
        "best_generation": evaluation.best_generation,
        "best_copy_count": evaluation.best_copy_count,
        "best_purity": evaluation.best_purity,
        "best_approximate_purity": evaluation.best_approximate_purity,
        "best_repeat_count": evaluation.best_repeat_count,
        "best_repeat_purity": evaluation.best_repeat_purity,
        "best_mass_balance": evaluation.best_mass_balance,
        "best_similarity": evaluation.best_similarity,
        "best_score": evaluation.best_score,
        "is_replicator": evaluation.is_replicator,
        "is_fission_like": evaluation.is_fission_like,
    }


def _evaluate_candidates(
    candidates: list[Candidate],
    config: ReplicatorSearchConfig,
    birth_mask: Any,
    survival_mask: Any,
) -> tuple[
    dict[ShapeKey, ReplicatorEvaluation],
    dict[ShapeKey, np.ndarray],
    dict[ShapeKey, RolloutTermination],
]:
    """Batch native rollouts with independent deterministic termination.

    The transition remains one JAX batch call for all currently active
    candidates. Terminal worlds are carried forward unchanged in that batch,
    while component analysis and exact-state recurrence checks happen on the
    host between transitions. This lets short-lived candidates stop without
    forcing the rest of the population to stop with them.
    """

    unique_candidates: list[Candidate] = []
    seen: set[ShapeKey] = set()
    for candidate in candidates:
        if candidate.key not in seen:
            seen.add(candidate.key)
            unique_candidates.append(candidate)
    initial_worlds = _worlds_for_candidates(unique_candidates, config.world_size)
    current = jnp.asarray(initial_worlds, dtype=jnp.uint8)
    best: dict[ShapeKey, ReplicatorEvaluation] = {}
    best_states: dict[ShapeKey, np.ndarray] = {}
    terminal_reports: dict[ShapeKey, RolloutTermination] = {}
    active = np.ones(len(unique_candidates), dtype=bool)
    state_history: list[dict[bytes, int]] = [
        {np.ascontiguousarray(world, dtype=np.uint8).tobytes(): 0} for world in initial_worlds
    ]
    for generation in range(config.evaluation_steps + 1):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        host = np.asarray(jax.device_get(current), dtype=np.uint8)
        components_by_world = detect_components_batch(
            host[active_indices],
            min_component_cells=1,
            backend=config.component_backend,
        )
        for local_index, index_value in enumerate(active_indices):
            index = int(index_value)
            candidate = unique_candidates[index]
            observation = _observation(
                candidate,
                host[index],
                components_by_world[local_index],
                generation,
                config.min_purity,
                rotation_invariant=config.rotation_invariant,
                reflection_invariant=config.reflection_invariant,
            )
            previous = best.get(candidate.key)
            if previous is None or _evaluation_rank(observation) > _evaluation_rank(previous):
                best[candidate.key] = observation
                state = host[index].copy()
                state.setflags(write=False)
                best_states[candidate.key] = state

            if observation.is_replicator:
                terminal_reports[candidate.key] = RolloutTermination(
                    candidate_key=candidate.key,
                    generation=generation,
                    reason="replicator",
                )
                active[index] = False
                continue

            if config.terminate_on_terminal and generation > 0:
                if not np.any(host[index]):
                    terminal_reports[candidate.key] = RolloutTermination(
                        candidate_key=candidate.key,
                        generation=generation,
                        reason="extinct",
                    )
                    active[index] = False
                    continue
                fingerprint = np.ascontiguousarray(host[index], dtype=np.uint8).tobytes()
                previous_generation = state_history[index].get(fingerprint)
                if previous_generation is not None:
                    period = generation - previous_generation
                    if period <= config.max_cycle_period:
                        reason = "stable" if period == 1 else f"cycle_{period}"
                        terminal_reports[candidate.key] = RolloutTermination(
                            candidate_key=candidate.key,
                            generation=generation,
                            reason=reason,
                            period=period,
                        )
                        active[index] = False
                        continue
                else:
                    state_history[index][fingerprint] = generation

        if generation < config.evaluation_steps:
            if not np.any(active):
                break
            next_current = batched_step(current, birth_mask, survival_mask)
            current = jnp.where(
                jnp.asarray(active)[:, None, None],
                next_current,
                current,
            )
    for index, candidate in enumerate(unique_candidates):
        if candidate.key not in terminal_reports:
            terminal_reports[candidate.key] = RolloutTermination(
                candidate_key=candidate.key,
                generation=config.evaluation_steps,
                reason="horizon",
            )
    return best, best_states, terminal_reports


def run_replicator_search(config: ReplicatorSearchConfig) -> ReplicatorSearchResult:
    """Run the deterministic evolutionary search and return its evidence."""

    start = time.perf_counter()
    birth_mask, survival_mask = rule_to_masks(parse_rule(NATIVE_RULE))
    population = initial_population(config)
    evaluations: dict[ShapeKey, ReplicatorEvaluation] = {}
    history: list[TraceRow] = []
    trace: list[TraceRow] = []
    terminal_reports: dict[ShapeKey, RolloutTermination] = {}
    best_candidate = population[0]
    best_evaluation = ReplicatorEvaluation(
        candidate_key=best_candidate.key,
        initial_cells=best_candidate.cell_count,
        best_generation=0,
        best_copy_count=0,
        best_purity=0.0,
        best_approximate_purity=0.0,
        best_repeat_count=0,
        best_repeat_purity=0.0,
        best_repeat_key=None,
        best_mass_balance=0.0,
        best_similarity=0.0,
        best_score=0.0,
        is_replicator=False,
        is_fission_like=False,
    )
    found_search_generation: int | None = None
    best_state = _worlds_for_candidates([best_candidate], config.world_size)[0]
    best_state.setflags(write=False)

    for search_generation in range(config.generations + 1):
        current_evaluations, current_states, current_terminal_reports = _evaluate_candidates(
            population,
            config,
            birth_mask,
            survival_mask,
        )
        evaluations.update(current_evaluations)
        terminal_reports.update(current_terminal_reports)
        unique_population = {candidate.key: candidate for candidate in population}
        trace.extend(
            _trace_row(search_generation, candidate, current_evaluations[candidate.key])
            for candidate in sorted(
                unique_population.values(),
                key=lambda item: shape_key_sort_key(item.key),
            )
        )
        scored = [(candidate, current_evaluations[candidate.key]) for candidate in population]
        scored.sort(key=lambda item: shape_key_sort_key(item[0].key))
        scored.sort(key=lambda item: _evaluation_rank(item[1]), reverse=True)
        generation_best_candidate, generation_best_evaluation = scored[0]
        if _better_evaluation(
            generation_best_candidate,
            generation_best_evaluation,
            best_candidate,
            best_evaluation,
        ):
            best_candidate = generation_best_candidate
            best_evaluation = generation_best_evaluation
            best_state = current_states[generation_best_candidate.key]
        if generation_best_evaluation.is_replicator and found_search_generation is None:
            found_search_generation = search_generation
        history.append(
            {
                "search_generation": search_generation,
                "population_size": len(population),
                "unique_evaluations_total": len(evaluations),
                "best_score": generation_best_evaluation.best_score,
                "best_copy_count": generation_best_evaluation.best_copy_count,
                "best_purity": generation_best_evaluation.best_purity,
                "best_approximate_purity": generation_best_evaluation.best_approximate_purity,
                "best_repeat_count": generation_best_evaluation.best_repeat_count,
                "best_repeat_purity": generation_best_evaluation.best_repeat_purity,
                "best_mass_balance": generation_best_evaluation.best_mass_balance,
                "best_similarity": generation_best_evaluation.best_similarity,
                "best_evaluation_generation": generation_best_evaluation.best_generation,
                "best_is_replicator": generation_best_evaluation.is_replicator,
                "best_is_fission_like": generation_best_evaluation.is_fission_like,
                "best_height": generation_best_candidate.matrix.shape[0],
                "best_width": generation_best_candidate.matrix.shape[1],
                "best_cells": generation_best_candidate.cell_count,
                "best_packed_hex": generation_best_candidate.key.packed_hex,
                "terminal_rollouts": len(current_terminal_reports),
                "terminal_replicators": sum(
                    report.reason == "replicator" for report in current_terminal_reports.values()
                ),
                "terminal_stable": sum(
                    report.reason == "stable" for report in current_terminal_reports.values()
                ),
                "terminal_cycles": sum(
                    report.reason.startswith("cycle_")
                    for report in current_terminal_reports.values()
                ),
                "terminal_extinct": sum(
                    report.reason == "extinct" for report in current_terminal_reports.values()
                ),
                "terminal_horizon": sum(
                    report.reason == "horizon" for report in current_terminal_reports.values()
                ),
            }
        )
        if found_search_generation is not None or search_generation == config.generations:
            break
        elite_count = min(config.elite_count, len(scored))
        elites = [candidate for candidate, _ in scored[:elite_count]]
        next_population = list(elites)
        for child_index in range(config.population_size - elite_count):
            parent_index = deterministic_uint64(
                config.seed,
                "replicator-parent-selection",
                search_generation * max(1, config.population_size) + child_index,
            ) % len(elites)
            next_population.append(
                mutate_candidate(
                    elites[int(parent_index)],
                    config,
                    search_generation=search_generation,
                    child_index=child_index,
                )
            )
        population = next_population
    return ReplicatorSearchResult(
        config=config,
        best_candidate=best_candidate,
        best_evaluation=best_evaluation,
        best_state=best_state,
        found=best_evaluation.is_replicator,
        found_search_generation=found_search_generation,
        history=history,
        trace=trace,
        evaluations=evaluations,
        elapsed_seconds=time.perf_counter() - start,
        terminal_reports=terminal_reports,
    )


def matrix_to_text(matrix: Any) -> str:
    """Format a binary pattern as ``#``/``.`` rows."""

    values = np.asarray(matrix) != 0
    if values.ndim != 2 or values.size == 0 or not np.any(values):
        raise ValueError("matrix must be a non-empty nonzero two-dimensional array")
    return "\n".join("".join("#" if cell else "." for cell in row) for row in values)


def _shape_key_json(key: ShapeKey) -> dict[str, Any]:
    """Serialize one exact key for search metadata."""

    return {
        "height": key.height,
        "width": key.width,
        "packed_hex": key.packed_hex,
    }


def write_search_artifacts(
    result: ReplicatorSearchResult,
    output_dir: str | Path,
    *,
    timestamp: str | None = None,
) -> Path:
    """Write compact search metadata, history, and the best pattern."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    best = result.best_evaluation
    manifest = {
        "experiment": "morphology_interactions",
        "search": "replicator",
        "version": 1,
        "config": result.config.as_dict(),
        "base_rule": format_rule(parse_rule(NATIVE_RULE)),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_version": getattr(jax, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "found": result.found,
        "found_search_generation": result.found_search_generation,
        "elapsed_seconds": result.elapsed_seconds,
        "evaluated_unique_candidates": len(result.evaluations),
        "trace_rows": len(result.trace),
        "terminal_rollouts": len(result.terminal_reports),
        "terminal_reason_counts": dict(
            sorted(Counter(report.reason for report in result.terminal_reports.values()).items())
        ),
        "best_pattern": matrix_to_text(result.best_candidate.matrix),
        "best_shape_key": _shape_key_json(result.best_candidate.key),
        "best_state_generation": best.best_generation,
        "best_state_alive_cells": int(np.count_nonzero(result.best_state)),
        "best_evaluation": {
            "initial_cells": best.initial_cells,
            "best_generation": best.best_generation,
            "best_copy_count": best.best_copy_count,
            "best_purity": best.best_purity,
            "best_approximate_purity": best.best_approximate_purity,
            "best_repeat_count": best.best_repeat_count,
            "best_repeat_purity": best.best_repeat_purity,
            "best_repeat_shape_key": (
                None if best.best_repeat_key is None else _shape_key_json(best.best_repeat_key)
            ),
            "best_mass_balance": best.best_mass_balance,
            "best_similarity": best.best_similarity,
            "best_score": best.best_score,
            "is_replicator": best.is_replicator,
            "is_fission_like": best.is_fission_like,
        },
    }
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = tuple(result.history[0].keys()) if result.history else ()
    with (target / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(result.history)
    trace_fields = tuple(result.trace[0].keys()) if result.trace else ()
    with (target / "trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        if trace_fields:
            writer.writeheader()
            writer.writerows(result.trace)
    terminal_fields = (
        "candidate_height",
        "candidate_width",
        "candidate_packed_hex",
        "terminal_generation",
        "terminal_reason",
        "period",
    )
    terminal_rows = [
        {
            "candidate_height": key.height,
            "candidate_width": key.width,
            "candidate_packed_hex": key.packed_hex,
            "terminal_generation": report.generation,
            "terminal_reason": report.reason,
            "period": report.period,
        }
        for key, report in sorted(
            result.terminal_reports.items(),
            key=lambda item: shape_key_sort_key(item[0]),
        )
    ]
    with (target / "terminal.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=terminal_fields)
        writer.writeheader()
        writer.writerows(terminal_rows)
    (target / "identity_audit.json").write_text(
        json.dumps(
            audit_encodings(
                result.evaluations,
                universe_seed=result.config.seed,
                identity_dim=32,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (target / "best_pattern.txt").write_text(
        matrix_to_text(result.best_candidate.matrix) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(target / "best_pattern.npz", grid=result.best_candidate.matrix)
    np.savez_compressed(target / "best_state.npz", grid=result.best_state)
    return target


def build_parser() -> argparse.ArgumentParser:
    """Build the replicator-search CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-size", type=int, default=9)
    parser.add_argument("--world-size", type=int, default=64)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--evaluation-steps", type=int, default=32)
    parser.add_argument("--mutations-per-child", type=int, default=2)
    parser.add_argument("--initial-live-cells", type=int, default=8)
    parser.add_argument("--min-purity", type=float, default=0.90)
    parser.add_argument("--max-cycle-period", type=int, default=2)
    parser.add_argument(
        "--no-terminal-termination",
        action="store_true",
        help="run every rollout to evaluation_steps for a fixed-horizon ablation",
    )
    parser.add_argument(
        "--component-backend",
        choices=("auto", "python", "scipy"),
        default="auto",
    )
    parser.add_argument("--reflection-invariant", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a deterministic native-Life replicator search."""

    args = build_parser().parse_args(argv)
    config = ReplicatorSearchConfig(
        seed=args.seed,
        candidate_size=args.candidate_size,
        world_size=args.world_size,
        population_size=args.population_size,
        elite_count=args.elite_count,
        generations=args.generations,
        evaluation_steps=args.evaluation_steps,
        mutations_per_child=args.mutations_per_child,
        initial_live_cells=args.initial_live_cells,
        min_purity=args.min_purity,
        terminate_on_terminal=not args.no_terminal_termination,
        max_cycle_period=args.max_cycle_period,
        component_backend=args.component_backend,
        reflection_invariant=args.reflection_invariant,
    )
    result = run_replicator_search(config)
    output_dir = args.output_dir
    if output_dir is not None:
        write_search_artifacts(result, output_dir)
    print(
        json.dumps(
            {
                "found": result.found,
                "found_search_generation": result.found_search_generation,
                "best_pattern": matrix_to_text(result.best_candidate.matrix),
                "best_evaluation": {
                    "best_generation": result.best_evaluation.best_generation,
                    "best_copy_count": result.best_evaluation.best_copy_count,
                    "best_purity": result.best_evaluation.best_purity,
                    "best_approximate_purity": result.best_evaluation.best_approximate_purity,
                    "best_repeat_count": result.best_evaluation.best_repeat_count,
                    "best_repeat_purity": result.best_evaluation.best_repeat_purity,
                    "best_score": result.best_evaluation.best_score,
                    "is_replicator": result.best_evaluation.is_replicator,
                    "is_fission_like": result.best_evaluation.is_fission_like,
                },
                "evaluated_unique_candidates": len(result.evaluations),
                "trace_rows": len(result.trace),
                "terminal_rollouts": len(result.terminal_reports),
                "terminal_reason_counts": dict(
                    sorted(
                        Counter(
                            report.reason for report in result.terminal_reports.values()
                        ).items()
                    )
                ),
                "elapsed_seconds": result.elapsed_seconds,
                "output_dir": None if output_dir is None else str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
