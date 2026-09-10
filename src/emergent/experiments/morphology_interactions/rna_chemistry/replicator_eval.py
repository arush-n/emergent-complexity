"""Replication evidence evaluation under the RNA-inspired chemistry.

This is deliberately an evaluator rather than a replication mechanism.  It
recognizes repeated exact morphology descendants in the ordinary CA state and
never inserts a copy/replicate action into the chemistry.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..canonical import ShapeKey, canonicalize_component
from ..components import detect_components
from ..replicator_search import Candidate
from .config import RNAExperimentConfig
from .engine import RNAChemistryEngine


@dataclass(frozen=True)
class LineageEvaluation:
    """Repeated-copy evidence from one candidate rollout."""

    candidate_key: ShapeKey
    initial_cells: int
    max_exact_copy_count: int
    first_double_generation: int | None
    max_copy_generation: int
    max_copy_purity: float
    daughter_persistence: int
    repeated_double_generations: int
    replication_depth: int
    max_alive_cells: int
    debris_fraction_at_best: float
    is_replication_like: bool
    is_sustained: bool


def _center_candidate(candidate: Candidate, config: RNAExperimentConfig) -> np.ndarray:
    """Place one candidate in the center of a fresh toroidal world."""

    result = np.zeros((config.height, config.width), dtype=np.uint8)
    row = (config.height - candidate.matrix.shape[0]) // 2
    col = (config.width - candidate.matrix.shape[1]) // 2
    result[
        row : row + candidate.matrix.shape[0],
        col : col + candidate.matrix.shape[1],
    ] = candidate.matrix
    return result


def _copy_count(
    candidate: Candidate,
    grid: np.ndarray,
    *,
    component_backend: str,
    rotation_invariant: bool,
    reflection_invariant: bool,
) -> tuple[int, int]:
    components = detect_components(grid, backend=component_backend)
    exact = 0
    for component in components:
        key = canonicalize_component(
            component,
            grid_shape=grid.shape,
            rotation_invariant=rotation_invariant,
            reflection_invariant=reflection_invariant,
        )
        exact += int(key == candidate.key)
    return exact, int(np.count_nonzero(grid))


def evaluate_candidate(
    candidate: Candidate,
    config: RNAExperimentConfig,
    *,
    min_copy_purity: float = 0.90,
    persistence_steps: int = 8,
) -> LineageEvaluation:
    """Evaluate exact-copy depth and persistence for one RNA-world candidate."""

    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a morphology-interaction Candidate")
    if not 0.0 <= float(min_copy_purity) <= 1.0:
        raise ValueError("min_copy_purity must be between 0 and 1")
    if not isinstance(persistence_steps, int) or persistence_steps < 1:
        raise ValueError("persistence_steps must be positive")
    initial = _center_candidate(candidate, config)
    engine = RNAChemistryEngine(config, initial_grid=initial)
    copy_history: list[int] = []
    alive_history: list[int] = []
    for generation in range(config.steps + 1):
        current = np.asarray(engine.grid if generation else initial, dtype=np.uint8)
        count, alive = _copy_count(
            candidate,
            current,
            component_backend=config.component_backend,
            rotation_invariant=config.rotation_invariant,
            reflection_invariant=config.reflection_invariant,
        )
        copy_history.append(count)
        alive_history.append(alive)
        if generation < config.steps:
            engine.step()
    max_copy_count = max(copy_history, default=0)
    max_copy_generation = max(
        range(len(copy_history)),
        key=lambda index: (copy_history[index], -index),
        default=0,
    )
    best_alive = max(1, alive_history[max_copy_generation])
    max_copy_purity = min(1.0, max_copy_count * candidate.cell_count / best_alive)
    first_double = next(
        (generation for generation, count in enumerate(copy_history) if count >= 2),
        None,
    )
    repeated_double = sum(count >= 2 for count in copy_history)
    persistence = 0
    if first_double is not None:
        persistence = sum(
            count >= 1
            for count in copy_history[first_double + 1 : first_double + 1 + persistence_steps]
        )
    replication_depth = int(np.floor(np.log2(max(1, max_copy_count))))
    debris_fraction = max(
        0.0,
        1.0 - (max_copy_count * candidate.cell_count) / best_alive,
    )
    is_replication_like = max_copy_count >= 2 and max_copy_purity >= min_copy_purity
    is_sustained = is_replication_like and (
        persistence >= min(2, persistence_steps) and replication_depth >= 2
    )
    return LineageEvaluation(
        candidate_key=candidate.key,
        initial_cells=candidate.cell_count,
        max_exact_copy_count=max_copy_count,
        first_double_generation=first_double,
        max_copy_generation=max_copy_generation,
        max_copy_purity=float(max_copy_purity),
        daughter_persistence=persistence,
        repeated_double_generations=repeated_double,
        replication_depth=replication_depth,
        max_alive_cells=max(alive_history, default=0),
        debris_fraction_at_best=float(debris_fraction),
        is_replication_like=is_replication_like,
        is_sustained=is_sustained,
    )


def compare_candidate_alphas(
    candidate: Candidate,
    config: RNAExperimentConfig,
    alphas: Iterable[float] = (0.0, 0.5, 1.0),
) -> list[dict[str, int | float | bool]]:
    """Evaluate one candidate under controlled structured/mixed/scrambled laws."""

    rows: list[dict[str, int | float | bool]] = []
    for alpha in alphas:
        value = float(alpha)
        if not 0.0 <= value <= 1.0:
            raise ValueError("alphas must lie between 0 and 1")
        run_config = RNAExperimentConfig(**{**config.as_dict(), "alpha": value})
        evaluation = evaluate_candidate(candidate, run_config)
        rows.append(
            {
                "alpha": value,
                "max_exact_copy_count": evaluation.max_exact_copy_count,
                "first_double_generation": (
                    -1
                    if evaluation.first_double_generation is None
                    else evaluation.first_double_generation
                ),
                "max_copy_purity": evaluation.max_copy_purity,
                "daughter_persistence": evaluation.daughter_persistence,
                "replication_depth": evaluation.replication_depth,
                "debris_fraction_at_best": evaluation.debris_fraction_at_best,
                "is_replication_like": evaluation.is_replication_like,
                "is_sustained": evaluation.is_sustained,
            }
        )
    return rows
