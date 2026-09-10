"""Discrete one-cell sensitivity analysis over morphology interaction space."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ...io.patterns import pattern_from_text
from .canonical import (
    ShapeKey,
    canonical_matrix,
    canonical_matrix_from_grid,
    shape_key_from_matrix,
    shape_key_sort_key,
)
from .components import MOORE_OFFSETS
from .interaction import make_interaction_universe, make_pair_interaction
from .sweep import parse_alphas


@dataclass(frozen=True)
class ShapePerturbation:
    """One valid add/remove-one-cell morphology candidate."""

    label: str
    key: ShapeKey
    matrix: np.ndarray


def _canonical_input(shape: ShapeKey | Any) -> tuple[ShapeKey, np.ndarray]:
    if isinstance(shape, ShapeKey):
        from .canonical import matrix_from_shape_key

        return shape, matrix_from_shape_key(shape)
    matrix = canonical_matrix_from_grid(shape)
    return shape_key_from_matrix(matrix), matrix


def _is_single_8_connected(coordinates: set[tuple[int, int]]) -> bool:
    """Return whether local coordinates form exactly one Moore component.

    Sensitivity candidates are local, unwrapped morphologies.  Using the
    toroidal detector here would incorrectly join cells on opposite edges of
    the candidate's bounding matrix, so this small non-toroidal check is
    intentional.
    """

    if not coordinates:
        return False
    visited = {min(coordinates)}
    pending = [next(iter(visited))]
    while pending:
        row, col = pending.pop()
        for row_delta, col_delta in MOORE_OFFSETS:
            neighbor = (row + row_delta, col + col_delta)
            if neighbor in coordinates and neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) == len(coordinates)


def single_cell_perturbations(shape: ShapeKey | Any) -> list[ShapePerturbation]:
    """Generate deterministic add/remove-one-cell candidates around ``shape``."""

    _, matrix = _canonical_input(shape)
    live_coordinates = {tuple(value) for value in np.argwhere(matrix != 0).tolist()}
    candidates: dict[ShapeKey, ShapePerturbation] = {}

    for row, col in sorted(live_coordinates):
        for row_delta in (-1, 0, 1):
            for col_delta in (-1, 0, 1):
                if (row_delta, col_delta) == (0, 0):
                    continue
                added = (row + row_delta, col + col_delta)
                if added in live_coordinates:
                    continue
                candidate_coordinates = live_coordinates | {added}
                if not _is_single_8_connected(candidate_coordinates):
                    continue
                coordinates = sorted(candidate_coordinates)
                candidate_matrix = canonical_matrix(np.asarray(coordinates, dtype=np.int64))
                key = shape_key_from_matrix(candidate_matrix)
                candidates.setdefault(
                    key,
                    ShapePerturbation(f"add:{added[0]},{added[1]}", key, candidate_matrix),
                )

    for removed in sorted(live_coordinates):
        if len(live_coordinates) == 1:
            continue
        candidate_coordinates = live_coordinates - {removed}
        # Removing a bridge must produce two organisms in the real engine,
        # not a disconnected union treated as one hypothetical species.
        if not _is_single_8_connected(candidate_coordinates):
            continue
        coordinates = sorted(candidate_coordinates)
        candidate_matrix = canonical_matrix(np.asarray(coordinates, dtype=np.int64))
        key = shape_key_from_matrix(candidate_matrix)
        candidates.setdefault(
            key,
            ShapePerturbation(
                f"remove:{removed[0]},{removed[1]}",
                key,
                candidate_matrix,
            ),
        )
    return sorted(candidates.values(), key=lambda item: (shape_key_sort_key(item.key), item.label))


def run_sensitivity(
    shape_a: ShapeKey | Any,
    shape_b: ShapeKey | Any,
    *,
    alphas: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
    identity_dim: int = 32,
    beta: float = 1.0,
    base_rule: str = "B3/S23",
    max_rule_changes: int = 2,
    interaction_threshold: float = 0.35,
) -> list[dict[str, Any]]:
    """Measure ``||v(A',B)-v(A,B)||`` for one-cell morphology changes."""

    key_a, matrix_a = _canonical_input(shape_a)
    key_b, _ = _canonical_input(shape_b)
    alpha_values = [float(alpha) for alpha in alphas]
    if any(not 0.0 <= alpha <= 1.0 for alpha in alpha_values):
        raise ValueError("alphas must be between 0 and 1")
    universe = make_interaction_universe(seed, identity_dim=identity_dim, beta=beta)
    vector_a = universe.encoder.encode(key_a)
    vector_b = universe.encoder.encode(key_b)
    perturbations = single_cell_perturbations(matrix_a)
    rows: list[dict[str, Any]] = []
    for perturbation in perturbations:
        vector_perturbed = universe.encoder.encode(perturbation.key)
        morphology_distance = float(np.linalg.norm(vector_perturbed - vector_a))
        for alpha in alpha_values:
            base_interaction = make_pair_interaction(
                key_a,
                key_b,
                vector_a,
                vector_b,
                universe=universe,
                alpha=alpha,
                base_rule=base_rule,
                max_rule_changes=max_rule_changes,
                interaction_threshold=interaction_threshold,
            )
            perturbed_interaction = make_pair_interaction(
                perturbation.key,
                key_b,
                vector_perturbed,
                vector_b,
                universe=universe,
                alpha=alpha,
                base_rule=base_rule,
                max_rule_changes=max_rule_changes,
                interaction_threshold=interaction_threshold,
            )
            interaction_distance = float(
                np.linalg.norm(perturbed_interaction.final_vector - base_interaction.final_vector)
            )
            rows.append(
                {
                    "alpha": alpha,
                    "perturbation": perturbation.label,
                    "morphology_distance": morphology_distance,
                    "interaction_distance": interaction_distance,
                    "sensitivity": interaction_distance,
                    "base_rule": base_interaction.local_rule_text,
                    "perturbed_rule": perturbed_interaction.local_rule_text,
                }
            )
    return rows


def _shape_text(text: str) -> np.ndarray:
    return np.asarray(pattern_from_text(text.replace("/", "\n")), dtype=np.uint8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-a", default="##/##")
    parser.add_argument("--shape-b", default=".#./###")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-rule-changes", type=int, default=2)
    parser.add_argument("--interaction-threshold", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = run_sensitivity(
        _shape_text(args.shape_a),
        _shape_text(args.shape_b),
        alphas=parse_alphas(args.alphas),
        seed=args.seed,
        identity_dim=args.identity_dim,
        beta=args.beta,
        max_rule_changes=args.max_rule_changes,
        interaction_threshold=args.interaction_threshold,
    )
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("artifacts/experiments/morphology_interactions") / f"sensitivity_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0].keys()) if rows else ()
    with (output_dir / "sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(rows)} sensitivity rows to {output_dir / 'sensitivity.csv'}")


if __name__ == "__main__":
    main()
