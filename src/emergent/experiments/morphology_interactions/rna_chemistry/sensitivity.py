"""Discrete morphology-to-chemistry derivative measurements."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ....io.patterns import pattern_from_text
from ..canonical import (
    ShapeKey,
    canonical_matrix_from_grid,
    matrix_from_shape_key,
    shape_key_from_matrix,
)
from ..sensitivity import single_cell_perturbations
from .chemistry import PairChemistry, evaluate_pair_chemistry, make_chemistry_universe
from .sequence import sequence_from_shape_key


def _canonical_input(shape: ShapeKey | Any) -> ShapeKey:
    if isinstance(shape, ShapeKey):
        return shape
    return shape_key_from_matrix(canonical_matrix_from_grid(shape))


def _levenshtein(first: np.ndarray, second: np.ndarray) -> int:
    """Return deterministic edit distance for variable-length sequences."""

    previous = list(range(second.size + 1))
    for first_index, first_value in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_value in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + int(first_value != second_value),
                )
            )
        previous = current
    return previous[-1]


def _aggregate_interaction(chemistry: PairChemistry) -> np.ndarray:
    """Compose all site vectors into one comparable pair-level observable."""

    if not chemistry.sites:
        return np.zeros(18, dtype=np.float32)
    return np.sum(
        np.asarray([site.final_vector for site in chemistry.sites], dtype=np.float32),
        axis=0,
    )


def run_sensitivity(
    shape_a: ShapeKey | Any,
    shape_b: ShapeKey | Any,
    *,
    alphas: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
    binding_seed_length: int = 4,
    minimum_binding_length: int = 4,
    allow_gu_wobble: bool = True,
    max_mismatches: int = 1,
    binding_energy_threshold: float = -5.0,
    motif_length: int = 4,
    calibration_size: int = 512,
) -> list[dict[str, int | float]]:
    """Measure ``morphology -> sequence -> interaction`` changes.

    The parent morphology experiment supplies only valid one-cell edits.  A
    bridge removal is excluded when it would create two components, matching
    the engine's species definition instead of inventing a disconnected
    chemistry target.
    """

    key_a = _canonical_input(shape_a)
    key_b = _canonical_input(shape_b)
    alpha_values = [float(alpha) for alpha in alphas]
    if not alpha_values or any(not 0.0 <= alpha <= 1.0 for alpha in alpha_values):
        raise ValueError("alphas must contain values between 0 and 1")
    sequence_a = sequence_from_shape_key(key_a)
    sequence_b = sequence_from_shape_key(key_b)
    universe = make_chemistry_universe(seed, calibration_size=calibration_size)
    perturbations = single_cell_perturbations(key_a)
    base_by_alpha = {
        alpha: evaluate_pair_chemistry(
            sequence_a,
            sequence_b,
            universe=universe,
            alpha=alpha,
            seed_length=binding_seed_length,
            minimum_length=minimum_binding_length,
            allow_gu_wobble=allow_gu_wobble,
            max_mismatches=max_mismatches,
            binding_energy_threshold=binding_energy_threshold,
            motif_length=motif_length,
        )
        for alpha in alpha_values
    }
    rows: list[dict[str, int | float]] = []
    for perturbation in perturbations:
        sequence_perturbed = sequence_from_shape_key(perturbation.key)
        sequence_edit_distance = _levenshtein(sequence_a.bases, sequence_perturbed.bases)
        for alpha in alpha_values:
            base = base_by_alpha[alpha]
            perturbed = evaluate_pair_chemistry(
                sequence_perturbed,
                sequence_b,
                universe=universe,
                alpha=alpha,
                seed_length=binding_seed_length,
                minimum_length=minimum_binding_length,
                allow_gu_wobble=allow_gu_wobble,
                max_mismatches=max_mismatches,
                binding_energy_threshold=binding_energy_threshold,
                motif_length=motif_length,
            )
            base_vector = _aggregate_interaction(base)
            perturbed_vector = _aggregate_interaction(perturbed)
            rows.append(
                {
                    "alpha": alpha,
                    "perturbation": perturbation.label,
                    "morphology_cells_a": int(np.count_nonzero(matrix_from_shape_key(key_a))),
                    "morphology_cells_perturbed": int(
                        np.count_nonzero(matrix_from_shape_key(perturbation.key))
                    ),
                    "sequence_length_a": sequence_a.length,
                    "sequence_length_perturbed": sequence_perturbed.length,
                    "sequence_edit_distance": sequence_edit_distance,
                    "base_binding_sites": base.successful_binding_sites,
                    "perturbed_binding_sites": perturbed.successful_binding_sites,
                    "interaction_distance": float(np.linalg.norm(perturbed_vector - base_vector)),
                    "interaction_sensitivity": float(
                        np.linalg.norm(perturbed_vector - base_vector)
                    ),
                }
            )
    return rows


def _shape_text(text: str) -> np.ndarray:
    return np.asarray(pattern_from_text(text.replace("/", "\n")), dtype=np.uint8)


def build_parser() -> argparse.ArgumentParser:
    """Build the one-cell RNA chemistry sensitivity CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-a", default="##/##")
    parser.add_argument("--shape-b", default=".#./###")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run sensitivity measurements and write a compact CSV."""

    args = build_parser().parse_args(argv)
    rows = run_sensitivity(
        _shape_text(args.shape_a),
        _shape_text(args.shape_b),
        alphas=[float(value) for value in args.alphas.split(",") if value.strip()],
        seed=args.seed,
        calibration_size=args.calibration_size,
    )
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            Path("artifacts/experiments/morphology_interactions/rna_chemistry")
            / f"sensitivity_{stamp}"
        )
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
