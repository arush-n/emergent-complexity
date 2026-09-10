"""Scaling and interaction-entropy audits for RNA-inspired chemistry."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np

from ..canonical import ShapeKey, canonicalize_grid, matrix_from_shape_key
from ..encoding import deterministic_uint64
from ..interaction import interaction_vector_to_rule
from .chemistry import evaluate_pair_chemistry, make_chemistry_universe
from .sequence import sequence_from_shape_key


def _frontier(coordinates: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    for row, col in coordinates:
        for row_delta in (-1, 0, 1):
            for col_delta in (-1, 0, 1):
                if row_delta == 0 and col_delta == 0:
                    continue
                candidate = (row + row_delta, col + col_delta)
                if candidate not in coordinates:
                    candidates.add(candidate)
    return sorted(candidates)


def deterministic_connected_shape(seed: int, cell_count: int) -> ShapeKey:
    """Generate one connected morphology without process-randomized state."""

    if not isinstance(cell_count, int) or cell_count < 1:
        raise ValueError("cell_count must be positive")
    coordinates = {(0, 0)}
    for index in range(1, cell_count):
        candidates = _frontier(coordinates)
        choice = deterministic_uint64(seed, "rna-audit-growth", index) % len(candidates)
        coordinates.add(candidates[int(choice)])
    rows = [value[0] for value in coordinates]
    cols = [value[1] for value in coordinates]
    matrix = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=np.uint8)
    for row, col in coordinates:
        matrix[row - min(rows), col - min(cols)] = 1
    return canonicalize_grid(matrix)


def generate_audit_shapes(
    seed: int,
    *,
    count: int = 512,
    minimum_cells: int = 1,
    maximum_cells: int = 128,
) -> list[ShapeKey]:
    """Generate a deterministic diverse morphology corpus."""

    if count < 1 or minimum_cells < 1 or maximum_cells < minimum_cells:
        raise ValueError("invalid audit shape bounds")
    keys: dict[ShapeKey, None] = {}
    index = 0
    while len(keys) < count:
        span = maximum_cells - minimum_cells + 1
        cell_count = minimum_cells + int(deterministic_uint64(seed, "rna-audit-size", index) % span)
        key = deterministic_connected_shape(
            deterministic_uint64(seed, "rna-audit-shape-seed", index),
            cell_count,
        )
        keys.setdefault(key, None)
        index += 1
        if index > count * 100:
            break
    return list(keys)


def _correlation(first: Iterable[float], second: Iterable[float]) -> float:
    left = np.asarray(list(first), dtype=np.float64)
    right = np.asarray(list(second), dtype=np.float64)
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


@dataclass
class AuditResult:
    """Rows and aggregate results from one deterministic chemistry audit."""

    rows: list[dict[str, int | float]]
    summary: dict[str, Any]


def audit_interaction_space(
    *,
    seed: int = 42,
    pair_count: int = 10_000,
    shape_count: int = 512,
    minimum_cells: int = 1,
    maximum_cells: int = 128,
    sequence_mode: str = "local_surface",
    seed_length: int = 4,
    minimum_binding_length: int = 4,
    allow_gu_wobble: bool = True,
    binding_energy_threshold: float = -5.0,
    motif_length: int = 4,
    alpha: float = 0.0,
    site_max_rule_changes: int = 1,
    interaction_threshold: float = 0.35,
    effect_padding: int = 1,
    calibration_size: int = 512,
) -> AuditResult:
    """Measure 10,000 morphology-pair opportunities without a global pair sweep."""

    if not isinstance(pair_count, int) or pair_count < 1:
        raise ValueError("pair_count must be positive")
    if not isinstance(effect_padding, int) or effect_padding < 0:
        raise ValueError("effect_padding must be a non-negative integer")
    shapes = generate_audit_shapes(
        seed,
        count=shape_count,
        minimum_cells=minimum_cells,
        maximum_cells=maximum_cells,
    )
    sequences = {key: sequence_from_shape_key(key, mode=sequence_mode) for key in shapes}
    universe = make_chemistry_universe(
        seed,
        calibration_threshold=interaction_threshold,
        calibration_size=calibration_size,
    )
    rows: list[dict[str, int | float]] = []
    seen_sequences: set[tuple[int, bytes]] = set()
    seen_motifs: set[tuple[int, int, int]] = set()
    seen_vectors: set[bytes] = set()
    seen_rules: set[int] = set()
    candidate_counts: list[int] = []
    expected_counts: list[float] = []
    length_products: list[float] = []
    for pair_index in range(pair_count):
        first_index = deterministic_uint64(seed, "rna-audit-pair-a", pair_index) % len(shapes)
        second_index = deterministic_uint64(seed, "rna-audit-pair-b", pair_index) % len(shapes)
        first = sequences[shapes[int(first_index)]]
        second = sequences[shapes[int(second_index)]]
        chemistry = evaluate_pair_chemistry(
            first,
            second,
            universe=universe,
            alpha=alpha,
            seed_length=seed_length,
            minimum_length=minimum_binding_length,
            allow_gu_wobble=allow_gu_wobble,
            binding_energy_threshold=binding_energy_threshold,
            motif_length=motif_length,
        )
        expected = float(
            max(0, first.length - seed_length + 1)
            * max(0, second.length - seed_length + 1)
            / (4**seed_length)
        )
        candidate_counts.append(chemistry.candidate_seed_count)
        expected_counts.append(expected)
        length_products.append(float(first.length * second.length))
        for sequence in (first, second):
            seen_sequences.add((sequence.length, sequence.bases.tobytes()))
        site_rules: set[int] = set()
        for site in chemistry.sites:
            seen_motifs.add((site.motif.motif_length, *site.motif.symmetric_codes))
            seen_vectors.add(site.final_vector.tobytes())
            rule, rule_id = interaction_vector_to_rule(
                site.final_vector,
                "B3/S23",
                max_rule_changes=site_max_rule_changes,
                interaction_threshold=interaction_threshold,
            )
            del rule
            site_rules.add(rule_id)
            seen_rules.add(rule_id)
        potential_effect_area = chemistry.successful_binding_sites * (1 + 2 * effect_padding) ** 2
        rows.append(
            {
                "pair_index": pair_index,
                "cells_a": int(np.count_nonzero(matrix_from_shape_key(first.shape_key))),
                "cells_b": int(np.count_nonzero(matrix_from_shape_key(second.shape_key))),
                "sequence_length_a": first.length,
                "sequence_length_b": second.length,
                "length_product": first.length * second.length,
                "expected_seed_count": expected,
                "candidate_seed_count": chemistry.candidate_seed_count,
                "raw_site_count": chemistry.raw_site_count,
                "successful_binding_sites": chemistry.successful_binding_sites,
                "unique_site_rules": len(site_rules),
                "potential_site_effect_area": potential_effect_area,
            }
        )
    summary = {
        "experiment": "rna_chemistry_interaction_audit",
        "seed": seed,
        "pair_count": pair_count,
        "shape_count": len(shapes),
        "sequence_mode": sequence_mode,
        "seed_length": seed_length,
        "alpha": float(alpha),
        "effect_padding": effect_padding,
        "unique_shape_keys": len(shapes),
        "unique_chemical_sequences": len(seen_sequences),
        "unique_motif_pairs": len(seen_motifs),
        "unique_reaction_vectors": len(seen_vectors),
        "unique_effective_rules": len(seen_rules),
        "candidate_seed_expected_correlation": _correlation(expected_counts, candidate_counts),
        "candidate_seed_length_product_correlation": _correlation(
            length_products, candidate_counts
        ),
        "mean_candidate_seed_count": float(np.mean(candidate_counts)),
        "mean_expected_seed_count": float(np.mean(expected_counts)),
        "mean_successful_binding_sites": float(
            np.mean([row["successful_binding_sites"] for row in rows])
        ),
        "chemistry_calibration": universe.calibration_report(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    return AuditResult(rows, summary)


def write_audit(result: AuditResult, output_dir: str | Path) -> Path:
    """Write the audit summary and compact per-pair CSV."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = tuple(result.rows[0].keys()) if result.rows else ()
    with (target / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(result.rows)
    return target


def backend_report() -> dict[str, Any]:
    """Return the active JAX backend and device inventory."""

    return {
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "platforms": sorted({device.platform for device in jax.devices()}),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone 10,000-pair audit CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=10_000)
    parser.add_argument("--shapes", type=int, default=512)
    parser.add_argument("--min-cells", type=int, default=1)
    parser.add_argument("--max-cells", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-length", type=int, default=4)
    parser.add_argument(
        "--sequence-mode", choices=("local_surface", "exact_shape"), default="local_surface"
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--effect-padding", type=int, default=1)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the interaction-space audit and print its summary."""

    args = build_parser().parse_args(argv)
    result = audit_interaction_space(
        seed=args.seed,
        pair_count=args.pairs,
        shape_count=args.shapes,
        minimum_cells=args.min_cells,
        maximum_cells=args.max_cells,
        sequence_mode=args.sequence_mode,
        seed_length=args.seed_length,
        alpha=args.alpha,
        effect_padding=args.effect_padding,
        calibration_size=args.calibration_size,
    )
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            Path("artifacts/experiments/morphology_interactions/rna_chemistry") / f"audit_{stamp}"
        )
    write_audit(result, output_dir)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "summary": result.summary}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
