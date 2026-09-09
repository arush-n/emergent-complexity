"""Reproducible random-rule sweeps for 3D cellular automata."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np

from ..core3d.random import generate_random_grids_3d, random_rules_3d
from ..core3d.rules import format_rule_3d, rule_to_int_3d
from ..core3d.simulate import run_steps_batch_with_metrics_3d

RESULT_FIELDS = (
    "dimensions",
    "rule",
    "rule_id",
    "initial_condition",
    "steps_executed",
    "initial_alive_fraction",
    "final_alive_fraction",
    "final_changed_fraction",
    "mean_changed_fraction",
    "mean_birth_fraction",
    "mean_death_fraction",
    "fixed_point",
)


def _output_directory(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        target = Path(output_dir)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = Path("artifacts/experiments") / f"random_3d_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def run_random_3d_experiment(
    *,
    rules: int = 100,
    initial_conditions: int = 20,
    size: int = 64,
    steps: int = 500,
    density: float = 0.10,
    seed: int = 42,
    output_dir: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run a batched 3D random-rule sweep and write CSV plus manifest files.

    Each rule is evaluated on one JAX batch containing all requested initial
    conditions. Only final grids and compact transition metrics are retained;
    voxel trajectories are deliberately not stored.
    """

    if rules < 1 or initial_conditions < 1 or size < 1 or steps < 0:
        raise ValueError(
            "rules, initial_conditions, and size must be positive; steps cannot be negative"
        )
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between 0 and 1")

    output = _output_directory(output_dir)
    root_key = jax.random.key(seed)
    rule_key, grid_key = jax.random.split(root_key)
    sampled_rules = random_rules_3d(rule_key, rules)
    grid_keys = jax.random.split(grid_key, initial_conditions)
    initial_grids = generate_random_grids_3d(
        grid_keys,
        size,
        size,
        size,
        density,
    )
    initial_fractions = np.asarray(jax.device_get(initial_grids), dtype=np.uint8).mean(
        axis=(1, 2, 3)
    )
    rows: list[dict[str, Any]] = []

    for rule_index, rule in enumerate(sampled_rules):
        final_grids, metrics = run_steps_batch_with_metrics_3d(initial_grids, rule, steps)
        final_values = np.asarray(jax.device_get(final_grids), dtype=np.uint8)
        metric_values = np.asarray(jax.device_get(metrics), dtype=np.float64)
        total_cells = float(size**3)
        final_fractions = final_values.mean(axis=(1, 2, 3))
        if steps:
            changed_fraction = metric_values[:, :, 1] / total_cells
            birth_fraction = metric_values[:, :, 2] / total_cells
            death_fraction = metric_values[:, :, 3] / total_cells
            final_changed = changed_fraction[:, -1]
        else:
            changed_fraction = np.zeros((initial_conditions, 0))
            birth_fraction = changed_fraction.copy()
            death_fraction = changed_fraction.copy()
            final_changed = np.zeros(initial_conditions)

        for condition_index in range(initial_conditions):
            rows.append(
                {
                    "dimensions": 3,
                    "rule": format_rule_3d(rule),
                    "rule_id": rule_to_int_3d(rule),
                    "initial_condition": condition_index,
                    "steps_executed": steps,
                    "initial_alive_fraction": float(initial_fractions[condition_index]),
                    "final_alive_fraction": float(final_fractions[condition_index]),
                    "final_changed_fraction": float(final_changed[condition_index]),
                    "mean_changed_fraction": float(changed_fraction[condition_index].mean())
                    if steps
                    else 0.0,
                    "mean_birth_fraction": float(birth_fraction[condition_index].mean())
                    if steps
                    else 0.0,
                    "mean_death_fraction": float(death_fraction[condition_index].mean())
                    if steps
                    else 0.0,
                    "fixed_point": bool(final_changed[condition_index] == 0),
                }
            )
        if progress is not None:
            progress(rule_index + 1, rules)

    timestamp = datetime.now(UTC).isoformat()
    manifest = {
        "dimensions": 3,
        "grid_size": [size, size, size],
        "rules": rules,
        "initial_conditions_per_rule": initial_conditions,
        "steps": steps,
        "density": density,
        "seed": seed,
        "device": str(jax.devices()[0]),
        "dtype": "uint8",
        "timestamp": timestamp,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"output_dir": str(output), "manifest": manifest, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=int, default=100)
    parser.add_argument("--initial-conditions", type=int, default=20)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    def print_progress(done: int, total: int) -> None:
        print(f"Rules completed: {done} / {total}")

    result = run_random_3d_experiment(
        rules=args.rules,
        initial_conditions=args.initial_conditions,
        size=args.size,
        steps=args.steps,
        density=args.density,
        seed=args.seed,
        output_dir=args.output_dir,
        progress=print_progress,
    )
    print(f"Wrote {len(result['rows'])} rows to {result['output_dir']}")


if __name__ == "__main__":
    main()
