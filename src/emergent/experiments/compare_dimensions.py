"""Run explicitly configured, raw 2D/3D comparison experiments."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np

from ..core.random import generate_random_grids, random_rules
from ..core.rules import format_rule, parse_rule
from ..core.simulate import run_steps_with_metrics
from ..core3d.random import generate_random_grids_3d, random_rules_3d
from ..core3d.rules import format_rule_3d, parse_rule_3d
from ..core3d.simulate import run_steps_batch_with_metrics_3d


def _output_directory(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        target = Path(output_dir)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = Path("artifacts/experiments") / f"compare_dimensions_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def run_comparison(
    *,
    rules: int = 1,
    initial_conditions: int = 20,
    size_2d: int = 64,
    size_3d: int = 32,
    steps: int = 100,
    density: float = 0.10,
    seed: int = 42,
    rule_2d: str = "B3/S23",
    rule_3d: str = "B6/S5,6,7",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run raw statistics for independently configured 2D and 3D systems.

    When ``rules`` is one, the explicit rules are used. For larger samples,
    independent random rule lists are generated for each dimensionality; no
    correspondence between those rule lists is implied.
    """

    if rules < 1 or initial_conditions < 1 or size_2d < 1 or size_3d < 1 or steps < 0:
        raise ValueError(
            "rules, initial_conditions, and grid sizes must be positive; steps cannot be negative"
        )
    root_key = jax.random.key(seed)
    key_2d, key_3d, init_2d, init_3d = jax.random.split(root_key, 4)
    rules_2d = [parse_rule(rule_2d)] if rules == 1 else random_rules(key_2d, rules)
    rules_3d = [parse_rule_3d(rule_3d)] if rules == 1 else random_rules_3d(key_3d, rules)
    grids_2d = generate_random_grids(
        jax.random.split(init_2d, initial_conditions), size_2d, size_2d, density
    )
    grids_3d = generate_random_grids_3d(
        jax.random.split(init_3d, initial_conditions), size_3d, size_3d, size_3d, density
    )
    rows: list[dict[str, Any]] = []
    total_2d = float(size_2d**2)
    total_3d = float(size_3d**3)

    for index, rule in enumerate(rules_2d):
        # Run each 2D condition through the existing single-grid metrics API;
        # this keeps the comparison code explicit while 3D uses one batch.
        final_values = []
        metric_values = []
        for condition in grids_2d:
            final_grid, condition_metrics = run_steps_with_metrics(condition, rule, steps)
            final_values.append(np.asarray(jax.device_get(final_grid), dtype=np.uint8))
            metric_values.append(np.asarray(jax.device_get(condition_metrics), dtype=np.float64))
        for condition_index, (final_grid, condition_metrics) in enumerate(
            zip(final_values, metric_values)
        ):
            rows.append(
                {
                    "dimensions": 2,
                    "rule": format_rule(rule),
                    "rule_index": index,
                    "initial_condition": condition_index,
                    "grid_shape": f"{size_2d}x{size_2d}",
                    "steps": steps,
                    "initial_alive_fraction": float(np.asarray(grids_2d[condition_index]).mean()),
                    "final_alive_fraction": float(final_grid.mean()),
                    "mean_changed_fraction": float((condition_metrics[:, 1] / total_2d).mean())
                    if steps
                    else 0.0,
                }
            )

    for index, rule in enumerate(rules_3d):
        final_3d, metrics_3d = run_steps_batch_with_metrics_3d(grids_3d, rule, steps)
        final_3d_values = np.asarray(jax.device_get(final_3d), dtype=np.uint8)
        metric_3d_values = np.asarray(jax.device_get(metrics_3d), dtype=np.float64)
        for condition_index in range(initial_conditions):
            rows.append(
                {
                    "dimensions": 3,
                    "rule": format_rule_3d(rule),
                    "rule_index": index,
                    "initial_condition": condition_index,
                    "grid_shape": f"{size_3d}x{size_3d}x{size_3d}",
                    "steps": steps,
                    "initial_alive_fraction": float(np.asarray(grids_3d[condition_index]).mean()),
                    "final_alive_fraction": float(final_3d_values[condition_index].mean()),
                    "mean_changed_fraction": float(
                        (metric_3d_values[condition_index, :, 1] / total_3d).mean()
                    )
                    if steps
                    else 0.0,
                }
            )

    output = _output_directory(output_dir)
    manifest = {
        "dimensions": [2, 3],
        "grid_size_2d": [size_2d, size_2d],
        "grid_size_3d": [size_3d, size_3d, size_3d],
        "rules": rules,
        "initial_conditions_per_dimension": initial_conditions,
        "steps": steps,
        "density": density,
        "seed": seed,
        "rule_2d": rule_2d,
        "rule_3d": rule_3d,
        "device": str(jax.devices()[0]),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return {"output_dir": str(output), "manifest": manifest, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=int, default=1)
    parser.add_argument("--initial-conditions", type=int, default=20)
    parser.add_argument("--size-2d", type=int, default=64)
    parser.add_argument("--size-3d", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rule-2d", default="B3/S23")
    parser.add_argument("--rule-3d", default="B6/S5,6,7")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_comparison(
        rules=args.rules,
        initial_conditions=args.initial_conditions,
        size_2d=args.size_2d,
        size_3d=args.size_3d,
        steps=args.steps,
        density=args.density,
        seed=args.seed,
        rule_2d=args.rule_2d,
        rule_3d=args.rule_3d,
        output_dir=args.output_dir,
    )
    print(f"Wrote {len(result['rows'])} rows to {result['output_dir']}")


if __name__ == "__main__":
    main()
