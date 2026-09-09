"""Small raw random-rule sweeps for the 2D laboratory mode."""

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
from ..core.rules import format_rule
from ..core.simulate import run_steps_with_metrics


def run_random_2d_experiment(
    *,
    rules: int = 5,
    initial_conditions: int = 4,
    size: int = 24,
    steps: int = 50,
    density: float = 0.10,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a compact 2D sweep with the same raw row shape as the 3D runner."""

    if rules < 1 or initial_conditions < 1 or size < 1 or steps < 0:
        raise ValueError(
            "rules, initial_conditions, and size must be positive; steps cannot be negative"
        )
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between 0 and 1")
    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path("artifacts/experiments") / f"random_2d_{stamp}"
    else:
        output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root_key = jax.random.key(seed)
    rule_key, grid_key = jax.random.split(root_key)
    sampled_rules = random_rules(rule_key, rules)
    grids = generate_random_grids(
        jax.random.split(grid_key, initial_conditions), size, size, density
    )
    initial_values = np.asarray(jax.device_get(grids), dtype=np.uint8)
    total_cells = float(size**2)
    rows: list[dict[str, Any]] = []
    for rule_index, rule in enumerate(sampled_rules):
        for condition_index, grid in enumerate(grids):
            final, metrics = run_steps_with_metrics(grid, rule, steps)
            final_values = np.asarray(jax.device_get(final), dtype=np.uint8)
            metric_values = np.asarray(jax.device_get(metrics), dtype=np.float64)
            changed = metric_values[:, 1] / total_cells if steps else np.zeros(0)
            births = metric_values[:, 2] / total_cells if steps else np.zeros(0)
            deaths = metric_values[:, 3] / total_cells if steps else np.zeros(0)
            rows.append(
                {
                    "dimensions": 2,
                    "rule": format_rule(rule),
                    "rule_id": rule_index,
                    "initial_condition": condition_index,
                    "steps_executed": steps,
                    "initial_alive_fraction": float(initial_values[condition_index].mean()),
                    "final_alive_fraction": float(final_values.mean()),
                    "final_changed_fraction": float(changed[-1]) if steps else 0.0,
                    "mean_changed_fraction": float(changed.mean()) if steps else 0.0,
                    "mean_birth_fraction": float(births.mean()) if steps else 0.0,
                    "mean_death_fraction": float(deaths.mean()) if steps else 0.0,
                    "fixed_point": bool(not steps or changed[-1] == 0),
                }
            )
    manifest = {
        "dimensions": 2,
        "grid_size": [size, size],
        "rules": rules,
        "initial_conditions_per_rule": initial_conditions,
        "steps": steps,
        "density": density,
        "seed": seed,
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
    parser.add_argument("--rules", type=int, default=5)
    parser.add_argument("--initial-conditions", type=int, default=4)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_random_2d_experiment(
        rules=args.rules,
        initial_conditions=args.initial_conditions,
        size=args.size,
        steps=args.steps,
        density=args.density,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"Wrote {len(result['rows'])} rows to {result['output_dir']}")


if __name__ == "__main__":
    main()
