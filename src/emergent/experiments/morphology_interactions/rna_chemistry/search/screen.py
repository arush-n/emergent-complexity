"""Batch isolation controls for archived growth leads, with exact cycle pruning.

These are finite diagnostic assays. Reaching the assay horizon is inconclusive,
and a positive result is a reproduction lead requiring daughter-transfer tests.
The continuous discovery runner retains its unlimited world lifetime.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ...canonical import ShapeKey, matrix_from_shape_key
from ..chemistry import make_chemistry_universe
from ..config import RNAExperimentConfig
from ..engine import RNAChemistryEngine
from .run import shape_counts_and_components_batch, write_json
from .runtime import ExactCycle, prepare_batch, state_bytes, step_fields


def collect_leads(root: Path, *, minimum_cells: int = 6) -> list[dict]:
    """Deduplicate exact morphology leads while retaining their source trace."""

    unique: dict[ShapeKey, dict] = {}
    for path in sorted(root.glob("worker_[0-9][0-9][0-9]/events.jsonl")):
        with path.open() as handle:
            for line in handle:
                if not line.endswith("\n"):
                    break
                event = json.loads(line)
                if event.get("event") != "copy_growth_lead":
                    continue
                shape = event["shape"]
                if not isinstance(shape, dict):
                    continue
                key = ShapeKey(shape["height"], shape["width"], bytes.fromhex(shape["packed"]))
                cells = int(matrix_from_shape_key(key).sum())
                if cells < minimum_cells:
                    continue
                record = {
                    "key": key,
                    "cells": cells,
                    "copies": event["copies"],
                    "source_worker": path.parent.name,
                    "source_trial": event["trial"],
                    "source_trace_tick": event["trace_tick"],
                    "source_config_key": event.get("config_key"),
                }
                previous = unique.get(key)
                if previous is None or record["copies"] > previous["copies"]:
                    unique[key] = record
    return sorted(
        unique.values(),
        key=lambda record: (
            -record["cells"], -record["copies"],
            record["key"].height, record["key"].width, record["key"].packed,
        ),
    )


def screen_batch(
    keys: list[ShapeKey], config: RNAExperimentConfig, *, steps: int = 128,
) -> list[dict]:
    """Run matched isolated candidates under native, structured, mixed, scrambled laws.

    Only complete chemistry-state cycles prune assays. A repeating grid can
    later change while bindings expire or the warmup countdown advances.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    if not keys:
        return []
    universe = make_chemistry_universe(
        config.seed, beta=config.beta, calibration_size=config.calibration_size,
        calibration_threshold=config.interaction_threshold,
    )
    engines, grids, rows, cycles, targets = [], [], [], [], []
    for key in keys:
        matrix = matrix_from_shape_key(key)
        if key.height + 4 > config.height or key.width + 4 > config.width:
            raise ValueError("candidate needs at least a two-cell world margin")
        initial = np.zeros((config.height, config.width), np.uint8)
        row, col = (config.height - key.height) // 2, (config.width - key.width) // 2
        initial[row : row + key.height, col : col + key.width] = matrix
        for condition, enabled, alpha in (
            ("native", False, 0.0), ("structured", True, 0.0),
            ("mixed", True, 0.5), ("scrambled", True, 1.0),
        ):
            assay = replace(config, interactions_enabled=enabled, alpha=alpha, warmup_steps=0)
            engine = RNAChemistryEngine(assay, initial_grid=initial, universe=universe)
            engines.append(engine)
            grids.append(initial.copy())
            targets.append(key)
            cycles.append(ExactCycle(state_bytes(engine, initial)))
            rows.append({
                "shape": {"height": key.height, "width": key.width, "packed": key.packed.hex()},
                "condition": condition, "max_copies": 1, "first_double": None,
                "steps_executed": 0, "outcome": "inconclusive_horizon",
                "period": None, "purity_at_max": 1.0,
                "confirmed_replicator": False,
            })
    active = list(range(len(engines)))
    for _ in range(steps):
        if not active:
            break
        batch_engines = [engines[i] for i in active]
        prepared = prepare_batch(batch_engines, np.stack([grids[i] for i in active]))
        following = np.asarray(step_fields(
            np.stack([grids[i] for i in active]), np.stack([p.rules for p in prepared]),
        ))
        counts, _ = shape_counts_and_components_batch(
            following, [e.config for e in batch_engines], batch_engines,
        )
        continuing = []
        for local, index in enumerate(active):
            engine, report, key = engines[index], rows[index], targets[index]
            engine.generation += 1
            grids[index] = following[local]
            report["steps_executed"] = engine.generation
            copies = counts[local].get(key, 0)
            if copies > report["max_copies"]:
                report["max_copies"] = copies
                report["purity_at_max"] = (
                    copies * int(matrix_from_shape_key(key).sum())
                    / max(1, int(following[local].sum()))
                )
            if copies >= 2 and report["first_double"] is None:
                report["first_double"] = engine.generation
            period = cycles[index].observe(state_bytes(engine, following[local]))
            if period is not None:
                report["period"] = period
                report["outcome"] = "periodic" if np.any(following[local]) else "extinct"
            else:
                continuing.append(index)
        active = continuing
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4, help="candidates; four controls each")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument(
        "--steps", type=int, default=128,
        help="diagnostic horizon; unfinished is inconclusive",
    )
    parser.add_argument("--minimum-cells", type=int, default=6)
    args = parser.parse_args()
    if min(args.candidates, args.batch_size, args.steps, args.minimum_cells) < 1:
        parser.error("counts must be positive")
    source = json.loads((args.run_directory / "worker_000/manifest.json").read_text())
    config = replace(RNAExperimentConfig(**source["config"]), width=args.size, height=args.size)
    leads = [
        lead for lead in collect_leads(args.run_directory, minimum_cells=args.minimum_cells)
        if lead["key"].height + 4 <= args.size and lead["key"].width + 4 <= args.size
    ][:args.candidates]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "manifest.json", {
        "source_run": str(args.run_directory.resolve()), "config": config.as_dict(),
        "assay_steps": args.steps, "candidate_count": len(leads),
        "initialization": "one isolated candidate; warmup zero; matched four-condition controls",
        "selection": "largest exact growth leads first; not an unbiased sample",
        "interpretation": "negative isolation does not exclude ecology-dependent reproduction",
    })
    totals = []
    with (args.output_dir / "screen.jsonl").open("w", buffering=1) as handle:
        for start in range(0, len(leads), args.batch_size):
            batch = leads[start : start + args.batch_size]
            results = screen_batch([item["key"] for item in batch], config, steps=args.steps)
            for index, result in enumerate(results):
                origin = {k: v for k, v in batch[index // 4].items() if k != "key"}
                row = origin | result
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                totals.append(row)
    summary = {
        "assays": len(totals), "candidate_count": len(leads),
        "isolated_reproduction_leads": sum(row["first_double"] is not None for row in totals),
        "pruned_exact_cycles": sum(row["period"] is not None for row in totals),
        "inconclusive": sum(row["outcome"] == "inconclusive_horizon" for row in totals),
        "transitions_executed": sum(row["steps_executed"] for row in totals),
        "confirmed_replicators": 0,
        "lineage_verification": "requires independent daughter-transfer assay",
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
