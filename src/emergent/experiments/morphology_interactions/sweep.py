"""Run reproducible alpha/seed sweeps over the interaction landscape."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import MorphologyExperimentConfig
from .engine import initial_grid_from_config, run_experiment
from .persistence import ARTIFACT_ROOT


def parse_alphas(text: str) -> list[float]:
    """Parse comma-separated alpha values and validate their order."""

    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("alphas must be comma-separated numbers") from exc
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("alphas must contain at least one value between 0 and 1")
    return values


def parse_seeds(text: str) -> list[int]:
    """Parse ``0:20`` as ``range(0, 20)`` or a comma-separated seed list."""

    text = text.strip()
    if ":" in text:
        pieces = text.split(":")
        if len(pieces) not in (2, 3):
            raise ValueError("seed ranges must have the form start:stop[:step]")
        try:
            values = range(*(int(piece) for piece in pieces))
        except ValueError as exc:
            raise ValueError("seed ranges must contain integers") from exc
        result = list(values)
    else:
        try:
            result = [int(item.strip()) for item in text.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError("seeds must be integers") from exc
    if not result or any(value < 0 for value in result):
        raise ValueError("seeds must contain at least one non-negative integer")
    return result


def run_alpha_sweep(
    *,
    alphas: list[float],
    seeds: list[int],
    width: int = 128,
    height: int = 128,
    density: float = 0.10,
    base_rule: str = "B3/S23",
    steps: int = 5_000,
    warmup_steps: int = 100,
    interaction_radius: int = 2,
    effect_padding: int = 1,
    max_rule_changes: int = 2,
    interaction_threshold: float = 0.35,
    identity_dim: int = 32,
    output_dir: str | Path | None = None,
    include_control: bool = False,
) -> list[dict[str, Any]]:
    """Run every alpha/seed using the exact same initial grid per seed."""

    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        root = ARTIFACT_ROOT / f"sweep_{stamp}"
    else:
        root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    conditions = [(False, "control") for _ in range(1)] if include_control else []
    conditions.extend((True, f"alpha_{alpha:g}") for alpha in alphas)

    for seed in seeds:
        base = MorphologyExperimentConfig(
            width=width,
            height=height,
            seed=seed,
            density=density,
            base_rule=base_rule,
            steps=steps,
            warmup_steps=warmup_steps,
            identity_dim=identity_dim,
            alpha=0.0,
            interaction_radius=interaction_radius,
            effect_padding=effect_padding,
            max_rule_changes=max_rule_changes,
            interaction_threshold=interaction_threshold,
        )
        initial = initial_grid_from_config(base)
        for enabled, label in conditions:
            alpha = 0.0 if not enabled else float(label.removeprefix("alpha_"))
            config = MorphologyExperimentConfig(
                **{
                    **base.as_dict(),
                    "alpha": alpha,
                    "interactions_enabled": enabled,
                }
            )
            run_dir = root / f"seed_{seed}" / label
            result = run_experiment(config, initial_grid=initial, output_dir=run_dir)
            rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "interactions_enabled": enabled,
                    "final_alive_cells": result.summary["final_alive_cells"],
                    "unique_species_total": result.summary["unique_species_total"],
                    "unique_interaction_pairs_total": result.summary[
                        "unique_interaction_pairs_total"
                    ],
                    "unique_local_rules_total": result.summary["unique_local_rules_total"],
                    "output_dir": str(run_dir),
                }
            )

    fields = tuple(rows[0].keys()) if rows else ()
    with (root / "alpha_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seeds", default="0:20")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--base-rule", default="B3/S23")
    parser.add_argument("--identity-dim", type=int, default=32)
    parser.add_argument("--interaction-radius", type=int, default=2)
    parser.add_argument("--effect-padding", type=int, default=1)
    parser.add_argument("--max-rule-changes", type=int, default=2)
    parser.add_argument("--interaction-threshold", type=float, default=0.35)
    parser.add_argument("--include-control", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = run_alpha_sweep(
        alphas=parse_alphas(args.alphas),
        seeds=parse_seeds(args.seeds),
        width=args.size,
        height=args.size,
        density=args.density,
        base_rule=args.base_rule,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        identity_dim=args.identity_dim,
        interaction_radius=args.interaction_radius,
        effect_padding=args.effect_padding,
        max_rule_changes=args.max_rule_changes,
        interaction_threshold=args.interaction_threshold,
        output_dir=args.output_dir,
        include_control=args.include_control,
    )
    print(f"completed {len(rows)} runs")


if __name__ == "__main__":
    main()
