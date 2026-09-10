"""Controlled alpha sweeps for the RNA-inspired chemistry."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .config import RNAExperimentConfig
from .engine import RNAChemistryEngine, initial_grid_from_config


def parse_alphas(value: str) -> list[float]:
    """Parse comma-separated alpha values with deterministic ordering."""

    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("alphas must be comma-separated numbers") from exc
    if not values or any(not 0.0 <= item <= 1.0 for item in values):
        raise ValueError("alphas must contain values between 0 and 1")
    return values


def run_alpha_sweep(
    alphas: Iterable[float],
    config: RNAExperimentConfig,
) -> list[dict[str, int | float]]:
    """Run each alpha with exactly the same initial grid and universe seed."""

    alpha_values = [float(alpha) for alpha in alphas]
    if any(not 0.0 <= alpha <= 1.0 for alpha in alpha_values):
        raise ValueError("alphas must contain values between 0 and 1")
    initial = initial_grid_from_config(config)
    rows: list[dict[str, int | float]] = []
    for alpha in alpha_values:
        run_config = RNAExperimentConfig(**{**config.as_dict(), "alpha": alpha})
        result = RNAChemistryEngine(run_config, initial_grid=initial).run()
        row: dict[str, int | float] = {"alpha": alpha}
        row.update(result.summary)
        rows.append(row)
    return rows


def write_sweep(rows: list[dict[str, int | float]], output_dir: str | Path) -> Path:
    """Write one compact alpha comparison."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0].keys()) if rows else ()
    with (target / "alpha.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    (target / "summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    """Build the alpha sweep CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default="0,0.5,1")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument(
        "--sequence-mode", choices=("local_surface", "exact_shape"), default="local_surface"
    )
    parser.add_argument("--accessibility-mode", choices=("none", "simplified_fold"), default="none")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a controlled alpha sweep."""

    args = build_parser().parse_args(argv)
    config = RNAExperimentConfig(
        width=args.size,
        height=args.size,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        density=args.density,
        sequence_mode=args.sequence_mode,
        accessibility_mode=args.accessibility_mode,
    )
    rows = run_alpha_sweep(parse_alphas(args.alphas), config)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            Path("artifacts/experiments/morphology_interactions/rna_chemistry") / f"sweep_{stamp}"
        )
    write_sweep(rows, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
