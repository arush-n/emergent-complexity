"""Run one standalone RNA-inspired chemistry universe.

Example::

    python -m emergent.experiments.morphology_interactions.rna_chemistry.run \
        --size 64 --steps 500 --warmup-steps 40 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RNAExperimentConfig
from .engine import default_output_directory, run_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build the primary CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--base-rule", default="B3/S23")
    parser.add_argument(
        "--sequence-mode", choices=("local_surface", "exact_shape"), default="local_surface"
    )
    parser.add_argument("--seed-length", type=int, default=4)
    parser.add_argument("--minimum-binding-length", type=int, default=4)
    parser.add_argument("--max-mismatches", type=int, default=1)
    parser.add_argument("--no-gu-wobble", action="store_true")
    parser.add_argument("--stacking-bonus", type=float, default=-0.5)
    parser.add_argument("--binding-energy-threshold", type=float, default=-5.0)
    parser.add_argument("--motif-length", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--site-max-rule-changes", type=int, default=1)
    parser.add_argument("--interaction-threshold", type=float, default=0.35)
    parser.add_argument("--interaction-radius", type=int, default=2)
    parser.add_argument("--effect-padding", type=int, default=1)
    parser.add_argument("--accessibility-mode", choices=("none", "simplified_fold"), default="none")
    parser.add_argument("--binding-lifetime-mode", choices=("instant", "energy"), default="instant")
    parser.add_argument("--max-binding-lifetime", type=int, default=16)
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--min-component-cells", type=int, default=1)
    parser.add_argument(
        "--component-backend",
        choices=("auto", "python", "scipy"),
        default="auto",
    )
    parser.add_argument("--no-spatialization", action="store_true")
    parser.add_argument("--disable-interactions", action="store_true")
    parser.add_argument(
        "--initial-condition", choices=("random", "patterns", "npz", "api"), default="random"
    )
    parser.add_argument("--pattern", dest="pattern_name", default="glider")
    parser.add_argument("--pattern-row", type=int, default=None)
    parser.add_argument("--pattern-col", type=int, default=None)
    parser.add_argument("--initial-condition-path", type=str, default=None)
    parser.add_argument("--native-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> RNAExperimentConfig:
    """Resolve CLI values into one immutable configuration."""

    size = args.size
    width = args.width if args.width is not None else (size if size is not None else 128)
    height = args.height if args.height is not None else (size if size is not None else 128)
    return RNAExperimentConfig(
        width=width,
        height=height,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        density=args.density,
        base_rule=args.base_rule,
        min_component_cells=args.min_component_cells,
        component_backend=args.component_backend,
        sequence_mode=args.sequence_mode,
        binding_seed_length=args.seed_length,
        minimum_binding_length=args.minimum_binding_length,
        allow_gu_wobble=not args.no_gu_wobble,
        max_mismatches=args.max_mismatches,
        stacking_bonus=args.stacking_bonus,
        binding_energy_threshold=args.binding_energy_threshold,
        reaction_motif_length=args.motif_length,
        beta=args.beta,
        calibration_size=args.calibration_size,
        alpha=args.alpha,
        site_max_rule_changes=args.site_max_rule_changes,
        interaction_threshold=args.interaction_threshold,
        interaction_radius=args.interaction_radius,
        effect_padding=args.effect_padding,
        accessibility_mode=args.accessibility_mode,
        binding_lifetime_mode=args.binding_lifetime_mode,
        max_binding_lifetime=args.max_binding_lifetime,
        spatialize_sites=not args.no_spatialization,
        metrics_every=args.metrics_every,
        snapshot_every=args.snapshot_every,
        detect_every=args.detect_every,
        interactions_enabled=not args.disable_interactions,
        initial_condition=args.initial_condition,
        pattern_name=args.pattern_name,
        pattern_row=args.pattern_row,
        pattern_col=args.pattern_col,
        initial_condition_path=args.initial_condition_path,
        native_base_url=args.native_base_url,
    )


def main(argv: list[str] | None = None) -> None:
    """Run one experiment and print its summary."""

    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    output_dir = args.output_dir or default_output_directory(config)
    result = run_experiment(config, output_dir=output_dir)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "summary": result.summary}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
