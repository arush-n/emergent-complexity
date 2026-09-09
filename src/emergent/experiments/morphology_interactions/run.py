"""Run one standalone deterministic morphology-interaction experiment.

Example::

    python -m emergent.experiments.morphology_interactions.run \
        --size 128 --steps 5000 --warmup-steps 100 --seed 42 --alpha 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MorphologyExperimentConfig
from .engine import run_experiment
from .persistence import default_output_directory


def build_parser() -> argparse.ArgumentParser:
    """Build the primary CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=None, help="square grid size")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--base-rule", default="B3/S23")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--identity-dim", type=int, default=32)
    parser.add_argument("--interaction-radius", type=int, default=2)
    parser.add_argument("--effect-padding", type=int, default=1)
    parser.add_argument("--max-rule-changes", type=int, default=2)
    parser.add_argument("--interaction-threshold", type=float, default=0.35)
    parser.add_argument("--min-component-cells", type=int, default=1)
    parser.add_argument(
        "--component-backend",
        choices=("auto", "python", "scipy"),
        default="auto",
        help="host component detector; auto selects by grid density",
    )
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--disable-interactions", action="store_true")
    parser.add_argument(
        "--initial-condition",
        choices=("random", "patterns", "npz", "api"),
        default="random",
    )
    parser.add_argument("--pattern", dest="pattern_name", default="glider")
    parser.add_argument("--pattern-row", type=int, default=None)
    parser.add_argument("--pattern-col", type=int, default=None)
    parser.add_argument("--initial-condition-path", type=str, default=None)
    parser.add_argument("--native-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> MorphologyExperimentConfig:
    """Convert parsed CLI values into one immutable resolved configuration."""

    size = args.size
    width = args.width if args.width is not None else (size if size is not None else 128)
    height = args.height if args.height is not None else (size if size is not None else 128)
    return MorphologyExperimentConfig(
        width=width,
        height=height,
        seed=args.seed,
        density=args.density,
        base_rule=args.base_rule,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        min_component_cells=args.min_component_cells,
        component_backend=args.component_backend,
        identity_dim=args.identity_dim,
        alpha=args.alpha,
        beta=args.beta,
        interaction_radius=args.interaction_radius,
        effect_padding=args.effect_padding,
        max_rule_changes=args.max_rule_changes,
        interaction_threshold=args.interaction_threshold,
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
    """Run the requested experiment and print its artifact summary."""

    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    output_dir = args.output_dir or default_output_directory(config)
    result = run_experiment(config, output_dir=output_dir)
    payload = {
        "output_dir": str(output_dir),
        "summary": result.summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
