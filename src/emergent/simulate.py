"""Compatibility exports for the compact ``emergent.simulate`` API."""

from .core.simulate import (
    generate_batched_trajectory,
    generate_rule_trajectories,
    generate_trajectory,
    run_batched_until_stable,
    run_steps,
    run_steps_dynamic_with_metrics,
    run_steps_jit,
    run_steps_with_metrics,
    run_until_stable,
    run_until_stable_with_metrics,
    simulate,
    simulate_rule,
    simulate_rule_batch,
    simulate_rules,
)

__all__ = [
    "generate_batched_trajectory",
    "generate_trajectory",
    "generate_rule_trajectories",
    "run_batched_until_stable",
    "run_steps",
    "run_steps_dynamic_with_metrics",
    "run_steps_jit",
    "run_steps_with_metrics",
    "run_until_stable",
    "run_until_stable_with_metrics",
    "simulate",
    "simulate_rule",
    "simulate_rule_batch",
    "simulate_rules",
]
