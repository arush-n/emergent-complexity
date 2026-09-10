from __future__ import annotations

from emergent.experiments.morphology_interactions.benchmarks.scale import run_scale_trial
from emergent.experiments.morphology_interactions.benchmarks.short import run_short_benchmark


def test_short_benchmark_returns_comparable_control_and_active_cases() -> None:
    rows = run_short_benchmark(
        num_envs=2,
        size=8,
        steps=1,
        warmup_steps=0,
        batch_size=1,
        shared_universe_seed=42,
        include_detection_interval=False,
    )

    assert [row["label"] for row in rows] == [
        "control_batched",
        "active_batched",
        "active_full_batch",
        "active_one_environment",
    ]
    assert all(row["backend"] for row in rows)
    assert all(row["transitions_per_second"] > 0.0 for row in rows)


def test_scale_trial_reports_parallel_identity_and_rule_diversity(tmp_path) -> None:
    summary = run_scale_trial(
        num_envs=2,
        size=8,
        steps=1,
        warmup_steps=0,
        identity_dim=8,
        max_rule_changes=2,
        batch_size=2,
        pair_capacity=16,
        component_backend="python",
        output_dir=tmp_path / "scale",
    )

    assert summary["parallel"]["num_envs"] == 2
    assert summary["encoding_audit"]["identity_dim"] == 8
    assert summary["encoding_audit"]["exact_identity_collisions"] == 0
    assert summary["global_diversity"]["unique_local_rules"] >= 0
    assert (tmp_path / "scale" / "generations.csv").exists()
    assert (tmp_path / "scale" / "final_grids.npz").exists()
