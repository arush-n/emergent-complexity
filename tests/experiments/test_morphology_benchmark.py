from __future__ import annotations

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
