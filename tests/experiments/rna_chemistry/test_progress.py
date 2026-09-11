from emergent.experiments.morphology_interactions.canonical import ShapeKey
from emergent.experiments.morphology_interactions.rna_chemistry.search.runtime import (
    MorphologyProgress,
)


def test_progress_prunes_oscillating_counts_and_resets_for_growth():
    key = ShapeKey(1, 1, b"\x80")
    other = ShapeKey(1, 2, b"\xc0")
    tracker = MorphologyProgress(3)
    assert not tracker.observe({key: 2})
    assert not tracker.observe({key: 1})
    assert not tracker.observe({key: 2})
    assert tracker.observe({key: 1})
    assert not tracker.observe({key: 3})
    assert tracker.idle == 0
    assert not tracker.observe({other: 1})
    assert tracker.idle == 0


def test_disabled_progress_never_prunes():
    tracker = MorphologyProgress(0)
    assert not any(tracker.observe({}) for _ in range(100))


def test_runner_refills_nonproductive_moving_world(tmp_path, monkeypatch):
    import json

    import numpy as np

    from emergent.core import random as native_random
    from emergent.experiments.morphology_interactions.rna_chemistry.search.run import (
        parser,
        run_worker,
    )

    original = native_random.random_grid
    calls = 0

    def initialize(key, height, width, density):
        nonlocal calls
        calls += 1
        if calls != 1:
            return original(key, height, width, density)
        grid = np.zeros((height, width), np.uint8)
        grid[5:8, 5:8] = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]
        return grid

    monkeypatch.setattr(native_random, "random_grid", initialize)
    args = parser().parse_args([
        "--output-dir", str(tmp_path), "--worker-id", "0", "--workers", "1",
        "--batch-size", "1", "--size", "24", "--max-ticks", "12",
        "--warmup-steps", "0", "--stagnation-window", "4", "--strategy-schedule", "fixed",
    ])
    run_worker(args)
    events = [
        json.loads(line)
        for line in (tmp_path / "worker_000/events.jsonl").read_text().splitlines()
    ]
    pruned = next(event for event in events if event.get("reason") == "search_stagnation")
    assert pruned["prune_is_heuristic"]
    assert pruned["outcome"] == "search_pruned"
    assert pruned["grid_period"] is None
    assert pruned["period"] is None
    assert pruned["idle_analysis_steps"] == 4
    assert any(event.get("replaces_trial") == pruned["trial"] for event in events)
