import json

import numpy as np
import pytest

from emergent.core.rules import parse_rule, rule_to_masks
from emergent.core.step import batched_step
from emergent.experiments.morphology_interactions.rna_chemistry.search.runtime import pack_grids
from emergent.experiments.morphology_interactions.rna_chemistry.search.viewer.data import (
    CONWAY,
    TraceReader,
    native_next,
    packed,
)


def test_viewer_counterfactual_matches_native_conway():
    grids = np.random.default_rng(31).integers(0, 2, (4, 9, 11), dtype=np.uint8)
    birth, survival = rule_to_masks(parse_rule("B3/S23"))
    np.testing.assert_array_equal(native_next(grids), batched_step(grids, birth, survival))


def test_reader_follows_chunks_and_does_not_double_count_events(tmp_path):
    (tmp_path / "launcher.json").write_text("{}")
    worker = tmp_path / "worker_000"
    worker.mkdir()
    (worker / "manifest.json").write_text(json.dumps({"backend": "cpu", "config": {}}))
    events = worker / "events.jsonl"
    lead = {"event": "copy_growth_lead", "shape": [1, 1, "80"], "trial": 0}
    events.write_text(json.dumps(lead) + "\n" + '{"event":')
    grid = np.zeros((2, 1, 5, 5), dtype=np.uint8)
    grid[:, 0, 1:3, 1:3] = 1
    rules = np.full(grid.shape, CONWAY, np.uint32)
    np.savez_compressed(
        worker / "trace_000000000000.npz",
        before=grid,
        after=grid,
        rules=rules,
        trials=np.array([[0], [0]]),
        ages=np.array([[0], [1]]),
    )
    reader = TraceReader(tmp_path)
    overview = reader.overview()
    assert len(overview["worlds"]) == 1
    assert overview["worlds"][0]["effect_cells"] == 0
    assert overview["lead_shapes"][0]["alerts"] == 1
    assert overview["confirmed_replicators"] is None
    with events.open("a") as handle:
        handle.write('"terminal","reason":"stable"}\n')
    reader.overview_time = 0
    updated = reader.overview()
    assert updated["lead_shapes"][0]["alerts"] == 1
    assert updated["terminal_reasons"] == {"stable": 1}
    frames = reader.frames(0, 0, 0)
    assert frames["frames"][0]["after"] == packed(grid[0, 0])
    assert frames["frames"][0]["age"] == 1
    assert frames["frames"][1]["tick"] == 1
    assert frames["max_tick"] == 1
    with pytest.raises(ValueError):
        reader.frames(0, 999)
    with pytest.raises(ValueError):
        reader.frames(-1, 0)


def test_reader_decodes_bitpacked_trace(tmp_path):
    (tmp_path / "launcher.json").write_text("{}")
    worker = tmp_path / "worker_000"
    worker.mkdir()
    (worker / "manifest.json").write_text(json.dumps({"backend": "cpu", "config": {}}))
    grid = np.zeros((1, 1, 7, 9), dtype=np.uint8)
    grid[0, 0, 2, 4] = 1
    rules = np.full(grid.shape, CONWAY, np.uint32)
    np.savez_compressed(
        worker / "trace_000000000000.npz",
        before_bits=pack_grids(grid),
        after_bits=pack_grids(grid),
        height=np.int32(7),
        width=np.int32(9),
        rules=rules,
        trials=np.array([[4]]),
        ages=np.array([[6]]),
    )
    reader = TraceReader(tmp_path)
    frames = reader.frames(0, 0)
    assert frames["height"] == 7
    assert frames["width"] == 9
    assert frames["frames"][0]["after"] == packed(grid[0, 0])


def test_reader_prefers_current_live_snapshot_over_stale_trace(tmp_path):
    (tmp_path / "launcher.json").write_text("{}")
    worker = tmp_path / "worker_000"
    worker.mkdir()
    (worker / "manifest.json").write_text(json.dumps({"backend": "cpu", "config": {}}))

    archived = np.zeros((1, 1, 6, 6), dtype=np.uint8)
    archived[0, 0, 1:3, 1:3] = 1
    rules = np.full(archived.shape, CONWAY, np.uint32)
    np.savez_compressed(
        worker / "trace_000000000000.npz",
        before=archived,
        after=archived,
        rules=rules,
        trials=np.array([[11]]),
        ages=np.array([[40]]),
    )

    live = np.zeros((1, 6, 6), dtype=np.uint8)
    live[0, 4, 5] = 1
    np.savez_compressed(
        worker / "live.npz",
        grids_bits=pack_grids(live),
        trials=np.array([12]),
        ages=np.array([2]),
        tick=np.int64(43),
        height=np.int32(6),
        width=np.int32(6),
    )

    reader = TraceReader(tmp_path)
    world = reader.overview()["worlds"][0]
    assert world["live"] is True
    assert world["trial"] == 12
    assert world["age"] == 2
    assert world["tick"] == 43
    assert world["grid"] == packed(live[0])
    assert world["archive_trial"] == 11
    assert world["archive_lag_ticks"] == 43
    assert world["archive_matches_live"] is False

    frames = reader.frames(0, 0)
    assert frames["live_frame"]["trial"] == 12
    assert frames["live_frame"]["tick"] == 43
    assert frames["live_frame"]["after"] == packed(live[0])
    assert frames["live_frame"]["trace_trial"] == 11

    def reject_archive_load(_path):
        raise AssertionError("live polling must not decode archived rule arrays")

    reader.load = reject_archive_load
    live_only = reader.frames(0, 0, live_only=True)
    assert live_only["frames"] == []
    assert live_only["live_frame"]["trial"] == 12
    assert live_only["live_frame"]["after"] == packed(live[0])
    assert live_only["max_tick"] == 0
    assert live_only["live_frame"]["trace_trial"] == 11
    assert live_only["live_frame"]["archive_lag_ticks"] == 43
