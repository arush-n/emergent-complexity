"""Bounded, cached artifact reads for the experiment viewer."""

from __future__ import annotations

import bisect
import json
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from ..runtime import unpack_grids

CONWAY = (1 << 3) | (1 << 11) | (1 << 12)


def packed(grid: np.ndarray) -> str:
    return np.packbits(grid).tobytes().hex()


def native_next(grids: np.ndarray) -> np.ndarray:
    """Read-only counterfactual for archived transitions, never a simulation backend."""
    counts = sum(
        np.roll(grids, (r, c), axis=(-2, -1)) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c
    )
    return ((counts == 3) | ((grids != 0) & (counts == 2))).astype(np.uint8)


def trace_grids(data: dict, name: str) -> np.ndarray:
    """Decode a trace grid, retaining compatibility with pre-packed runs."""

    if name in data:
        return np.asarray(data[name], dtype=np.uint8)
    packed_name = f"{name}_bits"
    if packed_name not in data:
        raise ValueError(f"trace is missing {name!r}")
    height = int(np.asarray(data["height"]).item())
    width = int(np.asarray(data["width"]).item())
    return unpack_grids(data[packed_name], height=height, width=width)


class TraceReader:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not (self.root / "launcher.json").is_file():
            raise ValueError("run directory must contain launcher.json")
        self.lock = threading.RLock()
        self.cache: OrderedDict[Path, dict] = OrderedDict()
        self.live_cache: dict[Path, tuple[int, dict]] = {}
        self.overview_cache = None
        self.overview_time = 0.0
        self.event_offsets = {}
        self.lead_shapes = Counter()
        self.terminal_reasons = Counter()
        self.strategy_counts = Counter()
        self.interesting_shape_count = 0
        self.interesting_interaction_count = 0
        self.trial_strategies = {}
        self.recent_leads = []
        self.recent_interactions = []

    def worker(self, index: int) -> Path:
        if index < 0 or index > 999:
            raise ValueError("invalid worker")
        path = self.root / f"worker_{index:03d}"
        if not path.is_dir():
            raise ValueError("unknown worker")
        return path

    def load(self, path: Path) -> dict:
        if path not in self.cache:
            with np.load(path, allow_pickle=False) as data:
                self.cache[path] = {name: data[name] for name in data.files}
            while len(self.cache) > 12:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return self.cache[path]

    def load_live(self, worker: Path) -> dict | None:
        """Load the atomically published current state for one worker."""

        path = worker / "live.npz"
        if not path.exists():
            return None
        stamp = path.stat().st_mtime_ns
        cached = self.live_cache.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        with np.load(path, allow_pickle=False) as data:
            value = {name: data[name] for name in data.files}
        self.live_cache[path] = (stamp, value)
        return value

    def read_events(self, worker: Path) -> None:
        path = worker / "events.jsonl"
        if not path.exists():
            return
        with path.open("rb") as handle:
            handle.seek(self.event_offsets.get(path, 0))
            while line := handle.readline():
                if not line.endswith(b"\n"):
                    break
                event = json.loads(line)
                self.event_offsets[path] = handle.tell()
                if event["event"] == "start":
                    strategy = event.get("strategy_key", "legacy")
                    self.strategy_counts[strategy] += 1
                    self.trial_strategies[int(event["trial"])] = strategy
                elif event["event"] == "terminal":
                    self.terminal_reasons[event["reason"]] += 1
                elif event["event"] == "copy_growth_lead":
                    shape = event["shape"]
                    if isinstance(shape, dict):
                        shape_key = (shape["height"], shape["width"], shape["packed"])
                    else:
                        # Compatibility with traces written before shape notes
                        # included cell and sequence metadata.
                        shape_key = tuple(shape)
                    self.lead_shapes[shape_key] += 1
                    self.recent_leads.append(event | {"worker": int(worker.name[-3:])})
                    self.recent_leads = self.recent_leads[-100:]
                elif event["event"] == "interesting_shape":
                    self.interesting_shape_count += 1
                elif event["event"] == "interesting_interaction":
                    self.interesting_interaction_count += 1
                    self.recent_interactions.append(
                        event | {"worker": int(worker.name[-3:])}
                    )
                    self.recent_interactions = self.recent_interactions[-100:]

    def overview(self) -> dict:
        with self.lock:
            now = time.time()
            if self.overview_cache is not None and now - self.overview_time < 2:
                return self.overview_cache
            worlds = []
            statuses = []
            configs = []
            for worker in sorted(self.root.glob("worker_[0-9][0-9][0-9]")):
                self.read_events(worker)
                manifest_path = worker / "manifest.json"
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text())
                configs.append(manifest)
                if (worker / "status.json").exists():
                    status = json.loads((worker / "status.json").read_text())
                    statuses.append(
                        status
                        | {
                            "stale_seconds": now - (worker / "status.json").stat().st_mtime,
                            "stopped": (worker / "stopped.json").exists(),
                        }
                    )
                traces = sorted(worker.glob("trace_*.npz"))
                live_path = worker / "live.npz"
                live = self.load_live(worker)
                archive_path = traces[-1] if traces else None
                archive_trials = archive_ages = None
                archive_tick = None
                archive_lag = None if archive_path is None else now - archive_path.stat().st_mtime
                if archive_path is not None:
                    # When a live snapshot exists, read only trace metadata.
                    # Decoding the large archived grids here made the viewer
                    # both stale and unnecessarily memory hungry.
                    with np.load(archive_path, allow_pickle=False) as data:
                        archive_trials = np.asarray(data["trials"])
                        archive_ages = np.asarray(data["ages"])
                    archive_tick = int(archive_path.stem.split("_")[1]) + len(archive_ages) - 1

                if live is not None:
                    live_grids = unpack_grids(
                        live["grids_bits"],
                        height=int(np.asarray(live["height"]).item()),
                        width=int(np.asarray(live["width"]).item()),
                    )
                    live_trials = np.asarray(live["trials"], dtype=np.int64)
                    live_ages = np.asarray(live["ages"], dtype=np.int64)
                    live_tick = int(np.asarray(live["tick"]).item())
                    live_height = int(np.asarray(live["height"]).item())
                    live_width = int(np.asarray(live["width"]).item())
                    live_lag = now - live_path.stat().st_mtime
                    for slot, grid in enumerate(live_grids):
                        trial = int(live_trials[slot])
                        age = int(live_ages[slot])
                        archive_trial = (
                            None
                            if archive_trials is None
                            else int(archive_trials[-1, slot])
                        )
                        archive_age = (
                            None
                            if archive_ages is None
                            else int(archive_ages[-1, slot]) + 1
                        )
                        archive_matches_live = (
                            archive_tick == live_tick
                            and archive_trial == trial
                            and archive_age == age
                        )
                        worlds.append(
                            {
                                "worker": int(worker.name[-3:]),
                                "slot": slot,
                                "trial": trial,
                                "strategy": self.trial_strategies.get(trial, "unknown"),
                                "age": age,
                                "tick": live_tick,
                                "height": live_height,
                                "width": live_width,
                                "grid": packed(grid),
                                "alive": int(grid.sum()),
                                "changed": None,
                                "rule_cells": None,
                                "effect_cells": None,
                                "live": True,
                                "live_lag_seconds": live_lag,
                                "archive_lag_seconds": archive_lag,
                                "archive_lag_ticks": (
                                    None
                                    if archive_tick is None
                                    else live_tick - archive_tick
                                ),
                                "archive_tick": archive_tick,
                                "archive_trial": archive_trial,
                                "archive_age": archive_age,
                                "archive_matches_live": archive_matches_live,
                            }
                        )
                elif archive_path is not None:
                    data = self.load(archive_path)
                    before_chunk = trace_grids(data, "before")
                    after_chunk = trace_grids(data, "after")
                    before = before_chunk[-1]
                    after = after_chunk[-1]
                    rules = data["rules"][-1]
                    counterfactual = native_next(before)
                    for slot, grid in enumerate(after):
                        trial = int(data["trials"][-1, slot])
                        age = int(data["ages"][-1, slot]) + 1
                        worlds.append(
                            {
                                "worker": int(worker.name[-3:]),
                                "slot": slot,
                                "trial": trial,
                                "strategy": self.trial_strategies.get(trial, "unknown"),
                                "age": age,
                                "tick": archive_tick,
                                "height": grid.shape[0],
                                "width": grid.shape[1],
                                "grid": packed(grid),
                                "alive": int(grid.sum()),
                                "changed": int(np.count_nonzero(grid != before[slot])),
                                "rule_cells": int(np.count_nonzero(rules[slot] != CONWAY)),
                                "effect_cells": int(np.count_nonzero(grid != counterfactual[slot])),
                                "live": False,
                                "live_lag_seconds": None,
                                "archive_lag_seconds": archive_lag,
                                "archive_lag_ticks": 0,
                                "archive_tick": archive_tick,
                                "archive_trial": trial,
                                "archive_age": age,
                                "archive_matches_live": False,
                            }
                        )
            shapes = []
            for (height, width, code), count in self.lead_shapes.most_common(8):
                cells = np.unpackbits(np.frombuffer(bytes.fromhex(code), np.uint8))[
                    : height * width
                ]
                shapes.append(
                    {
                        "height": height,
                        "width": width,
                        "grid": code,
                        "cells": int(cells.sum()),
                        "alerts": count,
                    }
                )
            self.overview_cache = {
                "run": self.root.name,
                "updated": now,
                "worlds": worlds,
                "statuses": statuses,
                "lead_shapes": shapes,
                "recent_leads": self.recent_leads[-20:],
                "terminal_reasons": dict(self.terminal_reasons),
                "strategy_counts": dict(self.strategy_counts),
                "interesting_shape_count": self.interesting_shape_count,
                "interesting_interaction_count": self.interesting_interaction_count,
                "recent_interactions": self.recent_interactions[-20:],
                "config": configs[0]["config"] if configs else {},
                "backend": configs[0]["backend"] if configs else "unknown",
                "live_snapshot_available": any(world["live"] for world in worlds),
                "live_lag_seconds": max(
                    (world["live_lag_seconds"] or 0 for world in worlds), default=0
                ),
                "archive_lag_seconds": max(
                    (world["archive_lag_seconds"] or 0 for world in worlds), default=0
                ),
                "confirmed_replicators": None,
                "verification": "Lineage verification not implemented; growth alerts are leads.",
            }
            self.overview_time = now
            return self.overview_cache

    def frames(
        self, worker_id: int, slot: int, tick: int | None = None, *, live_only: bool = False
    ) -> dict:
        with self.lock:
            worker = self.worker(worker_id)
            paths = sorted(worker.glob("trace_*.npz"))
            live = self.load_live(worker)
            live_frame = None
            if live is not None:
                live_grids = unpack_grids(
                    live["grids_bits"],
                    height=int(np.asarray(live["height"]).item()),
                    width=int(np.asarray(live["width"]).item()),
                )
                if not 0 <= slot < live_grids.shape[0]:
                    raise ValueError("unknown slot")
                live_tick = int(np.asarray(live["tick"]).item())
                live_trial = int(np.asarray(live["trials"])[slot])
                live_age = int(np.asarray(live["ages"])[slot])
                live_frame = {
                    "tick": live_tick,
                    "trial": live_trial,
                    "age": live_age,
                    "before": None,
                    "after": packed(live_grids[slot]),
                    "rules": None,
                    "effect": None,
                    "alive": int(live_grids[slot].sum()),
                    "effect_cells": None,
                    "live": True,
                }
            if not paths or (live_only and live_frame is not None):
                if live_frame is None:
                    raise ValueError("waiting for the first live snapshot or flushed trace")
                max_tick = 0
                if paths:
                    # NPZ members load lazily: metadata is tiny; the dense
                    # rule arrays and archived grids stay on disk.
                    with np.load(paths[-1], allow_pickle=False) as metadata:
                        trials = metadata["trials"]
                        max_tick = int(paths[-1].stem.split("_")[1]) + len(trials) - 1
                        live_frame["trace_trial"] = int(trials[-1, slot])
                    live_frame["trace_tick"] = max_tick
                    live_frame["archive_lag_ticks"] = live_frame["tick"] - max_tick
                return {
                    "frames": [],
                    "height": live_grids.shape[1],
                    "width": live_grids.shape[2],
                    "max_tick": max_tick,
                    "worker": worker_id,
                    "slot": slot,
                    "live_frame": live_frame,
                }
            starts = [int(path.stem.split("_")[1]) for path in paths]
            index = (
                len(paths) - 1 if tick is None else max(0, bisect.bisect_right(starts, tick) - 1)
            )
            data = self.load(paths[index])
            before_chunk = trace_grids(data, "before")
            after_chunk = trace_grids(data, "after")
            if not 0 <= slot < after_chunk.shape[1]:
                raise ValueError("unknown slot")
            before = before_chunk[:, slot]
            after = after_chunk[:, slot]
            different = after != native_next(before)
            frames = [
                {
                    "tick": starts[index] + i,
                    "trial": int(data["trials"][i, slot]),
                    "age": int(data["ages"][i, slot]) + 1,
                    "before": packed(before[i]),
                    "after": packed(after[i]),
                    "rules": data["rules"][i, slot].ravel().tolist(),
                    "effect": packed(different[i]),
                    "alive": int(after[i].sum()),
                    "effect_cells": int(different[i].sum()),
                }
                for i in range(len(after))
            ]
            last_data = data if index == len(paths) - 1 else self.load(paths[-1])
            max_tick = starts[-1] + len(last_data["ages"]) - 1
            if live_frame is not None:
                live_frame["trace_trial"] = int(data["trials"][-1, slot])
                live_frame["trace_tick"] = max_tick
                live_frame["archive_lag_ticks"] = live_frame["tick"] - max_tick
            return {
                "frames": frames,
                "height": after.shape[1],
                "width": after.shape[2],
                "max_tick": max_tick,
                "worker": worker_id,
                "slot": slot,
                "live_frame": live_frame,
            }
