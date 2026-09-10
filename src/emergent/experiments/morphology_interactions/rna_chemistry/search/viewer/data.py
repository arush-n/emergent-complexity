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
                    self.lead_shapes[tuple(event["shape"])] += 1
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
                if not traces:
                    continue
                data = self.load(traces[-1])
                before_chunk = trace_grids(data, "before")
                after_chunk = trace_grids(data, "after")
                before = before_chunk[-1]
                after = after_chunk[-1]
                rules = data["rules"][-1]
                counterfactual = native_next(before)
                for slot, grid in enumerate(after):
                    worlds.append(
                        {
                            "worker": int(worker.name[-3:]),
                            "slot": slot,
                            "trial": int(data["trials"][-1, slot]),
                            "strategy": self.trial_strategies.get(
                                int(data["trials"][-1, slot]), "unknown"
                            ),
                            "age": int(data["ages"][-1, slot]) + 1,
                            "tick": int(traces[-1].stem.split("_")[1]) + len(data["ages"]) - 1,
                            "height": grid.shape[0],
                            "width": grid.shape[1],
                            "grid": packed(grid),
                            "alive": int(grid.sum()),
                            "changed": int(np.count_nonzero(grid != before[slot])),
                            "rule_cells": int(np.count_nonzero(rules[slot] != CONWAY)),
                            "effect_cells": int(np.count_nonzero(grid != counterfactual[slot])),
                            "archive_lag_seconds": now - traces[-1].stat().st_mtime,
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
                "confirmed_replicators": None,
                "verification": "Lineage verification not implemented; growth alerts are leads.",
            }
            self.overview_time = now
            return self.overview_cache

    def frames(self, worker_id: int, slot: int, tick: int | None = None) -> dict:
        with self.lock:
            worker = self.worker(worker_id)
            paths = sorted(worker.glob("trace_*.npz"))
            if not paths:
                raise ValueError("waiting for the first flushed trace")
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
            return {
                "frames": frames,
                "height": after.shape[1],
                "width": after.shape[2],
                "max_tick": starts[-1] + len(self.load(paths[-1])["ages"]) - 1,
                "worker": worker_id,
                "slot": slot,
            }
