"""Regenerate RNA chemistry and check every archived world and local rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .....core.random import key_from_seed, random_grid
from ..chemistry import make_chemistry_universe
from ..config import RNAExperimentConfig
from ..engine import RNAChemistryEngine
from .runtime import prepare, step_fields, unpack_grids


def _trace_grids(trace: object, name: str) -> np.ndarray:
    """Read both current bit-packed traces and older raw-grid traces."""

    files = getattr(trace, "files", ())
    if name in files:
        return np.asarray(trace[name], dtype=np.uint8)
    packed_name = f"{name}_bits"
    if packed_name not in files:
        raise ValueError(f"trace is missing {name!r}")
    height = int(np.asarray(trace["height"]).item())
    width = int(np.asarray(trace["width"]).item())
    return unpack_grids(trace[packed_name], height=height, width=width)


def verify(root: Path) -> dict[str, int]:
    manifest = json.loads((root / "manifest.json").read_text())
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    default_config = RNAExperimentConfig(**manifest["config"])
    starts = {event["trial"]: event for event in events if event["event"] == "start"}
    endings = {
        event["trial"]: event["generation"] for event in events if event["event"] == "terminal"
    }
    universes = {}
    engines = {}
    states = {}
    transitions = 0
    chunks = 0
    for path in sorted(root.glob("trace_*.npz")):
        with np.load(path, allow_pickle=False) as trace:
            before_chunk = _trace_grids(trace, "before")
            after_chunk = _trace_grids(trace, "after")
            for time_index in range(len(trace["trials"])):
                for slot, trial_value in enumerate(trace["trials"][time_index]):
                    trial = int(trial_value)
                    before = before_chunk[time_index, slot]
                    after = after_chunk[time_index, slot]
                    rules = trace["rules"][time_index, slot]
                    if trial not in engines:
                        start = starts.get(trial, {})
                        config = RNAExperimentConfig(
                            **start.get("config", default_config.as_dict())
                        )
                        universe = universes.get(config.seed)
                        if universe is None:
                            universe = make_chemistry_universe(
                                config.seed,
                                beta=config.beta,
                                calibration_threshold=config.interaction_threshold,
                                calibration_size=config.calibration_size,
                            )
                            universes[config.seed] = universe
                        initial_seed = int(
                            start.get(
                                "initial_seed",
                                # Compatibility with traces written before the
                                # keyed scheduler: those seeds were sequential.
                                manifest.get("seed", default_config.seed) + trial + 1,
                            )
                        )
                        initial = np.asarray(
                            random_grid(
                                key_from_seed(initial_seed),
                                config.height,
                                config.width,
                                config.density,
                            ),
                            dtype=np.uint8,
                        )
                        states[trial] = initial
                        engines[trial] = RNAChemistryEngine(
                            config, initial_grid=initial, universe=universe
                        )
                    engine = engines[trial]
                    if engine.generation != int(trace["ages"][time_index, slot]):
                        raise ValueError(f"missing/reordered trace for trial {trial}")
                    np.testing.assert_array_equal(states[trial], before)
                    expected_rules, _ = prepare(engine, before)
                    np.testing.assert_array_equal(expected_rules, rules)
                    expected = np.asarray(step_fields(before[None], rules[None]))[0]
                    np.testing.assert_array_equal(expected, after)
                    engine.generation += 1
                    states[trial] = after.copy()
                    if engine.generation == endings.get(trial):
                        del engines[trial]
                        del states[trial]
                    transitions += 1
        chunks += 1
    return {"verified_transitions": transitions, "verified_chunks": chunks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.worker_directory)))


if __name__ == "__main__":
    main()
