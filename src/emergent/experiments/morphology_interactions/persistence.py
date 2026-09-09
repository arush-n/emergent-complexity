"""Reproducible artifact writing for morphology-interaction runs."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .engine import ExperimentResult
from .interaction import STRUCTURED_WEIGHT_SCALE
from .metrics import GENERATION_FIELDS, novelty_rates

ARTIFACT_ROOT = Path("artifacts/experiments/morphology_interactions")


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def default_output_directory(config: Any) -> Path:
    """Return the conventional timestamped directory for a new run."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ARTIFACT_ROOT / f"{timestamp}_seed{config.seed}_alpha{config.alpha:g}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _species_rows(result: ExperimentResult) -> list[dict[str, Any]]:
    return [
        {
            "species_id": record.species_id,
            "height": record.height,
            "width": record.width,
            "cells": record.live_cells,
            "first_seen": record.first_seen_generation,
            "last_seen": record.last_seen_generation,
            "observations": record.observations,
        }
        for record in result.species_registry.records_by_id()
    ]


def _interaction_rows(result: ExperimentResult) -> list[dict[str, Any]]:
    return [
        {
            "species_a": result.species_registry[pair.species_a].species_id,
            "species_b": result.species_registry[pair.species_b].species_id,
            "first_seen": interaction.first_seen_generation,
            "encounters": interaction.encounters,
            "interaction_strength": interaction.strength,
            "local_rule": interaction.local_rule_text,
        }
        for pair, interaction in sorted(result.interaction_cache.items(), key=lambda item: item[0])
    ]


def write_artifacts(
    result: ExperimentResult,
    output_dir: str | Path | None = None,
    *,
    timestamp: str | None = None,
) -> Path:
    """Write the requested manifest, summaries, CSVs, and sparse snapshots."""

    target = default_output_directory(result.config) if output_dir is None else Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    initial = np.asarray(result.initial_grid, dtype=np.uint8)
    final = np.asarray(result.final_grid, dtype=np.uint8)
    manifest = {
        "experiment": "morphology_interactions",
        "version": 1,
        "config": result.config.as_dict(),
        "universe_seed": result.universe.universe_seed,
        "base_rule": result.config.base_rule,
        "structured_weight_scale": STRUCTURED_WEIGHT_SCALE,
        "jax_device": [str(device) for device in jax.devices()],
        "python_version": platform.python_version(),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "initial_grid_sha256": _sha256(initial),
        "final_grid_sha256": _sha256(final),
    }
    _write_json(target / "manifest.json", manifest)

    summary = dict(result.summary)
    summary.update(
        {
            "initial_grid_sha256": _sha256(initial),
            "final_grid_sha256": _sha256(final),
            "grid_shape": [int(initial.shape[0]), int(initial.shape[1])],
        }
    )
    _write_json(target / "summary.json", summary)
    _write_csv(target / "generations.csv", GENERATION_FIELDS, result.generation_records)
    _write_csv(
        target / "novelty.csv",
        ("generation", "window", "new_species_rate", "new_pair_rate", "new_rule_rate"),
        novelty_rates(result.generation_records),
    )
    _write_csv(
        target / "species.csv",
        ("species_id", "height", "width", "cells", "first_seen", "last_seen", "observations"),
        _species_rows(result),
    )
    _write_csv(
        target / "interactions.csv",
        (
            "species_a",
            "species_b",
            "first_seen",
            "encounters",
            "interaction_strength",
            "local_rule",
        ),
        _interaction_rows(result),
    )
    _write_json(
        target / "species_keys.json",
        [
            {
                "species_id": record.species_id,
                "height": record.key.height,
                "width": record.key.width,
                "packed_hex": record.key.packed_hex,
            }
            for record in result.species_registry.records_by_id()
        ],
    )

    generations = np.asarray(sorted(result.snapshots), dtype=np.int64)
    frames = np.asarray(
        [result.snapshots[int(generation)] for generation in generations],
        dtype=np.uint8,
    )
    np.savez_compressed(target / "snapshots.npz", generations=generations, frames=frames)
    return target


save_artifacts = write_artifacts
