"""Morphology-derived species registry and recurrence bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .canonical import ShapeKey

if TYPE_CHECKING:
    from .encoding import MorphologyEncoder


@dataclass
class SpeciesRecord:
    """Mutable observations for one exact morphology identity."""

    key: ShapeKey
    first_seen_generation: int
    observations: int = 0
    live_cells: int = 0
    height: int = 0
    width: int = 0
    interaction_vector: np.ndarray | None = None
    last_seen_generation: int | None = None
    independent_components: int = 0
    species_id: int = -1


class SpeciesRegistry(dict[ShapeKey, SpeciesRecord]):
    """A deterministic insertion-ordered mapping from keys to species records."""

    def observe(
        self,
        key: ShapeKey,
        *,
        generation: int,
        cell_count: int,
    ) -> tuple[SpeciesRecord, bool]:
        """Register one component observation and return ``(record, is_new)``."""

        if not isinstance(key, ShapeKey):
            raise TypeError("key must be a ShapeKey")
        if not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if not isinstance(cell_count, int) or cell_count < 1:
            raise ValueError("cell_count must be a positive integer")

        record = self.get(key)
        is_new = record is None
        if record is None:
            record = SpeciesRecord(
                key=key,
                first_seen_generation=generation,
                observations=1,
                live_cells=cell_count,
                height=key.height,
                width=key.width,
                last_seen_generation=generation,
                independent_components=1,
                species_id=len(self),
            )
            self[key] = record
        else:
            record.observations += 1
            record.independent_components += 1
            record.last_seen_generation = generation
        return record, is_new

    def ensure_encoding(
        self,
        record: SpeciesRecord,
        encoder: MorphologyEncoder,
    ) -> np.ndarray:
        """Derive and cache the fixed-size identity vector for ``record``."""

        if record.interaction_vector is None:
            vector = np.asarray(encoder.encode(record.key), dtype=np.float64)
            vector.setflags(write=False)
            record.interaction_vector = vector
        return record.interaction_vector

    def records_by_id(self) -> list[SpeciesRecord]:
        """Return records in deterministic first-seen species-id order."""

        return sorted(self.values(), key=lambda record: record.species_id)


def register_species(
    registry: SpeciesRegistry,
    key: ShapeKey,
    *,
    generation: int,
    cell_count: int,
) -> tuple[SpeciesRecord, bool]:
    """Functional-looking convenience wrapper around :meth:`observe`."""

    return registry.observe(key, generation=generation, cell_count=cell_count)
