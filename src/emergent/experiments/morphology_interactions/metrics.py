"""Per-generation measurements and novelty summaries for the experiment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .components import Component
from .interaction import PairInteraction, PairKey
from .species import SpeciesRegistry

GENERATION_FIELDS = (
    "generation",
    "alive_cells",
    "alive_fraction",
    "component_count",
    "unique_species_total",
    "species_observed_this_step",
    "new_species_this_step",
    "active_interaction_count",
    "unique_interaction_pairs_total",
    "unique_local_rules_total",
    "mean_component_size",
    "max_component_size",
    "mean_interaction_strength",
    "births",
    "deaths",
    "changed_cells",
)


@dataclass
class MetricsTracker:
    """Accumulate compact rows without retaining every full grid frame."""

    total_cells: int
    records: list[dict[str, int | float]] = field(default_factory=list)
    _seen_rule_ids: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.total_cells, int) or self.total_cells <= 0:
            raise ValueError("total_cells must be a positive integer")

    def record(
        self,
        *,
        generation: int,
        previous_grid: Any,
        next_grid: Any,
        components: Iterable[Component],
        species_keys: Iterable[Any],
        new_species_count: int,
        registry: SpeciesRegistry,
        active_interactions: Iterable[PairInteraction],
        interaction_cache: Mapping[PairKey, PairInteraction],
        detection_performed: bool = True,
    ) -> dict[str, int | float]:
        """Record one transition and all compact morphology/interaction counts."""

        previous = np.asarray(previous_grid, dtype=np.uint8)
        current = np.asarray(next_grid, dtype=np.uint8)
        if previous.shape != current.shape or previous.size != self.total_cells:
            raise ValueError("transition grids must have the configured cell count")
        component_list = list(components) if detection_performed else []
        observed_keys = set(species_keys) if detection_performed else set()
        active = list(active_interactions)
        births = int(np.count_nonzero((previous == 0) & (current != 0)))
        deaths = int(np.count_nonzero((previous != 0) & (current == 0)))
        changed = births + deaths
        sizes = [component.cell_count for component in component_list]
        strengths = [float(item.strength) for item in active]
        for item in active:
            self._seen_rule_ids.add(int(item.rule_id))
        row: dict[str, int | float] = {
            "generation": int(generation),
            "alive_cells": int(np.count_nonzero(current)),
            "alive_fraction": float(np.count_nonzero(current) / self.total_cells),
            "component_count": len(component_list),
            "unique_species_total": len(registry),
            "species_observed_this_step": len(observed_keys),
            "new_species_this_step": int(new_species_count),
            "active_interaction_count": len(active),
            "unique_interaction_pairs_total": len(interaction_cache),
            "unique_local_rules_total": len(self._seen_rule_ids),
            "mean_component_size": float(np.mean(sizes)) if sizes else 0.0,
            "max_component_size": int(max(sizes)) if sizes else 0,
            "mean_interaction_strength": float(np.mean(strengths)) if strengths else 0.0,
            "births": births,
            "deaths": deaths,
            "changed_cells": changed,
        }
        self.records.append(row)
        return row

    def observe_interactions(self, interactions: Iterable[PairInteraction]) -> None:
        """Include encountered local rules even when metrics are sampled sparsely."""

        self._seen_rule_ids.update(int(item.rule_id) for item in interactions)

    def summary(
        self,
        final_grid: Any,
        registry: SpeciesRegistry,
        cache: Mapping[Any, Any],
        *,
        last_generation: int | None = None,
    ) -> dict[str, int | float]:
        """Return endpoint and cumulative novelty statistics."""

        final = np.asarray(final_grid, dtype=np.uint8)
        latest = self.records[-1] if self.records else {}
        return {
            "generations_recorded": len(self.records),
            "final_alive_cells": int(np.count_nonzero(final)),
            "final_alive_fraction": float(np.count_nonzero(final) / self.total_cells),
            "unique_species_total": len(registry),
            "unique_interaction_pairs_total": len(cache),
            "unique_local_rules_total": int(len(self._seen_rule_ids)),
            "total_new_species": int(
                sum(int(row["new_species_this_step"]) for row in self.records)
            ),
            "total_active_interactions": int(
                sum(int(row["active_interaction_count"]) for row in self.records)
            ),
            "last_generation": int(
                latest.get("generation", 0) if last_generation is None else last_generation
            ),
        }


def transition_counts_numpy(previous_grid: Any, next_grid: Any) -> dict[str, int]:
    """Compute native-style transition counts on host arrays."""

    previous = np.asarray(previous_grid) != 0
    current = np.asarray(next_grid) != 0
    if previous.shape != current.shape:
        raise ValueError("transition grids must have equal shapes")
    births = int(np.count_nonzero(~previous & current))
    deaths = int(np.count_nonzero(previous & ~current))
    return {
        "births": births,
        "deaths": deaths,
        "changed_cells": births + deaths,
    }


def novelty_rates(
    records: Iterable[Mapping[str, int | float]],
    *,
    window: int = 100,
) -> list[dict[str, int | float]]:
    """Calculate windowed new-species/pair/rule rates for later analysis."""

    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    rows = list(records)
    result: list[dict[str, int | float]] = []
    for index, row in enumerate(rows):
        start = max(0, index - window + 1)
        span = max(
            1,
            int(row["generation"]) - (int(rows[start - 1]["generation"]) if start > 0 else 0),
        )
        previous = rows[start - 1] if start > 0 else None
        new_species_total = int(row["unique_species_total"]) - (
            int(previous["unique_species_total"]) if previous else 0
        )
        new_pairs_total = int(row["unique_interaction_pairs_total"]) - (
            int(previous["unique_interaction_pairs_total"]) if previous else 0
        )
        new_rules_total = int(row["unique_local_rules_total"]) - (
            int(previous["unique_local_rules_total"]) if previous else 0
        )
        result.append(
            {
                "generation": int(row["generation"]),
                "window": span,
                "new_species_rate": new_species_total / span,
                "new_pair_rate": new_pairs_total / span,
                "new_rule_rate": new_rules_total / span,
            }
        )
    return result
