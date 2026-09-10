"""Compact metrics for RNA-inspired morphology chemistry runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..canonical import ShapeKey
from ..components import Component
from .chemistry import PairChemistry
from .sequence import ChemicalSequence

GENERATION_FIELDS = (
    "generation",
    "alive_cells",
    "alive_fraction",
    "component_count",
    "unique_species_total",
    "chemical_sequences_total",
    "species_observed_this_step",
    "new_species_this_step",
    "mean_component_size",
    "max_component_size",
    "mean_sequence_length",
    "max_sequence_length",
    "candidate_seed_count",
    "active_binding_sites",
    "unique_pair_chemistries_total",
    "unique_motifs_total",
    "unique_site_rules_total",
    "interaction_covered_cells",
    "mean_binding_score",
    "births",
    "deaths",
    "changed_cells",
    "detection_performed",
)


@dataclass
class ChemicalMetricsTracker:
    """Accumulate sampled rows and global chemistry novelty sets."""

    total_cells: int
    records: list[dict[str, int | float]] = field(default_factory=list)
    seen_species: set[ShapeKey] = field(default_factory=set, init=False)
    seen_motifs: set[tuple[int, int, int]] = field(default_factory=set, init=False)
    seen_pair_chemistries: set[tuple[ShapeKey, ShapeKey]] = field(default_factory=set, init=False)
    motifs_by_species: dict[ShapeKey, set[tuple[int, int, int]]] = field(
        default_factory=dict,
        init=False,
    )
    seen_rule_ids: set[int] = field(default_factory=set, init=False)
    seen_sequences: set[tuple[int, bytes]] = field(default_factory=set, init=False)
    total_active_sites: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.total_cells, int) or self.total_cells < 1:
            raise ValueError("total_cells must be a positive integer")

    def observe_chemistry(
        self,
        *,
        observations: list[tuple[Component, ShapeKey, ChemicalSequence]],
        pair_chemistries: list[PairChemistry],
        active_sites: list[Any],
        detection_performed: bool,
    ) -> None:
        """Update cumulative chemistry state for one detection pass.

        Detection and metric sampling are deliberately separate controls.  A
        run with ``metrics_every > 1`` must not lose species, motif, or site
        discoveries that occur between sampled rows.
        """

        if detection_performed:
            for _, key, _ in observations:
                self.seen_species.add(key)
            for _, _, sequence in observations:
                self.seen_sequences.add((sequence.length, sequence.bases.tobytes()))
            for chemistry in pair_chemistries:
                self.seen_pair_chemistries.add(chemistry.pair_key)
                for site in chemistry.sites:
                    motif_key = (site.motif.motif_length, *site.motif.symmetric_codes)
                    self.seen_motifs.add(motif_key)
                    for species_key in chemistry.pair_key:
                        self.motifs_by_species.setdefault(species_key, set()).add(motif_key)
        self.seen_rule_ids.update(int(active.effect.rule_id) for active in active_sites)
        self.total_active_sites += len(active_sites)

    def record(
        self,
        *,
        generation: int,
        previous_grid: Any,
        next_grid: Any,
        components: list[Component] | tuple[Component, ...],
        observations: list[tuple[Component, ShapeKey, ChemicalSequence]],
        new_species_count: int,
        pair_chemistries: list[PairChemistry],
        active_sites: list[Any],
        owner_map: Any,
        detection_performed: bool,
        chemistry_already_observed: bool = False,
    ) -> dict[str, int | float]:
        """Record one transition and update cumulative chemistry sets."""

        previous = np.asarray(previous_grid, dtype=np.uint8)
        current = np.asarray(next_grid, dtype=np.uint8)
        if previous.shape != current.shape or previous.size != self.total_cells:
            raise ValueError("transition grids must match the configured world")
        observed_sequences = [item[2] for item in observations] if detection_performed else []
        if not chemistry_already_observed:
            self.observe_chemistry(
                observations=observations,
                pair_chemistries=pair_chemistries,
                active_sites=active_sites,
                detection_performed=detection_performed,
            )
        active_binding_sites = len(active_sites)
        births = int(np.count_nonzero((previous == 0) & (current != 0)))
        deaths = int(np.count_nonzero((previous != 0) & (current == 0)))
        component_sizes = [component.cell_count for component in components]
        sequence_lengths = [sequence.length for sequence in observed_sequences]
        binding_scores = [float(active.effect.interaction.binding_score) for active in active_sites]
        covered = int(np.count_nonzero(np.asarray(owner_map) >= 0))
        row: dict[str, int | float] = {
            "generation": int(generation),
            "alive_cells": int(np.count_nonzero(current)),
            "alive_fraction": float(np.count_nonzero(current) / self.total_cells),
            "component_count": len(components) if detection_performed else 0,
            "unique_species_total": len(self.seen_species),
            "chemical_sequences_total": len(self.seen_sequences),
            "species_observed_this_step": len(observed_sequences),
            "new_species_this_step": int(new_species_count),
            "mean_component_size": float(np.mean(component_sizes)) if component_sizes else 0.0,
            "max_component_size": int(max(component_sizes)) if component_sizes else 0,
            "mean_sequence_length": float(np.mean(sequence_lengths)) if sequence_lengths else 0.0,
            "max_sequence_length": int(max(sequence_lengths)) if sequence_lengths else 0,
            "candidate_seed_count": int(
                sum(item.candidate_seed_count for item in pair_chemistries)
            ),
            "active_binding_sites": active_binding_sites,
            "unique_pair_chemistries_total": len(self.seen_pair_chemistries),
            "unique_motifs_total": len(self.seen_motifs),
            "unique_site_rules_total": len(self.seen_rule_ids),
            "interaction_covered_cells": covered,
            "mean_binding_score": float(np.mean(binding_scores)) if binding_scores else 0.0,
            "births": births,
            "deaths": deaths,
            "changed_cells": births + deaths,
            "detection_performed": int(detection_performed),
        }
        self.records.append(row)
        return row

    def summary(
        self,
        final_grid: Any,
        *,
        species_count: int,
        pair_count: int,
        last_generation: int,
    ) -> dict[str, int | float]:
        """Return cumulative and endpoint metrics."""

        final = np.asarray(final_grid, dtype=np.uint8)
        return {
            "generations_recorded": len(self.records),
            "final_alive_cells": int(np.count_nonzero(final)),
            "final_alive_fraction": float(np.count_nonzero(final) / self.total_cells),
            "unique_species_total": int(species_count),
            "unique_chemical_sequences_total": len(self.seen_sequences),
            "unique_pair_chemistries_total": int(pair_count),
            "unique_motifs_total": len(self.seen_motifs),
            "unique_site_rules_total": len(self.seen_rule_ids),
            "total_active_binding_sites": self.total_active_sites,
            "last_generation": int(last_generation),
        }


def transition_counts(previous_grid: Any, next_grid: Any) -> dict[str, int]:
    """Return native-style birth/death counts."""

    previous = np.asarray(previous_grid) != 0
    current = np.asarray(next_grid) != 0
    if previous.shape != current.shape:
        raise ValueError("transition grids must have equal shapes")
    births = int(np.count_nonzero(~previous & current))
    deaths = int(np.count_nonzero(previous & ~current))
    return {"births": births, "deaths": deaths, "changed_cells": births + deaths}
