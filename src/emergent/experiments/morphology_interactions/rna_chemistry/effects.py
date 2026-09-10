"""Map bounded site reactions onto spatially local Life-rule effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ....core.rules import Rule, format_rule, rule_to_masks
from ..canonical import (
    matrix_from_shape_key,
    unwrap_toroidal_coordinates,
)
from ..components import Component
from ..interaction import build_interaction_zone, interaction_vector_to_rule, toroidal_dilate
from .chemistry import SiteInteraction
from .sequence import ChemicalSequence


@dataclass(frozen=True)
class SiteEffect:
    """One bounded local B/S rule derived from one chemical binding site."""

    interaction: SiteInteraction
    local_rule: Rule
    rule_id: int
    birth_mask: np.ndarray
    survival_mask: np.ndarray

    def __post_init__(self) -> None:
        birth = np.asarray(self.birth_mask, dtype=np.uint8)
        survival = np.asarray(self.survival_mask, dtype=np.uint8)
        if birth.shape != (9,) or survival.shape != (9,):
            raise ValueError("site rule masks must have shape (9,)")
        birth.setflags(write=False)
        survival.setflags(write=False)
        object.__setattr__(self, "birth_mask", birth)
        object.__setattr__(self, "survival_mask", survival)

    @property
    def strength(self) -> float:
        """Return the continuous reaction strength for overlap resolution."""

        return self.interaction.strength

    @property
    def local_rule_text(self) -> str:
        """Return the canonical local rule text."""

        return format_rule(self.local_rule)

    @property
    def sort_key(self) -> tuple[Any, ...]:
        """Stable identity used for equal-strength site overlaps."""

        motif = self.interaction.motif
        site = self.interaction.site
        return (
            motif.symmetric_codes,
            site.start_a,
            site.start_b,
            site.end_a,
            site.end_b,
            self.rule_id,
        )


def site_interaction_to_effect(
    interaction: SiteInteraction,
    base_rule: Rule | str = "B3/S23",
    *,
    max_rule_changes: int = 1,
    interaction_threshold: float = 0.35,
) -> SiteEffect:
    """Convert one site vector to a small deterministic local Life rule."""

    local_rule, rule_id = interaction_vector_to_rule(
        interaction.final_vector,
        base_rule,
        max_rule_changes=max_rule_changes,
        interaction_threshold=interaction_threshold,
    )
    birth, survival = rule_to_masks(local_rule)
    return SiteEffect(
        interaction=interaction,
        local_rule=local_rule,
        rule_id=rule_id,
        birth_mask=np.asarray(birth, dtype=np.uint8),
        survival_mask=np.asarray(survival, dtype=np.uint8),
    )


def _transformed_coordinate(
    shape: tuple[int, int],
    coordinate: tuple[int, int],
    rotation: int,
    reflected: bool,
) -> tuple[int, int]:
    """Apply the same transform used by ``np.rot90``/``np.fliplr``."""

    row, col = coordinate
    height, width = shape
    for _ in range(rotation % 4):
        row, col = width - 1 - col, row
        height, width = width, height
    if reflected:
        col = width - 1 - col
    return row, col


def component_site_coordinate(
    component: Component,
    sequence: ChemicalSequence,
    position: int,
    grid_shape: tuple[int, int],
) -> tuple[int, int] | None:
    """Map one sequence position to a toroidal world coordinate when possible."""

    if position < 0 or position >= sequence.length:
        return None
    coordinate = tuple(int(value) for value in sequence.site_coordinates[position])
    if coordinate[0] < 0 or coordinate[1] < 0:
        return None
    values = unwrap_toroidal_coordinates(component.coordinates, grid_shape)
    root = tuple(
        int(value)
        for value in component.coordinates[
            np.argmin(component.coordinates[:, 0] * grid_shape[1] + component.coordinates[:, 1])
        ]
    )
    local_min = values.min(axis=0)
    local_values = values - local_min
    local_shape = (int(local_values[:, 0].max()) + 1, int(local_values[:, 1].max()) + 1)
    canonical = matrix_from_shape_key(sequence.shape_key)
    target = {tuple(int(value) for value in item) for item in local_values.tolist()}

    transforms: list[tuple[np.ndarray, tuple[int, int]]] = []
    for rotation in range(4):
        rotated = np.rot90(canonical, k=rotation).copy()
        transforms.append(
            (rotated, _transformed_coordinate(canonical.shape, coordinate, rotation, False))
        )
        reflected = np.fliplr(rotated).copy()
        transforms.append(
            (reflected, _transformed_coordinate(canonical.shape, coordinate, rotation, True))
        )
    for candidate, transformed_site in transforms:
        if candidate.shape != local_shape:
            continue
        candidate_coordinates = {
            tuple(int(value) for value in item) for item in np.argwhere(candidate != 0)
        }
        if candidate_coordinates != target:
            continue
        unwrapped = np.asarray(transformed_site, dtype=np.int64) + local_min
        return (
            (root[0] + int(unwrapped[0])) % int(grid_shape[0]),
            (root[1] + int(unwrapped[1])) % int(grid_shape[1]),
        )
    return None


def site_effect_zone(
    component_a: Component,
    component_b: Component,
    sequence_a: ChemicalSequence,
    sequence_b: ChemicalSequence,
    interaction: SiteInteraction,
    grid_shape: tuple[int, int],
    *,
    interaction_radius: int = 2,
    effect_padding: int = 1,
    spatialize_sites: bool = True,
) -> np.ndarray:
    """Anchor one site effect near its mapped sequence positions.

    If a canonical-to-world transform cannot be recovered (for example for a
    deliberately abstract exact-shape header position), the function falls
    back to the complete encounter zone.  This preserves Stage 1 chemistry
    validation while supporting Stage 2 spatialized sites whenever possible.
    """

    encounter_zone = build_interaction_zone(
        component_a,
        component_b,
        grid_shape,
        interaction_radius=interaction_radius,
        effect_padding=effect_padding,
    )
    if not spatialize_sites:
        return encounter_zone
    site = interaction.site
    anchors = np.zeros(grid_shape, dtype=bool)
    positions_a = range(site.start_a, site.end_a)
    positions_b = range(site.end_b - 1, site.start_b - 1, -1)
    for position in positions_a:
        coordinate = component_site_coordinate(component_a, sequence_a, position, grid_shape)
        if coordinate is not None:
            anchors[coordinate] = True
    for position in positions_b:
        coordinate = component_site_coordinate(component_b, sequence_b, position, grid_shape)
        if coordinate is not None:
            anchors[coordinate] = True
    if not np.any(anchors):
        return encounter_zone
    localized = toroidal_dilate(anchors, interaction_radius + effect_padding)
    localized &= encounter_zone
    return localized if np.any(localized) else encounter_zone


def resolve_site_owner_map(
    zones: list[tuple[SiteEffect, Any]],
    *,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, list[SiteEffect]]:
    """Resolve overlapping site zones by strength, then chemical identity."""

    height, width = grid_shape
    merged: dict[tuple[Any, ...], tuple[SiteEffect, np.ndarray]] = {}
    for effect, zone in zones:
        values = np.asarray(zone, dtype=bool)
        if values.shape != (height, width):
            raise ValueError("site zones must match grid_shape")
        key = effect.sort_key
        if key in merged:
            merged[key] = (effect, merged[key][1] | values)
        else:
            merged[key] = (effect, values.copy())
    ordered = [
        item[0]
        for item in sorted(
            merged.values(),
            key=lambda item: item[0].sort_key,
        )
        if np.any(item[1])
    ]
    zone_by_key = {effect.sort_key: merged[effect.sort_key][1] for effect in ordered}
    owner_map = np.full((height, width), -1, dtype=np.int32)
    best_strength = np.full((height, width), -np.inf, dtype=np.float32)
    for index, effect in enumerate(ordered):
        zone = zone_by_key[effect.sort_key]
        replace = zone & (float(effect.strength) > best_strength)
        owner_map[replace] = index
        best_strength[replace] = float(effect.strength)
    return owner_map, ordered


def site_rule_arrays(
    effects: list[SiteEffect],
    *,
    capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack site rules for V1's one-pass JAX local step."""

    if capacity is None:
        table_size = len(effects)
    else:
        if not isinstance(capacity, int) or capacity < len(effects) or capacity < 1:
            raise ValueError("capacity must contain every site effect and be positive")
        table_size = capacity
    birth = np.zeros((table_size, 9), dtype=np.uint8)
    survival = np.zeros((table_size, 9), dtype=np.uint8)
    if effects:
        birth[: len(effects)] = np.asarray(
            [effect.birth_mask for effect in effects],
            dtype=np.uint8,
        )
        survival[: len(effects)] = np.asarray(
            [effect.survival_mask for effect in effects],
            dtype=np.uint8,
        )
    return birth, survival


def effect_rule_ids(effects: list[SiteEffect]) -> set[int]:
    """Return the distinct local rules represented by a site collection."""

    return {int(effect.rule_id) for effect in effects}
