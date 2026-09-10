"""Toroidal 8-connected component detection for Life grids."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover - exercised only in minimal installs
    _ndimage = None

MOORE_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (row_delta, col_delta)
    for row_delta in (-1, 0, 1)
    for col_delta in (-1, 0, 1)
    if (row_delta, col_delta) != (0, 0)
)
SCIPY_DENSE_THRESHOLD = 0.20

# Reused immutable connectivity stencils avoid allocating the same small
# arrays on every SciPy detection pass.
_SCIPY_STRUCTURE_2D = np.ones((3, 3), dtype=np.uint8)
_SCIPY_STRUCTURE_2D.setflags(write=False)
_SCIPY_STRUCTURE_3D = np.zeros((3, 3, 3), dtype=np.uint8)
_SCIPY_STRUCTURE_3D[1, :, :] = 1
_SCIPY_STRUCTURE_3D.setflags(write=False)


@dataclass(frozen=True)
class Component:
    """An 8-connected live component in ``(row, column)`` world coordinates."""

    coordinates: np.ndarray
    cell_count: int

    def __post_init__(self) -> None:
        values = np.asarray(self.coordinates, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("component coordinates must have shape (cell_count, 2)")
        if values.shape[0] != int(self.cell_count) or values.shape[0] == 0:
            raise ValueError("cell_count must equal the number of non-empty coordinates")
        if len({tuple(value) for value in values.tolist()}) != values.shape[0]:
            raise ValueError("component coordinates must be unique")
        ordered = values[np.lexsort((values[:, 1], values[:, 0]))].copy()
        ordered.setflags(write=False)
        object.__setattr__(self, "coordinates", ordered)
        object.__setattr__(self, "cell_count", int(self.cell_count))

    @classmethod
    def _from_sorted_coordinates(cls, coordinates: np.ndarray) -> Component:
        """Build a detector-owned component whose coordinates are scan-sorted."""

        values = np.asarray(coordinates, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("component coordinates must be non-empty with shape (n, 2)")
        # The detector created this array solely for this component. Avoiding
        # a second copy matters for sparse soups containing many components.
        values.setflags(write=False)
        component = object.__new__(cls)
        object.__setattr__(component, "coordinates", values)
        object.__setattr__(component, "cell_count", int(values.shape[0]))
        return component


def _as_binary_grid(grid: Any) -> np.ndarray:
    values = np.asarray(grid)
    if values.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    return values != 0


def _detect_components_python(
    live: np.ndarray,
    *,
    min_component_cells: int,
) -> list[Component]:
    """Reference flood-fill implementation used as the portable fallback."""

    height, width = live.shape
    visited = np.zeros_like(live, dtype=bool)
    components: list[Component] = []

    for row in range(height):
        for col in range(width):
            if not live[row, col] or visited[row, col]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            coordinates: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                coordinates.append((current_row, current_col))
                for row_delta, col_delta in MOORE_OFFSETS:
                    neighbor = (
                        (current_row + row_delta) % height,
                        (current_col + col_delta) % width,
                    )
                    if live[neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            if len(coordinates) >= min_component_cells:
                components.append(
                    Component(
                        np.asarray(coordinates, dtype=np.int64),
                        len(coordinates),
                    )
                )
    return components


def _find_root(parent: np.ndarray, label: int) -> int:
    """Find a union-find root with path compression."""

    root = label
    while parent[root] != root:
        root = int(parent[root])
    while parent[label] != label:
        next_label = int(parent[label])
        parent[label] = root
        label = next_label
    return root


def _union_labels(parent: np.ndarray, first: int, second: int) -> None:
    """Merge two nonzero connected-component labels deterministically."""

    if first == 0 or second == 0:
        return
    first_root = _find_root(parent, first)
    second_root = _find_root(parent, second)
    if first_root == second_root:
        return
    # Keeping the smaller root makes the relabeling independent of the order
    # in which duplicate seam contacts are encountered.
    if first_root > second_root:
        first_root, second_root = second_root, first_root
    parent[second_root] = first_root


def _components_from_labels(
    labels: np.ndarray,
    *,
    min_component_cells: int,
) -> list[Component]:
    """Convert one labeled grid into the public, scan-ordered components."""

    _, width = labels.shape
    flat_labels = labels.reshape(-1)
    live_positions = np.flatnonzero(flat_labels)
    if live_positions.size == 0:
        return []
    live_labels = flat_labels[live_positions]

    # Separate singleton cells before sorting the larger groups. Sparse soups
    # commonly contain thousands of singletons; sorting every live label and
    # allocating one coordinate array per singleton was a measurable host
    # bottleneck. ``inverse`` lets us identify them in one C-backed pass while
    # preserving the exact final scan-order sort below.
    _, inverse, counts = np.unique(
        live_labels,
        return_inverse=True,
        return_counts=True,
    )
    components: list[Component] = []

    if min_component_cells <= 1:
        singleton_positions = live_positions[counts[inverse] == 1]
        if singleton_positions.size:
            singleton_coordinates = np.empty((singleton_positions.size, 2), dtype=np.int64)
            singleton_coordinates[:, 0] = singleton_positions // width
            singleton_coordinates[:, 1] = singleton_positions % width
            for index in range(singleton_coordinates.shape[0]):
                components.append(
                    Component._from_sorted_coordinates(
                        singleton_coordinates[index : index + 1]
                    )
                )

    # Only the non-singleton cells need a label sort. This is substantially
    # smaller than the full live-cell sort for fragmented random soups.
    multi_mask = counts[inverse] > 1
    multi_labels = live_labels[multi_mask]
    multi_positions = live_positions[multi_mask]
    if multi_labels.size:
        order = np.argsort(multi_labels, kind="stable")
        sorted_labels = multi_labels[order]
        sorted_positions = multi_positions[order]
        group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1]
        group_ends = np.r_[group_starts[1:], sorted_labels.size]
    else:
        group_starts = np.empty(0, dtype=np.int64)
        group_ends = np.empty(0, dtype=np.int64)

    for start, end in zip(group_starts, group_ends):
        if end - start < min_component_cells:
            continue
        positions = sorted_positions[start:end]
        rows = positions // width
        cols = positions % width
        coordinates = np.column_stack((rows, cols)).astype(np.int64, copy=False)
        components.append(Component._from_sorted_coordinates(coordinates))
    # Union roots are determined by label IDs, not necessarily by the first
    # coordinate of the final toroidal component.  Restore the reference
    # detector's scan-order contract explicitly.
    components.sort(key=lambda component: tuple(component.coordinates[0]))
    return components


def _select_backend(live: np.ndarray, backend: str) -> str:
    """Choose a host backend without changing component semantics."""

    if backend != "auto":
        return backend
    if _ndimage is None or live.size == 0:
        return "python"
    density = float(np.count_nonzero(live)) / float(live.size)
    return "scipy" if density >= SCIPY_DENSE_THRESHOLD else "python"


def _periodic_label_pairs(live: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return live label pairs touching the two toroidal seams.

    Candidate generation is vectorized with NumPy rolls. The old code did up
    to three Python modulo operations for every boundary cell, even when the
    boundary was empty. Python is still used for the disjoint-set unions, but
    only for actual candidate pairs.
    """

    pairs: list[np.ndarray] = []

    top_live = live[0]
    top_labels = labels[0]
    bottom_live = live[-1]
    bottom_labels = labels[-1]
    for delta in (-1, 0, 1):
        shifted_live = np.roll(bottom_live, -delta)
        mask = top_live & shifted_live
        if np.any(mask):
            pairs.append(
                np.column_stack((top_labels[mask], np.roll(bottom_labels, -delta)[mask]))
            )

    left_live = live[:, 0]
    left_labels = labels[:, 0]
    right_live = live[:, -1]
    right_labels = labels[:, -1]
    for delta in (-1, 0, 1):
        shifted_live = np.roll(right_live, -delta)
        mask = left_live & shifted_live
        if np.any(mask):
            pairs.append(
                np.column_stack((left_labels[mask], np.roll(right_labels, -delta)[mask]))
            )

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.concatenate(pairs, axis=0).astype(np.int32, copy=False)


def _detect_components_scipy(
    live: np.ndarray,
    *,
    min_component_cells: int,
) -> list[Component]:
    """Label an 8-connected grid in C, then merge the toroidal seams."""

    if _ndimage is None:  # pragma: no cover - guarded by the dispatcher
        raise RuntimeError("SciPy is not available for the scipy component backend")

    labels, label_count = _ndimage.label(live, structure=_SCIPY_STRUCTURE_2D)
    if label_count == 0:
        return []

    # ``ndimage.label`` handles the expensive interior flood fill in C.  A
    # component can still cross each periodic seam, so merge labels that are
    # Moore neighbors across the top/bottom and left/right boundaries.
    periodic_pairs = _periodic_label_pairs(live, labels)
    if not periodic_pairs.size:
        return _components_from_labels(
            labels,
            min_component_cells=min_component_cells,
        )
    parent = np.arange(label_count + 1, dtype=np.int32)
    for first, second in periodic_pairs:
        _union_labels(parent, int(first), int(second))

    root_by_label = np.arange(label_count + 1, dtype=np.int32)
    for label in range(1, label_count + 1):
        root_by_label[label] = _find_root(parent, label)
    merged_labels = root_by_label[labels]

    return _components_from_labels(
        merged_labels,
        min_component_cells=min_component_cells,
    )


def detect_components_batch(
    grids: Any,
    *,
    min_component_cells: int = 1,
    backend: str = "auto",
) -> list[list[Component]]:
    """Detect toroidal components for a batch in one C-backed labeling call.

    The first dimension is treated as independent environments.  When SciPy
    is available, all environments share one ``ndimage.label`` invocation;
    only periodic seam merging and conversion into component objects remain on
    the host.  The Python backend is a portable correctness fallback.
    """

    if not isinstance(min_component_cells, int):
        raise TypeError("min_component_cells must be an integer")
    if min_component_cells < 1:
        raise ValueError("min_component_cells must be at least 1")
    if not isinstance(backend, str) or backend not in {"auto", "python", "scipy"}:
        raise ValueError("backend must be one of: auto, python, scipy")

    values = np.asarray(grids)
    if values.ndim != 3:
        raise ValueError("grids must have shape (batch, height, width)")
    if any(dimension < 1 for dimension in values.shape):
        raise ValueError("grids dimensions must be positive")
    live = values != 0
    if _select_backend(live, backend) == "python":
        return [
            _detect_components_python(
                grid,
                min_component_cells=min_component_cells,
            )
            for grid in live
        ]
    if _ndimage is None:  # pragma: no cover - guarded by the dispatcher
        raise RuntimeError("SciPy is not available for the scipy component backend")

    # The first axis is intentionally isolated: only the middle slice has
    # neighbors in the batch dimension, while the two spatial axes use the
    # ordinary 8-connected Moore structure.
    labels, label_count = _ndimage.label(live, structure=_SCIPY_STRUCTURE_3D)
    if label_count == 0:
        return [[] for _ in range(values.shape[0])]

    batch_size = values.shape[0]
    boundary_pairs: list[np.ndarray] = []
    for environment in range(batch_size):
        pairs = _periodic_label_pairs(live[environment], labels[environment])
        if pairs.size:
            boundary_pairs.append(pairs)

    if not boundary_pairs:
        return [
            _components_from_labels(
                labels[environment],
                min_component_cells=min_component_cells,
            )
            for environment in range(batch_size)
        ]

    parent = np.arange(label_count + 1, dtype=np.int32)
    for pairs in boundary_pairs:
        for first, second in pairs:
            _union_labels(parent, int(first), int(second))

    boundary_labels = [pairs.reshape(-1) for pairs in boundary_pairs]
    root_by_label = np.arange(label_count + 1, dtype=np.int32)
    for label in np.unique(np.concatenate(boundary_labels)):
        root_by_label[label] = _find_root(parent, int(label))
    merged_labels = root_by_label[labels]
    return [
        _components_from_labels(
            merged_labels[environment],
            min_component_cells=min_component_cells,
        )
        for environment in range(batch_size)
    ]


def detect_components(
    grid: Any,
    *,
    min_component_cells: int = 1,
    backend: str = "auto",
) -> list[Component]:
    """Return all toroidal 8-connected live components in scan order.

    ``backend='auto'`` selects the SciPy C implementation for dense grids and
    otherwise uses the explicit NumPy flood fill, which is faster for sparse
    grids with many tiny components.  Both paths use the same toroidal
    Moore-neighborhood semantics and return the same scan order.
    """

    if not isinstance(min_component_cells, int):
        raise TypeError("min_component_cells must be an integer")
    if min_component_cells < 1:
        raise ValueError("min_component_cells must be at least 1")
    if not isinstance(backend, str) or backend not in {"auto", "python", "scipy"}:
        raise ValueError("backend must be one of: auto, python, scipy")

    live = _as_binary_grid(grid)
    selected_backend = _select_backend(live, backend)
    if selected_backend == "scipy":
        return _detect_components_scipy(
            live,
            min_component_cells=min_component_cells,
        )
    return _detect_components_python(live, min_component_cells=min_component_cells)


# The reference backend is intentionally public within this isolated
# experiment so benchmark and parity tests can quantify the optional
# acceleration without importing anything from emergent.core.
detect_components_python = _detect_components_python


# Descriptive aliases make the analysis API convenient without duplicating
# implementation or introducing any dependency into ``emergent.core``.
connected_components = detect_components
extract_components = detect_components
find_connected_components = detect_components
