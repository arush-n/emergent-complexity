"""Small, conservative workload checks for synchronous browser experiments."""

from __future__ import annotations

import operator

BROWSER_MAX_CELL_UPDATES = 100_000_000


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def estimate_cell_updates(
    *,
    dimensions: int,
    rules: int,
    initial_conditions: int,
    size: int,
    steps: int,
) -> int:
    """Estimate the number of cell-generations in a cubic experiment."""

    dimension_count = _integer(dimensions, "dimensions", minimum=1)
    if dimension_count not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    rule_count = _integer(rules, "rules", minimum=1)
    condition_count = _integer(initial_conditions, "initial_conditions", minimum=1)
    grid_size = _integer(size, "size", minimum=1)
    step_count = _integer(steps, "steps")
    return rule_count * condition_count * (grid_size**dimension_count) * step_count


def validate_browser_workload(
    *,
    dimensions: int,
    rules: int,
    initial_conditions: int,
    size: int,
    steps: int,
    limit: int = BROWSER_MAX_CELL_UPDATES,
) -> int:
    """Reject synchronous browser jobs that exceed a safe cell-update budget."""

    maximum = _integer(limit, "limit", minimum=1)
    work = estimate_cell_updates(
        dimensions=dimensions,
        rules=rules,
        initial_conditions=initial_conditions,
        size=size,
        steps=steps,
    )
    if work > maximum:
        raise ValueError(
            "This run is too large for the interactive server. "
            f"Estimated workload: {work:,} cell-updates; browser limit: {maximum:,}. "
            "Use the CLI experiment runner for large sweeps."
        )
    return work
