import pytest

from emergent.experiments.workload import (
    BROWSER_MAX_CELL_UPDATES,
    INTERACTIVE_MAX_CELL_UPDATES,
    estimate_cell_updates,
    estimate_interactive_cell_updates,
    validate_browser_workload,
    validate_interactive_workload,
)


def test_estimate_cell_updates_uses_dimensions() -> None:
    assert estimate_cell_updates(
        dimensions=3,
        rules=2,
        initial_conditions=3,
        size=4,
        steps=5,
    ) == 2 * 3 * 4**3 * 5


def test_browser_workload_limit_is_conservative_and_explicit() -> None:
    assert validate_browser_workload(
        dimensions=3,
        rules=1,
        initial_conditions=1,
        size=8,
        steps=10,
    ) == 5_120
    with pytest.raises(ValueError, match="Use the CLI experiment runner"):
        validate_browser_workload(
            dimensions=3,
            rules=100,
            initial_conditions=100,
            size=96,
            steps=2_000,
        )
    assert BROWSER_MAX_CELL_UPDATES == 100_000_000


def test_interactive_workload_accounts_for_neighborhood_size() -> None:
    assert estimate_interactive_cell_updates(dimensions=2, total_cells=64, steps=2) == 1_024
    assert estimate_interactive_cell_updates(dimensions=3, total_cells=512, steps=2) == 26_624
    assert validate_interactive_workload(
        dimensions=2,
        total_cells=64,
        steps=2,
    ) == 1_024
    assert INTERACTIVE_MAX_CELL_UPDATES == 100_000_000
    with pytest.raises(ValueError, match="interactive server"):
        validate_interactive_workload(dimensions=3, total_cells=128**3, steps=2)
