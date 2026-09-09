"""Small experiment-building blocks built on the core simulator."""

from .sampling import sample_initial_grids, sample_rules
from .trajectories import (
    generate_batched_trajectories,
    generate_rule_trajectories,
    generate_trajectories,
)
from .workload import (
    BROWSER_MAX_CELL_UPDATES,
    INTERACTIVE_MAX_CELL_UPDATES,
    estimate_cell_updates,
    estimate_interactive_cell_updates,
    validate_browser_workload,
    validate_interactive_steps,
    validate_interactive_workload,
)

__all__ = [
    "generate_batched_trajectories",
    "generate_rule_trajectories",
    "generate_trajectories",
    "BROWSER_MAX_CELL_UPDATES",
    "INTERACTIVE_MAX_CELL_UPDATES",
    "estimate_cell_updates",
    "estimate_interactive_cell_updates",
    "sample_initial_grids",
    "sample_rules",
    "validate_browser_workload",
    "validate_interactive_steps",
    "validate_interactive_workload",
]
