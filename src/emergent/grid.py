"""Compatibility exports for the compact ``emergent.grid`` API."""

from .core.grid import Grid, as_grid, empty_grid, validate_dimensions

__all__ = ["Grid", "as_grid", "empty_grid", "validate_dimensions"]
