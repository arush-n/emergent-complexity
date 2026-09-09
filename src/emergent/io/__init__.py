"""Persistence and small known-pattern utilities."""

from .patterns import (
    BLINKER,
    BLOCK,
    GLIDER,
    PATTERNS,
    get_pattern,
    pattern_from_text,
    place_pattern,
)
from .serialization import (
    SavedState,
    export_state_json,
    import_state_json,
    load_grid,
    load_state,
    save_grid,
    save_state,
    state_to_dict,
)
from .serialization_3d import (
    SavedState3D,
    export_state_json_3d,
    import_state_json_3d,
    load_grid_3d,
    load_state_3d,
    save_grid_3d,
    save_state_3d,
    state_to_dict_3d,
)

__all__ = [
    "BLINKER",
    "BLOCK",
    "GLIDER",
    "PATTERNS",
    "SavedState",
    "SavedState3D",
    "export_state_json",
    "export_state_json_3d",
    "get_pattern",
    "import_state_json",
    "import_state_json_3d",
    "load_grid",
    "load_grid_3d",
    "load_state",
    "load_state_3d",
    "pattern_from_text",
    "place_pattern",
    "save_grid",
    "save_grid_3d",
    "save_state",
    "save_state_3d",
    "state_to_dict",
    "state_to_dict_3d",
]
