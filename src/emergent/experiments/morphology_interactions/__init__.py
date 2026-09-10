"""Deterministic morphology-dependent interaction experiments on native Life."""

from .canonical import (
    ShapeKey,
    canonical_matrix,
    canonical_matrix_from_grid,
    canonicalize,
    canonicalize_component,
    canonicalize_grid,
    matrix_from_shape_key,
    shape_key_from_matrix,
    unwrap_toroidal_coordinates,
)
from .components import (
    Component,
    connected_components,
    detect_components,
    detect_components_batch,
    find_connected_components,
)
from .config import MorphologyExperimentConfig
from .encoding import (
    MorphologyEncoder,
    audit_encodings,
    encode_shape,
    exact_identity_vector,
    make_encoder,
    morphology_statistics,
)
from .engine import Engine, ExperimentResult, MorphologyInteractionEngine, run_experiment
from .interaction import (
    InteractionUniverse,
    PairInteraction,
    PairKey,
    build_interaction_zone,
    calculate_interaction,
    calculate_interaction_vector,
    find_interacting_pairs,
    interaction,
    interaction_vector,
    interaction_vector_to_rule,
    make_interaction_universe,
    make_pair_interaction,
    make_pair_key,
    scrambled_interaction,
    structured_interaction,
)
from .local_step import experimental_step, resolve_owner_map, step_with_interactions
from .metrics import MetricsTracker, novelty_rates
from .persistence import write_artifacts
from .species import SpeciesRecord, SpeciesRegistry


def __getattr__(name: str):
    """Lazily expose optional runners without perturbing ``python -m`` imports."""

    if name == "ParallelExecutionConfig":
        from .parallel import ParallelExecutionConfig

        return ParallelExecutionConfig
    if name == "ParallelExperimentResult":
        from .parallel import ParallelExperimentResult

        return ParallelExperimentResult
    if name in {"ParallelEngine", "run_parallel"}:
        from .parallel import ParallelEngine, run_parallel

        return ParallelEngine if name == "ParallelEngine" else run_parallel
    replicator_exports = {
        "Candidate",
        "ReplicatorEvaluation",
        "ReplicatorSearchConfig",
        "ReplicatorSearchResult",
        "evaluate_candidate_state",
        "run_replicator_search",
        "write_search_artifacts",
    }
    if name in replicator_exports:
        from . import replicator_search

        return getattr(replicator_search, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Component",
    "Candidate",
    "Engine",
    "ExperimentResult",
    "InteractionUniverse",
    "MetricsTracker",
    "MorphologyEncoder",
    "MorphologyExperimentConfig",
    "MorphologyInteractionEngine",
    "PairInteraction",
    "PairKey",
    "ParallelEngine",
    "ParallelExecutionConfig",
    "ParallelExperimentResult",
    "ReplicatorEvaluation",
    "ReplicatorSearchConfig",
    "ReplicatorSearchResult",
    "ShapeKey",
    "SpeciesRecord",
    "SpeciesRegistry",
    "build_interaction_zone",
    "audit_encodings",
    "calculate_interaction",
    "calculate_interaction_vector",
    "canonical_matrix",
    "canonical_matrix_from_grid",
    "canonicalize",
    "canonicalize_component",
    "canonicalize_grid",
    "connected_components",
    "detect_components",
    "detect_components_batch",
    "encode_shape",
    "exact_identity_vector",
    "evaluate_candidate_state",
    "experimental_step",
    "find_interacting_pairs",
    "find_connected_components",
    "interaction",
    "interaction_vector",
    "interaction_vector_to_rule",
    "make_encoder",
    "make_interaction_universe",
    "make_pair_interaction",
    "make_pair_key",
    "matrix_from_shape_key",
    "morphology_statistics",
    "novelty_rates",
    "resolve_owner_map",
    "run_experiment",
    "run_replicator_search",
    "run_parallel",
    "scrambled_interaction",
    "shape_key_from_matrix",
    "step_with_interactions",
    "structured_interaction",
    "unwrap_toroidal_coordinates",
    "write_artifacts",
    "write_search_artifacts",
]
