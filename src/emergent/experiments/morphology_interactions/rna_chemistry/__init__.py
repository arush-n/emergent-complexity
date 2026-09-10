"""RNA-inspired deterministic chemistry embedded in Conway-like dynamics.

This package is intentionally separate from the V1 morphology-interaction
experiment.  It reuses V1's exact ``ShapeKey`` and local Life stepping APIs,
but introduces variable-length four-symbol chemical sequences, complementary
binding sites, accessibility, and composable per-site effects.
"""

from .accessibility import AccessibilityCache, compute_accessibility, fold_pairs
from .alphabet import (
    BASE_TO_INT,
    BASES,
    complement_base,
    hamming_distance,
    kmer_code,
    reverse_complement,
    sequence_entropy,
    validate_bases,
)
from .binding import BindingResult, BindingSite, find_binding_sites, scan_binding_sites
from .chemistry import (
    ChemistryUniverse,
    PairChemistry,
    SiteInteraction,
    evaluate_pair_chemistry,
    make_chemistry_universe,
)
from .config import RNAExperimentConfig
from .effects import SiteEffect, site_interaction_to_effect
from .engine import RNAChemistryEngine, RNAExperimentResult, run_experiment, write_artifacts
from .sequence import ChemicalSequence, sequence_from_shape, sequence_from_shape_key

__all__ = [
    "BASES",
    "BASE_TO_INT",
    "AccessibilityCache",
    "BindingResult",
    "BindingSite",
    "ChemicalSequence",
    "ChemistryUniverse",
    "PairChemistry",
    "RNAExperimentConfig",
    "RNAExperimentResult",
    "RNAChemistryEngine",
    "SiteEffect",
    "SiteInteraction",
    "compute_accessibility",
    "complement_base",
    "evaluate_pair_chemistry",
    "find_binding_sites",
    "fold_pairs",
    "hamming_distance",
    "kmer_code",
    "make_chemistry_universe",
    "reverse_complement",
    "run_experiment",
    "scan_binding_sites",
    "site_interaction_to_effect",
    "sequence_entropy",
    "sequence_from_shape",
    "sequence_from_shape_key",
    "validate_bases",
    "write_artifacts",
]
