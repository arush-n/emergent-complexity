# RNA-Inspired Morphology Chemistry

This is a standalone experiment built on the repository's native Conway's
Game of Life implementation. It does not change the V1 morphology-interaction
experiment or the normal simulator.

The experiment gives each canonical morphology a deterministic four-symbol
chemical sequence:

```text
morphology → ShapeKey → A/C/G/U sequence
                         ↓
                 complementary binding sites
                         ↓
                    reaction motifs
                         ↓
                 small local Life-rule effects
```

The sequence is an interface, not the species identity. Exact identity remains
the canonical `ShapeKey`. In the default `local_surface` mode, one nucleotide
is generated for each reactive boundary cell, so larger or more structured
morphologies can expose more potential sites without receiving a larger
per-site action budget. `exact_shape` is a reversible sequence control.

Binding uses antiparallel A/U and C/G pairing, optional G/U wobble, seed and
extend matching, dimensionless pair scores, and an adjacent-pair stacking
contribution. Optional simplified folding reduces accessibility for internally
paired bases. Reaction motifs are projected through one universe-fixed JAX
kernel, with a calibrated scrambled BLAKE2b control and deterministic alpha
interpolation.

The dense numerical work is JAX-compiled: alignment scores, fold windows,
motif features, chemistry projections, alpha mixing, and each full-grid Life
transition. Host-side work is limited to ragged component extraction,
canonical identity, spatial encounter assembly, and persistent caches. All
active site rules are applied in one grid transition per generation.

Every successful site produces at most the configured small number of local
B/S changes. Complexity comes from multiple sites composing in separate local
zones, not from giving large organisms arbitrary global powers. No explicit
replication action is present.

## Run

```bash
python -m emergent.experiments.morphology_interactions.rna_chemistry.run \
  --size 64 \
  --steps 500 \
  --warmup-steps 40 \
  --seed 42
```

Useful controls include `--sequence-mode exact_shape`,
`--accessibility-mode simplified_fold`, `--alpha 0.5`, and
`--disable-interactions` for the native Conway control.

Set `--binding-lifetime-mode energy` to give stronger deterministic duplexes
larger lifetime values. The default `instant` mode is the cleanest baseline;
the lifetime value is recorded with each site so persistence comparisons are
explicit rather than stochastic.

## Scaling audit

Run the first chemistry experiment before long evolutionary searches:

```bash
python -m emergent.experiments.morphology_interactions.rna_chemistry.audit \
  --pairs 10000 \
  --seed 42
```

The audit records sequence lengths, expected and observed seed opportunities,
successful binding sites, motif diversity, reaction-vector diversity, local
rule diversity, calibration values, and the active JAX backend. The expected
seed opportunity for seed length `k` is compared with
`(L_A-k+1)(L_B-k+1)/4**k`.

The chemistry derivative check uses valid one-cell morphology edits:

```bash
python -m emergent.experiments.morphology_interactions.rna_chemistry.sensitivity \
  --shape-a '##/##' --shape-b '.#./###'
```

It reports sequence edit distance, binding-site changes, and pair-level
interaction distance for each alpha condition.

## Controlled comparison

```bash
python -m emergent.experiments.morphology_interactions.rna_chemistry.sweep \
  --alphas 0,0.5,1 \
  --size 64 \
  --steps 500
```

All alpha conditions reuse the same initial grid and universe seed. The
structured and scrambled site vectors are calibrated on a deterministic corpus
before interpolation so alpha primarily changes interaction organization,
rather than simply changing how often thresholds are crossed.

## Outputs

For continuous sampling across parallel environments, use the
[continuous search runner](search/README.md). Worlds run until their complete
dynamic state repeats, and finished slots are refilled. It archives every
grid transition and applied local rule for deterministic replay.

Runs write to:

```text
artifacts/experiments/morphology_interactions/rna_chemistry/<run-id>/
```

The bundle contains a resolved manifest, summary, per-generation metrics,
species and pair chemistry tables, and sparse grid snapshots. Audit runs write
`summary.json` and `pairs.csv`.

The optional `reference_vienna.py` module can compare the simplified chemistry
with ViennaRNA when the separate `RNA` Python bindings are installed. ViennaRNA
is never required for, or called by, the simulation loop.

This should be described as an **RNA-inspired artificial chemistry embedded in
Conway-like dynamics**, not as a literal RNA simulation.
