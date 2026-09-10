# Native-Life self-replication search

`replicator_search.py` is an experiment-local, deterministic evolutionary
probe for compact self-replicating or fission-like patterns. It uses the
repository's native `B3/S23` transition and never enables the morphology
interaction physics. That separation makes a search result attributable to
ordinary Conway dynamics rather than to a hand-designed interaction effect.

## Operational definition of a hit

The search starts each candidate as one connected component in an otherwise
empty toroidal world. A candidate is marked as a hit only when, at a later
generation, the world contains:

1. at least two distinct components;
2. at least two components with the candidate's exact canonical `ShapeKey`; and
3. at least `min_purity` of all live cells belonging to those exact copies.

The default purity threshold is `0.90`. The evaluation also records total
mass balance, similarity of the closest component, and the first generation
at which the evidence appeared. The current criterion is intentionally
structural: it establishes repeatable production of exact copies, but does
not yet prove cell-level parent/offspring lineage or rule out every possible
fission interpretation. A future temporal-identity experiment can add those
stronger tests without changing this search.

The scorer separately records `is_fission_like` when a later state contains
two or more exact repeated components with high purity even if their key is
different from the starting candidate. This is useful as a lead generator:
an input can split into two nearly conserved daughter morphologies without
being a true self-replicator. It never changes the stricter `found` flag.

## Search procedure

Each search generation does the following:

1. keep the known `block`, `blinker`, and `glider` patterns as controls;
2. fill the remaining population with deterministic connected shapes grown by
   a seed-derived BLAKE2b word stream;
3. place each candidate in an independent centered world;
4. roll all worlds out in one native JAX `batched_step` call per generation;
5. detect toroidal components on the host and score exact canonical copies;
6. retain the highest-ranked candidates; and
7. create children with deterministic one-cell additions/removals that remain
   connected and within the candidate canvas.

No Python `hash`, wall-clock value, or unseeded random generator affects the
search. The same configuration and software environment therefore produce
the same candidate order, rollouts, score history, and best pattern.

The implementation is intentionally batched across candidates. Component
analysis remains on the host because it is an experiment-specific analysis
step; the physical transition remains the repository's native JAX function.

## Run a short search

From the repository root:

```bash
python -m emergent.experiments.morphology_interactions.replicator_search \
  --seed 42 \
  --candidate-size 9 \
  --world-size 64 \
  --population-size 32 \
  --elite-count 8 \
  --generations 20 \
  --evaluation-steps 32 \
  --component-backend auto \
  --output-dir artifacts/experiments/morphology_interactions/replicator_search/seed42
```

The result is printed as JSON. When `--output-dir` is provided, the directory
contains:

```text
manifest.json       resolved search and software metadata
summary.json        hit/fission flags and best-evidence summary
history.csv         best score per evolutionary generation
trace.csv           every unique candidate's best evidence per generation
best_pattern.txt    human-readable #/. candidate
best_pattern.npz    exact binary candidate matrix
best_state.npz      exact world at the best evaluation generation
```

`history.csv` is the compact progress trace for generation winners.
`trace.csv` preserves the exact packed candidate identity and all scored
copy/fission fields for every unique candidate evaluated in each evolutionary
generation, so a large run can be audited without retaining every grid frame.

`found=false` is an honest bounded-search result, not evidence that Conway's
Game of Life has no self-replicators. Known engineered Life replicators are
much larger and more specialized than the compact candidate family searched
by this first probe. Increase canvas size, rollout length, population, and
search generations deliberately, recording each resolved configuration.

## Independent inspection

The public `evaluate_candidate_state` function can score the `grid` array in
`best_state.npz` with the same exact-copy criterion. This is useful for
checking a claimed hit outside the evolutionary loop. The normal
morphology-interaction tests also cover deterministic native stepping,
component semantics, and replay.
