# Deterministic morphology-dependent interactions

This is a standalone research experiment built on the repository's native
Conway's Game of Life (CGOL) implementation. It asks whether an emergent
cellular-automaton morphology can serve as a deterministic identity in a very
large interaction space:

- identical instantaneous shapes recover identical interaction physics;
- new shapes query new parts of the interaction landscape; and
- the same fixed universe can range from structured to apparently random
  while replaying bit-for-bit.

The experiment is isolated under this package. It does not modify the normal
simulator, frontend, server, or `emergent.core` behavior.

Detailed design, benchmarking, requirements, and search notes are available in
the scoped [docs](docs/) directory.

The RNA-inspired chemistry extension is a separate experiment with its own
configuration, artifacts, tests, and [README](rna_chemistry/README.md).

## Model

The physical state is a binary toroidal grid `X_t`. The default transition is
the repository's native `B3/S23` JAX step. After optional warmup, the engine
executes this order for every transition:

```text
X_t
  -> toroidal 8-connected components
  -> exact canonical shape keys
  -> species registration and fixed-dimensional encodings
  -> spatially close component pairs
  -> cached deterministic pair interactions
  -> local pair-specific B/S rule tables
  -> deterministic overlap resolution
  -> one JAX grid transition
  -> metrics and sparse snapshots
  -> X_(t+1)
```

Formally:

```text
X_(t+1) = G(X_t, I(X_t))
```

where `I(X_t)` contains only interactions between components within the
configured toroidal Chebyshev radius. Outside an encounter zone, the native
CGOL rule is used unchanged.

## Morphology identity

An organism is an 8-connected component of live cells using the same Moore
neighborhood that CGOL uses for physics. Components wrap across both grid
axes. The component detector has two equivalent host implementations:

- `python`: explicit NumPy flood fill, the portable reference;
- `scipy`: C-backed labeling plus deterministic periodic-seam merging; and
- `auto`: Python for sparse grids and SciPy for dense grids.

The exact identity is never a random vector. A component is unwrapped into a
local coordinate system, translated to its bounding box, optionally rotated
through 0/90/180/270 degrees, and serialized as:

```python
ShapeKey(height, width, packed_bytes)
```

The default is translation- and rotation-invariant but reflection-sensitive:

```text
rotation_invariant  = True
reflection_invariant = False
```

The packed canonical matrix is the scientific identity. A registry record
tracks first/last observation, recurrence, independent component count, and a
cached encoding. Species have no permanent organism ID: reappearing
morphology recovers the same `ShapeKey`.

For audits and downstream analyses, `exact_identity_vector()` exposes an
injective variable-length representation containing the canonical dimensions
and every canonical cell bit. Its length grows with bounding-box area. The
fixed-dimensional `MorphologyEncoder.encode_scaled()` view adds an explicit
`cell_count ** exponent` magnitude (square-root scaling by default). The
ordinary `encode()` path remains unit-normalized so the baseline alpha sweep
does not silently confound landscape structure with interaction magnitude.

## Interaction landscape

Each canonical shape also receives a fixed-dimensional encoding `z_A` with
default dimension 32. The encoding combines a universe-seeded Fourier-like
coordinate embedding with explicit, comparably scaled morphology statistics:
a bounded `log1p(cell_count)`, bounding-box density, bounded aspect ratio,
perimeter, and horizontal/vertical symmetry. Frequencies are generated once
when the universe is created. The bounded transforms prevent a large
transient component's raw cell count or aspect ratio from suppressing its
geometric information after unit normalization.

For each unordered pair of species, the universe caches two 18-channel
vectors:

```text
u(A,B) = tanh(beta / sqrt(d) * z_A^T W_k z_B)
q(A,B) = BLAKE2b(universe_seed, canonical_A, canonical_B, channel)
v_alpha(A,B) = (1-alpha) * u(A,B) + alpha * q(A,B)
```

The `W_k` matrices are fixed and symmetric. Their fixed gain is calibrated
once so representative structured and scrambled channels have comparable
marginal mean magnitude and default-threshold activation. It is not fitted
per pair, generation, or run, and is recorded as `structured_weight_scale`
in the manifest. `q` is deterministic pseudorandomness, not the identity.
The unordered pair key is sorted before both cache lookup and BLAKE2b input,
so symmetric mode satisfies `F(A,B) == F(B,A)` exactly. This calibration is
important for alpha sweeps: changing alpha should primarily change landscape
structure/predictability rather than simply turn on many more rule channels.

The 18 channels are nine birth and nine survival decisions. A pair changes at
most `max_rule_changes=2` channels, only when a channel magnitude reaches
`interaction_threshold=0.35`, and only when the requested bit differs from
base CGOL. Thus a pair creates a small deterministic perturbation such as
`B36/S23`, rather than an arbitrary object command.

Only components within `interaction_radius=2` interact. Their encounter mask
is dilated by `effect_padding=1`. If zones overlap, the greatest mean absolute
interaction strength wins; exact ties use lexicographic canonical pair order.

## Running the experiment

Run one artifact-producing universe:

```bash
python -m emergent.experiments.morphology_interactions.run \
  --size 128 --steps 5000 --warmup-steps 100 \
  --seed 42 --alpha 0.5 --component-backend auto
```

Run the pure-CGOL control with identical initial-condition parameters:

```bash
python -m emergent.experiments.morphology_interactions.run \
  --size 128 --steps 5000 --warmup-steps 100 \
  --seed 42 --disable-interactions
```

Use `--warmup-steps 0` for controlled encounters. Initial states can be
selected with `--initial-condition random|patterns|npz|api`.

Sweep the four primary landscape conditions with the same grid per seed:

```bash
python -m emergent.experiments.morphology_interactions.sweep \
  --alphas 0,0.25,0.5,0.75,1 --seeds 0:20 \
  --size 128 --warmup-steps 100 --steps 5000 --include-control
```

Measure one-cell discrete sensitivity:

```bash
python -m emergent.experiments.morphology_interactions.sensitivity \
  --shape-a '##/##' --shape-b '.#./###' \
  --alphas 0,0.25,0.5,0.75,1
```

Run the short compile-aware performance probe:

```bash
python -m emergent.experiments.morphology_interactions.benchmarks.short \
  --num-envs 128 --size 32 --steps 40 --warmup-steps 5 \
  --batch-size 64 \
  --output-dir artifacts/experiments/morphology_interactions/benchmarks/short_cpu
```

## Parallel environments

`parallel.py` runs independent environments in lockstep. It keeps each
environment's registry and pair cache separate, while batching the grid,
owner-map, and padded pair-rule tables into one JAX call per chunk. Component
labeling is batched on the host when the selected SciPy backend is active.

```bash
python -m emergent.experiments.morphology_interactions.parallel \
  --num-envs 1000 --size 24 --steps 20 --warmup-steps 5 \
  --identity-dim 32 --max-rule-changes 2 \
  --batch-size 250 --host-workers 0 --pair-capacity 256 \
  --shared-universe-seed 42 --no-metrics
```

Guidance:

- `batch_size` controls device memory and JAX compilation shapes. Values in
  the 128–512 range are a useful starting point for many environments.
- `shared_universe_seed` reuses one fixed interaction law across environments
  while their resolved seeds still produce independent random initial grids.
  Omit it when each environment should have its own universe law.
- `host_workers=0` is deliberately serial. Canonicalization and interaction
  bookkeeping are Python-heavy, and extra threads were slower in benchmarks.
  Explicit values greater than one are available for measurement, not assumed
  to improve throughput.
- `detect_every > 1` reduces host analysis frequency and can greatly increase
  throughput, but changes the physical experiment and must be recorded as a
  parameter.
- `pair_capacity` is fixed per JAX shape. The runner raises if a world needs
  more active pair slots instead of silently dropping interactions.
- `--no-metrics` skips morphology analysis for interaction-disabled controls
  and avoids unnecessary per-generation host transfers.

The parallel runner is an experiment-local API:

```python
from emergent.experiments.morphology_interactions.parallel import run_parallel

result = run_parallel(
    config,
    num_envs=512,
    batch_size=256,
    shared_universe_seed=42,
    collect_metrics=False,
)
```

For a larger high-dimensional trial with automatic diversity and throughput
artifacts, use the scale probe:

```bash
python -m emergent.experiments.morphology_interactions.benchmarks.scale \
  --num-envs 512 --size 64 --steps 200 --warmup-steps 40 \
  --identity-dim 256 --max-rule-changes 6 --batch-size 128 \
  --pair-capacity 1024 --shared-universe-seed 42 \
  --output-dir artifacts/experiments/morphology_interactions/benchmarks/scale_d256
```

This keeps one fixed universe law shared across all environments, while each
environment receives its own deterministic starting grid. It reports global
unique species, pair laws, effective local rules, per-generation metrics, and
an exact/empirical identity-vector audit. Increasing `identity_dim` expands
the structured interaction coordinates; increasing `max_rule_changes` expands
how much of the 18-channel interaction vector can affect local B/S physics.
Both are explicit physical parameters and must be held fixed in comparisons.

## Self-replication search

The optional `replicator_search.py` runner is a deterministic evolutionary
probe for compact self-replicating or fission-like patterns under native
Conway dynamics. It keeps each candidate as one connected morphology, rolls a
population out in batched JAX worlds, and rewards later states containing at
least two exact copies of the candidate with little non-copy debris. It does
not use morphology interactions, so a hit is attributable to the native
`B3/S23` substrate.

The result also reports `is_fission_like` for a promising lead whose later
state contains repeated daughter components with a different key. That lead
signal is never promoted to the stricter `found` flag.

This is a strict, inspectable structural criterion, not yet a complete
cell-lineage proof. A no-hit short search is only a bounded negative result;
known engineered Life replicators are generally much larger than the first
candidate canvas. The full search protocol, artifact format, and validation
guidance are in [docs/replicator_search.md](docs/replicator_search.md).

```bash
python -m emergent.experiments.morphology_interactions.replicator_search \
  --candidate-size 9 --world-size 64 --population-size 32 \
  --elite-count 8 --generations 20 --evaluation-steps 32 \
  --output-dir artifacts/experiments/morphology_interactions/replicator_search/seed42
```

Search artifacts include the initial `best_pattern.npz` and the exact
`best_state.npz` at the score's recorded generation, so a claimed hit can be
rescored independently with `evaluate_candidate_state`. `history.csv` tracks
generation winners, while `trace.csv` records every unique candidate's exact
identity and scored evidence at each search generation. `identity_audit.json`
is generated automatically and records exact-key/vector counts, empirical
collision counts for the fixed-dimensional encodings, and the measured error
in the explicit size-scaled vector norm.

## Artifacts and reproducibility

Scalar runs write:

```text
artifacts/experiments/morphology_interactions/<run-id>/
  manifest.json
  summary.json
  generations.csv
  novelty.csv
  species.csv
  interactions.csv
  species_keys.json
  snapshots.npz
```

`manifest.json` records the resolved configuration, universe seed, base rule,
structured-weight scale, software versions, JAX devices, timestamp, and
initial/final grid hashes. Timestamps are metadata only and never enter the
dynamics.

`species.csv` and `interactions.csv` contain compact recurrence and encounter
records. Exact packed identities are stored in `species_keys.json`, not
expanded into CSV. `snapshots.npz` stores only sampled generations plus a
final frame.

Given the same initial grid, universe seed, configuration, and software
versions, the simulation is deterministic. The implementation never uses
Python's randomized `hash()`, wall-clock state, or an unseeded random draw in
the transition.

## Testing

Run all tests and static checks from the repository root:

```bash
python -m pytest -q
ruff check .
ruff format --check src/emergent/experiments/morphology_interactions tests/experiments
python -m compileall -q src tests
```

The tests cover component topology, canonical identity, deterministic pair
laws, local-rule stepping, native CGOL compatibility, replay, sensitivity,
and parallel execution. The complete implementation checklist is in
[docs/requirements.md](docs/requirements.md).
