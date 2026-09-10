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

The detailed design contract is in [docs/design.md](docs/design.md), and the
performance protocol is in [docs/benchmarking.md](docs/benchmarking.md). The
package also contains the scoped short probe at `benchmarks/short.py`.

The package layout keeps the required implementation modules importable at
the top level and groups only supporting material:

```text
morphology_interactions/
  *.py                 implementation and stable CLI entry points
  docs/                design contract and benchmark protocol
  benchmarks/          short compile-aware performance probe
```

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

## Configuration and initial conditions

`MorphologyExperimentConfig` is an immutable, validated configuration. Its
resolved value is written to every scalar-run manifest. Important defaults are:

```text
world                 128 x 128
seed                  42
density               0.10
base_rule             B3/S23
steps                 5000
warmup_steps          100
alpha                 0.5
beta                  1.0
interaction_radius    2
effect_padding        1
max_rule_changes      2
interaction_threshold 0.35
detect_every          1
```

Warmup runs native CGOL only. Set `warmup_steps=0` for controlled encounters.
The scalar runner supports:

- `random`: native deterministic JAX random grid;
- `patterns`: repository patterns such as `block`, `blinker`, and `glider`;
- `npz`: an exact saved grid; and
- `api`: one-time import through the native HTTP session/render endpoints.

The API adapter is not used in the hot loop.

Version 1 intentionally uses instantaneous morphology identity. A glider's
phases may therefore be different species keys. Temporal or spatiotemporal
identity is a separate future experiment, not silently mixed into these
results.

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
rescored independently with `evaluate_candidate_state`.

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

## Measurements and scientific controls

Each sampled generation records live cells/fraction, component count, total
and new species, active interactions, total pairs and local rules, component
size statistics, interaction strength, births, deaths, and changed cells.
The key novelty rates are:

```text
d(new species)/dt
d(new encountered pairs)/dt
d(new local rules)/dt
```

The primary controls hold the initial grid, native rule, size, warmup, and
duration fixed:

```text
control:    interactions disabled
structured: alpha = 0.00
mixed:      alpha = 0.50
scrambled:  alpha = 1.00
```

The sensitivity command measures morphology distance
`||z_A' - z_A||` against interaction distance
`||v(A',B) - v(A,B)||` for valid single-cell additions/removals. The
structured condition should generally retain more local correlation than the
scrambled condition; that is an empirical hypothesis, not a guaranteed result.

## Testing

Run all tests and static checks from the repository root:

```bash
python -m pytest -q
ruff check .
ruff format --check src/emergent/experiments/morphology_interactions tests/experiments
python -m compileall -q src tests
```

The experiment tests cover:

- toroidal 8-connected components and batched detector equivalence;
- translation, rotation, reflection, and exact packed identities;
- deterministic structured/scrambled/interpolated interactions;
- bounded local B/S projection and deterministic overlap ownership;
- exact native CGOL compatibility when no zones are active;
- species/pair reappearance and symmetric interactions;
- structured/scrambled marginal calibration and bounded large-shape statistics;
- connected single-cell sensitivity candidates and sparse-metrics accounting;
- scalar-versus-batched replay parity; and
- randomized complete-run determinism.

## Version 1 requirement checklist

This is the implementation checklist for the experiment specification. The
repository root `experiment.md` is an ignored local note in this checkout;
this package README is the version-controlled contract.

- [x] Native `parse_rule`, `rule_to_masks`, `neighbor_count_jit`, and
  `step_jit` are imported; normal CGOL is not duplicated.
- [x] The standalone package and requested modules live under
  `src/emergent/experiments/morphology_interactions/`.
- [x] Tests live under `tests/experiments/` and generated data under
  `artifacts/experiments/morphology_interactions/`.
- [x] Toroidal 8-connected component detection is implemented.
- [x] Exact translation-aware canonical shape identity is implemented, with
  configurable rotation and reflection invariance.
- [x] Reappearing morphology recovers the same species key and cached law.
- [x] Fixed-dimensional universe-seeded morphology encodings are implemented.
- [x] Structured, deterministic-scrambled, and alpha-interpolated pair laws
  are implemented.
- [x] Pair laws are bounded 18-channel vectors and small local B/S changes.
- [x] Spatial encounter zones, toroidal distance, and deterministic overlap
  resolution are implemented.
- [x] One JAX local-rule transition is used per generation, with one batched
  transition for parallel environments.
- [x] Interactions-disabled execution matches native CGOL bit-for-bit.
- [x] Species and pair interaction caches are implemented.
- [x] Warmup and random/pattern/NPZ/API initial-condition modes are implemented.
- [x] Reproducible manifest, summary, CSV, packed-key, and sparse-snapshot
  artifacts are implemented.
- [x] Novel species, pair, and local-rule rates are recorded.
- [x] Alpha sweep and one-cell sensitivity runners are implemented.
- [x] Scalar, local-step, deterministic replay, batched-parity, and lint gates
  pass.
- [x] Optional CPU host acceleration and batched JAX execution support
  hundreds/thousands of environments without modifying the native simulator.
- [x] Scoped design/performance documentation and a compile-aware short
  benchmark probe are included.

## Current verification snapshot

On the development machine, the full suite passes with 103 tests. A final CPU
scale check using 1,000 environments, 24×24 grids, 20 steps, batch size 250,
shared universe seed 42, and metrics disabled completed at approximately
3,500 active-interaction transitions/sec. The corresponding interaction-
disabled native control completed at approximately 44,000 transitions/sec.

The scoped short probe also measured 128 environments at 32×32 for 40 steps:
about 1,698 active transitions/sec with 64-environment batches versus 41,432
native-control transitions/sec. Explicit SciPy component labeling reached
about 1,756 active transitions/sec, while four host workers reached about
1,491, so serial host preparation remains the default. `detect_every=2` reached
about 3,083 transitions/sec but is a different physical experiment.

Metal support is environment-dependent. The repository's normal environment
currently exposes only `cpu:0`; a compatible `jax-metal` trial previously
matched CPU output bit-for-bit but was slower for this morphology-heavy
workload because host analysis and device transfers dominated.
