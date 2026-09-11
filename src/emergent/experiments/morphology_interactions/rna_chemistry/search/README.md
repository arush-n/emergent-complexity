# Continuous parallel RNA chemistry search

Use [RNA Observatory](viewer/README.md) to watch worlds, replay individual
transitions, and inspect the exact local rules and their effects.

This runner samples deterministic RNA chemistry worlds indefinitely. Each
environment has **no generation horizon**. Its slot receives a fresh seeded
world only after extinction, an exact unchanged transition, an exact dense-grid
cycle, or an exact full chemistry-state cycle. Other environments continue at
their existing ages. A finite toroidal world will eventually repeat under this
deterministic chemistry, but the required time can be extremely long.

```bash
.venv/bin/python -m emergent.experiments.morphology_interactions.rna_chemistry.search.run \
  --output-dir artifacts/experiments/morphology_interactions/rna_chemistry/search/continuous \
  --workers 8 --batch-size 64 --size 128 \
  --binding-lifetime-mode energy
```

There are `workers × batch-size` simultaneous environments. Each process
prepares spatial chemistry independently, then advances its entire batch in
one JAX call. Finished slots are immediately refilled. Process counts and
batch sizes affect resources, not per-trial physics. The default rotating
schedule gives each replacement a different deterministic strategy/configuration
(density, folding, lifetime, and interaction radius). Use
`--strategy-schedule fixed` for a single-configuration control.

Every trial has a stable `trial_key`, resolved `config_key`, initial-state key,
and composite test key. Duplicate `(configuration, initial state)` tests are
skipped. Terminal worlds are recorded as failed trials (`failure: true`) with
their exact reason (`extinct`, `stable`, `repeat`, `grid_repeat`,
or `unchanged`) and then
replaced. The
replacement `start` event points to the failed trial and records its new
strategy/configuration. Universe seed and initial-grid seed are recorded
separately.

The same RNA laws apply to every morphology: variable-length surface sequences,
complementary binding sites, motifs, bounded effects per site, optional folding,
and energy-dependent binding lifetimes. The default initializer is a random
soup, so partners and potential feedstock can already be present. This is a
raw sampling search, not an evolutionary optimizer or a claim of autonomous
replication in isolation.

## Exact termination

The runner keeps exact Brent detectors for dense-grid and full-chemistry-state
cycles. A morphology-multiset detector remains available as an opt-in diagnostic
with `--evict-morphology-repeats`, but is disabled by default: a moving organism
can revisit the same morphology composition while its spatial state is still
changing and should have time to evolve. All detectors support any period
without a hash-collision risk and store one anchor per environment. Chemistry
caches are periodically evicted without changing dynamics.

Unchanged grids, grid repeats, fixed points, extinction, and full-state periodic
worlds are recorded separately. An
`--max-ticks` option exists for benchmarks only: it interrupts the worker and
does **not** classify remaining worlds as terminal. Normal runs omit it.

Unchanged-grid eviction is exact: consecutive dense grid states are compared
with a JAX predicate. There is no low-activity threshold or arbitrary
stagnation window. A world that keeps changing without an exact dense-grid or
full-state recurrence remains active.

## Trace and evidence

The output contains a Python source archive with SHA-256 checksums, launch
arguments, and one directory per worker. Each worker records:

- `manifest.json`: full chemistry configuration, calibration, versions, and device;
- `events.jsonl`: keyed starts, failed terminal trials, copy-growth leads, and
  first-seen interesting shapes/interactions;
- `status.json`: live throughput, ages, terminal failures, unique test keys, and
  chemistry activity;
- `live.npz`: atomically replaced compact snapshot of the current grid, trial,
  age, and tick for every active slot; this is what the live viewer uses for
  current tiles;
- `trace_*.npz`: every grid before/after every transition and its applied rule field;
  grids are little-endian bit-packed on disk to keep large-world archival practical;
- `stopped.json`: present only after a graceful worker stop or failure cleanup.

Rule-field integers use bits 0–8 for births and 9–17 for survival. Their shape
is `(ticks, environments, height, width)`. Trial IDs and source ages identify
each transition, including slot replacements. Chemistry can be regenerated
from initial seed, recorded configuration, and archived source. Every growth
lead and interesting chemistry event includes the trial strategy, resolved
configuration key, generation, and `trace_tick`, so the corresponding grid and
per-cell rule field can be opened directly in the viewer or replayed exactly.
Shape notes contain the exact packed morphology and sequence length; interaction
notes contain binding-site counts, energies, motifs, and site-effect strengths.

The dense grid transition is JAX-backed and stays as a device array between
steps. Chemistry and component lists are ragged, so exact canonicalization,
binding-site selection, and event serialization remain deterministic host-side
bookkeeping. Batched component labeling and one batched JAX transition keep
that unavoidable boundary narrow.

## Throughput guidance

Use `--component-backend scipy` for search workloads. The fixed-shape Life
transition is already one JAX operation over each worker batch; the limiting
stage is usually exact component extraction and chemistry preparation. On
CPU, several moderate process shards generally outperform one very large
batch because they parallelize that host-side work. A useful starting point
is `--workers 4 --batch-size 8`, then measure on the target machine.

`--detect-every 2` or greater reduces analysis overhead, but it is a declared
physics parameter: interaction fields are refreshed less often. It should be
compared as a separate condition, not treated as an invisible optimization.
The manifest records the active JAX backend and devices. If the environment
does not provide a Metal JAX plugin, JAX will report `cpu`; install and select
that backend separately before attributing a speed change to GPU execution.

## Screening growth leads

Use the isolation assay before promoting population-count alerts to replication
candidates. It deduplicates exact shapes, selects larger leads, preserves their
source worker/trial/trace tick, and runs each candidate under four matched
conditions in JAX batches: native Conway and RNA alpha 0, 0.5, and 1.

```bash
.venv/bin/python -m emergent.experiments.morphology_interactions.rna_chemistry.search.screen \
  artifacts/experiments/morphology_interactions/rna_chemistry/search/continuous \
  --output-dir artifacts/experiments/morphology_interactions/rna_chemistry/search/isolation \
  --candidates 24 --batch-size 4 --size 64 --steps 128
```

Each assay starts from one centered copy with zero warmup and no surrounding
soup. Complete chemistry-state recurrence prunes finished assays; remaining
assays at `--steps` are inconclusive. This diagnostic horizon does not impose
an age limit on continuous discovery worlds. `screen.jsonl` records each
condition, exact-copy counts, purity, termination, and source trace location.

A positive isolation result still needs daughter-transfer and repeated-lineage
tests. A negative result does not rule out reproduction requiring partners or
feedstock. A single connected organism has no pair interaction until it splits
or encounters another component; compare ecological and isolated assays before
concluding that its chemistry cannot support reproduction. Selection of the
largest observed leads is deliberately targeted, not an unbiased alpha sweep.

```bash
.venv/bin/python -m emergent.experiments.morphology_interactions.rna_chemistry.search.replay \
  artifacts/experiments/morphology_interactions/rna_chemistry/search/continuous/worker_000
```

Replay reconstructs RNA bindings and every rule field, then checks every next
grid exactly. It verifies the flushed prefix of an active run. Traces flush
every four batch ticks by default, and on graceful shutdown. Set
`--trace-chunk 1` for per-tick persistence. An abrupt kill can lose the buffered
suffix. Storage grows with runtime because exact traces are retained.

Copy-growth events track any exact morphology reaching two or four simultaneous
components. These are **leads**, not verified self-replicators: common debris
can also increase in count, and population growth alone does not establish
parentage or daughter reproductive capability. The complete chemistry trace is
retained for subsequent isolation and repeated-lineage validation.

Stop the launcher using Ctrl-C or `kill -TERM <pid>` with the PID from
`launcher.json`. It asks workers to stop after their current tick and flush.
Interrupted worlds are not marked terminal. Automatic checkpoint resume is
not implemented; the archived prefix supports deterministic replay.
