# Continuous parallel RNA chemistry search

Use [RNA Observatory](viewer/README.md) to watch worlds, replay individual
transitions, and inspect the exact local rules and their effects.

This runner samples deterministic RNA chemistry worlds indefinitely. Each
environment has **no generation horizon**. Once its grid repeats, its canonical
morphology multiset repeats, its complete dynamic state repeats, or an exact
transition leaves its grid unchanged, its slot receives a fresh seeded world.
Other environments continue
at their existing ages. A finite toroidal world will eventually repeat under
this deterministic chemistry, but the required time can be extremely long.

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
`morphology_repeat`, or `unchanged`) and then
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

The runner keeps three exact Brent detectors. A grid-only detector quickly evicts
spatially repeating oscillators or movers. A morphology detector compares the
sorted multiset of canonical shapes and evicts worlds whose organism-level
composition repeats even when objects have moved. A full-state detector compares
the serialized grid, warmup/detection phase, persistent binding sites, remaining
lifetimes, zones, rules, strengths, and tie ordering. All support any period
without a hash-collision risk and store one anchor per environment. Terminal
events record `grid_period` or `morphology_period` when those fast paths fire.
Chemistry caches are periodically evicted without changing dynamics.

Unchanged grids, grid repeats, morphology repeats, fixed points, extinction, and
full-state periodic worlds are recorded separately. An
`--max-ticks` option exists for benchmarks only: it interrupts the worker and
does **not** classify remaining worlds as terminal. Normal runs omit it.

Unchanged-grid eviction is exact: consecutive dense grid states are compared
with a JAX predicate. There is no low-activity threshold or arbitrary
stagnation window. A world that keeps changing without an exact grid or
morphology recurrence remains active.

## Trace and evidence

The output contains a Python source archive with SHA-256 checksums, launch
arguments, and one directory per worker. Each worker records:

- `manifest.json`: full chemistry configuration, calibration, versions, and device;
- `events.jsonl`: keyed starts, failed terminal trials, copy-growth leads, and
  first-seen interesting shapes/interactions;
- `status.json`: live throughput, ages, terminal failures, unique test keys, and
  chemistry activity;
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
