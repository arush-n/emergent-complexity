# Performance and short benchmark protocol

The experiment has two different costs:

```text
host analysis:  components, canonical keys, registries, pair zones
device step:   one native or local-rule JAX transition
```

The host path is intentionally straightforward and correct first. The device
path computes neighbor counts once and uses a padded pair-rule table in one
JAX call. Benchmarks must report both the native control and active path; the
active path includes the morphology analysis that gives the experiment its
semantics.

## Short probe

Run the scoped probe from the repository root:

```bash
python -m emergent.experiments.morphology_interactions.benchmarks.short \
  --num-envs 128 --size 32 --steps 40 --warmup-steps 5 \
  --batch-size 64 \
  --output-dir artifacts/experiments/morphology_interactions/benchmarks/short_cpu
```

The probe:

1. builds one deterministic initial batch and reuses it for comparable cases;
2. compiles the relevant JAX paths before timing;
3. measures a batched native control;
4. measures active morphology interactions at the requested batch size;
5. measures the same active workload as one full batch; and
6. measures `detect_every=2` as an explicitly different-physics profile.

It also reports a one-environment active case. The output includes backend,
device, elapsed seconds, environments/second, transitions/second, and mean
final population. When `--output-dir` is supplied, `summary.json` and
`cases.csv` are written below that directory. Benchmark files are generated
artifacts and are ignored by Git.

The timing starts after a compile warmup and ends only after the batched result
has been copied to the host, so asynchronous JAX work is included. It does
not include deterministic initial-grid construction or the one-time compile.

## Interpreting results

Use `control_batched` to measure the ceiling of the native batched JAX path.
Use `active_batched` for the normal experiment cost. Compare
`active_full_batch` against `active_batched` to choose a batch size that fits
device memory without adding unnecessary calls. The one-environment case is a
diagnostic, not a claim that scalar execution is preferable for sweeps.

`detect_every > 1` can increase throughput by reducing host work, but it
changes when `I(X_t)` is refreshed and therefore changes the physical model.
It must remain a recorded configuration parameter and should not be called a
transparent optimization.

## Scaling procedure

Start with the following small matrix:

```text
64 x 64,   500 generations,   1–32 environments
128 x 128, 5,000 generations, 1–128 environments
256 x 256, 10,000 generations, only after profiling
```

For many independent environments, increase `num_envs` and `batch_size`
together, then compare host workers `0`, `2`, and `4`. Host threads are not
assumed to help because canonicalization and Python cache bookkeeping can
contend; measure before selecting them. Increase `pair_capacity` if a dense
world needs more active pair slots.

The `scipy` component backend is useful for dense grids, while `python` is the
portable reference and may be faster for sparse grids. `auto` selects between
them by density. Backend comparisons are valid only when their component lists
and complete transition outputs agree.

## Accelerator note

The experiment uses JAX's selected default backend and does not require a GPU.
On Apple Metal, device stepping can be faster in isolation while the full
experiment remains host-bound by component extraction and transfer overhead.
Always compare end-to-end transitions/second, not only a standalone local-step
kernel. Record `jax_backend` and `jax_devices` with any published benchmark.

## Recording results

Record the resolved configuration, backend, devices, batch size, component
backend, detection interval, and elapsed time with every benchmark. Compare
native control and active morphology runs using the same initial batch. Report
both environments per second and environment-generations per second.

The high-dimensional parallel scale probe additionally records global species,
pair, and effective-rule diversity plus an exact/empirical encoding audit. Its
artifacts are suitable for comparing interaction-vector dimensions and local
rule-change caps across machines without embedding machine-specific results in
the documentation.
