# Experiments

The CLI experiment runners write manifests and CSV files under the requested
output directory. Browser experiment requests compute rows in memory and
return JSON; they never accept a server output path.

Browser workloads are validated before execution and a process-local lock
allows one synchronous browser experiment at a time. Larger sweeps should use
the CLI, where the machine's available memory and runtime can be managed
deliberately.

The deterministic morphology-dependent interaction experiment is documented in
[`src/emergent/experiments/morphology_interactions/README.md`](../src/emergent/experiments/morphology_interactions/README.md).
Its detailed design and short performance protocol stay inside that isolated
package:

- [`docs/design.md`](../src/emergent/experiments/morphology_interactions/docs/design.md)
- [`docs/benchmarking.md`](../src/emergent/experiments/morphology_interactions/docs/benchmarking.md)

Run its short compile-aware probe with:

```bash
python -m emergent.experiments.morphology_interactions.benchmarks.short \
  --num-envs 128 --size 32 --steps 40 --warmup-steps 5 --batch-size 64 \
  --shared-universe-seed 42
```

Use `--shared-universe-seed 42` when the environments should query one common
fixed interaction law. See the experiment's
[`docs/benchmarking.md`](../src/emergent/experiments/morphology_interactions/docs/benchmarking.md)
for the current short-probe results and interpretation.

```bash
python -m emergent.experiments.random_3d \
  --rules 100 --initial-conditions 20 --size 64 --steps 500 \
  --density 0.10 --seed 42
```
