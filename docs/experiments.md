# Experiments

The CLI experiment runners write manifests and CSV files under the requested
output directory. Browser experiment requests compute rows in memory and
return JSON; they never accept a server output path.

Browser workloads are validated before execution and a process-local lock
allows one synchronous browser experiment at a time. Larger sweeps should use
the CLI, where the machine's available memory and runtime can be managed
deliberately.

These utilities measure cellular dynamics, not verified biological phenomena.
Exploratory morphology/RNA replication searches are excluded from the published
tree pending research validation. Passing implementation tests or reproducing
a known positive control is not a new spontaneous-replication discovery.

Example random-rule survey:

```bash
python -m emergent.experiments.random_3d \
  --rules 100 --initial-conditions 20 --size 64 --steps 500 \
  --density 0.10 --seed 42
```
