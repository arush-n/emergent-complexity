# RNA Observatory

A read-only live view of the continuous search. It reads completed trace
chunks, never runs or modifies simulation physics, and uses a bounded chunk
cache to avoid repeatedly decompressing the archive.

```bash
.venv/bin/python -m emergent.experiments.morphology_interactions.rna_chemistry.search.viewer \
  artifacts/experiments/morphology_interactions/rna_chemistry/search/continuous_seed42_512 \
  --port 8777
```

Open **http://127.0.0.1:8777**. Select a worker and world, then use the timeline
and Play trace to inspect history. Follow latest returns to the newest flushed
frame. Hover over a cell for the exact B/S rule used to produce it.

- Teal: live cells.
- Amber: cells assigned a rule different from Conway.
- Pink: cells whose next state actually differs from a Conway-only transition.

The rule overlay refers to the input of the displayed transition. It does not
necessarily describe the next rule that will be chosen for the displayed grid.
The Conway comparison is a one-step counterfactual from the same input state,
not a separate long-running control experiment.

Archived batch ticks continue across slot replacements; the displayed trial
ID and per-trial age identify exactly which world is shown. The page updates
every three seconds, but the simulation flush interval controls actual frame
availability. Stopped workers and stale heartbeats are indicated separately.

Growth alerts are not verified replicators. The current search tracks copies
of initially present morphologies and does not perform independent lineage
validation. The viewer explicitly shows that validation is not yet evaluated,
rather than presenting an invented count of confirmed replicators.

The service listens only on localhost. It does not change the repository's
normal simulator or frontend.
