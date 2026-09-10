# Version 1 requirements

This checklist records the implementation contract for the isolated
`morphology_interactions` experiment.

- [x] Native CGOL rules and JAX stepping are imported from `emergent.core`.
- [x] The normal simulator, frontend, server, and core packages are untouched.
- [x] Toroidal 8-connected components use the Moore neighborhood.
- [x] Exact canonical `ShapeKey` identity supports configurable rotation and
  reflection invariance.
- [x] Reappearing morphologies recover the same species and pair law.
- [x] Fixed-dimensional structured and deterministic-scrambled interactions
  are alpha-interpolated and symmetric.
- [x] Exact identity vectors are injective and size-aware; fixed interaction
  vectors have an explicit size-scaled analysis view.
- [x] Pair vectors map to bounded local birth/survival rule changes.
- [x] Toroidal encounter zones and deterministic overlap resolution are used.
- [x] One JAX transition handles all active local rules per generation.
- [x] Interaction-disabled execution matches native CGOL exactly.
- [x] Warmup, random, pattern, NPZ, and one-time API initial states work.
- [x] Species and pair laws are cached and artifacts are reproducible.
- [x] Species, pair, local-rule, recurrence, and novelty metrics are recorded.
- [x] Alpha sweep, sensitivity, parallel, and native-Life search runners exist.
- [x] Scalar and batched replay parity is tested.
- [x] High-dimensional parallel scale probing and encoding audits are available.

Validation from the repository root:

```bash
python -m pytest -q
ruff check .
ruff format --check src/emergent/experiments/morphology_interactions tests/experiments
python -m compileall -q src tests
```
