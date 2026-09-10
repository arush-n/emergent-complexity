# Design and scientific protocol

This document is the detailed design record for the standalone
`morphology_interactions` experiment. The package-level [README](../README.md)
is the quick-start guide; this page explains the invariants that make results
interpretable.

## Scope and isolation

The experiment is an adapter around the repository's native Conway's Game of
Life implementation. It imports `parse_rule`, `rule_to_masks`,
`neighbor_count_jit`, and `step_jit` from `emergent.core`. It does not modify
the core step, server, frontend, or normal experiment behavior.

The required public module names remain at the package level so existing
commands and imports stay stable:

```text
components.py       toroidal component detection
canonical.py        exact morphology identity
species.py          recurrence registry and encoding cache
encoding.py         fixed-dimensional morphology vectors
interaction.py      deterministic pair laws and zones
local_step.py       one-grid and batched local-rule transitions
engine.py           scalar execution order and state
metrics.py          measurements and novelty rates
persistence.py      reproducible artifact bundles
native_api.py       one-time native-server state import
run.py              single-run CLI
sweep.py            alpha/seed CLI
sensitivity.py      one-cell morphology derivative CLI
parallel.py         lockstep multi-environment execution
replicator_search.py deterministic native-Life self-replication probe
```

Research notes and performance probes are kept in the scoped subfolders:

```text
docs/design.md
docs/benchmarking.md
docs/replicator_search.md
benchmarks/short.py
```

## State transition

At source state `X_t`, the engine performs detection, identity, interaction,
zone construction, conflict resolution, and one JAX transition in that order:

```text
X_t
  -> toroidal components
  -> canonical ShapeKeys
  -> registry/encodings
  -> nearby component pairs
  -> cached pair vectors/rules
  -> owner_map and pair tables
  -> one local-step JAX call
  -> X_(t+1)
```

The default rule is native `B3/S23` everywhere. An interaction only replaces
the local birth/survival lookup inside its resolved owner zone. Neighbor counts
are computed once, and the grid is advanced once per generation.

## Identity contract

An organism is one toroidal 8-connected component under the Moore neighborhood.
The exact identity is a canonical binary matrix packed into:

```python
ShapeKey(height, width, packed)
```

Canonicalization removes translation and, by default, identifies the four
rotations while retaining mirror chirality. Identity is not the lossy vector
used by the interaction law. A reappearing key obtains the same species record
and pair-cache entry.

For identity audits, `exact_identity_vector()` is an injective,
variable-length representation containing the canonical dimensions and all
canonical bits; its length grows with bounding-box area. The encoder also
exposes `encode_scaled()`, an explicit size-aware view whose norm scales as
`cell_count ** exponent` (square-root scaling by default). Ordinary `encode()`
remains unit-normalized for controlled alpha comparisons. Neither
fixed-dimensional view replaces `ShapeKey` as the exact identity.

V1 identity is instantaneous. A glider's phases can therefore be different
species. When two components become Moore-adjacent they become one component;
pair effects are consequently encounter effects around distinct nearby shapes,
not persistent identities after merger.

## Interaction law

The universe creates fixed Fourier frequencies, phases, and symmetric matrices
from one seed. For canonical shape encodings `z_A` and `z_B`:

```text
u_k(A,B) = tanh(beta / sqrt(d) * z_A.T @ W_k @ z_B)
q(A,B)   = deterministic BLAKE2b pair/channel values in [-1, 1]
v_alpha = (1 - alpha) * u + alpha * q
```

`STRUCTURED_WEIGHT_SCALE` is a fixed, documented calibration constant. It
aligns the representative structured and scrambled marginal mean magnitudes
and default-threshold activation rates. It is not estimated from a run or
adapted to a pair. The calibration targets the default `beta=1`; changing
`beta` is intentionally a strength change and should not be presented as a
pure alpha comparison.

The pair vector has 18 channels: nine birth and nine survival decisions. At
most `max_rule_changes` channels whose magnitude exceeds the threshold and
disagrees with base CGOL are selected. This deliberately bounds local rule
changes for V1. The continuous pair vector remains in artifacts and is not
reduced to the rule ID for identity or recurrence analysis.

Only component pairs within toroidal Chebyshev distance `interaction_radius`
are considered. Their local encounter cells are dilated by `effect_padding`.
Overlapping zones select the largest mean absolute pair-vector strength; exact
ties use canonical pair order. No random choice occurs during this resolution.

## Scientific controls

For an alpha sweep, create one deterministic initial grid per seed and reuse it
for all conditions:

```text
control       interactions_enabled=False
structured    alpha=0.00
mixed         alpha=0.50
scrambled     alpha=1.00
```

Keep world dimensions, density, base rule, warmup, duration, detector cadence,
rule-change cap, and threshold fixed. Compare at least:

- species discovery rate;
- new pair discovery rate;
- new effective-rule discovery rate;
- recurrence and component-size distributions;
- active interaction counts and strength distributions; and
- final population/activity summaries.

The local-rule projection has only 172 possible subsets of at most two changed
bits over 18 channels. It is therefore a bounded behavioral projection, not a
claim that the effective world has one million distinct local rules. The full
18-dimensional vectors and pair identities should be analyzed separately.

## Determinism contract

Replay requires the same initial grid, universe seed, resolved configuration,
software versions, and native numerical backend. Simulation behavior must not
depend on wall-clock time, Python's randomized `hash()`, or unseeded random
draws. Timestamps in manifests are metadata only.

The most important gates are:

1. no active zones equals repeated native `step_jit` bit-for-bit;
2. translating or rotating a shape gives the configured same key;
3. repeated and reversed pair queries produce byte-identical laws;
4. overlap ownership is strength-then-key deterministic; and
5. complete replay produces the same final grid, discovery order, caches, and
   sparse snapshots.

## Version 1 boundary

The first scientific question is what instantaneous morphology-dependent local
physics does. Temporal organism identity, learned encoders, direct operator
effects, movement commands, reproduction tracking, and 3D extensions are out
of scope for this version. Adding any of them would require a separately named
effect or identity mode and a new control protocol.
