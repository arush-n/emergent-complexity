# Emergent Complexity Cellular Automata Lab

This repository is a small, reproducible cellular-automata laboratory. It provides a JAX simulation engine for Conway's Game of Life, arbitrary binary outer-totalistic `B/S` rules, keyed random initialization and rule generation, batched execution, trajectories, a thin FastAPI server, and a browser interface with 2D, 3D, comparison, and experiment modes.

It deliberately does not classify rules or implement research analysis, machine learning, complexity metrics, or automated searches.

## Install

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The default dependency is CPU-capable JAX. Installing an accelerator-specific JAX build later is optional; the code does not assume CUDA or any particular device.

## Run the simulator

```bash
emergent-server
```

or:

```bash
python -m emergent.server.app
```

Open <http://127.0.0.1:8000>. The first JAX operation may take a moment while the shape-specific computation is compiled.

The site is a client of the Python API. Drawing, rule editing, playback, reset, randomization, and export all go through the backend; the cellular-automaton transition itself remains in JAX. Three.js/WebGL renders living 3D voxels with one `InstancedMesh`, while exact cross-sections are requested separately from the server.

The Evolution Trace panel is a lightweight diagnostic view, not a research
classifier. It plots the live-cell fraction and the fraction of cells changed
by the last generation returned by the backend. The graph keeps the most recent
180 points in the browser and resets when the simulation is rewound or resized.
It also separates each transition into births and deaths. Playback batches the
work between browser renders, so the UI speed control supports up to 600
generations per second without making one HTTP request per generation.

## Deploy online

The repository includes a CPU Docker deployment and a `render.yaml` Blueprint.
The container serves both the FastAPI backend and the static frontend, so the
browser still talks to the same-origin `/api` endpoints.

Run the same image locally:

```bash
docker build -t emergent-complexity .
docker run --rm -p 8000:8000 emergent-complexity
```

For a public Render deployment, push the repository to a Git provider, create a
new Render Blueprint from that repository, and use the included `render.yaml`.
It creates one Docker web service with `/api/health` as its health check. No
database or external service is required; sessions remain in process memory.

## Test and benchmark

```bash
pytest
python benchmarks/benchmark_step.py
python benchmarks/benchmark_3d.py --max-size 128 --steps 20
```

The benchmarks warm the JIT and call `block_until_ready()` before timing. The 3D benchmark separates compile time from steady-state milliseconds per generation, increases sizes gradually, and writes CSV/JSON results under `artifacts/benchmarks/`. Use `--max-memory-gb` to set a conservative allocation limit.

For reproducible raw 3D rule sweeps:

```bash
python -m emergent.experiments.random_3d \
  --rules 100 --initial-conditions 20 --size 64 --steps 500 \
  --density 0.10 --seed 42
```

For an explicitly configured 2D/3D comparison:

```bash
python -m emergent.experiments.compare_dimensions \
  --rules 1 --initial-conditions 20 --size-2d 64 --size-3d 32 \
  --rule-2d B3/S23 --rule-3d B6/S5,6,7 --steps 500 --seed 42
```

## Architecture

```text
src/emergent/core/        pure JAX 2D grids, rules, transitions, scans, PRNG, measurements
src/emergent/core3d/      pure JAX 3D grids, 26-neighbor rules, scans, batching, PRNG
src/emergent/experiments/ small sampling, raw rule sweeps, and dimension comparisons
src/emergent/io/          2D/3D patterns plus JSON/NPY/NPZ persistence
src/emergent/server/      FastAPI models, separate 2D/3D in-memory sessions, HTTP adapter
frontend/                 static HTML/CSS/JS UI, SVG diagnostics, Canvas, and WebGL voxel view
tests/                    numerical, persistence, batching, and API tests
examples/                 importable Python usage examples
benchmarks/               warm-JIT timing script
```

The `emergent.rules`, `emergent.random`, and `emergent.simulate` modules re-export the 2D API so compact examples remain convenient. The 3D API is available from `emergent.core3d`, `emergent.three_d`, or the package root with explicit `_3d` names. The canonical implementations are under `emergent.core` and `emergent.core3d`.

## Rule representation

Rules are parsed outside compiled functions:

```python
from emergent.core.rules import parse_rule, rule_to_masks

rule = parse_rule("B3/S23")
birth, survival = rule_to_masks(rule)
```

Each mask has nine `uint8` entries, indexed by live-neighbor count `0..8`. A dead cell uses the birth mask; a live cell uses the survival mask. The transition uses the eight-cell Moore neighborhood and excludes the center cell.

The default boundary is toroidal (wraparound), implemented with `jax.numpy.roll`. This avoids special edge cases and is part of the simulator semantics.

Every rule can also be encoded as an 18-bit integer. Birth bits occupy positions `0..8`; survival bits occupy `9..17`:

```python
from emergent.core.rules import int_to_rule, rule_to_int

rule_id = rule_to_int(rule)
same_rule = int_to_rule(rule_id)
```

Presets are available as `CONWAY = "B3/S23"` and `HIGHLIFE = "B36/S23"`.

## 3D rules and API

3D grids use shape `(depth, height, width)`, uint8 state values, toroidal boundaries, and all 26 cells in the surrounding 3 x 3 x 3 cube. A `Rule3D` has two masks of length 27. Counts 0 through 9 may use compact notation for convenience, but the formatter always emits unambiguous comma-separated counts:

```python
import jax

from emergent.three_d import (
    generate_trajectory_3d,
    parse_rule_3d,
    random_grid_3d,
)

rule = parse_rule_3d("B6/S5,6,7")
grid = random_grid_3d(jax.random.key(42), 32, 32, 32, 0.04)
trajectory = generate_trajectory_3d(grid, rule, steps=100, record_every=10)
print(trajectory.shape)  # (11, 32, 32, 32)
```

For only the endpoint, use `run_steps_3d`. For many initial states, use `run_steps_batch_3d` with `(batch, depth, height, width)` input. `generate_batched_trajectory_3d` returns `(batch, time, depth, height, width)` and supports `record_every` so full 3D histories are only retained when explicitly requested. `simulate_rules_3d` returns `(rule, batch, depth, height, width)` final states.

3D rules have 54 independent decisions and can be encoded with `rule_to_int_3d`/`int_to_rule_3d` in `[0, 2**54)`. `random_rule_3d` and `random_rules_3d` use explicit JAX keys and never enumerate that space.

The 3D Python persistence helpers are `save_state_3d`/`load_state_3d` for complete JSON or NPZ states, and `save_grid_3d`/`load_grid_3d` for standalone arrays. The browser's 3D export is intentionally coordinate-based and capped for rendering safety; it does not replace the full Python NPZ format.

## Website workflow

The top navigation switches between:

- `2D Life`: draw cells on Canvas, run Conway/HighLife/arbitrary rules, inspect population/activity/birth/death traces, and save/load JSON.
- `3D Life`: inspect a JAX-backed voxel state with orbit/pan/zoom camera controls, bounds, front/side/top views, a 27-count rule editor, exact X/Y/Z slice view, deterministic randomization, fixed-point runs, capture, and configuration-link copying.
- `Compare`: advance independent 2D and 3D sessions side by side with shared seed/density controls and separately chosen rules.
- `Experiments`: run small raw random-rule batches, filter/sort results, export CSV, and open a selected rule in the relevant simulator.

The 3D browser view loads Three.js from a pinned public CDN module. The Python numerical engine and all experiment runners work without the website.

## Python API

```python
import jax

from emergent.random import random_grid
from emergent.rules import parse_rule
from emergent.simulate import generate_trajectory

key = jax.random.key(42)
rule = parse_rule("B3/S23")
grid = random_grid(key, height=128, width=128, density=0.20)
trajectory = generate_trajectory(grid, rule, steps=500)
print(trajectory.shape)  # (501, 128, 128)
```

Random functions receive explicit keys. To create 100 independent initial states:

```python
import jax

from emergent.random import generate_random_grids
from emergent.simulate import generate_batched_trajectory

keys = jax.random.split(jax.random.key(42), 100)
initial_grids = generate_random_grids(keys, 128, 128, 0.20)
trajectories = generate_batched_trajectory(
    initial_grids,
    parse_rule("B3/S23"),
    steps=500,
)
```

Batch layouts are consistent: grids are `(batch, height, width)` and trajectories are `(batch, time, height, width)`. A single trajectory is `(time, height, width)` and includes the initial frame at index zero. `run_steps` returns only the final state. `simulate_rules(rules, initial_grids, steps)` returns final states shaped `(rule, batch, height, width)`.

For parallel runs that should stop when they reach a fixed point:

```python
from emergent.simulate import run_batched_until_stable

final_grids, steps_taken, settled = run_batched_until_stable(
    initial_grids,
    parse_rule("B3/S23"),
    max_steps=10_000,
)
```

`steps_taken` and `settled` are per-grid arrays. A grid that oscillates or
continues changing is bounded by `max_steps`; periodic behavior is not silently
called an endpoint. For per-generation UI or experiment diagnostics,
`run_steps_with_metrics` returns rows shaped `(steps, 4)` containing
`[alive, changed, births, deaths]`.

The equivalent runnable example is `python examples/batch_until_stable.py`.

The core transition is functional and can be used directly:

```python
from emergent.core.step import batched_step, step_jit

birth, survival = rule_to_masks(rule)
next_grid = step_jit(grid, birth, survival)
next_batch = batched_step(initial_grids, birth, survival)
```

## Persistence and patterns

`save_state`/`load_state` support complete `.json` and metadata-bearing `.npz` states. `save_grid`/`load_grid` support standalone `.npy` and `.npz` grids. JSON is also the browser export format. `block`, `blinker`, and `glider` are available from `emergent.io.patterns`; `place_pattern` returns a new grid and never mutates its input.

## Known limitations

The server uses in-memory single-process session stores. It has no authentication, database, distributed execution, or multi-user coordination. 3D browser rendering transmits living coordinates and explicitly samples only the visualization when a configurable cap is exceeded; JAX continues simulating the complete grid. A fresh shape intentionally creates a new session and may trigger a shape-specific JAX compilation. The browser's 3D export is compact coordinate metadata; use Python NPZ/JSON persistence for a complete dense 3D state.
