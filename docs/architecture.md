# Architecture

The browser is a visualization client. The FastAPI layer owns short-lived
mutable sessions, while the pure JAX modules remain the numerical source of
truth.

```text
Browser
  │  JSON metadata + compact binary render payloads
  ▼
FastAPI session API
  │
  ▼
Pure JAX simulation core
  ├── emergent.core       2D
  └── emergent.core3d     3D
```

2D render responses use little-endian `np.packbits` data. 3D render responses
use unsigned-byte `(depth, height, width)` coordinate triples and are capped
only for visualization. Exact 3D state export remains NPZ.

The backend does not own playback. It stores grid, rule, generation, initial
seed, initial density, and transition measurements. The browser owns speed,
timers, and whether playback is running.
