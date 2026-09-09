# Benchmarks

The benchmark scripts separate JAX compilation from warm execution and call
`block_until_ready()` before timing. The 3D size list grows in 64-cell steps
past the standard candidates until `--max-size`, while the memory guard keeps
the probe conservative.

```bash
python benchmarks/benchmark_3d.py --max-size 512 --steps 20
```

The end-to-end benchmark can measure simulation, binary render extraction,
binary serialization, metadata JSON encoding, and browser interaction
separately.
