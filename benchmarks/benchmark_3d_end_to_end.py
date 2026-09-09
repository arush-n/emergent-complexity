"""Measure the non-JAX parts of the interactive 3D pipeline.

This is intentionally separate from ``benchmark_3d.py``. The engine benchmark
answers how quickly JAX advances a volume; this script measures the additional
cost of extracting living coordinates, serializing compact render bytes, and
optionally round-tripping a browser step through the running application.

Examples::

    python benchmarks/benchmark_3d_end_to_end.py --size 64 --steps 10
    python benchmarks/benchmark_3d_end_to_end.py --mode browser --steps 10
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax

from emergent.server.sessions_3d import SessionStore3D


def benchmark_server_pipeline(
    *,
    size: int,
    steps: int,
    repeats: int,
    density: float,
    max_voxels: int,
) -> list[dict[str, Any]]:
    """Measure one server-side step and its render payload separately."""

    store = SessionStore3D()
    session_id = store.create(
        depth=size,
        height=size,
        width=size,
        density=density,
        seed=42,
        rule="B6/S5,6,7",
        session_id="benchmark-3d",
    )
    # Warm the shape-specific JAX compilation before collecting steady-state
    # measurements. The warm-up is still a real server transition.
    store.step(session_id, steps=1)

    rows: list[dict[str, Any]] = []
    for _ in range(repeats):
        step_start = time.perf_counter()
        store.step(session_id, steps=steps)
        payload = store.payload(session_id)
        render_content, render_metadata = store.render_bytes(
            session_id,
            max_voxels=max_voxels,
        )
        metadata_start = time.perf_counter()
        encoded_metadata = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        metadata_ms = (time.perf_counter() - metadata_start) * 1000
        end_to_end_ms = (time.perf_counter() - step_start) * 1000
        rows.append(
            {
                "size": size,
                "shape": f"{size}x{size}x{size}",
                "cells": size**3,
                "steps": steps,
                "jax_device": str(jax.devices()[0]),
                "simulation_ms": payload["simulation_ms"],
                "render_extract_ms": render_metadata["render_extract_ms"],
                "binary_serialization_ms": render_metadata["serialization_ms"],
                "metadata_json_serialization_ms": metadata_ms,
                "end_to_end_ms": end_to_end_ms,
                "generations_per_second": steps / (end_to_end_ms / 1000)
                if end_to_end_ms
                else 0.0,
                "alive": payload["alive"],
                "rendered_voxels": render_metadata["rendered_voxels"],
                "render_sampled": render_metadata["render_sampled"],
                "binary_bytes": len(render_content),
                "metadata_json_bytes": len(encoded_metadata),
            }
        )
    return rows


def benchmark_browser(*, url: str, steps: int, repeats: int) -> list[dict[str, Any]]:
    """Measure warm 3D step requests in a real Chromium browser.

    Playwright is an optional development dependency. The browser must be able
    to reach a running local server, for example ``emergent-server``.
    """

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "browser benchmarking requires the optional Playwright dependency; "
            "install with `python -m pip install -e '.[browser]'`"
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(url, wait_until="domcontentloaded")
        page.locator('[data-mode="3d"]').click()
        page.locator('[id="viewport-3d"] canvas').wait_for(state="visible")
        page.wait_for_function(
            "() => document.getElementById('3d-status-message').textContent.includes('Ready')"
        )
        page.locator('[id="3d-view-menu"] summary').click()
        page.locator('[id="3d-performance-input"]').check()

        page.locator('[id="3d-step-button"]').click()
        page.wait_for_function(
            "() => Number(document.getElementById('3d-generation-value').textContent) >= 1"
        )
        rows: list[dict[str, Any]] = []
        for index in range(repeats):
            before = time.perf_counter()
            for offset in range(steps):
                page.locator('[id="3d-step-button"]').click()
                expected = 2 + index * steps + offset
                page.wait_for_function(
                    "expected => Number(document.getElementById("
                    "'3d-generation-value').textContent) === expected",
                    arg=expected,
                )
            elapsed_ms = (time.perf_counter() - before) * 1000
            overlay = page.locator('[id="3d-performance-overlay"]').inner_text()
            rows.append(
                {
                    "steps": steps,
                    "browser_elapsed_ms": elapsed_ms,
                    "browser_generations_per_second": steps / (elapsed_ms / 1000)
                    if elapsed_ms
                    else 0.0,
                    "overlay": overlay,
                }
            )
        browser.close()
    return rows


def _write_rows(output_dir: Path, prefix: str, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"{prefix}_{stamp}.csv"
    json_path = output_dir / f"{prefix}_{stamp}.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "benchmark": prefix,
                "timestamp": datetime.now(UTC).isoformat(),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("server", "browser", "both"), default="server")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--density", type=float, default=0.04)
    parser.add_argument("--max-voxels", type=int, default=75_000)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks"))
    args = parser.parse_args()
    if min(args.size, args.steps, args.repeats, args.max_voxels) < 1:
        raise SystemExit("size, steps, repeats, and max-voxels must be positive")

    if args.mode in {"server", "both"}:
        rows = benchmark_server_pipeline(
            size=args.size,
            steps=args.steps,
            repeats=args.repeats,
            density=args.density,
            max_voxels=args.max_voxels,
        )
        _write_rows(args.output_dir, "benchmark_3d_server_pipeline", rows)
        mean = sum(float(row["end_to_end_ms"]) for row in rows) / len(rows)
        print(f"server pipeline: {mean:.2f} ms/request over {args.steps} generations")

    if args.mode in {"browser", "both"}:
        rows = benchmark_browser(url=args.url, steps=args.steps, repeats=args.repeats)
        _write_rows(args.output_dir, "benchmark_3d_browser", rows)
        mean = sum(float(row["browser_elapsed_ms"]) for row in rows) / len(rows)
        print(f"browser pipeline: {mean:.2f} ms/request over {args.steps} generations")


if __name__ == "__main__":
    main()
