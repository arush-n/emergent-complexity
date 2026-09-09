from pathlib import Path
from runpy import run_path

benchmark_sizes = run_path(
    str(Path(__file__).parents[1] / "benchmarks" / "benchmark_3d.py")
)["benchmark_sizes"]


def test_benchmark_sizes_extend_to_requested_maximum() -> None:
    assert benchmark_sizes(128) == (16, 32, 48, 64, 96, 128)
    sizes = benchmark_sizes(512)
    assert sizes[-1] == 512
    assert 320 in sizes
    assert 384 in sizes
    assert 448 in sizes
