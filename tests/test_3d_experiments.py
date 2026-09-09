from pathlib import Path

from emergent.experiments.compare_dimensions import run_comparison
from emergent.experiments.random_3d import run_random_3d_experiment


def test_random_3d_experiment_writes_reproducible_artifacts(tmp_path: Path) -> None:
    first = run_random_3d_experiment(
        rules=2,
        initial_conditions=2,
        size=4,
        steps=2,
        density=0.2,
        seed=42,
        output_dir=tmp_path / "first",
    )
    second = run_random_3d_experiment(
        rules=2,
        initial_conditions=2,
        size=4,
        steps=2,
        density=0.2,
        seed=42,
        output_dir=tmp_path / "second",
    )
    assert first["rows"] == second["rows"]
    assert (tmp_path / "first" / "manifest.json").exists()
    assert (tmp_path / "first" / "results.csv").exists()


def test_dimension_comparison_writes_raw_rows(tmp_path: Path) -> None:
    result = run_comparison(
        rules=1,
        initial_conditions=2,
        size_2d=4,
        size_3d=4,
        steps=1,
        density=0.1,
        seed=3,
        output_dir=tmp_path,
    )
    assert {row["dimensions"] for row in result["rows"]} == {2, 3}
    assert len(result["rows"]) == 4
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "results.csv").exists()
