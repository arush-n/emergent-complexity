"""Optional offline validation against ViennaRNA Python bindings."""

from __future__ import annotations

from typing import Any

from .alphabet import bases_to_string


def viennarna_available() -> bool:
    """Return whether the optional ``RNA`` module can be imported."""

    try:
        import RNA  # noqa: F401
    except ImportError:
        return False
    return True


def _result_energy(result: Any) -> tuple[str, float]:
    if hasattr(result, "structure") and hasattr(result, "energy"):
        return str(result.structure), float(result.energy)
    if isinstance(result, tuple) and len(result) >= 2:
        return str(result[0]), float(result[1])
    raise ValueError("ViennaRNA returned an unsupported result object")


def reference_duplex_energy(sequence_a: Any, sequence_b: Any) -> float:
    """Return ViennaRNA's duplex energy for two A/C/G/U sequences."""

    if not viennarna_available():
        raise RuntimeError("ViennaRNA Python bindings are not installed")
    import RNA

    result = RNA.duplexfold(bases_to_string(sequence_a), bases_to_string(sequence_b))
    return _result_energy(result)[1]


def reference_duplex(sequence_a: Any, sequence_b: Any) -> tuple[str, float]:
    """Return ViennaRNA duplex structure and energy."""

    if not viennarna_available():
        raise RuntimeError("ViennaRNA Python bindings are not installed")
    import RNA

    result = RNA.duplexfold(bases_to_string(sequence_a), bases_to_string(sequence_b))
    return _result_energy(result)


def reference_fold(sequence: Any) -> tuple[str, float]:
    """Return ViennaRNA minimum-free-energy fold and energy."""

    if not viennarna_available():
        raise RuntimeError("ViennaRNA Python bindings are not installed")
    import RNA

    result = RNA.fold(bases_to_string(sequence))
    return _result_energy(result)
