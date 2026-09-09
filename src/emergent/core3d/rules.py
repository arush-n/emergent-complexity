"""Binary outer-totalistic rules for a 26-neighbor Moore neighborhood."""

from __future__ import annotations

import operator
import re
from collections.abc import Iterable
from dataclasses import dataclass

import jax.numpy as jnp

DEFAULT_3D_RULE = "B6/S5,6,7"
_RULE_PATTERN = re.compile(r"^B([^/]*)\s*/\s*S(.*)$", re.IGNORECASE)
_MAX_COUNT = 26
_MASK_LENGTH = _MAX_COUNT + 1
_RULE_BITS = _MASK_LENGTH * 2


def _normalise_mask(values: Iterable[bool], name: str) -> tuple[bool, ...]:
    values_tuple = tuple(values)
    if len(values_tuple) != _MASK_LENGTH:
        raise ValueError(f"{name} must contain exactly 27 entries for counts 0 through 26")
    result: list[bool] = []
    for value in values_tuple:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{name} entries must be boolean values")
        try:
            numeric = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} entries must be boolean values") from exc
        if numeric not in (0, 1) or value != numeric:
            raise ValueError(f"{name} entries must be boolean values")
        result.append(bool(numeric))
    return tuple(result)


@dataclass(frozen=True)
class Rule3D:
    """A binary 3D outer-totalistic rule with two length-27 masks."""

    birth: tuple[bool, ...]
    survival: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "birth", _normalise_mask(self.birth, "birth"))
        object.__setattr__(self, "survival", _normalise_mask(self.survival, "survival"))


def validate_rule_3d(rule: Rule3D) -> bool:
    """Validate a 3D rule and return ``True``."""

    if not isinstance(rule, Rule3D):
        raise TypeError("rule must be a Rule3D instance")
    _normalise_mask(rule.birth, "birth")
    _normalise_mask(rule.survival, "survival")
    return True


def _parse_counts(text: str, section: str) -> set[int]:
    if not text:
        return set()

    if "," in text:
        tokens = [token.strip() for token in text.split(",")]
        if any(not token or not token.isdigit() for token in tokens):
            raise ValueError(f"{section} counts must be comma-separated integers from 0 to 26")
    else:
        # Legacy compact notation is intentionally narrow: one digit means a
        # single-digit count, while an entire two-digit section may represent
        # one count from 10 through 26. Longer sections remain a sequence of
        # single-digit counts. There is no heuristic scan of adjacent digits.
        if not text.isdigit():
            raise ValueError(f"{section} counts must be comma-separated integers from 0 to 26")
        if len(text) == 2:
            value = int(text)
            if 10 <= value <= _MAX_COUNT:
                tokens = [text]
            else:
                raise ValueError(
                    f"{section} two-digit counts must be between 10 and 26 or comma-separated"
                )
        else:
            tokens = list(text)

    counts = [int(token) for token in tokens]
    if len(set(counts)) != len(counts):
        raise ValueError(f"{section} counts must not be repeated")
    if any(count < 0 or count > _MAX_COUNT for count in counts):
        raise ValueError(f"{section} counts must be between 0 and 26")
    return set(counts)


def parse_rule_3d(text: str) -> Rule3D:
    """Parse a 3D rule such as ``B6/S5,6,7``.

    Comma-separated notation is canonical. For compatibility, compact
    single-digit notation such as ``B6/S567`` and a two-digit section such as
    ``B10/S10`` are also accepted.
    """

    if not isinstance(text, str):
        raise TypeError("rule text must be a string")
    match = _RULE_PATTERN.fullmatch(text.strip())
    if match is None:
        raise ValueError("invalid 3D rule; expected B<counts>/S<counts>")
    birth_counts = _parse_counts(match.group(1).strip(), "birth")
    survival_counts = _parse_counts(match.group(2).strip(), "survival")
    return Rule3D(
        tuple(count in birth_counts for count in range(_MASK_LENGTH)),
        tuple(count in survival_counts for count in range(_MASK_LENGTH)),
    )


def format_rule_3d(rule: Rule3D) -> str:
    """Format a rule with unambiguous comma-separated counts."""

    validate_rule_3d(rule)
    birth = ",".join(str(count) for count, enabled in enumerate(rule.birth) if enabled)
    survival = ",".join(str(count) for count, enabled in enumerate(rule.survival) if enabled)
    return f"B{birth}/S{survival}"


def rule_to_masks_3d(rule: Rule3D) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert a 3D rule to JAX-compatible uint8 masks."""

    validate_rule_3d(rule)
    return (
        jnp.asarray(rule.birth, dtype=jnp.uint8),
        jnp.asarray(rule.survival, dtype=jnp.uint8),
    )


def masks_to_rule_3d(birth: Iterable[bool], survival: Iterable[bool]) -> Rule3D:
    """Construct a 3D rule from two 27-entry masks."""

    return Rule3D(tuple(birth), tuple(survival))


def rule_to_int_3d(rule: Rule3D) -> int:
    """Encode a 3D rule as an integer in ``[0, 2**54)``."""

    validate_rule_3d(rule)
    value = 0
    for count, enabled in enumerate(rule.birth):
        value |= int(enabled) << count
    for count, enabled in enumerate(rule.survival):
        value |= int(enabled) << (_MASK_LENGTH + count)
    return value


def int_to_rule_3d(value: int) -> Rule3D:
    """Decode a 54-bit integer into a ``Rule3D``."""

    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError("rule value must be an integer") from exc
    if integer < 0 or integer >= 2**_RULE_BITS:
        raise ValueError("rule value must satisfy 0 <= value < 2**54")
    birth = tuple(bool(integer & (1 << count)) for count in range(_MASK_LENGTH))
    survival = tuple(bool(integer & (1 << (_MASK_LENGTH + count))) for count in range(_MASK_LENGTH))
    return Rule3D(birth, survival)
