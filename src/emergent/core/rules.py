"""Binary outer-totalistic Moore-neighborhood rule representation."""

from __future__ import annotations

import operator
import re
from collections.abc import Iterable
from dataclasses import dataclass

import jax.numpy as jnp

CONWAY = "B3/S23"
HIGHLIFE = "B36/S23"

_RULE_PATTERN = re.compile(r"^B([0-8]*)\s*/\s*S([0-8]*)$", re.IGNORECASE)


def _normalise_mask(values: Iterable[bool], name: str) -> tuple[bool, ...]:
    """Validate and canonicalise one nine-entry rule mask."""

    values_tuple = tuple(values)
    if len(values_tuple) != 9:
        raise ValueError(f"{name} must contain exactly nine entries for counts 0 through 8")

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
class Rule:
    """A binary outer-totalistic rule represented by two length-nine masks.

    ``birth[count]`` determines whether a dead cell is born with ``count``
    live Moore neighbors. ``survival[count]`` determines whether a live cell
    remains alive with that count. Counts are ordered from zero through eight.
    """

    birth: tuple[bool, ...]
    survival: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "birth", _normalise_mask(self.birth, "birth"))
        object.__setattr__(self, "survival", _normalise_mask(self.survival, "survival"))


def validate_rule(rule: Rule) -> bool:
    """Validate ``rule`` and return ``True``.

    Invalid rules raise ``TypeError`` or ``ValueError`` with a message that is
    suitable for configuration and API validation.
    """

    if not isinstance(rule, Rule):
        raise TypeError("rule must be a Rule instance")
    _normalise_mask(rule.birth, "birth")
    _normalise_mask(rule.survival, "survival")
    return True


def parse_rule(text: str) -> Rule:
    """Parse a rule string such as ``B3/S23`` or ``B36/S23``.

    The parser is case-insensitive, accepts whitespace around the slash, and
    permits empty sections such as ``B2/S``. Repeated neighbor counts are
    rejected because they do not add meaning to a rule.
    """

    if not isinstance(text, str):
        raise TypeError("rule text must be a string")
    match = _RULE_PATTERN.fullmatch(text.strip())
    if match is None:
        raise ValueError("invalid rule; expected the form B<counts>/S<counts> with counts 0-8")

    birth_text, survival_text = match.groups()
    if len(set(birth_text)) != len(birth_text):
        raise ValueError("birth counts must not be repeated")
    if len(set(survival_text)) != len(survival_text):
        raise ValueError("survival counts must not be repeated")

    birth_counts = {int(character) for character in birth_text}
    survival_counts = {int(character) for character in survival_text}
    return Rule(
        tuple(count in birth_counts for count in range(9)),
        tuple(count in survival_counts for count in range(9)),
    )


def format_rule(rule: Rule) -> str:
    """Format a rule as its canonical ``B.../S...`` string."""

    validate_rule(rule)
    birth = "".join(str(count) for count, enabled in enumerate(rule.birth) if enabled)
    survival = "".join(str(count) for count, enabled in enumerate(rule.survival) if enabled)
    return f"B{birth}/S{survival}"


def rule_to_masks(rule: Rule) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert a rule into JAX-compatible ``uint8`` birth and survival masks."""

    validate_rule(rule)
    return (
        jnp.asarray(rule.birth, dtype=jnp.uint8),
        jnp.asarray(rule.survival, dtype=jnp.uint8),
    )


def masks_to_rule(birth: Iterable[bool], survival: Iterable[bool]) -> Rule:
    """Construct a ``Rule`` from two nine-entry masks."""

    return Rule(tuple(birth), tuple(survival))


def rule_to_int(rule: Rule) -> int:
    """Encode a rule as an integer in ``[0, 2**18)``.

    Birth bits occupy positions 0 through 8 and survival bits occupy positions
    9 through 17. This order is stable and makes every binary rule uniquely
    representable.
    """

    validate_rule(rule)
    value = 0
    for count, enabled in enumerate(rule.birth):
        value |= int(enabled) << count
    for count, enabled in enumerate(rule.survival):
        value |= int(enabled) << (9 + count)
    return value


def int_to_rule(value: int) -> Rule:
    """Decode an 18-bit integer into a ``Rule``."""

    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError("rule value must be an integer") from exc
    if integer < 0 or integer >= 2**18:
        raise ValueError("rule value must satisfy 0 <= value < 2**18")
    birth = tuple(bool(integer & (1 << count)) for count in range(9))
    survival = tuple(bool(integer & (1 << (9 + count))) for count in range(9))
    return Rule(birth, survival)
