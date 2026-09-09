"""Compatibility exports for the compact ``emergent.rules`` API."""

from .core.rules import (
    CONWAY,
    HIGHLIFE,
    Rule,
    format_rule,
    int_to_rule,
    masks_to_rule,
    parse_rule,
    rule_to_int,
    rule_to_masks,
    validate_rule,
)

__all__ = [
    "CONWAY",
    "HIGHLIFE",
    "Rule",
    "format_rule",
    "int_to_rule",
    "masks_to_rule",
    "parse_rule",
    "rule_to_int",
    "rule_to_masks",
    "validate_rule",
]
