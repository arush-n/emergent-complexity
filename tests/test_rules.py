import pytest

from emergent.core.random import key_from_seed, random_rule, random_rules
from emergent.core.rules import (
    CONWAY,
    HIGHLIFE,
    Rule,
    format_rule,
    int_to_rule,
    parse_rule,
    rule_to_int,
    rule_to_masks,
)


def test_parse_and_format_conway() -> None:
    rule = parse_rule(CONWAY)
    assert rule.birth == (False, False, False, True, False, False, False, False, False)
    assert rule.survival == (False, False, True, True, False, False, False, False, False)
    assert format_rule(rule) == CONWAY


def test_parse_highlife_and_empty_sections() -> None:
    assert format_rule(parse_rule(HIGHLIFE)) == HIGHLIFE
    assert format_rule(parse_rule("b2/s")) == "B2/S"
    assert format_rule(parse_rule("B/S1357")) == "B/S1357"


@pytest.mark.parametrize("text", ["B9/S23", "B33/S23", "B3/23", "life", "", "B3/S2x"])
def test_reject_malformed_rules(text: str) -> None:
    with pytest.raises(ValueError):
        parse_rule(text)


def test_rule_masks_and_integer_round_trip() -> None:
    rule = parse_rule("B0248/S1357")
    birth, survival = rule_to_masks(rule)
    assert birth.tolist() == [1, 0, 1, 0, 1, 0, 0, 0, 1]
    assert survival.tolist() == [0, 1, 0, 1, 0, 1, 0, 1, 0]
    assert int_to_rule(rule_to_int(rule)) == rule
    assert int_to_rule(0) == Rule((False,) * 9, (False,) * 9)
    assert rule_to_int(int_to_rule(2**18 - 1)) == 2**18 - 1


def test_integer_range_is_checked() -> None:
    with pytest.raises(ValueError):
        int_to_rule(-1)
    with pytest.raises(ValueError):
        int_to_rule(2**18)


def test_random_rule_generation_is_keyed_and_representable() -> None:
    first = random_rule(key_from_seed(7))
    second = random_rule(key_from_seed(7))
    assert first == second
    assert len(random_rules(key_from_seed(9), 12)) == 12
    for rule in random_rules(key_from_seed(10), 8):
        assert int_to_rule(rule_to_int(rule)) == rule
