import pytest

from emergent.core3d.rules import (
    Rule3D,
    format_rule_3d,
    int_to_rule_3d,
    parse_rule_3d,
    rule_to_int_3d,
    rule_to_masks_3d,
)


def test_parse_and_format_3d_rule() -> None:
    rule = parse_rule_3d(" B6/S567 ")
    assert format_rule_3d(rule) == "B6/S5,6,7"
    assert rule.birth[6]
    assert rule.survival[5:8] == (True, True, True)
    birth, survival = rule_to_masks_3d(rule)
    assert birth.shape == (27,)
    assert survival.shape == (27,)


def test_parse_canonical_and_legacy_3d_notation() -> None:
    assert format_rule_3d(parse_rule_3d("B6/S5,6,7")) == "B6/S5,6,7"
    assert format_rule_3d(parse_rule_3d("B/S")) == "B/S"
    assert format_rule_3d(parse_rule_3d("B1/S2,3")) == "B1/S2,3"
    assert format_rule_3d(parse_rule_3d("B10/S10")) == "B10/S10"
    rule = parse_rule_3d("B6,9,10/S4,5,6")
    assert format_rule_3d(rule) == "B6,9,10/S4,5,6"
    assert format_rule_3d(parse_rule_3d("B10,11/S9,10,11")) == "B10,11/S9,10,11"


def test_compact_3d_sequences_do_not_use_adjacent_digit_heuristics() -> None:
    assert format_rule_3d(parse_rule_3d("B1/S234")) == "B1/S2,3,4"
    with pytest.raises(ValueError):
        parse_rule_3d("B1011/S")
    with pytest.raises(ValueError):
        parse_rule_3d("B27/S")


@pytest.mark.parametrize("text", ["B27,28/S", "B1,1/S", "B1,,2/S", "B/S0,27", "life"])
def test_invalid_3d_rules_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        parse_rule_3d(text)


def test_3d_rule_integer_round_trip() -> None:
    values = [0, 1, 2**27 - 1, 2**54 - 1, 0x123456789ABC]
    for value in values:
        assert rule_to_int_3d(int_to_rule_3d(value)) == value


def test_3d_rule_masks_are_distinct_from_2d_length() -> None:
    with pytest.raises(ValueError):
        Rule3D((True,) * 9, (False,) * 9)
