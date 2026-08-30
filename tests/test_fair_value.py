import pytest

from src.market_maker.strategy.fair_value import (
    calculate_fair_value,
)


def test_no_signal_no_inventory():
    fair_value = calculate_fair_value(
        mid_price=100.0,
        directional_score=0.0,
        position=0.0,
        max_position=100.0,
    )

    assert fair_value == pytest.approx(
        100.0
    )


def test_positive_signal_increases_fair_value():
    fair_value = calculate_fair_value(
        mid_price=100.0,
        directional_score=0.5,
        position=0.0,
        max_position=100.0,
        alpha=1.0,
    )

    assert fair_value > 100.0


def test_negative_signal_decreases_fair_value():
    fair_value = calculate_fair_value(
        mid_price=100.0,
        directional_score=-0.5,
        position=0.0,
        max_position=100.0,
        alpha=1.0,
    )

    assert fair_value < 100.0


def test_short_inventory_increases_fair_value():
    fair_value = calculate_fair_value(
        mid_price=100.0,
        directional_score=0.0,
        position=-50.0,
        max_position=100.0,
        inventory_skew_strength=1.0,
    )

    assert fair_value > 100.0


def test_long_inventory_decreases_fair_value():
    fair_value = calculate_fair_value(
        mid_price=100.0,
        directional_score=0.0,
        position=50.0,
        max_position=100.0,
        inventory_skew_strength=1.0,
    )

    assert fair_value < 100.0


def test_invalid_price():
    with pytest.raises(ValueError):
        calculate_fair_value(
            mid_price=0.0,
            directional_score=0.0,
            position=0.0,
            max_position=100.0,
        )