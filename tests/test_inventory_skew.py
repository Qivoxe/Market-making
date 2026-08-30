import pytest

from src.market_maker.strategy.inventory_skew import (
    calculate_inventory_skew,
)


def test_flat_position_has_no_skew():
    skew = calculate_inventory_skew(
        position=0.0,
        max_position=100.0,
    )

    assert skew == pytest.approx(0.0)


def test_short_position_creates_positive_skew():
    skew = calculate_inventory_skew(
        position=-50.0,
        max_position=100.0,
    )

    assert skew > 0.0


def test_long_position_creates_negative_skew():
    skew = calculate_inventory_skew(
        position=50.0,
        max_position=100.0,
    )

    assert skew < 0.0


def test_max_short_position():
    skew = calculate_inventory_skew(
        position=-100.0,
        max_position=100.0,
        skew_strength=0.25,
    )

    assert skew == pytest.approx(
        0.25
    )


def test_max_long_position():
    skew = calculate_inventory_skew(
        position=100.0,
        max_position=100.0,
        skew_strength=0.25,
    )

    assert skew == pytest.approx(
        -0.25
    )


def test_invalid_max_position():
    with pytest.raises(ValueError):
        calculate_inventory_skew(
            position=0.0,
            max_position=0.0,
        )