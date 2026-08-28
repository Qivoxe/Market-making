import pytest

from src.market_maker.strategy.inventory import (
    InventoryState,
    adjust_quotes_for_inventory,
    calculate_inventory_skew,
)


def test_zero_inventory_has_zero_skew():
    assert calculate_inventory_skew(
        position=0.0,
        max_position=100.0,
    ) == pytest.approx(0.0)


def test_positive_inventory_has_positive_skew():
    assert calculate_inventory_skew(
        position=50.0,
        max_position=100.0,
    ) == pytest.approx(0.5)


def test_negative_inventory_has_negative_skew():
    assert calculate_inventory_skew(
        position=-50.0,
        max_position=100.0,
    ) == pytest.approx(-0.5)


def test_position_cannot_exceed_limit():
    with pytest.raises(ValueError):
        calculate_inventory_skew(
            position=101.0,
            max_position=100.0,
        )


def test_invalid_max_position():
    with pytest.raises(ValueError):
        calculate_inventory_skew(
            position=0.0,
            max_position=0.0,
        )


def test_inventory_state():
    state = InventoryState(
        position=25.0,
        max_position=100.0,
    )

    assert state.position == 25.0
    assert state.max_position == 100.0


def test_zero_inventory_preserves_quotes():
    bid, ask = adjust_quotes_for_inventory(
        bid=99.0,
        ask=101.0,
        position=0.0,
        max_position=100.0,
    )

    assert bid == pytest.approx(99.0)
    assert ask == pytest.approx(101.0)


def test_long_position_shifts_quotes_down():
    bid, ask = adjust_quotes_for_inventory(
        bid=99.0,
        ask=101.0,
        position=50.0,
        max_position=100.0,
    )

    assert bid == pytest.approx(98.5)
    assert ask == pytest.approx(100.5)


def test_short_position_shifts_quotes_up():
    bid, ask = adjust_quotes_for_inventory(
        bid=99.0,
        ask=101.0,
        position=-50.0,
        max_position=100.0,
    )

    assert bid == pytest.approx(99.5)
    assert ask == pytest.approx(101.5)


def test_inventory_adjustment_preserves_spread():
    bid, ask = adjust_quotes_for_inventory(
        bid=99.0,
        ask=101.0,
        position=50.0,
        max_position=100.0,
    )

    assert ask - bid == pytest.approx(2.0)


def test_negative_skew_factor():
    with pytest.raises(ValueError):
        adjust_quotes_for_inventory(
            bid=99.0,
            ask=101.0,
            position=50.0,
            max_position=100.0,
            skew_factor=-0.1,
        )


def test_invalid_quotes():
    with pytest.raises(ValueError):
        adjust_quotes_for_inventory(
            bid=101.0,
            ask=99.0,
            position=0.0,
            max_position=100.0,
        )