import pytest

from src.market_maker.ml.signal import (
    TradingSignal,
    generate_signal,
)


def test_buy_signal():
    signal = generate_signal(
        down_probability=0.10,
        flat_probability=0.20,
        up_probability=0.70,
    )

    assert signal == TradingSignal(
        action="BUY",
        confidence=0.70,
    )


def test_sell_signal():
    signal = generate_signal(
        down_probability=0.70,
        flat_probability=0.20,
        up_probability=0.10,
    )

    assert signal == TradingSignal(
        action="SELL",
        confidence=0.70,
    )


def test_hold_signal():
    signal = generate_signal(
        down_probability=0.20,
        flat_probability=0.60,
        up_probability=0.20,
    )

    assert signal == TradingSignal(
        action="HOLD",
        confidence=0.60,
    )


def test_custom_threshold():
    signal = generate_signal(
        down_probability=0.10,
        flat_probability=0.20,
        up_probability=0.50,
        threshold=0.50,
    )

    assert signal.action == "BUY"
    assert signal.confidence == 0.50


def test_invalid_probability():
    with pytest.raises(ValueError):
        generate_signal(
            down_probability=-0.1,
            flat_probability=0.5,
            up_probability=0.6,
        )


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError):
        generate_signal(
            down_probability=0.2,
            flat_probability=0.2,
            up_probability=0.2,
        )


def test_invalid_threshold():
    with pytest.raises(ValueError):
        generate_signal(
            down_probability=0.2,
            flat_probability=0.2,
            up_probability=0.6,
            threshold=0.0,
        )