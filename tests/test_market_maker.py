import pytest

from src.market_maker.ml.signal import TradingSignal
from src.market_maker.strategy.market_maker import (
    Quote,
    generate_quotes,
)


def test_hold_generates_symmetric_quotes():
    signal = TradingSignal(
        action="HOLD",
        confidence=0.6,
    )

    quote = generate_quotes(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
    )

    assert quote == Quote(
        bid=99.0,
        ask=101.0,
    )


def test_buy_shifts_quotes_up():
    signal = TradingSignal(
        action="BUY",
        confidence=0.8,
    )

    quote = generate_quotes(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
    )

    assert quote.bid == pytest.approx(99.4)
    assert quote.ask == pytest.approx(101.4)


def test_sell_shifts_quotes_down():
    signal = TradingSignal(
        action="SELL",
        confidence=0.8,
    )

    quote = generate_quotes(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
    )

    assert quote.bid == pytest.approx(98.6)
    assert quote.ask == pytest.approx(100.6)


def test_quotes_preserve_spread():
    signal = TradingSignal(
        action="BUY",
        confidence=0.7,
    )

    quote = generate_quotes(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
    )

    assert quote.ask - quote.bid == pytest.approx(2.0)


def test_negative_mid_price():
    signal = TradingSignal(
        action="HOLD",
        confidence=0.6,
    )

    with pytest.raises(ValueError):
        generate_quotes(
            mid_price=-100.0,
            spread=2.0,
            signal=signal,
        )


def test_negative_spread():
    signal = TradingSignal(
        action="HOLD",
        confidence=0.6,
    )

    with pytest.raises(ValueError):
        generate_quotes(
            mid_price=100.0,
            spread=-1.0,
            signal=signal,
        )


def test_negative_shift_factor():
    signal = TradingSignal(
        action="HOLD",
        confidence=0.6,
    )

    with pytest.raises(ValueError):
        generate_quotes(
            mid_price=100.0,
            spread=2.0,
            signal=signal,
            shift_factor=-0.1,
        )


def test_unknown_signal():
    signal = TradingSignal(
        action="INVALID",
        confidence=0.6,
    )

    with pytest.raises(ValueError):
        generate_quotes(
            mid_price=100.0,
            spread=2.0,
            signal=signal,
        )

def test_buy_shift_scales_with_confidence():
    low_confidence = TradingSignal(
        action="BUY",
        confidence=0.5,
    )

    high_confidence = TradingSignal(
        action="BUY",
        confidence=1.0,
    )

    low_quote = generate_quotes(
        mid_price=100.0,
        spread=2.0,
        signal=low_confidence,
    )

    high_quote = generate_quotes(
        mid_price=100.0,
        spread=2.0,
        signal=high_confidence,
    )

    assert low_quote.bid == pytest.approx(99.25)
    assert low_quote.ask == pytest.approx(101.25)

    assert high_quote.bid == pytest.approx(99.5)
    assert high_quote.ask == pytest.approx(101.5)


def test_sell_shift_scales_with_confidence():
    low_confidence = TradingSignal(
        action="SELL",
        confidence=0.5,
    )

    high_confidence = TradingSignal(
        action="SELL",
        confidence=1.0,
    )

    low_quote = generate_quotes(
        mid_price=100.0,
        spread=2.0,
        signal=low_confidence,
    )

    high_quote = generate_quotes(
        mid_price=100.0,
        spread=2.0,
        signal=high_confidence,
    )

    assert low_quote.bid == pytest.approx(98.75)
    assert low_quote.ask == pytest.approx(100.75)

    assert high_quote.bid == pytest.approx(98.5)
    assert high_quote.ask == pytest.approx(100.5)


def test_invalid_confidence():
    signal = TradingSignal(
        action="BUY",
        confidence=1.5,
    )

    with pytest.raises(ValueError):
        generate_quotes(
            mid_price=100.0,
            spread=2.0,
            signal=signal,
        )

# def test_buy_shift_scales_with_confidence():
#     low_confidence = TradingSignal(
#         action="BUY",
#         confidence=0.5,
#     )

#     high_confidence = TradingSignal(
#         action="BUY",
#         confidence=1.0,
#     )

#     low_quote = generate_quotes(
#         mid_price=100.0,
#         spread=2.0,
#         signal=low_confidence,
#         confidence_scaling=True,
#     )

#     high_quote = generate_quotes(
#         mid_price=100.0,
#         spread=2.0,
#         signal=high_confidence,
#         confidence_scaling=True,
#     )

#     assert low_quote.bid == pytest.approx(99.25)
#     assert low_quote.ask == pytest.approx(101.25)

#     assert high_quote.bid == pytest.approx(99.5)
#     assert high_quote.ask == pytest.approx(101.5)


# def test_sell_shift_scales_with_confidence():
#     low_confidence = TradingSignal(
#         action="SELL",
#         confidence=0.5,
#     )

#     high_confidence = TradingSignal(
#         action="SELL",
#         confidence=1.0,
#     )

#     low_quote = generate_quotes(
#         mid_price=100.0,
#         spread=2.0,
#         signal=low_confidence,
#         confidence_scaling=True,
#     )

#     high_quote = generate_quotes(
#         mid_price=100.0,
#         spread=2.0,
#         signal=high_confidence,
#         confidence_scaling=True,
#     )

#     assert low_quote.bid == pytest.approx(98.75)
#     assert low_quote.ask == pytest.approx(100.75)

#     assert high_quote.bid == pytest.approx(98.5)
#     assert high_quote.ask == pytest.approx(100.5)        