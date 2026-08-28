import pytest
from src.market_maker.ml.signal import TradingSignal
from src.market_maker.strategy.engine import (
    StrategyDecision,
    make_strategy_decision,
)


def test_hold_zero_inventory():
    signal = TradingSignal(
        action="HOLD",
        confidence=0.8,
    )

    decision = make_strategy_decision(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
        position=0.0,
        max_position=100.0,
    )

    assert decision.quote.bid == pytest.approx(99.0)
    assert decision.quote.ask == pytest.approx(101.0)
    assert decision.position == 0.0
    assert decision.signal == signal


def test_buy_signal_shifts_quotes_up():
    signal = TradingSignal(
        action="BUY",
        confidence=0.8,
    )

    decision = make_strategy_decision(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
        position=0.0,
        max_position=100.0,
    )

    assert decision.quote.bid == pytest.approx(99.4)
    assert decision.quote.ask == pytest.approx(101.4)


def test_sell_signal_shifts_quotes_down():
    signal = TradingSignal(
        action="SELL",
        confidence=0.8,
    )

    decision = make_strategy_decision(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
        position=0.0,
        max_position=100.0,
    )

    assert decision.quote.bid == pytest.approx(98.6)
    assert decision.quote.ask == pytest.approx(100.6)


def test_long_inventory_shifts_quotes_down():
    signal = TradingSignal(
        action="HOLD",
        confidence=0.8,
    )

    decision = make_strategy_decision(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
        position=50.0,
        max_position=100.0,
    )

    assert decision.quote.bid == pytest.approx(98.5)
    assert decision.quote.ask == pytest.approx(100.5)


def test_short_inventory_shifts_quotes_up():
    signal = TradingSignal(
        action="HOLD",
        confidence=0.8,
    )

    decision = make_strategy_decision(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
        position=-50.0,
        max_position=100.0,
    )

    assert decision.quote.bid == pytest.approx(99.5)
    assert decision.quote.ask == pytest.approx(101.5)


def test_signal_and_inventory_combine():
    signal = TradingSignal(
        action="BUY",
        confidence=0.8,
    )

    decision = make_strategy_decision(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
        position=50.0,
        max_position=100.0,
    )

    assert decision.quote.bid == pytest.approx(98.9)
    assert decision.quote.ask == pytest.approx(100.9)


def test_position_limit():
    signal = TradingSignal(
        action="HOLD",
        confidence=0.8,
    )

    with pytest.raises(ValueError):
        make_strategy_decision(
            mid_price=100.0,
            spread=2.0,
            signal=signal,
            position=101.0,
            max_position=100.0,
        )


def test_invalid_max_position():
    signal = TradingSignal(
        action="HOLD",
        confidence=0.8,
    )

    with pytest.raises(ValueError):
        make_strategy_decision(
            mid_price=100.0,
            spread=2.0,
            signal=signal,
            position=0.0,
            max_position=0.0,
        )


def test_decision_is_immutable():
    signal = TradingSignal(
        action="HOLD",
        confidence=0.8,
    )

    decision = make_strategy_decision(
        mid_price=100.0,
        spread=2.0,
        signal=signal,
        position=0.0,
        max_position=100.0,
    )

    with pytest.raises(AttributeError):
        decision.position = 10.0