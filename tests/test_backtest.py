import pytest

from src.market_maker.backtest.engine import (
    BacktestEngine,
)
from src.market_maker.ml.signal import TradingSignal
from src.market_maker.strategy.engine import (
    StrategyDecision,
)
from src.market_maker.strategy.market_maker import Quote


def decision(
    action: str,
    bid: float = 99.0,
    ask: float = 101.0,
) -> StrategyDecision:
    signal = TradingSignal(
        action=action,
        confidence=0.8,
    )

    return StrategyDecision(
        quote=Quote(
            bid=bid,
            ask=ask,
        ),
        signal=signal,
        position=0.0,
    )


def test_initial_state():
    engine = BacktestEngine(
        initial_cash=10_000.0,
        max_position=100.0,
        order_size=1.0,
    )

    assert engine.cash == pytest.approx(10_000.0)
    assert engine.position == pytest.approx(0.0)
    assert engine.trades == []


def test_buy_execution():
    engine = BacktestEngine(
        initial_cash=10_000.0,
        order_size=1.0,
    )

    executed = engine.execute_bid(
        step=1,
        bid=99.0,
    )

    assert executed is True
    assert engine.position == pytest.approx(1.0)
    assert engine.cash == pytest.approx(9_901.0)
    assert len(engine.trades) == 1

    trade = engine.trades[0]

    assert trade.side == "BUY"
    assert trade.price == pytest.approx(99.0)
    assert trade.quantity == pytest.approx(1.0)


def test_sell_execution():
    engine = BacktestEngine(
        initial_cash=10_000.0,
        order_size=1.0,
    )

    executed = engine.execute_ask(
        step=1,
        ask=101.0,
    )

    assert executed is True
    assert engine.position == pytest.approx(-1.0)
    assert engine.cash == pytest.approx(10_101.0)

    trade = engine.trades[0]

    assert trade.side == "SELL"
    assert trade.price == pytest.approx(101.0)


def test_position_limit():
    engine = BacktestEngine(
        initial_cash=10_000.0,
        max_position=2.0,
        order_size=1.0,
    )

    assert engine.execute_bid(1, 99.0)
    assert engine.execute_bid(2, 99.0)
    assert not engine.execute_bid(3, 99.0)

    assert engine.position == pytest.approx(2.0)


def test_short_position_limit():
    engine = BacktestEngine(
        initial_cash=10_000.0,
        max_position=2.0,
        order_size=1.0,
    )

    assert engine.execute_ask(1, 101.0)
    assert engine.execute_ask(2, 101.0)
    assert not engine.execute_ask(3, 101.0)

    assert engine.position == pytest.approx(-2.0)


def test_mark_to_market():
    engine = BacktestEngine(
        initial_cash=10_000.0,
    )

    engine.execute_bid(
        step=1,
        bid=99.0,
    )

    equity = engine.mark_to_market(
        mid_price=100.0,
    )

    assert equity == pytest.approx(10_001.0)


def test_hold_produces_no_trade():
    engine = BacktestEngine(
        initial_cash=10_000.0,
    )

    result = engine.run(
        decisions=[
            decision("HOLD"),
        ],
        mid_prices=[100.0],
    )

    assert result.final_position == pytest.approx(0.0)
    assert result.final_cash == pytest.approx(
        10_000.0
    )
    assert result.final_equity == pytest.approx(
        10_000.0
    )
    assert result.pnl == pytest.approx(0.0)
    assert len(result.trades) == 0


def test_buy_backtest():
    engine = BacktestEngine(
        initial_cash=10_000.0,
        order_size=1.0,
    )

    result = engine.run(
        decisions=[
            decision("BUY", bid=99.0),
            decision("HOLD"),
        ],
        mid_prices=[
            100.0,
            101.0,
        ],
    )

    assert result.final_position == pytest.approx(1.0)
    assert result.final_cash == pytest.approx(9_901.0)
    assert result.final_equity == pytest.approx(
        10_002.0
    )
    assert result.pnl == pytest.approx(2.0)
    assert len(result.trades) == 1


def test_sell_backtest():
    engine = BacktestEngine(
        initial_cash=10_000.0,
        order_size=1.0,
    )

    result = engine.run(
        decisions=[
            decision("SELL", ask=101.0),
            decision("HOLD"),
        ],
        mid_prices=[
            100.0,
            99.0,
        ],
    )

    assert result.final_position == pytest.approx(-1.0)
    assert result.final_cash == pytest.approx(
        10_101.0
    )
    assert result.final_equity == pytest.approx(
        10_002.0
    )
    assert result.pnl == pytest.approx(2.0)
    assert len(result.trades) == 1


def test_mismatched_lengths():
    engine = BacktestEngine()

    with pytest.raises(ValueError):
        engine.run(
            decisions=[
                decision("BUY"),
            ],
            mid_prices=[
                100.0,
                101.0,
            ],
        )


def test_empty_backtest():
    engine = BacktestEngine()

    with pytest.raises(ValueError):
        engine.run(
            decisions=[],
            mid_prices=[],
        )


def test_invalid_initial_cash():
    with pytest.raises(ValueError):
        BacktestEngine(
            initial_cash=0.0,
        )


def test_invalid_max_position():
    with pytest.raises(ValueError):
        BacktestEngine(
            max_position=0.0,
        )


def test_invalid_order_size():
    with pytest.raises(ValueError):
        BacktestEngine(
            order_size=0.0,
        )


def test_invalid_bid():
    engine = BacktestEngine()

    with pytest.raises(ValueError):
        engine.execute_bid(
            step=1,
            bid=0.0,
        )


def test_invalid_ask():
    engine = BacktestEngine()

    with pytest.raises(ValueError):
        engine.execute_ask(
            step=1,
            ask=0.0,
        )