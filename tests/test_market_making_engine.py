from __future__ import annotations

import pytest

from src.market_maker.backtest.market_making_engine import (
    MarketMakingBacktestEngine,
)
from src.market_maker.ml.signal import generate_signal
from src.market_maker.orderbook.models import Side
from src.market_maker.strategy.engine import (
    StrategyDecision,
)
from src.market_maker.strategy.market_maker import Quote


def make_decision(
    action: str = "HOLD",
    confidence: float = 0.8,
) -> StrategyDecision:
    if action == "BUY":
        signal = generate_signal(
            down_probability=0.1,
            flat_probability=0.1,
            up_probability=0.8,
            threshold=0.6,
        )

    elif action == "SELL":
        signal = generate_signal(
            down_probability=0.8,
            flat_probability=0.1,
            up_probability=0.1,
            threshold=0.6,
        )

    else:
        signal = generate_signal(
            down_probability=0.1,
            flat_probability=0.8,
            up_probability=0.1,
            threshold=0.6,
        )

    return StrategyDecision(
        quote=Quote(
            bid=99.0,
            ask=101.0,
        ),
        signal=signal,
        position=0.0,
    )


def test_initialization() -> None:
    engine = MarketMakingBacktestEngine()

    assert engine.initial_cash == 10_000.0
    assert engine.cash == 10_000.0
    assert engine.position == 0.0
    assert engine.max_position == 100.0
    assert engine.order_size == 1.0


def test_invalid_initial_cash() -> None:
    with pytest.raises(ValueError):
        MarketMakingBacktestEngine(
            initial_cash=0.0,
        )


def test_invalid_max_position() -> None:
    with pytest.raises(ValueError):
        MarketMakingBacktestEngine(
            max_position=0.0,
        )


def test_invalid_order_size() -> None:
    with pytest.raises(ValueError):
        MarketMakingBacktestEngine(
            order_size=0.0,
        )


def test_mark_to_market() -> None:
    engine = MarketMakingBacktestEngine(
        initial_cash=10_000.0,
    )

    engine.cash = 9_900.0
    engine.position = 1.0

    equity = engine.mark_to_market(101.0)

    assert equity == pytest.approx(10_001.0)


def test_buy_quote_can_be_filled() -> None:
    engine = MarketMakingBacktestEngine(
        initial_cash=10_000.0,
        max_position=10.0,
        order_size=1.0,
    )

    decision = make_decision()

    bid_id, ask_id = engine.place_quotes(
        decision=decision,
    )

    assert bid_id != ask_id

    engine.process_market_order(
        side=Side.SELL,
        price=99.0,
        quantity=1.0,
        strategy_order_ids={bid_id, ask_id},
        step=1,
    )

    assert engine.position == pytest.approx(1.0)
    assert engine.cash == pytest.approx(9_901.0)
    assert len(engine.trades) == 1
    assert engine.trades[0].side == "BUY"
    assert engine.trades[0].price == pytest.approx(99.0)


def test_sell_quote_can_be_filled() -> None:
    engine = MarketMakingBacktestEngine(
        initial_cash=10_000.0,
        max_position=10.0,
        order_size=1.0,
    )

    decision = make_decision()

    bid_id, ask_id = engine.place_quotes(
        decision=decision,
    )

    engine.process_market_order(
        side=Side.BUY,
        price=101.0,
        quantity=1.0,
        strategy_order_ids={bid_id, ask_id},
        step=1,
    )

    assert engine.position == pytest.approx(-1.0)
    assert engine.cash == pytest.approx(10_101.0)
    assert len(engine.trades) == 1
    assert engine.trades[0].side == "SELL"
    assert engine.trades[0].price == pytest.approx(101.0)


def test_no_cross_means_no_fill() -> None:
    engine = MarketMakingBacktestEngine(
        initial_cash=10_000.0,
        max_position=10.0,
        order_size=1.0,
    )

    decision = make_decision()

    bid_id, ask_id = engine.place_quotes(
        decision=decision,
    )

    engine.process_market_order(
        side=Side.SELL,
        price=100.0,
        quantity=1.0,
        strategy_order_ids={bid_id, ask_id},
        step=1,
    )

    assert engine.position == pytest.approx(0.0)
    assert engine.cash == pytest.approx(10_000.0)
    assert len(engine.trades) == 0


def test_run_produces_equity_curve() -> None:
    engine = MarketMakingBacktestEngine(
        initial_cash=10_000.0,
        max_position=10.0,
        order_size=1.0,
    )

    decisions = [
        make_decision(),
        make_decision(),
        make_decision(),
    ]

    market_orders = [
        (Side.SELL, 99.0, 1.0),
        (Side.BUY, 101.0, 1.0),
        (Side.SELL, 99.0, 1.0),
    ]

    mid_prices = [
        100.0,
        100.0,
        100.0,
    ]

    result = engine.run(
        decisions=decisions,
        market_orders=market_orders,
        mid_prices=mid_prices,
    )

    assert len(result.equity_curve) == 3
    assert result.initial_cash == 10_000.0
    assert result.final_mid_price == 100.0
    assert result.final_position == pytest.approx(1.0)
    assert result.pnl == pytest.approx(3.0)


def test_run_rejects_mismatched_lengths() -> None:
    engine = MarketMakingBacktestEngine()

    decision = make_decision()

    with pytest.raises(ValueError):
        engine.run(
            decisions=[decision],
            market_orders=[],
            mid_prices=[100.0],
        )


def test_run_rejects_empty_input() -> None:
    engine = MarketMakingBacktestEngine()

    with pytest.raises(ValueError):
        engine.run(
            decisions=[],
            market_orders=[],
            mid_prices=[],
        )