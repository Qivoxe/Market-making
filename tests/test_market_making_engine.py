from src.market_maker.strategy.market_maker import Quote
from src.market_maker.strategy.engine import StrategyDecision
from src.market_maker.ml.signal import generate_signal
from src.market_maker.backtest.market_making_engine import MarketMakingBacktestEngine
from src.market_maker.orderbook.engine import ExchangeEngine
from src.market_maker.orderbook.models import Side
from src.market_maker.orderbook.models import Side
from src.market_maker.orderbook.engine import ExchangeEngine

def test_market_sell_hits_strategy_bid() -> None:
    exchange = ExchangeEngine()

    engine = MarketMakingBacktestEngine(
        initial_cash=10_000.0,
        max_position=10.0,
        order_size=1.0,
        exchange=exchange,
    )

    decision = make_decision()

    order_ids = engine.place_quotes(decision=decision)

    engine.process_market_order(
        side=Side.SELL,
        price=98.0,
        quantity=1.0,
        step=1,
        strategy_order_ids=order_ids,
    )

    assert engine.position == 1.0
    assert engine.cash == 9_901.0
    assert len(engine.trades) == 1
    assert engine.trades[0].side == "BUY"
    assert engine.trades[0].price == 99.0
    assert engine.trades[0].quantity == 1.0


def test_market_buy_hits_strategy_ask() -> None:
    exchange = ExchangeEngine()

    engine = MarketMakingBacktestEngine(
        initial_cash=10_000.0,
        max_position=10.0,
        order_size=1.0,
        exchange=exchange,
    )

    decision = make_decision()

    order_ids = engine.place_quotes(decision=decision)

    engine.process_market_order(
        side=Side.BUY,
        price=102.0,
        quantity=1.0,
        step=1,
        strategy_order_ids=order_ids,
    )

    assert engine.position == -1.0
    assert engine.cash == 10_101.0
    assert len(engine.trades) == 1
    assert engine.trades[0].side == "SELL"
    assert engine.trades[0].price == 101.0
    assert engine.trades[0].quantity == 1.0


def test_non_crossing_market_order_does_not_fill() -> None:
    exchange = ExchangeEngine()

    engine = MarketMakingBacktestEngine(
        initial_cash=10_000.0,
        max_position=10.0,
        order_size=1.0,
        exchange=exchange,
    )

    decision = make_decision()

    order_ids = engine.place_quotes(decision=decision)

    engine.process_market_order(
        side=Side.SELL,
        price=102.0,
        quantity=1.0,
        step=1,
        strategy_order_ids=order_ids,
    )

    assert engine.position == 0.0
    assert engine.cash == 10_000.0
    assert len(engine.trades) == 0


def test_quote_cancellation_prevents_future_fill() -> None:
    exchange = ExchangeEngine()

    engine = MarketMakingBacktestEngine(
        initial_cash=10_000.0,
        max_position=10.0,
        order_size=1.0,
        exchange=exchange,
    )

    decision = make_decision()

    order_ids = engine.place_quotes(decision=decision)

    engine.cancel_quotes(order_ids)

    engine.process_market_order(
        side=Side.SELL,
        price=98.0,
        quantity=1.0,
        step=1,
        strategy_order_ids=order_ids,
    )

    assert engine.position == 0.0
    assert engine.cash == 10_000.0
    assert len(engine.trades) == 0

def make_decision() -> StrategyDecision:
    signal = generate_signal(
        down_probability=0.1,
        flat_probability=0.8,
        up_probability=0.1,
        threshold=0.6,
    )

    return StrategyDecision(
        quote=Quote(bid=99.0, ask=101.0),
        signal=signal,
        position=0.0,
    )


