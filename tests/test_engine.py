from src.market_maker.orderbook.engine import ExchangeEngine
from src.market_maker.orderbook.models import Side, OrderStatus


def test_submit_resting_orders():
    engine = ExchangeEngine()

    buy = engine.submit_order(
        Side.BUY,
        100.0,
        50,
    )

    sell = engine.submit_order(
        Side.SELL,
        101.0,
        30,
    )

    assert buy.status == OrderStatus.ACTIVE
    assert sell.status == OrderStatus.ACTIVE

    assert engine.get_best_bid() == 100.0
    assert engine.get_best_ask() == 101.0
    assert engine.get_mid_price() == 100.5
    assert engine.get_spread() == 1.0


def test_full_execution():
    engine = ExchangeEngine()

    buy = engine.submit_order(
        Side.BUY,
        100.0,
        50,
    )

    sell = engine.submit_order(
        Side.SELL,
        100.0,
        50,
    )

    assert buy.status == OrderStatus.FILLED
    assert sell.status == OrderStatus.FILLED

    assert buy.remaining_quantity == 0
    assert sell.remaining_quantity == 0

    assert len(engine.trade_log) == 1

    trade = engine.trade_log[0]

    assert trade.price == 100.0
    assert trade.quantity == 50
    assert trade.buy_order_id == buy.order_id
    assert trade.sell_order_id == sell.order_id


def test_partial_execution():
    engine = ExchangeEngine()

    buy = engine.submit_order(
        Side.BUY,
        100.0,
        100,
    )

    sell = engine.submit_order(
        Side.SELL,
        100.0,
        40,
    )

    assert buy.status == OrderStatus.PARTIALLY_FILLED
    assert buy.remaining_quantity == 60

    assert sell.status == OrderStatus.FILLED
    assert sell.remaining_quantity == 0

    assert engine.get_best_bid() == 100.0


def test_cancel_order():
    engine = ExchangeEngine()

    order = engine.submit_order(
        Side.BUY,
        100.0,
        50,
    )

    assert engine.cancel_order(order.order_id) is True

    assert order.status == OrderStatus.CANCELLED
    assert engine.get_best_bid() is None

    assert engine.cancel_order(order.order_id) is False


def test_no_execution_when_prices_do_not_cross():
    engine = ExchangeEngine()

    buy = engine.submit_order(
        Side.BUY,
        100.0,
        50,
    )

    sell = engine.submit_order(
        Side.SELL,
        101.0,
        50,
    )

    assert buy.status == OrderStatus.ACTIVE
    assert sell.status == OrderStatus.ACTIVE

    assert buy.remaining_quantity == 50
    assert sell.remaining_quantity == 50

    assert len(engine.trade_log) == 0