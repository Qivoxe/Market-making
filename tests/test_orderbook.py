from src.market_maker.orderbook.models import (
    OrderBook,
    Side,
    OrderStatus,

)

def test_add_orders_and_best_price():
    book = OrderBook()

    book.add_order(Side.BUY, 100.0, 50)
    book.add_order(Side.BUY, 99.0, 100)

    book.add_order(Side.SELL, 101.0, 75)
    book.add_order(Side.SELL, 102.0, 100)

    assert book.get_best_bid() == 100.0
    assert book.get_best_ask() == 101.0
    assert book.get_mid_price() == 100.5
    assert book.get_spread() == 1.0
def test_full_match():
    book = OrderBook()

    buy_id = book.add_order(Side.BUY, 100.0, 50)
    sell_id = book.add_order(Side.SELL, 100.0, 50)

    buy_order = book.get_order(buy_id)
    sell_order = book.get_order(sell_id)

    assert buy_order is not None
    assert sell_order is not None

    assert buy_order.status == OrderStatus.FILLED
    assert sell_order.status == OrderStatus.FILLED

    assert buy_order.remaining_quantity == 0
    assert sell_order.remaining_quantity == 0

    assert len(book.trade_log) == 1

    trade = book.trade_log[0]

    assert trade.price == 100.0
    assert trade.quantity == 50
    assert trade.buy_order_id == buy_id
    assert trade.sell_order_id == sell_id


def test_partial_fill():
    book = OrderBook()

    buy_id = book.add_order(Side.BUY, 100.0, 100)
    sell_id = book.add_order(Side.SELL, 100.0, 40)

    buy_order = book.get_order(buy_id)
    sell_order = book.get_order(sell_id)

    assert buy_order is not None
    assert sell_order is not None

    assert buy_order.status == OrderStatus.PARTIALLY_FILLED
    assert buy_order.remaining_quantity == 60

    assert sell_order.status == OrderStatus.FILLED
    assert sell_order.remaining_quantity == 0

    assert len(book.trade_log) == 1
    assert book.trade_log[0].quantity == 40


def test_price_time_priority():
    book = OrderBook()

    first_buy = book.add_order(Side.BUY, 100.0, 50)
    second_buy = book.add_order(Side.BUY, 100.0, 100)

    sell_id = book.add_order(Side.SELL, 100.0, 75)

    first_order = book.get_order(first_buy)
    second_order = book.get_order(second_buy)
    sell_order = book.get_order(sell_id)

    assert first_order is not None
    assert second_order is not None
    assert sell_order is not None

    assert first_order.status == OrderStatus.FILLED
    assert first_order.remaining_quantity == 0

    assert second_order.status == OrderStatus.PARTIALLY_FILLED
    assert second_order.remaining_quantity == 75

    assert sell_order.status == OrderStatus.FILLED


def test_cancel_order():
    book = OrderBook()

    order_id = book.add_order(
        Side.BUY,
        100.0,
        50,
    )

    assert book.get_best_bid() == 100.0

    result = book.cancel_order(order_id)

    assert result is True

    order = book.get_order(order_id)

    assert order is not None
    assert order.status == OrderStatus.CANCELLED

    assert book.get_best_bid() is None


def test_no_match_when_prices_do_not_cross():
    book = OrderBook()

    buy_id = book.add_order(
        Side.BUY,
        100.0,
        50,
    )

    sell_id = book.add_order(
        Side.SELL,
        101.0,
        50,
    )

    assert len(book.trade_log) == 0

    buy_order = book.get_order(buy_id)
    sell_order = book.get_order(sell_id)

    assert buy_order is not None
    assert sell_order is not None

    assert buy_order.remaining_quantity == 50
    assert sell_order.remaining_quantity == 50

    assert book.get_best_bid() == 100.0
    assert book.get_best_ask() == 101.0
