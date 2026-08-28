from src.market_maker.orderbook.models import Order, Side, OrderStatus
from src.market_maker.orderbook.queue import PriceLevel


def test_price_level_fifo():
    level = PriceLevel(100.0)

    first = Order(
        order_id=1,
        side=Side.BUY,
        price=100.0,
        quantity=50,
    )

    second = Order(
        order_id=2,
        side=Side.BUY,
        price=100.0,
        quantity=100,
    )

    level.add(first)
    level.add(second)

    assert level.peek() == first
    assert level.pop() == first
    assert level.peek() == second


def test_price_level_total_quantity():
    level = PriceLevel(100.0)

    first = Order(1, Side.BUY, 100.0, 50)
    second = Order(2, Side.BUY, 100.0, 100)

    level.add(first)
    level.add(second)

    assert level.total_quantity == 150


def test_price_level_quantity_after_partial_fill():
    level = PriceLevel(100.0)

    order = Order(1, Side.BUY, 100.0, 100)

    level.add(order)

    order.fill(40)

    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.remaining_quantity == 60
    assert level.total_quantity == 60


def test_remove_order():
    level = PriceLevel(100.0)

    first = Order(1, Side.BUY, 100.0, 50)
    second = Order(2, Side.BUY, 100.0, 100)

    level.add(first)
    level.add(second)

    assert level.remove(first) is True

    assert level.peek() == second
    assert len(level) == 1


def test_remove_nonexistent_order():
    level = PriceLevel(100.0)

    order = Order(1, Side.BUY, 100.0, 50)

    assert level.remove(order) is False


def test_empty_price_level():
    level = PriceLevel(100.0)

    assert level.is_empty is True
    assert level.peek() is None
    assert level.pop() is None
    assert len(level) == 0