from src.market_maker.features.order_flow import calculate_order_flow
from src.market_maker.orderbook.book import OrderBook
from src.market_maker.orderbook.models import Order, Side


def test_order_flow_features():
    book = OrderBook()

    book.add_order_to_book(
        Order(1, Side.BUY, 100.0, 50)
    )

    book.add_order_to_book(
        Order(2, Side.BUY, 99.0, 100)
    )

    book.add_order_to_book(
        Order(3, Side.SELL, 101.0, 20)
    )

    book.add_order_to_book(
        Order(4, Side.SELL, 102.0, 80)
    )

    features = calculate_order_flow(book)

    assert features is not None
    assert features.bid_volume == 150
    assert features.ask_volume == 100
    assert features.mid_price == 100.5
    assert features.spread == 1.0
    assert features.imbalance == 0.2


def test_balanced_order_flow():
    book = OrderBook()

    book.add_order_to_book(
        Order(1, Side.BUY, 100.0, 50)
    )

    book.add_order_to_book(
        Order(2, Side.SELL, 101.0, 50)
    )

    features = calculate_order_flow(book)

    assert features is not None
    assert features.imbalance == 0.0


def test_bid_heavy_order_flow():
    book = OrderBook()

    book.add_order_to_book(
        Order(1, Side.BUY, 100.0, 100)
    )

    book.add_order_to_book(
        Order(2, Side.SELL, 101.0, 25)
    )

    features = calculate_order_flow(book)

    assert features is not None
    assert features.imbalance == 0.6


def test_ask_heavy_order_flow():
    book = OrderBook()

    book.add_order_to_book(
        Order(1, Side.BUY, 100.0, 25)
    )

    book.add_order_to_book(
        Order(2, Side.SELL, 101.0, 100)
    )

    features = calculate_order_flow(book)

    assert features is not None
    assert features.imbalance == -0.6


def test_no_features_without_market():
    book = OrderBook()

    features = calculate_order_flow(book)

    assert features is None


def test_invalid_levels():
    book = OrderBook()

    try:
        calculate_order_flow(book, levels=0)
        assert False
    except ValueError:
        assert True