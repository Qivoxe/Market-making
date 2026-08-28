from src.market_maker.orderbook.book import OrderBook
from src.market_maker.orderbook.models import Order, Side
from src.market_maker.strategy.strategy import MarketMakingStrategy


def test_generate_quote():
    book = OrderBook()

    book.add_order_to_book(
        Order(1, Side.BUY, 99.0, 50)
    )

    book.add_order_to_book(
        Order(2, Side.SELL, 101.0, 50)
    )

    strategy = MarketMakingStrategy(
        spread=2.0,
        order_size=10.0,
    )

    quote = strategy.generate_quote(book)

    assert quote is not None
    assert quote.bid_price == 99.0
    assert quote.ask_price == 101.0
    assert quote.bid_quantity == 10.0
    assert quote.ask_quantity == 10.0


def test_no_quote_without_market():
    book = OrderBook()

    strategy = MarketMakingStrategy()

    quote = strategy.generate_quote(book)

    assert quote is None


def test_invalid_spread():
    try:
        MarketMakingStrategy(spread=0)
        assert False
    except ValueError:
        assert True


def test_invalid_order_size():
    try:
        MarketMakingStrategy(order_size=0)
        assert False
    except ValueError:
        assert True

def test_positive_inventory_skews_quotes_down():
    book = OrderBook()

    book.add_order_to_book(
        Order(1, Side.BUY, 99.0, 50)
    )

    book.add_order_to_book(
        Order(2, Side.SELL, 101.0, 50)
    )

    strategy = MarketMakingStrategy(
        spread=2.0,
        order_size=10.0,
        inventory_skew=0.1,
    )

    strategy.inventory = 10.0

    quote = strategy.generate_quote(book)

    assert quote is not None
    assert quote.bid_price == 98.0
    assert quote.ask_price == 100.0


def test_negative_inventory_skews_quotes_up():
    book = OrderBook()

    book.add_order_to_book(
        Order(1, Side.BUY, 99.0, 50)
    )

    book.add_order_to_book(
        Order(2, Side.SELL, 101.0, 50)
    )

    strategy = MarketMakingStrategy(
        spread=2.0,
        order_size=10.0,
        inventory_skew=0.1,
    )

    strategy.inventory = -10.0

    quote = strategy.generate_quote(book)

    assert quote is not None
    assert quote.bid_price == 100.0
    assert quote.ask_price == 102.0        