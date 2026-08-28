from __future__ import annotations

from time import time

from .book import OrderBook
from .matching import MatchingEngine
from .models import Order, Side, Trade


class ExchangeEngine:
    def __init__(self) -> None:
        self.book = OrderBook()
        self.matcher = MatchingEngine(self.book)
        self.next_order_id = 1
        self.trade_log: list[Trade] = []

    def submit_order(
        self,
        side: Side,
        price: float,
        quantity: float,
    ) -> Order:
        order = Order(
            order_id=self.next_order_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=time(),
        )

        self.next_order_id += 1

        trades = self.matcher.process_order(order)

        self.trade_log.extend(trades)

        return order

    def cancel_order(self, order_id: int) -> bool:

        order = self.book.get_order(order_id)

        if order is None:
            return False

        if not order.is_active:
            return False

        removed = self.book.remove_order_from_book(order)

        if not removed:
            return False

        order.cancel()

        return True

    def get_order(self, order_id: int) -> Order | None:
        return self.book.get_order(order_id)

    def get_best_bid(self) -> float | None:
        return self.book.get_best_bid()

    def get_best_ask(self) -> float | None:
        return self.book.get_best_ask()

    def get_mid_price(self) -> float | None:
        return self.book.get_mid_price()

    def get_spread(self) -> float | None:
        return self.book.get_spread()