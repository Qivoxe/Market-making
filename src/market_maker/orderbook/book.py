from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

from sortedcontainers import SortedDict

from .models import Order, Side


class OrderBook:

    def __init__(self) -> None:

        self.bids: SortedDict[float, Deque[Order]] = SortedDict(
            lambda price: -price
        )


        self.asks: SortedDict[float, Deque[Order]] = SortedDict()


        self.orders: Dict[int, Order] = {}


    def add_order_to_book(self, order: Order) -> None:

        book = (
            self.bids
            if order.side == Side.BUY
            else self.asks
        )

        price_level = book.setdefault(
            order.price,
            deque(),
        )

        price_level.append(order)

        self.orders[order.order_id] = order

    def remove_order_from_book(self, order: Order) -> bool:

        book = (
            self.bids
            if order.side == Side.BUY
            else self.asks
        )

        price_level = book.get(order.price)

        if price_level is None:
            return False

        try:
            price_level.remove(order)
        except ValueError:
            return False


        if not price_level:
            del book[order.price]

        return True


    def get_best_bid(self) -> Optional[float]:


        if not self.bids:
            return None

        return self.bids.peekitem(0)[0]

    def get_best_ask(self) -> Optional[float]:

        if not self.asks:
            return None

        return self.asks.peekitem(0)[0]

    def get_mid_price(self) -> Optional[float]:

        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is None or best_ask is None:
            return None

        return (best_bid + best_ask) / 2

    def get_spread(self) -> Optional[float]:


        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is None or best_ask is None:
            return None

        return best_ask - best_bid

    def get_bid_queue(
        self,
        price: float,
    ) -> Optional[Deque[Order]]:

        return self.bids.get(price)

    def get_ask_queue(
        self,
        price: float,
    ) -> Optional[Deque[Order]]:

        return self.asks.get(price)

    def get_bid_depth(
        self,
        levels: int = 5,
    ) -> list[tuple[float, float]]:

        if levels <= 0:
            return []

        depth = []

        for price, orders in list(self.bids.items())[:levels]:
            total_quantity = sum(
                order.remaining_quantity
                for order in orders
            )

            depth.append(
                (price, total_quantity)
            )

        return depth

    def get_ask_depth(
        self,
        levels: int = 5,
    ) -> list[tuple[float, float]]:

        if levels <= 0:
            return []

        depth = []

        for price, orders in list(self.asks.items())[:levels]:
            total_quantity = sum(
                order.remaining_quantity
                for order in orders
            )

            depth.append(
                (price, total_quantity)
            )

        return depth

    def get_order(
        self,
        order_id: int,
    ) -> Optional[Order]:

        return self.orders.get(order_id)

    def active_order_count(self) -> int:
        return sum(
            order.is_active
            for order in self.orders.values()
        )

    def __len__(self) -> int:
        return self.active_order_count()