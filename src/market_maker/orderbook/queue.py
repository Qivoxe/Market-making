from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from .models import Order


class PriceLevel:
    """
    FIFO queue for orders at a single price.

    Orders are executed in time priority:
    first order in -> first order out.
    """

    def __init__(self, price: float) -> None:
        if price <= 0:
            raise ValueError("Price must be greater than zero.")

        self.price = price
        self.orders: Deque[Order] = deque()

    def add(self, order: Order) -> None:
        """
        Add an order to the back of the queue.
        """

        if order.price != self.price:
            raise ValueError(
                "Order price does not match price level."
            )

        self.orders.append(order)

    def peek(self) -> Optional[Order]:
        """
        Return the oldest order without removing it.
        """

        if not self.orders:
            return None

        return self.orders[0]

    def pop(self) -> Optional[Order]:
        """
        Remove and return the oldest order.
        """

        if not self.orders:
            return None

        return self.orders.popleft()

    def remove(self, order: Order) -> bool:
        """
        Remove a specific order from the queue.

        Returns True if the order was found.
        """

        try:
            self.orders.remove(order)
            return True
        except ValueError:
            return False

    @property
    def total_quantity(self) -> float:
        """
        Total remaining quantity at this price level.
        """

        return sum(
            order.remaining_quantity
            for order in self.orders
        )

    @property
    def is_empty(self) -> bool:
        return not self.orders

    def __len__(self) -> int:
        return len(self.orders)