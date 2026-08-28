from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.market_maker.orderbook.book import OrderBook


@dataclass
class Quote:
    bid_price: Optional[float]
    ask_price: Optional[float]
    bid_quantity: float
    ask_quantity: float


class MarketMakingStrategy:
    def __init__(
        self,
        spread: float = 1.0,
        order_size: float = 1.0,
        inventory_skew: float = 0.01,
    ) -> None:
        if spread <= 0:
            raise ValueError("Spread must be greater than zero.")

        if order_size <= 0:
            raise ValueError("Order size must be greater than zero.")

        if inventory_skew < 0:
            raise ValueError("Inventory skew cannot be negative.")

        self.spread = spread
        self.order_size = order_size
        self.inventory_skew = inventory_skew
        self.inventory = 0.0

    def generate_quote(
        self,
        book: OrderBook,
    ) -> Optional[Quote]:
        mid_price = book.get_mid_price()

        if mid_price is None:
            return None

        reservation_price = (
            mid_price
            - self.inventory * self.inventory_skew
        )

        half_spread = self.spread / 2

        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread

        return Quote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_quantity=self.order_size,
            ask_quantity=self.order_size,
        )