from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Deque, Dict, Optional

from sortedcontainers import SortedDict


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    order_id: int
    side: Side
    price: float
    quantity: float
    timestamp: float = field(default_factory=time)

    remaining_quantity: float = field(init=False)
    status: OrderStatus = field(
        default=OrderStatus.ACTIVE,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("Price must be greater than zero.")

        if self.quantity <= 0:
            raise ValueError("Quantity must be greater than zero.")

        self.remaining_quantity = self.quantity

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        return self.status in {
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
        }

    def fill(self, quantity: float) -> None:
        if quantity <= 0:
            raise ValueError("Fill quantity must be greater than zero.")

        if quantity > self.remaining_quantity:
            raise ValueError(
                "Fill quantity exceeds remaining quantity."
            )

        self.remaining_quantity -= quantity

        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        if self.is_filled:
            raise ValueError("Cannot cancel a filled order.")

        self.status = OrderStatus.CANCELLED


@dataclass
class Trade:
    trade_id: int
    price: float
    quantity: float
    buy_order_id: int
    sell_order_id: int
    timestamp: float = field(default_factory=time)


class OrderBook:
    def __init__(self) -> None:
        # Highest bid first.
        self.bids: SortedDict[float, Deque[Order]] = SortedDict(
            lambda price: -price
        )

        # Lowest ask first.
        self.asks: SortedDict[float, Deque[Order]] = SortedDict()

        # All orders, including filled/cancelled orders.
        self.orders: Dict[int, Order] = {}

        # Historical trade log.
        self.trade_log: list[Trade] = []

        self.next_order_id: int = 1
        self.next_trade_id: int = 1

    def add_order(
        self,
        side: Side,
        price: float,
        quantity: float,
    ) -> int:
        """
        Add a limit order to the order book.

        The incoming order first attempts to match
        against the opposite side.

        Any remaining quantity is added to the book.
        """

        order_id = self.next_order_id
        self.next_order_id += 1

        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
        )

        self.orders[order_id] = order

        if side == Side.BUY:
            self._match_buy(order)
        else:
            self._match_sell(order)

        return order_id

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an active order.

        Returns True if cancellation succeeds.
        Returns False if the order doesn't exist
        or is already inactive.
        """

        order = self.orders.get(order_id)

        if order is None:
            return False

        if not order.is_active:
            return False

        book = (
            self.bids
            if order.side == Side.BUY
            else self.asks
        )

        price_level = book.get(order.price)

        if price_level is not None:
            try:
                price_level.remove(order)
            except ValueError:
                pass

            if not price_level:
                del book[order.price]

        order.cancel()

        return True

    def _match_buy(self, incoming_order: Order) -> None:
        """
        Match an incoming BUY order against the best asks.
        """

        while (
            incoming_order.remaining_quantity > 0
            and self.asks
        ):
            best_ask_price = self.get_best_ask()

            if best_ask_price is None:
                break

            # BUY cannot execute above its limit price.
            if incoming_order.price < best_ask_price:
                break

            ask_queue = self.asks[best_ask_price]

            while (
                incoming_order.remaining_quantity > 0
                and ask_queue
            ):
                resting_order = ask_queue[0]

                trade_quantity = min(
                    incoming_order.remaining_quantity,
                    resting_order.remaining_quantity,
                )

                self._execute_trade(
                    buy_order=incoming_order,
                    sell_order=resting_order,
                    price=resting_order.price,
                    quantity=trade_quantity,
                )

                if resting_order.is_filled:
                    ask_queue.popleft()

            if not ask_queue:
                del self.asks[best_ask_price]

        # IMPORTANT:
        # Any unfilled quantity becomes a resting bid.
        if incoming_order.remaining_quantity > 0:
            price_level = self.bids.setdefault(
                incoming_order.price,
                deque(),
            )

            price_level.append(incoming_order)

    def _match_sell(self, incoming_order: Order) -> None:
        """
        Match an incoming SELL order against the best bids.
        """

        while (
            incoming_order.remaining_quantity > 0
            and self.bids
        ):
            best_bid_price = self.get_best_bid()

            if best_bid_price is None:
                break

            # SELL cannot execute below its limit price.
            if incoming_order.price > best_bid_price:
                break

            bid_queue = self.bids[best_bid_price]

            while (
                incoming_order.remaining_quantity > 0
                and bid_queue
            ):
                resting_order = bid_queue[0]

                trade_quantity = min(
                    incoming_order.remaining_quantity,
                    resting_order.remaining_quantity,
                )

                self._execute_trade(
                    buy_order=resting_order,
                    sell_order=incoming_order,
                    price=resting_order.price,
                    quantity=trade_quantity,
                )

                if resting_order.is_filled:
                    bid_queue.popleft()

            if not bid_queue:
                del self.bids[best_bid_price]

        # IMPORTANT:
        # Any unfilled quantity becomes a resting ask.
        if incoming_order.remaining_quantity > 0:
            price_level = self.asks.setdefault(
                incoming_order.price,
                deque(),
            )

            price_level.append(incoming_order)

    def _execute_trade(
        self,
        buy_order: Order,
        sell_order: Order,
        price: float,
        quantity: float,
    ) -> None:
        """
        Execute a trade between two orders.
        """

        buy_order.fill(quantity)
        sell_order.fill(quantity)

        trade = Trade(
            trade_id=self.next_trade_id,
            price=price,
            quantity=quantity,
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
        )

        self.next_trade_id += 1
        self.trade_log.append(trade)

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

    def get_bid_depth(
        self,
        levels: int = 5,
    ) -> list[tuple[float, float]]:
        """
        Return aggregated quantity at the top N bid levels.
        """

        depth = []

        for price, orders in list(self.bids.items())[:levels]:
            total_quantity = sum(
                order.remaining_quantity
                for order in orders
            )

            depth.append((price, total_quantity))

        return depth

    def get_ask_depth(
        self,
        levels: int = 5,
    ) -> list[tuple[float, float]]:
        """
        Return aggregated quantity at the top N ask levels.
        """

        depth = []

        for price, orders in list(self.asks.items())[:levels]:
            total_quantity = sum(
                order.remaining_quantity
                for order in orders
            )

            depth.append((price, total_quantity))

        return depth

    def get_order(
        self,
        order_id: int,
    ) -> Optional[Order]:
        return self.orders.get(order_id)

    def __len__(self) -> int:
        return sum(
            order.is_active
            for order in self.orders.values()
        )