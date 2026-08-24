from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Optional

from .book import OrderBook
from .models import Order, Side, Trade


@dataclass
class MatchingEngine:

    book: OrderBook

    next_trade_id: int = 1

    def process_order(self, order: Order) -> list[Trade]:

        trades: list[Trade] = []

        if order.side == Side.BUY:
            self._match_buy(order, trades)
        else:
            self._match_sell(order, trades)


        if order.remaining_quantity > 0:
            self.book.add_order_to_book(order)

        return trades

    def _match_buy(
        self,
        incoming_order: Order,
        trades: list[Trade],
    ) -> None:

        while (
            incoming_order.remaining_quantity > 0
            and self.book.asks
        ):
            best_ask = self.book.get_best_ask()

            if best_ask is None:
                break

            if incoming_order.price < best_ask:
                break

            ask_queue = self.book.get_ask_queue(best_ask)

            if ask_queue is None:
                break

            while (
                incoming_order.remaining_quantity > 0
                and ask_queue
            ):
                resting_order = ask_queue[0]

                trade_quantity = min(
                    incoming_order.remaining_quantity,
                    resting_order.remaining_quantity,
                )

                trade = self._execute_trade(
                    buy_order=incoming_order,
                    sell_order=resting_order,
                    price=resting_order.price,
                    quantity=trade_quantity,
                )

                trades.append(trade)

                if resting_order.is_filled:
                    ask_queue.popleft()

            if not ask_queue:
                del self.book.asks[best_ask]

    def _match_sell(
        self,
        incoming_order: Order,
        trades: list[Trade],
    ) -> None:
        while (
            incoming_order.remaining_quantity > 0
            and self.book.bids
        ):
            best_bid = self.book.get_best_bid()

            if best_bid is None:
                break


            if incoming_order.price > best_bid:
                break

            bid_queue = self.book.get_bid_queue(best_bid)

            if bid_queue is None:
                break

            while (
                incoming_order.remaining_quantity > 0
                and bid_queue
            ):
                resting_order = bid_queue[0]

                trade_quantity = min(
                    incoming_order.remaining_quantity,
                    resting_order.remaining_quantity,
                )

                trade = self._execute_trade(
                    buy_order=resting_order,
                    sell_order=incoming_order,
                    price=resting_order.price,
                    quantity=trade_quantity,
                )

                trades.append(trade)

                if resting_order.is_filled:
                    bid_queue.popleft()

            if not bid_queue:
                del self.book.bids[best_bid]

    def _execute_trade(
        self,
        buy_order: Order,
        sell_order: Order,
        price: float,
        quantity: float,
    ) -> Trade:
        
        buy_order.fill(quantity)
        sell_order.fill(quantity)

        trade = Trade(
            trade_id=self.next_trade_id,
            price=price,
            quantity=quantity,
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
            timestamp=time(),
        )

        self.next_trade_id += 1

        return trade
