from __future__ import annotations

from dataclasses import dataclass

from src.market_maker.orderbook.book import OrderBook


@dataclass(frozen=True)
class OrderFlowFeatures:
    bid_volume: float
    ask_volume: float
    imbalance: float
    mid_price: float
    spread: float


def calculate_order_flow(
    book: OrderBook,
    levels: int = 5,
) -> OrderFlowFeatures | None:
    if levels <= 0:
        raise ValueError("Levels must be greater than zero.")

    mid_price = book.get_mid_price()
    spread = book.get_spread()

    if mid_price is None or spread is None:
        return None

    bid_depth = book.get_bid_depth(levels)
    ask_depth = book.get_ask_depth(levels)

    bid_volume = sum(
        quantity for _, quantity in bid_depth
    )

    ask_volume = sum(
        quantity for _, quantity in ask_depth
    )

    total_volume = bid_volume + ask_volume

    if total_volume == 0:
        imbalance = 0.0
    else:
        imbalance = (
            (bid_volume - ask_volume)
            / total_volume
        )

    return OrderFlowFeatures(
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        imbalance=imbalance,
        mid_price=mid_price,
        spread=spread,
    )