from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InventoryState:
    position: float
    max_position: float


def calculate_inventory_skew(
    position: float,
    max_position: float,
) -> float:
    if max_position <= 0:
        raise ValueError(
            "Max position must be greater than zero."
        )

    if abs(position) > max_position:
        raise ValueError(
            "Position cannot exceed max position."
        )

    return position / max_position


def adjust_quotes_for_inventory(
    bid: float,
    ask: float,
    position: float,
    max_position: float,
    skew_factor: float = 0.5,
) -> tuple[float, float]:
    if bid <= 0:
        raise ValueError(
            "Bid must be greater than zero."
        )

    if ask <= 0:
        raise ValueError(
            "Ask must be greater than zero."
        )

    if ask < bid:
        raise ValueError(
            "Ask must be greater than or equal to bid."
        )

    if skew_factor < 0:
        raise ValueError(
            "Skew factor must not be negative."
        )

    skew = calculate_inventory_skew(
        position,
        max_position,
    )

    spread = ask - bid
    shift = spread * skew_factor * skew

    adjusted_bid = bid - shift
    adjusted_ask = ask - shift

    adjusted_bid = max(
        adjusted_bid,
        0.000001,
    )

    if adjusted_ask <= adjusted_bid:
        adjusted_ask = (
            adjusted_bid
            + max(spread, 0.000001)
        )

    return adjusted_bid, adjusted_ask