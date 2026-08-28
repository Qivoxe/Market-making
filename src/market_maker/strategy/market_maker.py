from __future__ import annotations

from dataclasses import dataclass

from src.market_maker.ml.signal import TradingSignal


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float


def generate_quotes(
    mid_price: float,
    spread: float,
    signal: TradingSignal,
    shift_factor: float = 0.25,
    confidence_scaling: bool = False,
) -> Quote:
    if mid_price <= 0:
        raise ValueError("Mid price must be greater than zero.")

    if spread < 0:
        raise ValueError("Spread must not be negative.")

    if shift_factor < 0:
        raise ValueError("Shift factor must not be negative.")

    if not 0.0 <= signal.confidence <= 1.0:
        raise ValueError(
            "Signal confidence must be between 0 and 1."
        )

    half_spread = spread / 2.0

    if confidence_scaling:
        shift = (
            spread
            * shift_factor
            * signal.confidence
        )
    else:
        shift = spread * shift_factor

    if signal.action == "BUY":
        center = mid_price + shift
    elif signal.action == "SELL":
        center = mid_price - shift
    elif signal.action == "HOLD":
        center = mid_price
    else:
        raise ValueError("Unknown trading signal.")

    bid = center - half_spread
    ask = center + half_spread

    bid = max(bid, 0.000001)

    if ask <= bid:
        ask = bid + max(spread, 0.000001)

    return Quote(
        bid=bid,
        ask=ask,
    )