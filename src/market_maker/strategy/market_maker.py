from __future__ import annotations

from dataclasses import dataclass

from src.market_maker.ml.signal import TradingSignal


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float


def generate_quotes(
    *,
    mid_price: float,
    spread: float,
    signal: TradingSignal,
    shift_factor: float = 0.25,
    confidence_scaling: bool = False,
) -> Quote:
    """
    Generate bid/ask quotes around the mid price.

    Base quote:
        bid = mid_price - spread / 2
        ask = mid_price + spread / 2

    BUY:
        Shift both quotes upward.

    SELL:
        Shift both quotes downward.

    The quote shift is confidence-aware:

        shift = spread * shift_factor * confidence

    confidence_scaling is retained for API compatibility.
    """

    if mid_price <= 0:
        raise ValueError(
            "mid_price must be greater than 0."
        )

    if spread < 0:
        raise ValueError(
            "spread must be non-negative."
        )

    if shift_factor < 0:
        raise ValueError(
            "shift_factor must be non-negative."
        )

    if not 0.0 <= signal.confidence <= 1.0:
        raise ValueError(
            "signal confidence must be between 0 and 1."
        )

    if signal.action not in {"BUY", "SELL", "HOLD"}:
        raise ValueError(
            f"Unknown signal action: {signal.action}"
        )

    half_spread = spread / 2.0

    bid = mid_price - half_spread
    ask = mid_price + half_spread

    # Confidence-aware directional shift.
    shift = (
        spread
        * shift_factor
        * signal.confidence
    )

    if signal.action == "BUY":
        bid += shift
        ask += shift

    elif signal.action == "SELL":
        bid -= shift
        ask -= shift

    return Quote(
        bid=float(bid),
        ask=float(ask),
    )