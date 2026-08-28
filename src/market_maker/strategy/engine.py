from __future__ import annotations

from dataclasses import dataclass

from src.market_maker.ml.signal import TradingSignal
from src.market_maker.strategy.inventory import (
    adjust_quotes_for_inventory,
)
from src.market_maker.strategy.market_maker import (
    Quote,
    generate_quotes,
)


@dataclass(frozen=True)
class StrategyDecision:
    quote: Quote
    signal: TradingSignal
    position: float


def make_strategy_decision(
    mid_price: float,
    spread: float,
    signal: TradingSignal,
    position: float,
    max_position: float,
    shift_factor: float = 0.25,
    inventory_skew_factor: float = 0.5,
) -> StrategyDecision:
    if max_position <= 0:
        raise ValueError(
            "Max position must be greater than zero."
        )

    if abs(position) > max_position:
        raise ValueError(
            "Position cannot exceed max position."
        )

    quote = generate_quotes(
        mid_price=mid_price,
        spread=spread,
        signal=signal,
        shift_factor=shift_factor,
        confidence_scaling=True,
    )

    bid, ask = adjust_quotes_for_inventory(
        bid=quote.bid,
        ask=quote.ask,
        position=position,
        max_position=max_position,
        skew_factor=inventory_skew_factor,
    )

    final_quote = Quote(
        bid=bid,
        ask=ask,
    )

    return StrategyDecision(
        quote=final_quote,
        signal=signal,
        position=position,
    )