from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.market_maker.strategy.engine import StrategyDecision


@dataclass(frozen=True)
class Trade:
    step: int
    side: str
    price: float
    quantity: float


@dataclass(frozen=True)
class BacktestResult:
    initial_cash: float
    final_cash: float
    final_position: float
    final_mid_price: float
    final_equity: float
    pnl: float
    return_pct: float
    trades: tuple[Trade, ...]


class BacktestEngine:
    def __init__(
        self,
        initial_cash: float = 10_000.0,
        max_position: float = 100.0,
        order_size: float = 1.0,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError(
                "Initial cash must be greater than zero."
            )

        if max_position <= 0:
            raise ValueError(
                "Max position must be greater than zero."
            )

        if order_size <= 0:
            raise ValueError(
                "Order size must be greater than zero."
            )

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0.0

        self.max_position = max_position
        self.order_size = order_size

        self.trades: list[Trade] = []

    def execute_bid(
        self,
        step: int,
        bid: float,
    ) -> bool:
        if bid <= 0:
            raise ValueError(
                "Bid price must be greater than zero."
            )

        if (
            self.position + self.order_size
            > self.max_position
        ):
            return False

        self.cash -= bid * self.order_size
        self.position += self.order_size

        self.trades.append(
            Trade(
                step=step,
                side="BUY",
                price=bid,
                quantity=self.order_size,
            )
        )

        return True

    def execute_ask(
        self,
        step: int,
        ask: float,
    ) -> bool:
        if ask <= 0:
            raise ValueError(
                "Ask price must be greater than zero."
            )

        if (
            self.position - self.order_size
            < -self.max_position
        ):
            return False

        self.cash += ask * self.order_size
        self.position -= self.order_size

        self.trades.append(
            Trade(
                step=step,
                side="SELL",
                price=ask,
                quantity=self.order_size,
            )
        )

        return True

    def mark_to_market(
        self,
        mid_price: float,
    ) -> float:
        if mid_price <= 0:
            raise ValueError(
                "Mid price must be greater than zero."
            )

        return (
            self.cash
            + self.position * mid_price
        )

    def run(
        self,
        decisions: Sequence[StrategyDecision],
        mid_prices: Sequence[float],
    ) -> BacktestResult:
        if len(decisions) != len(mid_prices):
            raise ValueError(
                "Decisions and prices must have the same length."
            )

        if len(decisions) == 0:
            raise ValueError(
                "Backtest requires at least one decision."
            )

        for step, (
            decision,
            mid_price,
        ) in enumerate(
            zip(decisions, mid_prices),
            start=1,
        ):
            if mid_price <= 0:
                raise ValueError(
                    "Mid price must be greater than zero."
                )

            quote = decision.quote

            if decision.signal.action == "BUY":
                self.execute_bid(
                    step=step,
                    bid=quote.bid,
                )

            elif decision.signal.action == "SELL":
                self.execute_ask(
                    step=step,
                    ask=quote.ask,
                )

        final_mid_price = mid_prices[-1]

        final_equity = self.mark_to_market(
            final_mid_price
        )

        pnl = (
            final_equity
            - self.initial_cash
        )

        return_pct = pnl / self.initial_cash

        return BacktestResult(
            initial_cash=self.initial_cash,
            final_cash=self.cash,
            final_position=self.position,
            final_mid_price=final_mid_price,
            final_equity=final_equity,
            pnl=pnl,
            return_pct=return_pct,
            trades=tuple(self.trades),
        )
