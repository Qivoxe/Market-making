from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.market_maker.orderbook.engine import ExchangeEngine
from src.market_maker.orderbook.models import Side
from src.market_maker.strategy.engine import StrategyDecision


@dataclass(frozen=True)
class MarketMakingTrade:
    step: int
    side: str
    price: float
    quantity: float


@dataclass(frozen=True)
class MarketMakingResult:
    initial_cash: float
    final_cash: float
    final_position: float
    final_mid_price: float
    final_equity: float
    pnl: float
    return_pct: float
    trades: tuple[MarketMakingTrade, ...]
    equity_curve: tuple[float, ...]


class MarketMakingBacktestEngine:
    """
    Event-driven market-making backtest engine.

    The engine can operate on an externally supplied ExchangeEngine.
    This allows the market simulator and strategy to share exactly
    the same order book.
    """

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        max_position: float = 100.0,
        order_size: float = 1.0,
        exchange: ExchangeEngine | None = None,
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

        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.position = 0.0
        self.max_position = float(max_position)
        self.order_size = float(order_size)

        self.exchange = (
            exchange
            if exchange is not None
            else ExchangeEngine()
        )

        self.trades: list[MarketMakingTrade] = []

        # Track exchange trades already processed.
        self._processed_trade_ids: set[int] = set()

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

    def _apply_fill(
        self,
        *,
        side: Side,
        price: float,
        quantity: float,
        step: int,
    ) -> None:
        if price <= 0:
            raise ValueError(
                "Fill price must be greater than zero."
            )

        if quantity <= 0:
            raise ValueError(
                "Fill quantity must be greater than zero."
            )

        if side == Side.BUY:
            new_position = (
                self.position + quantity
            )

            if new_position > self.max_position:
                raise RuntimeError(
                    "Buy fill would exceed maximum position."
                )

            self.cash -= (
                price * quantity
            )

            self.position = new_position

            trade_side = "BUY"

        elif side == Side.SELL:
            new_position = (
                self.position - quantity
            )

            if new_position < -self.max_position:
                raise RuntimeError(
                    "Sell fill would exceed maximum position."
                )

            self.cash += (
                price * quantity
            )

            self.position = new_position

            trade_side = "SELL"

        else:
            raise ValueError(
                f"Unsupported fill side: {side}"
            )

        self.trades.append(
            MarketMakingTrade(
                step=step,
                side=trade_side,
                price=float(price),
                quantity=float(quantity),
            )
        )

    def _process_exchange_fills(
        self,
        *,
        strategy_order_ids: set[int],
        step: int,
    ) -> None:
        """
        Process newly generated exchange trades involving
        one of our strategy orders.
        """

        for trade in self.exchange.trade_log:
            if trade.trade_id in self._processed_trade_ids:
                continue

            self._processed_trade_ids.add(
                trade.trade_id
            )

            if (
                trade.buy_order_id
                in strategy_order_ids
            ):
                self._apply_fill(
                    side=Side.BUY,
                    price=trade.price,
                    quantity=trade.quantity,
                    step=step,
                )

            elif (
                trade.sell_order_id
                in strategy_order_ids
            ):
                self._apply_fill(
                    side=Side.SELL,
                    price=trade.price,
                    quantity=trade.quantity,
                    step=step,
                )

    def place_quotes(
        self,
        *,
        decision: StrategyDecision,
    ) -> tuple[int, int]:
        """
        Place both sides of the strategy quote.
        """

        bid_order = self.exchange.submit_order(
            Side.BUY,
            decision.quote.bid,
            self.order_size,
        )

        ask_order = self.exchange.submit_order(
            Side.SELL,
            decision.quote.ask,
            self.order_size,
        )

        return (
            bid_order.order_id,
            ask_order.order_id,
        )

    def cancel_quotes(
        self,
        order_ids: Sequence[int],
    ) -> None:
        for order_id in order_ids:
            order = self.exchange.get_order(
                order_id
            )

            if order is None:
                continue

            if order.is_active:
                self.exchange.cancel_order(
                    order_id
                )

    def process_market_order(
        self,
        *,
        side: Side,
        price: float,
        quantity: float,
        strategy_order_ids: set[int],
        step: int,
    ) -> None:
        """
        Submit an external market participant order
        into the same exchange used by the strategy.
        """

        if price <= 0:
            raise ValueError(
                "Market order price must be greater than zero."
            )

        if quantity <= 0:
            raise ValueError(
                "Market order quantity must be greater than zero."
            )

        self.exchange.submit_order(
            side,
            price,
            quantity,
        )

        self._process_exchange_fills(
            strategy_order_ids=strategy_order_ids,
            step=step,
        )

    def run(
        self,
        *,
        decisions: Sequence[StrategyDecision],
        market_orders: Sequence[
            tuple[Side, float, float]
        ],
        mid_prices: Sequence[float],
    ) -> MarketMakingResult:
        """
        Run an event-driven market-making backtest.

        For each event:

        1. Place strategy bid and ask.
        2. Submit external market order.
        3. Let the matching engine determine fills.
        4. Process strategy fills.
        5. Cancel remaining quotes.
        6. Mark portfolio to market.
        """

        if not (
            len(decisions)
            == len(market_orders)
            == len(mid_prices)
        ):
            raise ValueError(
                "Decisions, market orders, and prices "
                "must have the same length."
            )

        if len(decisions) == 0:
            raise ValueError(
                "Market-making backtest requires "
                "at least one event."
            )

        equity_curve: list[float] = []

        for step, (
            decision,
            market_order,
            mid_price,
        ) in enumerate(
            zip(
                decisions,
                market_orders,
                mid_prices,
            ),
            start=1,
        ):
            if mid_price <= 0:
                raise ValueError(
                    "Mid price must be greater than zero."
                )

            side, price, quantity = market_order

            bid_id, ask_id = self.place_quotes(
                decision=decision,
            )

            strategy_order_ids = {
                bid_id,
                ask_id,
            }

            self.process_market_order(
                side=side,
                price=price,
                quantity=quantity,
                strategy_order_ids=strategy_order_ids,
                step=step,
            )

            self.cancel_quotes(
                strategy_order_ids
            )

            equity = self.mark_to_market(
                mid_price
            )

            equity_curve.append(equity)

        final_mid_price = float(
            mid_prices[-1]
        )

        final_equity = self.mark_to_market(
            final_mid_price
        )

        pnl = (
            final_equity
            - self.initial_cash
        )

        return_pct = (
            pnl / self.initial_cash
        )

        return MarketMakingResult(
            initial_cash=self.initial_cash,
            final_cash=self.cash,
            final_position=self.position,
            final_mid_price=final_mid_price,
            final_equity=final_equity,
            pnl=pnl,
            return_pct=return_pct,
            trades=tuple(self.trades),
            equity_curve=tuple(equity_curve),
        )