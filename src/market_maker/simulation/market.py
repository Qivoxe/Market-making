
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.market_maker.features.order_flow import (
    OrderFlowFeatures,
    calculate_order_flow,
)
from src.market_maker.orderbook.engine import ExchangeEngine
from src.market_maker.orderbook.models import Side
from src.market_maker.simulation.regime import MarketRegime


@dataclass(frozen=True)
class MarketSnapshot:
    step: int
    mid_price: float
    spread: float
    bid_volume: float
    ask_volume: float
    imbalance: float


class MarketSimulator:
    def __init__(
        self,
        initial_price: float = 100.0,
        seed: int | None = None,
        regime: MarketRegime = MarketRegime.NORMAL,
    ) -> None:
        if initial_price <= 0:
            raise ValueError("initial_price must be positive")

        self.initial_price = float(initial_price)
        self.reference_price = float(initial_price)
        self.fair_price = float(initial_price)
        self.rng = np.random.default_rng(seed)
        self.exchange = ExchangeEngine()
        self.book = self.exchange.book
        self.step = 0
        self.regime = regime

        self._initialize_book()

    def _initialize_book(self) -> None:
        self.exchange.submit_order(
            Side.BUY,
            self.initial_price - 0.5,
            10.0,
        )
        self.exchange.submit_order(
            Side.SELL,
            self.initial_price + 0.5,
            10.0,
        )

    def _price_move(self) -> float:
        if self.regime == MarketRegime.NORMAL:
            return float(self.rng.normal(0.0, 0.5))

        if self.regime == MarketRegime.HIGH_VOLATILITY:
            return float(self.rng.normal(0.0, 2.0))

        if self.regime == MarketRegime.TRENDING_UP:
            return float(self.rng.normal(0.2, 0.5))

        if self.regime == MarketRegime.TRENDING_DOWN:
            return float(self.rng.normal(-0.2, 0.5))

        if self.regime == MarketRegime.MEAN_REVERTING:
            return float(
                0.2 * (self.fair_price - self.reference_price)
                + self.rng.normal(0.0, 0.5)
            )

        raise ValueError(f"Unsupported market regime: {self.regime}")

    def _generate_order(self) -> tuple[Side, float, float]:
        side = (
            Side.BUY
            if self.rng.random() < 0.5
            else Side.SELL
        )

        price_move = self._price_move()
        price = max(0.01, self.reference_price + price_move)
        quantity = float(self.rng.uniform(1.0, 10.0))

        return side, float(price), quantity

    def _ensure_book_liquidity(self) -> None:
        """Restore a basic two-sided book when a side is empty."""
        best_bid = self.exchange.get_best_bid()
        best_ask = self.exchange.get_best_ask()

        if best_bid is None:
            bid_price = max(0.01, self.reference_price - 0.5)

            if best_ask is not None:
                bid_price = min(bid_price, best_ask - 0.01)

            if bid_price > 0:
                self.exchange.submit_order(
                    Side.BUY,
                    bid_price,
                    10.0,
                )

        best_bid = self.exchange.get_best_bid()
        best_ask = self.exchange.get_best_ask()

        if best_ask is None:
            ask_price = max(
                self.reference_price + 0.5,
                (best_bid + 0.01) if best_bid is not None else 0.01,
            )

            self.exchange.submit_order(
                Side.SELL,
                ask_price,
                10.0,
            )

    def generate_market_order(self) -> tuple[Side, float, float]:
        """Generate an external order without submitting it."""
        return self._generate_order()

    def generate_market_event(
        self,
    ) -> tuple[Side, float, float, MarketSnapshot | None]:
        """
        Generate one external order and the current market snapshot.

        The external order is not submitted here. The runner places its
        quotes first, then submits the external order exactly once.
        """
        side, price, quantity = self._generate_order()

        self._ensure_book_liquidity()

        features: OrderFlowFeatures | None = calculate_order_flow(
            self.book
        )

        if features is None:
            return side, price, quantity, None

        snapshot = MarketSnapshot(
            step=self.step + 1,
            mid_price=features.mid_price,
            spread=features.spread,
            bid_volume=features.bid_volume,
            ask_volume=features.ask_volume,
            imbalance=features.imbalance,
        )

        return side, price, quantity, snapshot

    def step_market(self) -> MarketSnapshot | None:
        """Submit one simulated order and return the resulting snapshot."""
        side, price, quantity = self._generate_order()

        self.exchange.submit_order(
            side,
            price,
            quantity,
        )

        self._ensure_book_liquidity()

        features: OrderFlowFeatures | None = calculate_order_flow(
            self.book
        )

        if features is None:
            return None

        self.reference_price = features.mid_price
        self.step += 1

        return MarketSnapshot(
            step=self.step,
            mid_price=features.mid_price,
            spread=features.spread,
            bid_volume=features.bid_volume,
            ask_volume=features.ask_volume,
            imbalance=features.imbalance,
        )

    def generate_snapshots(self, count: int) -> list[MarketSnapshot]:
        if count <= 0:
            raise ValueError("count must be positive")

        snapshots: list[MarketSnapshot] = []

        while len(snapshots) < count:
            snapshot = self.step_market()

            if snapshot is not None:
                snapshots.append(snapshot)

        return snapshots

