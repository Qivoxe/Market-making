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
            raise ValueError(
                "Initial price must be greater than zero."
            )

        self.initial_price = initial_price
        self.reference_price = initial_price
        self.fair_price = initial_price
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
            return self.rng.normal(0.0, 0.5)

        if self.regime == MarketRegime.HIGH_VOLATILITY:
            return self.rng.normal(0.0, 2.0)

        if self.regime == MarketRegime.TRENDING_UP:
            return self.rng.normal(0.2, 0.5)

        if self.regime == MarketRegime.TRENDING_DOWN:
            return self.rng.normal(-0.2, 0.5)

        if self.regime == MarketRegime.MEAN_REVERTING:
            deviation = self.fair_price - self.reference_price
            return (
                0.2 * deviation
                + self.rng.normal(0.0, 0.5)
            )

        raise ValueError("Unsupported market regime.")

    def _generate_order(self) -> tuple[Side, float, float]:
        side = (
            Side.BUY
            if self.rng.random() < 0.5
            else Side.SELL
        )

        price_move = self._price_move()

        price = max(
            0.01,
            self.reference_price + price_move,
        )

        quantity = self.rng.uniform(1.0, 10.0)

        return side, price, quantity

    def step_market(self) -> MarketSnapshot | None:
        side, price, quantity = self._generate_order()

        self.exchange.submit_order(
            side,
            price,
            quantity,
        )

        features: OrderFlowFeatures | None = (
            calculate_order_flow(self.book)
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

    def generate_snapshots(
        self,
        count: int,
    ) -> list[MarketSnapshot]:
        if count <= 0:
            raise ValueError(
                "Count must be greater than zero."
            )

        snapshots: list[MarketSnapshot] = []

        while len(snapshots) < count:
            snapshot = self.step_market()

            if snapshot is not None:
                snapshots.append(snapshot)

        return snapshots