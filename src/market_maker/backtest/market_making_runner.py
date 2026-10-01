from __future__ import annotations

from dataclasses import dataclass

from src.market_maker.backtest.market_making_engine import (
    MarketMakingBacktestEngine,
    MarketMakingResult,
)
from src.market_maker.simulation.market import (
    MarketSimulator,
)
from src.market_maker.strategy.engine import (
    make_strategy_decision,
)
from src.market_maker.ml.signal import (
    TradingSignal,
)


@dataclass(frozen=True)
class MarketMakingRunConfig:
    steps: int = 100
    initial_price: float = 100.0
    seed: int | None = 42
    initial_cash: float = 10_000.0
    max_position: float = 100.0
    order_size: float = 1.0
    shift_factor: float = 0.25
    inventory_skew_factor: float = 0.5


def run_market_making_backtest(
    *,
    signal: TradingSignal,
    config: MarketMakingRunConfig | None = None,
) -> MarketMakingResult:
    """
    Run a simple event-driven market-making backtest
    using a fixed trading signal.

    This is an integration test of the simulator,
    strategy, order book, matching engine, and
    market-making backtest engine.
    """

    if config is None:
        config = MarketMakingRunConfig()

    if config.steps <= 0:
        raise ValueError(
            "Steps must be greater than zero."
        )

    simulator = MarketSimulator(
        initial_price=config.initial_price,
        seed=config.seed,
    )

    engine = MarketMakingBacktestEngine(
        initial_cash=config.initial_cash,
        max_position=config.max_position,
        order_size=config.order_size,
    )

    decisions = []
    market_orders = []
    mid_prices = []

    for _ in range(config.steps):
        (
            side,
            price,
            quantity,
            snapshot,
        ) = simulator.generate_market_event()

        if snapshot is None:
            continue

        decision = make_strategy_decision(
            mid_price=snapshot.mid_price,
            spread=snapshot.spread,
            signal=signal,
            position=engine.position,
            max_position=config.max_position,
            shift_factor=config.shift_factor,
            inventory_skew_factor=config.inventory_skew_factor,
        )

        decisions.append(decision)

        market_orders.append(
            (
                side,
                price,
                quantity,
            )
        )

        mid_prices.append(
            snapshot.mid_price
        )

        # Advance the simulator's market state using
        # the generated external order.
        simulator.exchange.submit_order(
            side,
            price,
            quantity,
        )

        simulator.step += 1

        simulator.reference_price = (
            simulator.exchange.get_mid_price()
            or simulator.reference_price
        )

    if not decisions:
        raise RuntimeError(
            "No valid market events were generated."
        )

    return engine.run(
        decisions=decisions,
        market_orders=market_orders,
        mid_prices=mid_prices,
    )