from __future__ import annotations

from dataclasses import dataclass

from src.market_maker.backtest.market_making_engine import (
    MarketMakingBacktestEngine,
    MarketMakingResult,
)
from src.market_maker.ml.signal import TradingSignal
from src.market_maker.simulation.market import MarketSimulator
from src.market_maker.strategy.engine import make_strategy_decision


@dataclass(frozen=True)
class MarketMakingRunConfig:
    steps: int = 100
    initial_price: float = 100.0
    seed: int = 42
    initial_cash: float = 10_000.0
    max_position: float = 100.0
    order_size: float = 1.0
    shift_factor: float = 0.25
    inventory_skew_factor: float = 0.5


def run_market_making_backtest(
    signal: TradingSignal,
    config: MarketMakingRunConfig | None = None,
) -> MarketMakingResult:
    """
    Run an event-driven market-making backtest.

    The simulator and backtest engine share the same ExchangeEngine.

    Each market event is processed exactly once:

    1. Generate market event.
    2. Generate strategy decision.
    3. Place strategy quotes.
    4. Submit external market order.
    5. Process fills.
    6. Cancel remaining quotes.
    7. Record mark-to-market equity.
    """

    if config is None:
        config = MarketMakingRunConfig()

    if config.steps <= 0:
        raise ValueError("steps must be positive")

    simulator = MarketSimulator(
        initial_price=config.initial_price,
        seed=config.seed,
    )

    exchange = simulator.exchange

    engine = MarketMakingBacktestEngine(
        initial_cash=config.initial_cash,
        max_position=config.max_position,
        order_size=config.order_size,
        exchange=exchange,
    )

    equity_curve: list[float] = []

    for _ in range(config.steps):
        side, price, quantity, snapshot = simulator.generate_market_event()

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

        # Place our quotes first.
        strategy_order_ids = engine.place_quotes(
            decision=decision,
        )

        # Submit the external market order exactly once.
        engine.process_market_order(
            side=side,
            price=price,
            quantity=quantity,
            step=snapshot.step,
            strategy_order_ids=strategy_order_ids,
        )

        # Remove any unfilled quotes.
        engine.cancel_quotes(strategy_order_ids)

        # Update simulator state after the event.
        simulator.step += 1

        current_mid = exchange.get_mid_price()

        if current_mid is not None:
            simulator.reference_price = current_mid
            mid_price = current_mid
        else:
            mid_price = snapshot.mid_price

        equity_curve.append(
            engine.mark_to_market(mid_price)
        )

    if not equity_curve:
        raise ValueError(
            "Market-making backtest produced no market events."
        )

    final_mid_price = simulator.reference_price
    final_equity = engine.mark_to_market(final_mid_price)
    pnl = final_equity - config.initial_cash
    return_pct = pnl / config.initial_cash

    return MarketMakingResult(
        initial_cash=config.initial_cash,
        final_cash=engine.cash,
        final_position=engine.position,
        final_mid_price=final_mid_price,
        final_equity=final_equity,
        pnl=pnl,
        return_pct=return_pct,
        trades=tuple(engine.trades),
        equity_curve=tuple(equity_curve),
    )