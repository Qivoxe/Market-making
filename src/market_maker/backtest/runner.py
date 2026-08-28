from __future__ import annotations

from dataclasses import dataclass

from src.market_maker.backtest.engine import (
    BacktestEngine,
    BacktestResult,
)
from src.market_maker.data.generator import (
    generate_market_dataset,
)
from src.market_maker.ml.signal import (
    TradingSignal,
    generate_signal,
)
from src.market_maker.simulation.market import (
    MarketSnapshot,
)
from src.market_maker.strategy.engine import (
    make_strategy_decision,
)


@dataclass(frozen=True)
class BacktestConfig:
    count_per_regime: int = 100
    initial_price: float = 100.0
    seed: int | None = 42

    initial_cash: float = 10_000.0
    max_position: float = 100.0
    order_size: float = 1.0

    signal_threshold: float = 0.60


def run_backtest(
    config: BacktestConfig | None = None,
) -> BacktestResult:
    if config is None:
        config = BacktestConfig()

    snapshots = generate_market_dataset(
        count_per_regime=config.count_per_regime,
        initial_price=config.initial_price,
        seed=config.seed,
    )

    if len(snapshots) == 0:
        raise ValueError(
            "Market dataset must contain snapshots."
        )

    backtest = BacktestEngine(
        initial_cash=config.initial_cash,
        max_position=config.max_position,
        order_size=config.order_size,
    )

    decisions = []
    mid_prices = []

    for snapshot in snapshots:
        signal = _generate_baseline_signal(
            snapshot=snapshot,
            threshold=config.signal_threshold,
        )

        decision = make_strategy_decision(
            mid_price=snapshot.mid_price,
            spread=snapshot.spread,
            signal=signal,
            position=backtest.position,
            max_position=config.max_position,
        )

        decisions.append(decision)
        mid_prices.append(snapshot.mid_price)

    return backtest.run(
        decisions=decisions,
        mid_prices=mid_prices,
    )


def _generate_baseline_signal(
    snapshot: MarketSnapshot,
    threshold: float,
) -> TradingSignal:
    if threshold <= 0.0:
        raise ValueError(
            "Signal threshold must be greater than zero."
        )

    imbalance = snapshot.imbalance

    if imbalance >= threshold:
        return generate_signal(
            down_probability=0.05,
            flat_probability=0.15,
            up_probability=0.80,
            threshold=0.60,
        )

    if imbalance <= -threshold:
        return generate_signal(
            down_probability=0.80,
            flat_probability=0.15,
            up_probability=0.05,
            threshold=0.60,
        )

    return generate_signal(
        down_probability=0.10,
        flat_probability=0.80,
        up_probability=0.10,
        threshold=0.60,
    )