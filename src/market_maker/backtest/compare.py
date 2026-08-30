from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.market_maker.backtest.engine import (
    BacktestEngine,
    BacktestResult,
)
from src.market_maker.backtest.report import (
    StrategyComparison,
    build_strategy_report,
    compare_strategies,
)
from src.market_maker.data.generator import (
    generate_market_dataset,
)
from src.market_maker.ml.model import load_model
from src.market_maker.ml.predictor import predict_signal
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
class ComparisonConfig:
    count_per_regime: int = 100
    initial_price: float = 100.0
    seed: int | None = 42
    initial_cash: float = 10_000.0
    max_position: float = 100.0
    order_size: float = 1.0
    signal_threshold: float = 0.60
    model_path: str = "models/random_forest.joblib"


def run_strategy(
    snapshots: list[MarketSnapshot],
    config: ComparisonConfig,
    use_ml: bool,
    model: Any | None = None,
) -> BacktestResult:
    engine = BacktestEngine(
        initial_cash=config.initial_cash,
        max_position=config.max_position,
        order_size=config.order_size,
    )

    if use_ml and model is None:
        raise ValueError(
            "ML model is required for ML strategy."
        )

    decisions = []
    mid_prices = []

    for snapshot in snapshots:
        if use_ml:
            signal = _generate_ml_signal(
                model=model,
                snapshot=snapshot,
                threshold=config.signal_threshold,
            )
        else:
            signal = _generate_baseline_signal(
                snapshot=snapshot,
                threshold=config.signal_threshold,
            )

        decision = make_strategy_decision(
            mid_price=snapshot.mid_price,
            spread=snapshot.spread,
            signal=signal,
            position=engine.position,
            max_position=config.max_position,
        )

        decisions.append(decision)
        mid_prices.append(snapshot.mid_price)

    return engine.run(
        decisions=decisions,
        mid_prices=mid_prices,
    )


def run_comparison(
    config: ComparisonConfig | None = None,
) -> StrategyComparison:
    if config is None:
        config = ComparisonConfig()

    snapshots = generate_market_dataset(
        count_per_regime=config.count_per_regime,
        initial_price=config.initial_price,
        seed=config.seed,
    )

    if not snapshots:
        raise ValueError(
            "Market dataset must contain snapshots."
        )

    model = load_model(
        config.model_path
    )

    baseline_result = run_strategy(
        snapshots=snapshots,
        config=config,
        use_ml=False,
    )

    ml_result = run_strategy(
        snapshots=snapshots,
        config=config,
        use_ml=True,
        model=model,
    )

    baseline_report = build_strategy_report(
        name="Baseline Strategy",
        initial_cash=baseline_result.initial_cash,
        pnl=baseline_result.pnl,
        trades=baseline_result.trades,
        equity_curve=baseline_result.equity_curve,
        final_position=baseline_result.final_position,
        max_position=config.max_position,
    )

    ml_report = build_strategy_report(
        name="ML Strategy",
        initial_cash=ml_result.initial_cash,
        pnl=ml_result.pnl,
        trades=ml_result.trades,
        equity_curve=ml_result.equity_curve,
        final_position=ml_result.final_position,
        max_position=config.max_position,
    )

    return compare_strategies(
        baseline=baseline_report,
        ml=ml_report,
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


def _generate_ml_signal(
    model: Any,
    snapshot: MarketSnapshot,
    threshold: float,
) -> TradingSignal:
    return predict_signal(
        model=model,
        mid_price=snapshot.mid_price,
        spread=snapshot.spread,
        bid_volume=snapshot.bid_volume,
        ask_volume=snapshot.ask_volume,
        imbalance=snapshot.imbalance,
        threshold=threshold,
    )