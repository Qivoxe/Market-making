from __future__ import annotations

from dataclasses import dataclass

from src.market_maker.backtest.engine import (
    BacktestEngine,
    BacktestResult,
)
from src.market_maker.data.generator import (
    generate_market_dataset,
)
from src.market_maker.ml.model import load_model
from src.market_maker.ml.predictor import predict_signal
from src.market_maker.strategy.engine import (
    make_strategy_decision,
)


@dataclass(frozen=True)
class MLBacktestConfig:
    count_per_regime: int = 100
    initial_price: float = 100.0
    seed: int | None = 42
    initial_cash: float = 10_000.0
    max_position: float = 100.0
    order_size: float = 1.0
    signal_threshold: float = 0.60
    model_path: str = "models/random_forest.joblib"


def run_ml_backtest(
    config: MLBacktestConfig | None = None,
) -> BacktestResult:
    if config is None:
        config = MLBacktestConfig()

    snapshots = generate_market_dataset(
        count_per_regime=config.count_per_regime,
        initial_price=config.initial_price,
        seed=config.seed,
    )

    if not snapshots:
        raise ValueError(
            "Market dataset must contain snapshots."
        )

    model = load_model(config.model_path)

    backtest = BacktestEngine(
        initial_cash=config.initial_cash,
        max_position=config.max_position,
        order_size=config.order_size,
    )

    decisions = []
    mid_prices = []

    for snapshot in snapshots:
        signal = predict_signal(
            model=model,
            mid_price=snapshot.mid_price,
            spread=snapshot.spread,
            bid_volume=snapshot.bid_volume,
            ask_volume=snapshot.ask_volume,
            imbalance=snapshot.imbalance,
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


if __name__ == "__main__":
    result = run_ml_backtest()

    print("ML Backtest")
    print("=" * 60)
    print(f"Initial Cash:   {result.initial_cash:.2f}")
    print(f"Final Cash:     {result.final_cash:.2f}")
    print(f"Final Position: {result.final_position:.2f}")
    print(f"Final Mid:      {result.final_mid_price:.4f}")
    print(f"Final Equity:   {result.final_equity:.2f}")
    print(f"PnL:            {result.pnl:.2f}")
    print(f"Return:         {result.return_pct * 100:.4f}%")
    print(f"Trades:         {len(result.trades)}")