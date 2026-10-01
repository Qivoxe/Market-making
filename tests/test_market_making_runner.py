from src.market_maker.backtest.market_making_runner import (
    MarketMakingRunConfig,
    run_market_making_backtest,
)
from src.market_maker.ml.signal import generate_signal


def test_market_making_runner() -> None:
    signal = generate_signal(
        down_probability=0.1,
        flat_probability=0.8,
        up_probability=0.1,
        threshold=0.6,
    )

    result = run_market_making_backtest(
        signal=signal,
        config=MarketMakingRunConfig(
            steps=25,
            seed=42,
        ),
    )

    assert result.initial_cash == 10_000.0
    assert len(result.equity_curve) > 0
    assert result.final_mid_price > 0