import pytest

from src.market_maker.backtest.compare import (
    ComparisonConfig,
    run_comparison,
)


def test_comparison_runs():
    result = run_comparison(
        ComparisonConfig(
            count_per_regime=10,
            initial_price=100.0,
            seed=42,
        )
    )

    assert result.baseline.name == (
        "Baseline Strategy"
    )

    assert result.ml.name == (
        "ML Strategy"
    )


def test_comparison_is_deterministic():
    config = ComparisonConfig(
        count_per_regime=10,
        initial_price=100.0,
        seed=42,
    )

    first = run_comparison(config)
    second = run_comparison(config)

    assert (
        first.baseline.metrics.total_pnl
        == pytest.approx(
            second.baseline.metrics.total_pnl
        )
    )

    assert (
        first.ml.metrics.total_pnl
        == pytest.approx(
            second.ml.metrics.total_pnl
        )
    )


def test_comparison_contains_risk_metrics():
    result = run_comparison(
        ComparisonConfig(
            count_per_regime=10,
            seed=42,
        )
    )

    assert (
        result.baseline.metrics.max_drawdown
        >= 0.0
    )

    assert (
        result.ml.metrics.max_drawdown
        >= 0.0
    )

    assert (
        result.baseline.metrics.trade_count
        >= 0
    )

    assert (
        result.ml.metrics.trade_count
        >= 0
    )