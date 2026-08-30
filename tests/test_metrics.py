import pytest

from src.market_maker.backtest.metrics import (
    calculate_metrics,
)


def test_metrics_calculate_return():
    metrics = calculate_metrics(
        initial_cash=10_000.0,
        pnl=100.0,
        trades=[],
        equity_curve=[
            10_000.0,
            10_050.0,
            10_100.0,
        ],
        final_position=10.0,
        max_position=100.0,
    )

    assert metrics.total_pnl == pytest.approx(100.0)
    assert metrics.return_pct == pytest.approx(0.01)


def test_metrics_calculate_drawdown():
    metrics = calculate_metrics(
        initial_cash=10_000.0,
        pnl=-100.0,
        trades=[],
        equity_curve=[
            10_000.0,
            10_200.0,
            10_100.0,
            10_050.0,
        ],
        final_position=0.0,
        max_position=100.0,
    )

    assert metrics.max_drawdown == pytest.approx(
        150.0
    )

    assert metrics.max_drawdown_pct == pytest.approx(
        150.0 / 10_200.0
    )


def test_inventory_utilization():
    metrics = calculate_metrics(
        initial_cash=10_000.0,
        pnl=0.0,
        trades=[],
        equity_curve=[
            10_000.0,
            10_001.0,
        ],
        final_position=-25.0,
        max_position=100.0,
    )

    assert metrics.inventory_utilization == pytest.approx(
        0.25
    )


def test_empty_equity_curve_rejected():
    with pytest.raises(ValueError):
        calculate_metrics(
            initial_cash=10_000.0,
            pnl=0.0,
            trades=[],
            equity_curve=[],
            final_position=0.0,
            max_position=100.0,
        )